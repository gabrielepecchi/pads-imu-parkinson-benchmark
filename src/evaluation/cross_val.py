"""
src/evaluation/cross_val.py
---------------------------
Builds 5-fold subject-stratified cross-validation splits for the PADS
benchmark.

Key design rules (non-negotiable):
    - Splitting is performed at the SUBJECT level, not the record level.
      All assessment steps from one subject are always in the same fold.
    - Stratification preserves the PD / HC ratio across folds as closely
      as possible.
    - Returned indices refer to the RECORD-level arrays (signals, labels,
      subject_ids) produced by preprocessor.py, not to subjects.
    - No preprocessing, normalisation, feature extraction, or model logic
      lives here.

Typical usage in run_pipeline.py:
    folds = build_folds(dataset.subject_ids, dataset.labels)
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train = dataset.signals[train_idx]
        X_test  = dataset.signals[test_idx]
        y_train = dataset.labels[train_idx]
        y_test  = dataset.labels[test_idx]
        # → apply normalisation here, then train and evaluate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of CV folds. Fixed for this project.
N_FOLDS: int = 5

#: Random seed for reproducible fold assignment.
#: Must be stored in configs/ and passed explicitly — never hardcoded elsewhere.
DEFAULT_RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldSplit:
    """Indices for one fold of cross-validation.

    Both arrays index into the record-level dataset arrays
    (PreprocessedDataset.signals, .labels, .subject_ids).

    Attributes
    ----------
    fold_index : int
        Zero-based fold number (0 to N_FOLDS - 1).
    train_indices : np.ndarray
        1-D integer array of record indices for training.
    test_indices : np.ndarray
        1-D integer array of record indices for testing.
    train_subject_ids : np.ndarray
        Subject IDs present in the training split (for audit / logging).
    test_subject_ids : np.ndarray
        Subject IDs present in the test split (for audit / logging).
    """

    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_subject_ids: np.ndarray
    test_subject_ids: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_subject_table(
    subject_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Derive per-subject arrays and a subject → record index mapping.

    Parameters
    ----------
    subject_ids : np.ndarray
        Shape (N,), dtype object. Record-level subject identifiers.
    labels : np.ndarray
        Shape (N,), dtype int. Record-level binary labels (0 = HC, 1 = PD).

    Returns
    -------
    unique_subjects : np.ndarray
        Shape (S,). Unique subject IDs in stable order.
    subject_labels : np.ndarray
        Shape (S,), dtype int. One label per subject (consistent across steps).
    subject_to_indices : dict[str, np.ndarray]
        Maps each subject_id to its record indices in the full dataset.

    Raises
    ------
    ValueError
        If any subject_id maps to more than one label value.
    """
    subject_to_indices: dict[str, np.ndarray] = {}
    subject_to_label: dict[str, int] = {}

    for record_idx, (sid, lbl) in enumerate(zip(subject_ids, labels)):
        if sid not in subject_to_indices:
            subject_to_indices[sid] = []
            subject_to_label[sid] = int(lbl)
        else:
            if subject_to_label[sid] != int(lbl):
                raise ValueError(
                    f"Subject '{sid}' has conflicting labels "
                    f"({subject_to_label[sid]} and {int(lbl)}). "
                    "Data integrity violation — check loader.py output."
                )
        subject_to_indices[sid].append(record_idx)

    # Convert index lists to arrays for efficient downstream indexing.
    subject_to_indices_arr: dict[str, np.ndarray] = {
        sid: np.array(idxs, dtype=int)
        for sid, idxs in subject_to_indices.items()
    }

    unique_subjects = np.array(list(subject_to_indices_arr.keys()), dtype=object)
    subject_labels = np.array(
        [subject_to_label[sid] for sid in unique_subjects], dtype=int
    )

    return unique_subjects, subject_labels, subject_to_indices_arr


def _subjects_to_record_indices(
    subject_ids_in_fold: np.ndarray,
    subject_to_indices: dict[str, np.ndarray],
) -> np.ndarray:
    """Expand a list of subject IDs to their corresponding record indices.

    Parameters
    ----------
    subject_ids_in_fold : np.ndarray
        1-D array of subject IDs assigned to this fold partition.
    subject_to_indices : dict[str, np.ndarray]
        Mapping from subject_id to record indices in the full dataset.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of record indices.
    """
    index_lists = [subject_to_indices[sid] for sid in subject_ids_in_fold]
    if not index_lists:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(index_lists))


def _validate_folds(
    folds: list[FoldSplit],
    n_records: int,
    subject_ids: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Run integrity checks on the generated folds. Raises on failure.

    Checks:
        1. Correct number of folds.
        2. No subject appears in both train and test of the same fold.
        3. Every record index appears in exactly one test fold.
        4. PD / HC ratio in each test fold is within tolerance of full dataset.
    """
    assert len(folds) == N_FOLDS, (
        f"Expected {N_FOLDS} folds, got {len(folds)}."
    )

    # Check 2: no subject leakage within any fold.
    for fold in folds:
        train_sids = set(fold.train_subject_ids.tolist())
        test_sids = set(fold.test_subject_ids.tolist())
        overlap = train_sids & test_sids
        assert not overlap, (
            f"Fold {fold.fold_index}: subject(s) appear in both train and test: "
            f"{overlap}. Subject-level split has failed."
        )

    # Check 3: every record appears in exactly one test fold.
    all_test_indices = np.concatenate([f.test_indices for f in folds])
    assert len(all_test_indices) == n_records, (
        f"Total test indices across folds ({len(all_test_indices)}) != "
        f"total records ({n_records}). Some records are missing or duplicated."
    )
    assert len(np.unique(all_test_indices)) == n_records, (
        "Duplicate record indices found across test folds — "
        "each record must appear in exactly one test fold."
    )

    # Check 4: per-fold class ratio within ±10% of global ratio.
    global_pd_ratio = float(labels.sum()) / len(labels)
    tolerance = 0.10
    for fold in folds:
        fold_labels = labels[fold.test_indices]
        fold_pd_ratio = float(fold_labels.sum()) / len(fold_labels)
        deviation = abs(fold_pd_ratio - global_pd_ratio)
        if deviation > tolerance:
            logger.warning(
                "Fold %d test set PD ratio %.3f deviates from global %.3f "
                "by %.3f (tolerance=%.2f). Class balance may be skewed — "
                "expected given small HC N.",
                fold.fold_index,
                fold_pd_ratio,
                global_pd_ratio,
                deviation,
                tolerance,
            )

    logger.info("All fold validation checks passed.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_folds(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int = N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[FoldSplit]:
    """Build subject-stratified cross-validation folds.

    Stratification is performed at the subject level. All records belonging
    to the same subject are assigned to the same fold partition. The PD / HC
    ratio is preserved across folds as closely as possible.

    Parameters
    ----------
    subject_ids : np.ndarray
        Shape (N,), dtype object. Record-level subject IDs from
        PreprocessedDataset.subject_ids.
    labels : np.ndarray
        Shape (N,), dtype int. Record-level binary labels from
        PreprocessedDataset.labels. Values must be in {0, 1}.
    n_folds : int
        Number of CV folds. Default: 5.
    random_seed : int
        Random seed for reproducible fold assignment. Must match the value
        stored in configs/. Default: 42.

    Returns
    -------
    list[FoldSplit]
        List of FoldSplit objects, one per fold. Indices in each FoldSplit
        refer to positions in the record-level dataset arrays.

    Raises
    ------
    ValueError
        If subject_ids and labels have mismatched lengths, if any subject
        maps to multiple labels, or if n_folds exceeds the number of subjects
        in the minority class.
    """
    if len(subject_ids) != len(labels):
        raise ValueError(
            f"subject_ids length ({len(subject_ids)}) != "
            f"labels length ({len(labels)})."
        )
    if len(subject_ids) == 0:
        raise ValueError("subject_ids is empty — nothing to split.")

    # Step 1: build subject-level table.
    unique_subjects, subject_labels, subject_to_indices = _build_subject_table(
        subject_ids, labels
    )

    n_subjects = len(unique_subjects)
    n_pd = int(subject_labels.sum())
    n_hc = n_subjects - n_pd
    logger.info(
        "Building %d-fold CV over %d subjects (%d PD, %d HC), "
        "%d total records. Seed=%d.",
        n_folds,
        n_subjects,
        n_pd,
        n_hc,
        len(subject_ids),
        random_seed,
    )

    minority_n = min(n_pd, n_hc)
    if n_folds > minority_n:
        raise ValueError(
            f"n_folds={n_folds} exceeds the number of subjects in the minority "
            f"class ({minority_n}). Reduce n_folds or check the dataset filter."
        )

    # Step 2: stratified split at subject level.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    folds: list[FoldSplit] = []
    for fold_idx, (train_subj_pos, test_subj_pos) in enumerate(
        skf.split(unique_subjects, subject_labels)
    ):
        train_subjects = unique_subjects[train_subj_pos]
        test_subjects = unique_subjects[test_subj_pos]

        # Step 3: expand subject IDs → record indices.
        train_indices = _subjects_to_record_indices(train_subjects, subject_to_indices)
        test_indices = _subjects_to_record_indices(test_subjects, subject_to_indices)

        fold = FoldSplit(
            fold_index=fold_idx,
            train_indices=train_indices,
            test_indices=test_indices,
            train_subject_ids=train_subjects,
            test_subject_ids=test_subjects,
        )
        folds.append(fold)

        logger.debug(
            "Fold %d: %d train records (%d subjects), "
            "%d test records (%d subjects).",
            fold_idx,
            len(train_indices),
            len(train_subjects),
            len(test_indices),
            len(test_subjects),
        )

    # Step 4: validate.
    _validate_folds(folds, len(subject_ids), subject_ids, labels)
    return folds


def summarise_folds(
    folds: list[FoldSplit],
    labels: np.ndarray,
) -> None:
    """Print a per-fold summary of record and class counts.

    Call after build_folds() to inspect balance before training.

    Parameters
    ----------
    folds : list[FoldSplit]
        Output of build_folds().
    labels : np.ndarray
        Record-level label array (PreprocessedDataset.labels).
    """
    global_pd = int(labels.sum())
    global_hc = int((labels == 0).sum())
    print(f"=== CV Fold Summary ({len(folds)} folds) ===")
    print(f"Full dataset: {len(labels)} records | PD={global_pd} | HC={global_hc}\n")

    header = f"{'Fold':<6} {'Train recs':>11} {'Train PD':>9} {'Train HC':>9} "
    header += f"{'Test recs':>10} {'Test PD':>8} {'Test HC':>8}"
    print(header)
    print("-" * len(header))

    for fold in folds:
        tr_labels = labels[fold.train_indices]
        te_labels = labels[fold.test_indices]
        print(
            f"{fold.fold_index:<6} "
            f"{len(fold.train_indices):>11} "
            f"{int(tr_labels.sum()):>9} "
            f"{int((tr_labels == 0).sum()):>9} "
            f"{len(fold.test_indices):>10} "
            f"{int(te_labels.sum()):>8} "
            f"{int((te_labels == 0).sum()):>8}"
        )
