"""
tests/test_cross_val.py
-----------------------
Tests for subject-level leakage prevention in cross_val.build_folds().
All data is synthetic; no real patient files are required.
"""

import numpy as np
import pytest

from src.evaluation.cross_val import build_folds, N_FOLDS


def _make_subject_arrays(n_pd: int = 10, n_hc: int = 10, steps_per_subject: int = 3):
    """Return (subject_ids, labels) record-level arrays for synthetic subjects."""
    subject_ids = []
    labels = []
    for i in range(n_pd):
        sid = f"pd_{i:03d}"
        subject_ids.extend([sid] * steps_per_subject)
        labels.extend([1] * steps_per_subject)
    for i in range(n_hc):
        sid = f"hc_{i:03d}"
        subject_ids.extend([sid] * steps_per_subject)
        labels.extend([0] * steps_per_subject)
    return np.array(subject_ids, dtype=object), np.array(labels, dtype=int)


def test_no_subject_leakage():
    """No subject may appear in both train and test within any fold."""
    subject_ids, labels = _make_subject_arrays(n_pd=10, n_hc=10, steps_per_subject=3)
    folds = build_folds(subject_ids, labels)

    for fold in folds:
        train_sids = set(subject_ids[fold.train_indices].tolist())
        test_sids = set(subject_ids[fold.test_indices].tolist())
        overlap = train_sids & test_sids
        assert not overlap, (
            f"Fold {fold.fold_index}: subjects appear in both train and test: {overlap}"
        )


def test_every_record_in_exactly_one_test_fold():
    """Every record index must appear in exactly one test fold."""
    subject_ids, labels = _make_subject_arrays(n_pd=10, n_hc=10, steps_per_subject=3)
    folds = build_folds(subject_ids, labels)

    n_records = len(subject_ids)
    all_test = np.concatenate([f.test_indices for f in folds])
    assert len(all_test) == n_records
    assert len(np.unique(all_test)) == n_records


def test_correct_number_of_folds():
    subject_ids, labels = _make_subject_arrays(n_pd=10, n_hc=10, steps_per_subject=2)
    folds = build_folds(subject_ids, labels)
    assert len(folds) == N_FOLDS


def test_train_test_indices_disjoint():
    """Train and test record indices must not overlap within any fold."""
    subject_ids, labels = _make_subject_arrays(n_pd=10, n_hc=10, steps_per_subject=4)
    folds = build_folds(subject_ids, labels)

    for fold in folds:
        overlap = set(fold.train_indices.tolist()) & set(fold.test_indices.tolist())
        assert not overlap, (
            f"Fold {fold.fold_index}: record indices appear in both train and test."
        )


def test_conflicting_subject_labels_raises():
    """build_folds must raise if a subject_id maps to two different labels."""
    subject_ids = np.array(["s001", "s001", "s002", "s002"], dtype=object)
    labels = np.array([0, 1, 0, 0], dtype=int)
    with pytest.raises((ValueError, AssertionError)):
        build_folds(subject_ids, labels)


def test_fold_subject_ids_audit_arrays():
    """FoldSplit.train_subject_ids and test_subject_ids must not overlap."""
    subject_ids, labels = _make_subject_arrays(n_pd=8, n_hc=8, steps_per_subject=2)
    folds = build_folds(subject_ids, labels)

    for fold in folds:
        train_set = set(fold.train_subject_ids.tolist())
        test_set = set(fold.test_subject_ids.tolist())
        assert not (train_set & test_set)
