"""
src/evaluation/metrics.py
--------------------------
Computes and aggregates evaluation metrics for PD / HC binary classification
on the PADS benchmark.

Contract
--------
- Receives per-fold ground-truth labels, predicted probabilities, and hard
  predicted labels. All inputs are validated before computation.
- Metrics computed per fold: Balanced Accuracy, AUROC, Sensitivity (Recall
  for PD, label=1), Specificity (Recall for HC, label=0).
- Aggregation across folds: mean ± standard deviation (ddof=1, unbiased).
- Aggregated confusion matrix: summed across all folds, then normalised by
  true-class totals (row-normalised), yielding a rate matrix.
- No CV logic, no model code, no preprocessing or feature extraction here.

Positive class: PD = label 1.  Negative class: HC = label 0.

Typical usage in run_pipeline.py:
    collector = MetricsCollector()
    for fold_idx, (y_true, proba, y_pred) in enumerate(fold_outputs):
        fold_result = compute_fold_metrics(fold_idx, y_true, proba, y_pred)
        collector.add_fold(fold_result)
    summary = collector.aggregate()
    print_summary(summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Index of the positive class (PD) in the probability matrix column.
PD_CLASS_INDEX: int = 1

#: Expected class values.
EXPECTED_CLASSES: frozenset[int] = frozenset({0, 1})

#: Standard deviation denominator: ddof=1 for unbiased sample SD.
SD_DDOF: int = 1


# ---------------------------------------------------------------------------
# Per-fold result
# ---------------------------------------------------------------------------

@dataclass
class FoldMetrics:
    """Metrics for a single cross-validation fold.

    Attributes
    ----------
    fold_index : int
        Zero-based fold number.
    balanced_accuracy : float
        Balanced accuracy: mean of sensitivity and specificity.
        Equivalent to ``(sensitivity + specificity) / 2``.
    auroc : float
        Area under the ROC curve, computed from predicted probabilities.
    sensitivity : float
        True positive rate for the PD class (label=1):
        ``TP / (TP + FN)``.
    specificity : float
        True negative rate for the HC class (label=0):
        ``TN / (TN + FP)``.
    confusion : np.ndarray
        Shape (2, 2), dtype int. Raw confusion matrix for this fold.
        Rows = true class, columns = predicted class.
        Layout: [[TN, FP], [FN, TP]].
    n_samples : int
        Total number of test samples in this fold.
    n_pd : int
        Number of PD samples (label=1) in this fold.
    n_hc : int
        Number of HC samples (label=0) in this fold.
    """

    fold_index: int
    balanced_accuracy: float
    auroc: float
    sensitivity: float
    specificity: float
    confusion: np.ndarray
    n_samples: int = field(init=False)
    n_pd: int = field(init=False)
    n_hc: int = field(init=False)

    def __post_init__(self) -> None:
        tn, fp, fn, tp = self.confusion.ravel()
        self.n_pd = int(fn + tp)
        self.n_hc = int(tn + fp)
        self.n_samples = self.n_pd + self.n_hc


# ---------------------------------------------------------------------------
# Aggregated result
# ---------------------------------------------------------------------------

@dataclass
class AggregatedMetrics:
    """Aggregated metrics across all cross-validation folds.

    All ``mean_*`` and ``sd_*`` attributes are computed from the per-fold
    scalar values using ddof=1 standard deviation.

    Attributes
    ----------
    mean_balanced_accuracy : float
    sd_balanced_accuracy : float
    mean_auroc : float
    sd_auroc : float
    mean_sensitivity : float
    sd_sensitivity : float
    mean_specificity : float
    sd_specificity : float
    confusion_sum : np.ndarray
        Shape (2, 2), dtype int. Confusion matrix summed across all folds.
        Layout: [[TN, FP], [FN, TP]].
    confusion_normalised : np.ndarray
        Shape (2, 2), dtype float64. Row-normalised confusion matrix:
        each row divided by its true-class total (normalised by support).
        Row 0 = HC rates: [TNR (specificity), FPR].
        Row 1 = PD rates: [FNR, TPR (sensitivity)].
    n_folds : int
        Number of folds included in the aggregation.
    fold_metrics : list[FoldMetrics]
        Per-fold results in fold-index order.
    """

    mean_balanced_accuracy: float
    sd_balanced_accuracy: float
    mean_auroc: float
    sd_auroc: float
    mean_sensitivity: float
    sd_sensitivity: float
    mean_specificity: float
    sd_specificity: float
    confusion_sum: np.ndarray
    confusion_normalised: np.ndarray
    n_folds: int
    fold_metrics: list[FoldMetrics]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_fold_inputs(
    fold_index: int,
    y_true: np.ndarray,
    proba: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Validate all inputs for a single fold. Raises on failure.

    Parameters
    ----------
    fold_index : int
        Fold number, used only for error messages.
    y_true : np.ndarray
        Shape (N,), dtype int. Ground-truth binary labels.
    proba : np.ndarray
        Shape (N, 2), dtype float64. Predicted class probabilities.
    y_pred : np.ndarray
        Shape (N,), dtype int. Hard predicted labels.

    Raises
    ------
    TypeError
        If any input is not a numpy ndarray.
    ValueError
        If shapes are inconsistent, label values are outside {0, 1},
        both classes are not present in y_true, or probabilities are
        invalid (non-finite, outside [0, 1], rows not summing to ~1).
    """
    tag = f"Fold {fold_index}"

    for name, arr in (("y_true", y_true), ("y_pred", y_pred), ("proba", proba)):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"{tag}: {name} must be a numpy ndarray, "
                f"got {type(arr).__name__}."
            )

    n = len(y_true)
    if n == 0:
        raise ValueError(f"{tag}: y_true is empty.")

    if y_true.ndim != 1:
        raise ValueError(
            f"{tag}: y_true must be 1-D, got shape {y_true.shape}."
        )
    if y_pred.ndim != 1:
        raise ValueError(
            f"{tag}: y_pred must be 1-D, got shape {y_pred.shape}."
        )
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(
            f"{tag}: proba must be shape (N, 2), got {proba.shape}."
        )
    if len(y_pred) != n:
        raise ValueError(
            f"{tag}: y_true length {n} != y_pred length {len(y_pred)}."
        )
    if proba.shape[0] != n:
        raise ValueError(
            f"{tag}: y_true length {n} != proba rows {proba.shape[0]}."
        )

    unique_true = set(y_true.tolist())
    if not unique_true.issubset(EXPECTED_CLASSES):
        raise ValueError(
            f"{tag}: y_true contains unexpected values: "
            f"{unique_true - EXPECTED_CLASSES}. Expected {{0, 1}}."
        )
    if unique_true != EXPECTED_CLASSES:
        raise ValueError(
            f"{tag}: y_true must contain both classes {{0, 1}}. "
            f"Found only: {unique_true}. "
            "AUROC and per-class metrics are undefined for single-class folds."
        )

    unique_pred = set(y_pred.tolist())
    if not unique_pred.issubset(EXPECTED_CLASSES):
        raise ValueError(
            f"{tag}: y_pred contains unexpected values: "
            f"{unique_pred - EXPECTED_CLASSES}. Expected subset of {{0, 1}}."
        )

    if not np.isfinite(proba).all():
        raise ValueError(f"{tag}: proba contains NaN or Inf values.")
    if (proba < 0.0).any() or (proba > 1.0).any():
        raise ValueError(
            f"{tag}: proba values must lie in [0, 1]. "
            f"Found min={proba.min():.6f}, max={proba.max():.6f}."
        )
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        bad = int(np.sum(~np.isclose(row_sums, 1.0, atol=1e-5)))
        raise ValueError(
            f"{tag}: {bad} row(s) in proba do not sum to 1.0 "
            f"(tolerance 1e-5). Expected valid probability simplex."
        )


def _validate_aggregation_inputs(fold_metrics_list: list[FoldMetrics]) -> None:
    """Validate the list of fold results before aggregation. Raises on failure."""
    if not fold_metrics_list:
        raise ValueError(
            "fold_metrics_list is empty — nothing to aggregate. "
            "Call add_fold() for each completed fold before aggregate()."
        )
    expected_indices = set(range(len(fold_metrics_list)))
    actual_indices = {fm.fold_index for fm in fold_metrics_list}
    if actual_indices != expected_indices:
        raise ValueError(
            f"fold_index values are not contiguous 0..{len(fold_metrics_list) - 1}. "
            f"Got: {sorted(actual_indices)}. Each fold must be added exactly once."
        )


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Compute sensitivity and specificity from binary labels.

    Sensitivity = TP / (TP + FN)  [true positive rate for PD]
    Specificity = TN / (TN + FP)  [true negative rate for HC]

    Parameters
    ----------
    y_true : np.ndarray
        Shape (N,), dtype int. Ground-truth labels in {0, 1}.
    y_pred : np.ndarray
        Shape (N,), dtype int. Predicted labels in {0, 1}.

    Returns
    -------
    tuple[float, float]
        (sensitivity, specificity). Both in [0.0, 1.0].
        Returns 0.0 for a class absent from y_true (undefined metric).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def compute_fold_metrics(
    fold_index: int,
    y_true: np.ndarray,
    proba: np.ndarray,
    y_pred: np.ndarray,
) -> FoldMetrics:
    """Compute all evaluation metrics for one cross-validation fold.

    Parameters
    ----------
    fold_index : int
        Zero-based fold number. Used for logging and result labelling.
    y_true : np.ndarray
        Shape (N,), dtype int. Ground-truth binary labels (0=HC, 1=PD).
        Must contain both classes.
    proba : np.ndarray
        Shape (N, 2), dtype float64. Predicted class probabilities from the
        model's predict_proba() output. Column 1 is the PD probability used
        for AUROC. Rows must sum to 1.0.
    y_pred : np.ndarray
        Shape (N,), dtype int. Hard predicted labels in {0, 1}, typically
        argmax(proba, axis=1).

    Returns
    -------
    FoldMetrics
        Scalar metrics for this fold and the raw confusion matrix.

    Raises
    ------
    TypeError
        If any input is not a numpy ndarray.
    ValueError
        If inputs are invalid (see _validate_fold_inputs for details).
    """
    _validate_fold_inputs(fold_index, y_true, proba, y_pred)

    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, proba[:, PD_CLASS_INDEX]))
    sensitivity, specificity = _compute_sensitivity_specificity(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    result = FoldMetrics(
        fold_index=fold_index,
        balanced_accuracy=balanced_acc,
        auroc=auroc,
        sensitivity=sensitivity,
        specificity=specificity,
        confusion=cm,
    )

    logger.info(
        "Fold %d metrics: BalAcc=%.4f, AUROC=%.4f, "
        "Sens=%.4f, Spec=%.4f | N=%d (PD=%d, HC=%d).",
        fold_index,
        balanced_acc,
        auroc,
        sensitivity,
        specificity,
        result.n_samples,
        result.n_pd,
        result.n_hc,
    )
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_fold_metrics(fold_metrics_list: list[FoldMetrics]) -> AggregatedMetrics:
    """Aggregate per-fold metrics into means, SDs, and a summed confusion matrix.

    Standard deviation uses ddof=1 (unbiased sample SD). With N_FOLDS=5,
    this divides by 4. For a single fold, SD is nan (undefined); the
    caller should check n_folds before interpreting SD values.

    Confusion matrix normalisation is performed on the summed matrix, not
    per-fold, to give stable rate estimates from the full test population.

    Parameters
    ----------
    fold_metrics_list : list[FoldMetrics]
        Per-fold results from compute_fold_metrics(), one entry per fold.
        Must be non-empty; fold_index values must be contiguous from 0.

    Returns
    -------
    AggregatedMetrics
        Mean ± SD for each scalar metric, summed and normalised confusion
        matrix, and the original per-fold results.

    Raises
    ------
    ValueError
        If fold_metrics_list is empty or fold_index values are inconsistent.
    """
    _validate_aggregation_inputs(fold_metrics_list)

    # Sort by fold_index for deterministic ordering.
    sorted_folds = sorted(fold_metrics_list, key=lambda fm: fm.fold_index)

    bal_accs = np.array([fm.balanced_accuracy for fm in sorted_folds])
    aurocs = np.array([fm.auroc for fm in sorted_folds])
    sensitivities = np.array([fm.sensitivity for fm in sorted_folds])
    specificities = np.array([fm.specificity for fm in sorted_folds])

    confusion_sum = np.sum(
        np.stack([fm.confusion for fm in sorted_folds], axis=0), axis=0
    ).astype(int)

    # Row-normalise by true-class totals (row sums).
    row_totals = confusion_sum.sum(axis=1, keepdims=True).astype(float)
    # Guard against zero-row edge case (should not occur with validated folds).
    row_totals_safe = np.where(row_totals == 0, 1.0, row_totals)
    confusion_normalised = confusion_sum.astype(float) / row_totals_safe

    n_folds = len(sorted_folds)
    ddof = SD_DDOF if n_folds > 1 else 0

    summary = AggregatedMetrics(
        mean_balanced_accuracy=float(np.mean(bal_accs)),
        sd_balanced_accuracy=float(np.std(bal_accs, ddof=ddof)),
        mean_auroc=float(np.mean(aurocs)),
        sd_auroc=float(np.std(aurocs, ddof=ddof)),
        mean_sensitivity=float(np.mean(sensitivities)),
        sd_sensitivity=float(np.std(sensitivities, ddof=ddof)),
        mean_specificity=float(np.mean(specificities)),
        sd_specificity=float(np.std(specificities, ddof=ddof)),
        confusion_sum=confusion_sum,
        confusion_normalised=confusion_normalised,
        n_folds=n_folds,
        fold_metrics=sorted_folds,
    )

    logger.info(
        "Aggregated metrics (%d folds): "
        "BalAcc=%.4f±%.4f, AUROC=%.4f±%.4f, "
        "Sens=%.4f±%.4f, Spec=%.4f±%.4f.",
        n_folds,
        summary.mean_balanced_accuracy,
        summary.sd_balanced_accuracy,
        summary.mean_auroc,
        summary.sd_auroc,
        summary.mean_sensitivity,
        summary.sd_sensitivity,
        summary.mean_specificity,
        summary.sd_specificity,
    )
    return summary


# ---------------------------------------------------------------------------
# Collector (stateful accumulator for run_pipeline.py)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Accumulates per-fold metrics during a cross-validation run.

    Designed to be instantiated once per model per pipeline run. After all
    folds have been processed, call aggregate() to obtain AggregatedMetrics.

    Example
    -------
    >>> collector = MetricsCollector()
    >>> for fold_idx, (y_true, proba, y_pred) in enumerate(fold_outputs):
    ...     fold_result = compute_fold_metrics(fold_idx, y_true, proba, y_pred)
    ...     collector.add_fold(fold_result)
    >>> summary = collector.aggregate()
    """

    def __init__(self) -> None:
        self._folds: list[FoldMetrics] = []

    def add_fold(self, fold_metrics: FoldMetrics) -> None:
        """Append a completed FoldMetrics result.

        Parameters
        ----------
        fold_metrics : FoldMetrics
            Result from compute_fold_metrics() for one fold.

        Raises
        ------
        TypeError
            If fold_metrics is not a FoldMetrics instance.
        ValueError
            If a FoldMetrics with the same fold_index has already been added.
        """
        if not isinstance(fold_metrics, FoldMetrics):
            raise TypeError(
                f"Expected FoldMetrics, got {type(fold_metrics).__name__}."
            )
        existing_indices = {fm.fold_index for fm in self._folds}
        if fold_metrics.fold_index in existing_indices:
            raise ValueError(
                f"Fold {fold_metrics.fold_index} has already been added. "
                "Each fold must be added exactly once."
            )
        self._folds.append(fold_metrics)
        logger.debug(
            "MetricsCollector: added fold %d (%d/%d folds so far).",
            fold_metrics.fold_index,
            len(self._folds),
            len(self._folds),  # total unknown until aggregate(); log as running count
        )

    def aggregate(self) -> AggregatedMetrics:
        """Aggregate all added folds and return AggregatedMetrics.

        Returns
        -------
        AggregatedMetrics
            Full aggregation from aggregate_fold_metrics().

        Raises
        ------
        ValueError
            If no folds have been added or fold indices are inconsistent.
        """
        return aggregate_fold_metrics(self._folds)

    @property
    def n_folds_added(self) -> int:
        """Number of folds added so far."""
        return len(self._folds)


# ---------------------------------------------------------------------------
# Display utility
# ---------------------------------------------------------------------------

def print_summary(summary: AggregatedMetrics, model_name: str = "") -> None:
    """Print a formatted summary of aggregated metrics to stdout.

    Intended for quick inspection during development and pipeline runs.
    Structured logging via the metrics logger is always emitted; this
    function provides a human-readable table in addition.

    Parameters
    ----------
    summary : AggregatedMetrics
        Output of aggregate_fold_metrics() or MetricsCollector.aggregate().
    model_name : str
        Optional model label printed in the header. Default: empty string.
    """
    header = f"=== Metrics Summary: {model_name} ({summary.n_folds} folds) ==="
    print(header)
    print(
        f"  Balanced Accuracy : {summary.mean_balanced_accuracy:.4f} "
        f"± {summary.sd_balanced_accuracy:.4f}"
    )
    print(
        f"  AUROC             : {summary.mean_auroc:.4f} "
        f"± {summary.sd_auroc:.4f}"
    )
    print(
        f"  Sensitivity (PD)  : {summary.mean_sensitivity:.4f} "
        f"± {summary.sd_sensitivity:.4f}"
    )
    print(
        f"  Specificity (HC)  : {summary.mean_specificity:.4f} "
        f"± {summary.sd_specificity:.4f}"
    )
    print()
    print("  Confusion matrix (summed, row-normalised):")
    print("              Pred HC   Pred PD")
    tn_r, fp_r = summary.confusion_normalised[0]
    fn_r, tp_r = summary.confusion_normalised[1]
    print(f"  True HC  :  {tn_r:.4f}    {fp_r:.4f}")
    print(f"  True PD  :  {fn_r:.4f}    {tp_r:.4f}")
    print()
    print("  Confusion matrix (raw counts summed across folds):")
    print("              Pred HC   Pred PD")
    tn, fp = summary.confusion_sum[0]
    fn, tp = summary.confusion_sum[1]
    print(f"  True HC  :  {tn:6d}    {fp:6d}")
    print(f"  True PD  :  {fn:6d}    {tp:6d}")
