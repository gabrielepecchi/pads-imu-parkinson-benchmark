"""
run_pipeline.py
---------------
Orchestrates the full PADS benchmark pipeline:

    load → preprocess → build folds → per-fold normalisation →
    feature extraction → train/evaluate LR → train/evaluate RF →
    train/evaluate CNN1D → aggregate metrics → save results

Normalisation contract
----------------------
- Feature normalisation (LR / RF): z-score per feature.
  StandardScaler is fit on X_train_features ONLY and applied to both
  X_train_features and X_test_features within each fold.
- Sequence normalisation (CNN1D): z-score per channel. Mean and std are
  computed from the valid (non-padded) timesteps of X_train_seq ONLY,
  using valid_lengths_train to exclude padding zeros. Statistics are then
  broadcast-applied to both train and test sequences within each fold.

Config contract
---------------
All parameters are supplied via a PipelineConfig dataclass or a YAML file
path passed on the command line. No dataset paths or hyperparameters are
hardcoded in this module.

Results
-------
Saved to <results_dir>/ (default: results/):
    metrics_summary.csv       — mean ± SD per model, all metrics
    per_fold_metrics.csv      — per-fold scalar metrics for all models
    confusion_<model>.npy     — raw summed confusion matrix (2×2 int)
    confusion_norm_<model>.npy — row-normalised confusion matrix (2×2 float)

Usage
-----
    # Programmatic
    config = PipelineConfig(data_root="data/pads")
    results = run_pipeline(config)

    # CLI
    python run_pipeline.py --data-root data/pads --results-dir results/
    python run_pipeline.py --config configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_pads
from src.data.preprocessor import preprocess_records
from src.evaluation.cross_val import build_folds, summarise_folds
from src.evaluation.metrics import (
    AggregatedMetrics,
    MetricsCollector,
    compute_fold_metrics,
    print_summary,
)
from src.features.extractor import extract_features
from src.models.cnn1d import CNN1DModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All pipeline parameters. No defaults contain dataset paths.

    Parameters
    ----------
    data_root : str
        Path to the root of the PADS dataset (must contain patients/ and
        movement/timeseries/).
    results_dir : str
        Directory where all output files will be written. Created if absent.
        Default: "results".
    n_folds : int
        Number of cross-validation folds. Default: 5.
    random_seed : int
        Global random seed for fold construction and all model defaults.
        Default: 42.
    sampling_rate_hz : float
        IMU sampling rate in Hz. Default: 100.0.
    highpass_cutoff_hz : float
        High-pass filter cutoff in Hz for accelerometer channels. Default: 0.5.
    filter_order : int
        Butterworth filter order. Default: 4.
    lr_C : float
        Logistic Regression inverse regularisation strength. Default: 1.0.
    lr_penalty : str
        Logistic Regression regularisation type. Default: "l2".
    lr_max_iter : int
        Logistic Regression max solver iterations. Default: 1000.
    rf_n_estimators : int
        Number of trees in the Random Forest. Default: 500.
    rf_max_features : str
        RF max features per split. Default: "sqrt".
    cnn_epochs : int
        CNN1D training epochs per fold. Default: 50.
    cnn_batch_size : int
        CNN1D mini-batch size. Default: 32.
    cnn_lr : float
        CNN1D Adam learning rate. Default: 1e-3.
    cnn_dropout : float
        CNN1D dropout probability. Default: 0.5.
    cnn_device : str | None
        PyTorch device string. None = auto-detect. Default: None.
    run_lr : bool
        Whether to run Logistic Regression. Default: True.
    run_rf : bool
        Whether to run Random Forest. Default: True.
    run_cnn : bool
        Whether to run CNN1D. Default: True.
    """

    data_root: str
    results_dir: str = "results"
    n_folds: int = 5
    random_seed: int = 42

    # Preprocessing
    sampling_rate_hz: float = 100.0
    highpass_cutoff_hz: float = 0.5
    filter_order: int = 4

    # Logistic Regression
    lr_C: float = 1.0
    lr_penalty: str = "l2"
    lr_max_iter: int = 1000

    # Random Forest
    rf_n_estimators: int = 500
    rf_max_features: str = "sqrt"

    # CNN1D
    cnn_epochs: int = 50
    cnn_batch_size: int = 32
    cnn_lr: float = 1e-3
    cnn_dropout: float = 0.5
    cnn_device: str | None = None

    # Model toggles
    run_lr: bool = True
    run_rf: bool = True
    run_cnn: bool = True


def _config_from_yaml(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file.

    The YAML must contain a mapping whose keys match PipelineConfig field
    names. Unknown keys raise ValueError; missing keys fall back to defaults
    (data_root is required).

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML contains keys not recognised by PipelineConfig, or if
        data_root is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    valid_keys = {f.name for f in PipelineConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {unknown}. "
            f"Valid keys: {valid_keys}."
        )
    if "data_root" not in raw:
        raise ValueError(
            f"Config file {path} must specify 'data_root'."
        )

    return PipelineConfig(**raw)


def _validate_config(cfg: PipelineConfig) -> None:
    """Raise ValueError for obviously invalid config values."""
    if not cfg.data_root:
        raise ValueError("data_root must be a non-empty string.")
    if cfg.n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {cfg.n_folds}.")
    if cfg.sampling_rate_hz <= 0:
        raise ValueError(f"sampling_rate_hz must be > 0, got {cfg.sampling_rate_hz}.")
    if not (cfg.run_lr or cfg.run_rf or cfg.run_cnn):
        raise ValueError("At least one model must be enabled (run_lr/run_rf/run_cnn).")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _fit_feature_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training feature matrix.

    Parameters
    ----------
    X_train : np.ndarray
        Shape (N_train, F), dtype float64.

    Returns
    -------
    StandardScaler
        Fitted scaler; apply with transform() to both train and test.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def _apply_feature_scaler(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Transform a feature matrix using a pre-fitted scaler.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler from _fit_feature_scaler().
    X : np.ndarray
        Shape (N, F), dtype float64.

    Returns
    -------
    np.ndarray
        Shape (N, F), dtype float64. Z-scored features.
    """
    return scaler.transform(X).astype(np.float64)


def _fit_sequence_scaler(
    X_train_seq: np.ndarray,
    valid_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from valid (non-padded) training timesteps only.

    For each training sample, only the first ``valid_lengths[i]`` time steps
    are included in the statistics. Padding zeros are excluded so they cannot
    bias the channel mean toward zero or inflate the std.

    Statistics are computed independently per channel (axis 2), producing a
    single mean and std per channel that is broadcast-safe for the full
    (N, max_len, 6) array.

    A std of zero (constant channel) is replaced with 1.0 to avoid
    division by zero; the result will be all zeros for that channel,
    which is the safe fallback.

    Parameters
    ----------
    X_train_seq : np.ndarray
        Shape (N_train, max_len, 6), dtype float32 or float64.
        Training fold sequences (pre-padded, not yet normalised).
    valid_lengths : np.ndarray
        Shape (N_train,), dtype int. Number of valid (non-padded) time steps
        per training sample. Obtained from dataset.valid_lengths[train_idx].

    Returns
    -------
    mean : np.ndarray
        Shape (1, 1, 6), dtype float64. Per-channel mean over valid timesteps.
    std : np.ndarray
        Shape (1, 1, 6), dtype float64. Per-channel std over valid timesteps
        (>= 1e-8).
    """
    # Concatenate only the valid rows from each training sample.
    # This excludes padding zeros from the statistics entirely.
    valid_rows: list[np.ndarray] = []
    for i, vl in enumerate(valid_lengths):
        valid_rows.append(X_train_seq[i, :int(vl), :].astype(np.float64))
    # Shape: (total_valid_timesteps, 6)
    all_valid = np.concatenate(valid_rows, axis=0)

    mean = all_valid.mean(axis=0, keepdims=True)   # (1, 6)
    std = all_valid.std(axis=0, keepdims=True)     # (1, 6)
    std = np.where(std < 1e-8, 1.0, std)

    # Reshape to (1, 1, 6) for broadcasting against (N, max_len, 6).
    mean = mean[np.newaxis, :, :]   # (1, 1, 6)
    std = std[np.newaxis, :, :]     # (1, 1, 6)
    return mean, std


def _apply_sequence_scaler(
    X_seq: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply per-channel z-score normalisation to a sequence array.

    Parameters
    ----------
    X_seq : np.ndarray
        Shape (N, max_len, 6), dtype float32 or float64.
    mean : np.ndarray
        Shape (1, 1, 6), dtype float64. From _fit_sequence_scaler().
    std : np.ndarray
        Shape (1, 1, 6), dtype float64. From _fit_sequence_scaler().

    Returns
    -------
    np.ndarray
        Shape (N, max_len, 6), dtype float32. Normalised sequences cast
        back to float32 for CNN memory efficiency.
    """
    normalised = (X_seq.astype(np.float64) - mean) / std
    return normalised.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-model fold loop
# ---------------------------------------------------------------------------

def _run_lr_fold(
    fold_idx: int,
    X_train_norm: np.ndarray,
    X_test_norm: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and evaluate LogisticRegressionModel on one fold.

    Parameters
    ----------
    fold_idx : int
        Zero-based fold index (used for logging only).
    X_train_norm : np.ndarray
        Shape (N_train, F), z-scored feature matrix.
    X_test_norm : np.ndarray
        Shape (N_test, F), z-scored feature matrix.
    y_train : np.ndarray
        Shape (N_train,), dtype int.
    y_test : np.ndarray
        Shape (N_test,), dtype int.
    cfg : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    y_test : np.ndarray
    proba : np.ndarray  shape (N_test, 2)
    y_pred : np.ndarray  shape (N_test,)
    """
    logger.info("[LR] Fold %d — fitting.", fold_idx)
    model = LogisticRegressionModel(
        C=cfg.lr_C,
        penalty=cfg.lr_penalty,
        max_iter=cfg.lr_max_iter,
        random_state=cfg.random_seed,
    )
    model.fit(X_train_norm, y_train)
    proba = model.predict_proba(X_test_norm)
    y_pred = model.predict(X_test_norm)
    return y_test, proba, y_pred


def _run_rf_fold(
    fold_idx: int,
    X_train_norm: np.ndarray,
    X_test_norm: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and evaluate RandomForestModel on one fold.

    Returns
    -------
    y_test, proba (N_test, 2), y_pred (N_test,)
    """
    logger.info("[RF] Fold %d — fitting.", fold_idx)
    model = RandomForestModel(
        n_estimators=cfg.rf_n_estimators,
        max_features=cfg.rf_max_features,
        random_state=cfg.random_seed,
    )
    model.fit(X_train_norm, y_train)
    proba = model.predict_proba(X_test_norm)
    y_pred = model.predict(X_test_norm)
    return y_test, proba, y_pred


def _run_cnn_fold(
    fold_idx: int,
    X_train_norm: np.ndarray,
    X_test_norm: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    max_len: int,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and evaluate CNN1DModel on one fold.

    Parameters
    ----------
    max_len : int
        Global sequence length (PreprocessedDataset.max_len).

    Returns
    -------
    y_test, proba (N_test, 2), y_pred (N_test,)
    """
    logger.info("[CNN] Fold %d — fitting.", fold_idx)
    model = CNN1DModel(
        max_len=max_len,
        epochs=cfg.cnn_epochs,
        batch_size=cfg.cnn_batch_size,
        lr=cfg.cnn_lr,
        dropout=cfg.cnn_dropout,
        random_state=cfg.random_seed,
        device=cfg.cnn_device,
    )
    model.fit(X_train_norm, y_train)
    proba = model.predict_proba(X_test_norm)
    y_pred = model.predict(X_test_norm)
    return y_test, proba, y_pred


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def _save_results(
    summaries: dict[str, AggregatedMetrics],
    results_dir: Path,
) -> None:
    """Persist metrics tables and confusion matrices to results_dir.

    Files written:
        metrics_summary.csv           — mean ± SD per model, all metrics
        per_fold_metrics.csv          — per-fold scalar metrics for all models
        confusion_<model>.npy         — raw summed confusion matrix (int)
        confusion_norm_<model>.npy    — row-normalised confusion matrix (float)

    Parameters
    ----------
    summaries : dict[str, AggregatedMetrics]
        Keys are model names ("LR", "RF", "CNN"). Values are aggregated
        metrics from MetricsCollector.aggregate().
    results_dir : Path
        Output directory. Created if it does not exist.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- metrics_summary.csv ---
    summary_path = results_dir / "metrics_summary.csv"
    summary_fields = [
        "model",
        "mean_balanced_accuracy", "sd_balanced_accuracy",
        "mean_auroc", "sd_auroc",
        "mean_sensitivity", "sd_sensitivity",
        "mean_specificity", "sd_specificity",
        "n_folds",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for model_name, summary in summaries.items():
            writer.writerow({
                "model": model_name,
                "mean_balanced_accuracy": f"{summary.mean_balanced_accuracy:.6f}",
                "sd_balanced_accuracy": f"{summary.sd_balanced_accuracy:.6f}",
                "mean_auroc": f"{summary.mean_auroc:.6f}",
                "sd_auroc": f"{summary.sd_auroc:.6f}",
                "mean_sensitivity": f"{summary.mean_sensitivity:.6f}",
                "sd_sensitivity": f"{summary.sd_sensitivity:.6f}",
                "mean_specificity": f"{summary.mean_specificity:.6f}",
                "sd_specificity": f"{summary.sd_specificity:.6f}",
                "n_folds": summary.n_folds,
            })
    logger.info("Saved metrics summary: %s", summary_path)

    # --- per_fold_metrics.csv ---
    fold_path = results_dir / "per_fold_metrics.csv"
    fold_fields = [
        "model", "fold_index",
        "balanced_accuracy", "auroc", "sensitivity", "specificity",
        "n_samples", "n_pd", "n_hc",
    ]
    with fold_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fields)
        writer.writeheader()
        for model_name, summary in summaries.items():
            for fm in summary.fold_metrics:
                writer.writerow({
                    "model": model_name,
                    "fold_index": fm.fold_index,
                    "balanced_accuracy": f"{fm.balanced_accuracy:.6f}",
                    "auroc": f"{fm.auroc:.6f}",
                    "sensitivity": f"{fm.sensitivity:.6f}",
                    "specificity": f"{fm.specificity:.6f}",
                    "n_samples": fm.n_samples,
                    "n_pd": fm.n_pd,
                    "n_hc": fm.n_hc,
                })
    logger.info("Saved per-fold metrics: %s", fold_path)

    # --- confusion matrices (.npy) ---
    for model_name, summary in summaries.items():
        tag = model_name.lower().replace(" ", "_")
        raw_path = results_dir / f"confusion_{tag}.npy"
        norm_path = results_dir / f"confusion_norm_{tag}.npy"
        np.save(raw_path, summary.confusion_sum)
        np.save(norm_path, summary.confusion_normalised)
        logger.info(
            "Saved confusion matrices for %s: %s, %s",
            model_name, raw_path, norm_path,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> dict[str, AggregatedMetrics]:
    """Execute the full PADS benchmark pipeline.

    Steps
    -----
    1.  Validate configuration.
    2.  Load raw records (loader.py).
    3.  Preprocess signals (preprocessor.py).
    4.  Build subject-stratified CV folds (cross_val.py).
    5.  Per-fold loop:
        a. Slice train/test arrays.
        b. Fit feature scaler on train features; normalise train + test.
        c. Fit sequence scaler on train sequences; normalise train + test.
        d. Extract features for LR / RF (extractor.py).
        e. Train + evaluate LR (logistic_regression.py + metrics.py).
        f. Train + evaluate RF (random_forest.py + metrics.py).
        g. Train + evaluate CNN (cnn1d.py + metrics.py).
    6.  Aggregate metrics per model (metrics.py).
    7.  Print summaries to stdout.
    8.  Save results to disk.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully populated pipeline configuration.

    Returns
    -------
    dict[str, AggregatedMetrics]
        Keys: "LR", "RF", "CNN" (only for enabled models).
        Values: aggregated cross-validation metrics.

    Raises
    ------
    ValueError
        If configuration is invalid.
    FileNotFoundError
        If data_root does not contain the expected PADS layout.
    """
    _validate_config(cfg)
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("PADS BENCHMARK PIPELINE — START")
    logger.info("data_root   : %s", cfg.data_root)
    logger.info("results_dir : %s", cfg.results_dir)
    logger.info("n_folds     : %d", cfg.n_folds)
    logger.info("random_seed : %d", cfg.random_seed)
    logger.info("models      : LR=%s  RF=%s  CNN=%s",
                cfg.run_lr, cfg.run_rf, cfg.run_cnn)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    logger.info("--- Step 1: Loading PADS records ---")
    records = load_pads(data_root=cfg.data_root)
    logger.info("Loaded %d records.", len(records))

    # ------------------------------------------------------------------
    # Step 2: Preprocess
    # ------------------------------------------------------------------
    logger.info("--- Step 2: Preprocessing ---")
    dataset = preprocess_records(
        records=records,
        sampling_rate_hz=cfg.sampling_rate_hz,
        cutoff_hz=cfg.highpass_cutoff_hz,
        filter_order=cfg.filter_order,
    )
    logger.info(
        "Preprocessed: signals %s, max_len=%d.",
        dataset.signals.shape, dataset.max_len,
    )

    # ------------------------------------------------------------------
    # Step 3: Build folds
    # ------------------------------------------------------------------
    logger.info("--- Step 3: Building CV folds ---")
    folds = build_folds(
        subject_ids=dataset.subject_ids,
        labels=dataset.labels,
        n_folds=cfg.n_folds,
        random_seed=cfg.random_seed,
    )
    summarise_folds(folds, dataset.labels)

    # Collectors — one per enabled model.
    collectors: dict[str, MetricsCollector] = {}
    if cfg.run_lr:
        collectors["LR"] = MetricsCollector()
    if cfg.run_rf:
        collectors["RF"] = MetricsCollector()
    if cfg.run_cnn:
        collectors["CNN"] = MetricsCollector()

    # ------------------------------------------------------------------
    # Step 5: Per-fold loop
    # ------------------------------------------------------------------
    logger.info("--- Step 5: Cross-validation loop ---")

    for fold in folds:
        fold_idx = fold.fold_index
        logger.info("===== Fold %d / %d =====", fold_idx + 1, cfg.n_folds)

        train_idx = fold.train_indices
        test_idx = fold.test_indices

        y_train = dataset.labels[train_idx]
        y_test = dataset.labels[test_idx]

        # --- 5a: Sequence arrays (used by CNN and as source for features) ---
        X_train_seq = dataset.signals[train_idx]   # (N_train, max_len, 6)
        X_test_seq = dataset.signals[test_idx]     # (N_test,  max_len, 6)

        # Valid lengths are needed by both the sequence scaler (step 5b) and
        # the feature extractor (step 5c), so slice them once here.
        valid_lengths_train = dataset.valid_lengths[train_idx]
        valid_lengths_test = dataset.valid_lengths[test_idx]

        # --- 5b: Sequence normalisation (CNN only) ---
        # Statistics are computed from valid (non-padded) training timesteps
        # only, then applied to both train and test sequences.
        seq_mean, seq_std = _fit_sequence_scaler(X_train_seq, valid_lengths_train)
        X_train_seq_norm = _apply_sequence_scaler(X_train_seq, seq_mean, seq_std)
        X_test_seq_norm = _apply_sequence_scaler(X_test_seq, seq_mean, seq_std)

        # --- 5c: Feature extraction (LR / RF) ---
        if cfg.run_lr or cfg.run_rf:
            fm_train = extract_features(
                signals=X_train_seq,
                valid_lengths=valid_lengths_train,
                sampling_rate_hz=cfg.sampling_rate_hz,
            )
            fm_test = extract_features(
                signals=X_test_seq,
                valid_lengths=valid_lengths_test,
                sampling_rate_hz=cfg.sampling_rate_hz,
            )

            # --- 5d: Feature normalisation (fit on train only) ---
            feat_scaler = _fit_feature_scaler(fm_train.X)
            X_train_feat_norm = _apply_feature_scaler(feat_scaler, fm_train.X)
            X_test_feat_norm = _apply_feature_scaler(feat_scaler, fm_test.X)

        # --- 5e: Logistic Regression ---
        if cfg.run_lr:
            y_true, proba, y_pred = _run_lr_fold(
                fold_idx=fold_idx,
                X_train_norm=X_train_feat_norm,
                X_test_norm=X_test_feat_norm,
                y_train=y_train,
                y_test=y_test,
                cfg=cfg,
            )
            fm = compute_fold_metrics(fold_idx, y_true, proba, y_pred)
            collectors["LR"].add_fold(fm)

        # --- 5f: Random Forest ---
        if cfg.run_rf:
            y_true, proba, y_pred = _run_rf_fold(
                fold_idx=fold_idx,
                X_train_norm=X_train_feat_norm,
                X_test_norm=X_test_feat_norm,
                y_train=y_train,
                y_test=y_test,
                cfg=cfg,
            )
            fm = compute_fold_metrics(fold_idx, y_true, proba, y_pred)
            collectors["RF"].add_fold(fm)

        # --- 5g: CNN1D ---
        if cfg.run_cnn:
            y_true, proba, y_pred = _run_cnn_fold(
                fold_idx=fold_idx,
                X_train_norm=X_train_seq_norm,
                X_test_norm=X_test_seq_norm,
                y_train=y_train,
                y_test=y_test,
                max_len=dataset.max_len,
                cfg=cfg,
            )
            fm = compute_fold_metrics(fold_idx, y_true, proba, y_pred)
            collectors["CNN"].add_fold(fm)

    # ------------------------------------------------------------------
    # Step 6: Aggregate and print
    # ------------------------------------------------------------------
    logger.info("--- Step 6: Aggregating metrics ---")
    summaries: dict[str, AggregatedMetrics] = {}
    for model_name, collector in collectors.items():
        summary = collector.aggregate()
        summaries[model_name] = summary
        print_summary(summary, model_name=model_name)

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    logger.info("--- Step 7: Saving results ---")
    _save_results(summaries, results_dir=Path(cfg.results_dir))

    elapsed = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1f s.", elapsed)
    logger.info("=" * 60)

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Supports two mutually exclusive invocation modes:
        --config path/to/pipeline.yaml
        --data-root path/to/pads  [plus optional overrides]
    """
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Run the full PADS PD/HC benchmark pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. All fields are optional except data_root.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory of the PADS dataset. Required if --config is not given.",
    )
    # Use SUPPRESS as the default for all overridable arguments so that
    # args.__dict__ only contains keys the user explicitly passed on the
    # command line. This lets _config_from_args detect exactly which flags
    # were provided and apply them as overrides on top of a --config YAML.
    _S = argparse.SUPPRESS
    parser.add_argument("--results-dir", type=str, default=_S,
                        help="Directory for output files.")
    parser.add_argument("--n-folds", type=int, default=_S)
    parser.add_argument("--random-seed", type=int, default=_S)
    parser.add_argument("--sampling-rate", type=float, default=_S)
    parser.add_argument("--highpass-cutoff", type=float, default=_S)
    parser.add_argument("--filter-order", type=int, default=_S)
    parser.add_argument("--lr-C", type=float, default=_S)
    parser.add_argument("--lr-penalty", type=str, default=_S)
    parser.add_argument("--lr-max-iter", type=int, default=_S)
    parser.add_argument("--rf-n-estimators", type=int, default=_S)
    parser.add_argument("--rf-max-features", type=str, default=_S)
    parser.add_argument("--cnn-epochs", type=int, default=_S)
    parser.add_argument("--cnn-batch-size", type=int, default=_S)
    parser.add_argument("--cnn-lr", type=float, default=_S)
    parser.add_argument("--cnn-dropout", type=float, default=_S)
    parser.add_argument("--cnn-device", type=str, default=_S)
    parser.add_argument("--no-lr", action="store_true", help="Disable Logistic Regression.")
    parser.add_argument("--no-rf", action="store_true", help="Disable Random Forest.")
    parser.add_argument("--no-cnn", action="store_true", help="Disable CNN1D.")
    return parser


# Mapping from argparse dest name → PipelineConfig field name.
# Only entries that differ between the two namings are listed; identical
# names are handled by the fallback identity mapping in _config_from_args.
_ARG_TO_CFG: dict[str, str] = {
    "results_dir": "results_dir",
    "n_folds": "n_folds",
    "random_seed": "random_seed",
    "sampling_rate": "sampling_rate_hz",
    "highpass_cutoff": "highpass_cutoff_hz",
    "filter_order": "filter_order",
    "lr_C": "lr_C",
    "lr_penalty": "lr_penalty",
    "lr_max_iter": "lr_max_iter",
    "rf_n_estimators": "rf_n_estimators",
    "rf_max_features": "rf_max_features",
    "cnn_epochs": "cnn_epochs",
    "cnn_batch_size": "cnn_batch_size",
    "cnn_lr": "cnn_lr",
    "cnn_dropout": "cnn_dropout",
    "cnn_device": "cnn_device",
}


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Construct a PipelineConfig from parsed CLI arguments.

    If --config is given, load YAML first, then apply every CLI argument
    that was explicitly provided on the command line as an override.
    Detection relies on argparse.SUPPRESS: suppressed defaults are absent
    from args.__dict__, so only user-supplied flags appear.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    ValueError
        If neither --config nor --data-root is supplied.
    """
    if args.config is not None:
        cfg = _config_from_yaml(args.config)
        # Apply every explicitly provided CLI flag as an override.
        args_dict = vars(args)
        if args.data_root is not None:
            cfg.data_root = args.data_root
        for arg_key, cfg_field in _ARG_TO_CFG.items():
            if arg_key in args_dict:
                setattr(cfg, cfg_field, args_dict[arg_key])
        # Boolean disable-flags: only override when the flag was passed.
        if args.no_lr:
            cfg.run_lr = False
        if args.no_rf:
            cfg.run_rf = False
        if args.no_cnn:
            cfg.run_cnn = False
    else:
        if args.data_root is None:
            raise ValueError(
                "Either --config or --data-root must be provided."
            )
        # argparse.SUPPRESS means absent keys fall back to PipelineConfig
        # defaults; use vars(args).get(key, default) for suppressed args.
        a = vars(args)
        cfg = PipelineConfig(
            data_root=args.data_root,
            results_dir=a.get("results_dir", "results"),
            n_folds=a.get("n_folds", 5),
            random_seed=a.get("random_seed", 42),
            sampling_rate_hz=a.get("sampling_rate", 100.0),
            highpass_cutoff_hz=a.get("highpass_cutoff", 0.5),
            filter_order=a.get("filter_order", 4),
            lr_C=a.get("lr_C", 1.0),
            lr_penalty=a.get("lr_penalty", "l2"),
            lr_max_iter=a.get("lr_max_iter", 1000),
            rf_n_estimators=a.get("rf_n_estimators", 500),
            rf_max_features=a.get("rf_max_features", "sqrt"),
            cnn_epochs=a.get("cnn_epochs", 50),
            cnn_batch_size=a.get("cnn_batch_size", 32),
            cnn_lr=a.get("cnn_lr", 1e-3),
            cnn_dropout=a.get("cnn_dropout", 0.5),
            cnn_device=a.get("cnn_device", None),
            run_lr=not args.no_lr,
            run_rf=not args.no_rf,
            run_cnn=not args.no_cnn,
        )
    return cfg


def main() -> None:
    """Entry point for CLI invocation."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        cfg = _config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
