"""
src/data/preprocessor.py
------------------------
Applies signal preprocessing to raw PADS SubjectRecords produced by loader.py.

Responsibilities:
    1. Apply a zero-phase Butterworth high-pass filter to accelerometer
       channels (indices 0–2) only. Gyroscope channels (indices 3–5) are
       passed through unchanged.
    2. Compute the global maximum sequence length across all records.
    3. Zero-pad every signal to that global maximum length (trailing zeros).
    4. Stack processed signals into a single (N, max_len, 6) float32 array.

This module performs NO normalisation, NO feature extraction, NO CV splitting,
and NO label manipulation. Normalisation is the pipeline's responsibility and
must be applied per fold inside run_pipeline.py.

Channel layout (inherited from loader.py):
    Index 0: accelerometer_x  → high-pass filtered
    Index 1: accelerometer_y  → high-pass filtered
    Index 2: accelerometer_z  → high-pass filtered
    Index 3: gyroscope_x      → unchanged
    Index 4: gyroscope_y      → unchanged
    Index 5: gyroscope_z      → unchanged

Sampling rate:
    PADS Apple Watch Series 4 records at 100 Hz.
    1024 samples = 10.24 s; 2048 samples = 20.48 s.
    Verify with inspect_sampling_rate() on first run if uncertain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.data.loader import SubjectRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Assumed sampling rate in Hz. Derived from PADS documentation:
#: 1024 samples / 10.24 s = 100 Hz. Verify on first run.
SAMPLING_RATE_HZ: float = 100.0

#: High-pass filter cutoff frequency in Hz.
#: Removes DC offset and slow gravitational drift from accelerometer signals.
HIGHPASS_CUTOFF_HZ: float = 0.5

#: Butterworth filter order. Order 4 gives a good roll-off without ringing.
FILTER_ORDER: int = 4

#: Accelerometer channel indices (must match loader.py channel order).
ACC_CHANNELS: tuple[int, ...] = (0, 1, 2)

#: Gyroscope channel indices (must match loader.py channel order).
GYR_CHANNELS: tuple[int, ...] = (3, 4, 5)


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedDataset:
    """Output of preprocess_records().

    Attributes
    ----------
    signals : np.ndarray
        Shape (N, max_len, 6), dtype float32.
        Accelerometer channels are high-pass filtered; gyroscope channels
        are unchanged. All sequences are zero-padded to max_len.
    labels : np.ndarray
        Shape (N,), dtype int. Binary labels: 1 = PD, 0 = HC.
        Aligned index-for-index with signals.
    subject_ids : np.ndarray
        Shape (N,), dtype object (str). Subject identifiers aligned with
        signals and labels. Required by cross_val.py for subject-level splits.
    step_ids : np.ndarray
        Shape (N,), dtype object (str). Assessment step IDs aligned with
        signals. Useful for per-step analysis and debugging.
    valid_lengths : np.ndarray
        Shape (N,), dtype int. Number of valid (non-padded) time steps per
        sample, recorded from each SubjectRecord.raw_signal before zero-
        padding. Aligned index-for-index with signals, labels, subject_ids,
        and step_ids. Pass directly to extractor.extract_features() as the
        required valid_lengths argument.
    max_len : int
        Global maximum sequence length used for padding. Computed from the
        full record list before any fold split; must not be recomputed
        per fold.
    sampling_rate_hz : float
        Sampling rate used during filtering, stored for downstream reference.
    """

    signals: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    step_ids: np.ndarray
    valid_lengths: np.ndarray
    max_len: int
    sampling_rate_hz: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_highpass_filter(
    cutoff_hz: float,
    fs: float,
    order: int,
) -> np.ndarray:
    """Design a zero-phase Butterworth high-pass filter (SOS form).

    Parameters
    ----------
    cutoff_hz : float
        Filter cutoff frequency in Hz.
    fs : float
        Sampling rate in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Second-order sections (SOS) array for use with sosfiltfilt.

    Raises
    ------
    ValueError
        If the normalised cutoff frequency is outside (0, 1).
    """
    nyquist = fs / 2.0
    normalised_cutoff = cutoff_hz / nyquist
    if not (0.0 < normalised_cutoff < 1.0):
        raise ValueError(
            f"Normalised cutoff frequency {normalised_cutoff:.4f} is outside (0, 1). "
            f"Check cutoff_hz={cutoff_hz} and fs={fs}."
        )
    sos = butter(order, normalised_cutoff, btype="highpass", output="sos")
    return sos


def _apply_highpass_to_signal(
    signal: np.ndarray,
    sos: np.ndarray,
) -> np.ndarray:
    """Apply high-pass filter to accelerometer channels; leave gyro unchanged.

    Parameters
    ----------
    signal : np.ndarray
        Raw signal array of shape (T, 6), dtype float32.
    sos : np.ndarray
        SOS filter coefficients from _build_highpass_filter().

    Returns
    -------
    np.ndarray
        Filtered signal of shape (T, 6), dtype float32. Only channels 0–2
        are modified; channels 3–5 are copied unchanged.
    """
    filtered = signal.copy()
    for ch in ACC_CHANNELS:
        # sosfiltfilt requires float64 for numerical stability; cast back after.
        filtered[:, ch] = sosfiltfilt(
            sos, signal[:, ch].astype(np.float64)
        ).astype(np.float32)
    return filtered


def _zero_pad(signal: np.ndarray, max_len: int) -> np.ndarray:
    """Zero-pad a signal to max_len along the time axis.

    Padding is applied at the end (trailing zeros). If the signal is already
    equal to max_len, it is returned unchanged. Signals longer than max_len
    are not truncated — this should not occur if max_len is derived from
    the full dataset; a warning is emitted instead.

    Parameters
    ----------
    signal : np.ndarray
        Shape (T, 6).
    max_len : int
        Target sequence length.

    Returns
    -------
    np.ndarray
        Shape (max_len, 6), dtype float32.
    """
    t = signal.shape[0]
    if t == max_len:
        return signal
    if t > max_len:
        raise ValueError(
            f"Signal length {t} exceeds max_len {max_len}. "
            "This should never occur when max_len is computed from the full "
            "dataset before any fold split. Check that preprocess_records() "
            "is called with the complete record list, not a fold subset."
        )
    pad_width = max_len - t
    return np.pad(signal, ((0, pad_width), (0, 0)), mode="constant", constant_values=0.0)


def _compute_max_length(records: list[SubjectRecord]) -> int:
    """Return the maximum sequence length across all records.

    Must be called on the FULL record list before any fold split so that
    padding is consistent across all folds.

    Parameters
    ----------
    records : list[SubjectRecord]
        Full list of SubjectRecords from load_pads().

    Returns
    -------
    int
        Maximum T across all raw_signal arrays.
    """
    return max(r.raw_signal.shape[0] for r in records)


def _validate_output(dataset: PreprocessedDataset) -> None:
    """Run basic sanity checks on the preprocessed output. Raises on failure."""
    n = len(dataset.labels)

    assert dataset.signals.shape == (n, dataset.max_len, 6), (
        f"signals shape {dataset.signals.shape} does not match "
        f"expected ({n}, {dataset.max_len}, 6)."
    )
    assert dataset.labels.shape == (n,), (
        f"labels shape {dataset.labels.shape} != ({n},)."
    )
    assert dataset.subject_ids.shape == (n,), (
        f"subject_ids shape {dataset.subject_ids.shape} != ({n},)."
    )
    assert dataset.step_ids.shape == (n,), (
        f"step_ids shape {dataset.step_ids.shape} != ({n},)."
    )
    assert dataset.valid_lengths.shape == (n,), (
        f"valid_lengths shape {dataset.valid_lengths.shape} != ({n},)."
    )
    assert int(dataset.valid_lengths.min()) >= 1, (
        "valid_lengths contains values < 1. Every sample must have at least "
        "one valid time step."
    )
    assert int(dataset.valid_lengths.max()) <= dataset.max_len, (
        f"valid_lengths max ({dataset.valid_lengths.max()}) exceeds "
        f"max_len ({dataset.max_len})."
    )
    assert set(dataset.labels.tolist()).issubset({0, 1}), (
        f"Unexpected label values: {set(dataset.labels.tolist())}."
    )
    assert not np.any(np.isnan(dataset.signals)), (
        "NaN values detected in preprocessed signals."
    )
    assert not np.any(np.isinf(dataset.signals)), (
        "Inf values detected in preprocessed signals."
    )

    logger.info(
        "Preprocessed dataset: %d records, max_len=%d, "
        "label distribution: PD=%d HC=%d.",
        n,
        dataset.max_len,
        int((dataset.labels == 1).sum()),
        int((dataset.labels == 0).sum()),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_records(
    records: list[SubjectRecord],
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
    cutoff_hz: float = HIGHPASS_CUTOFF_HZ,
    filter_order: int = FILTER_ORDER,
) -> PreprocessedDataset:
    """Preprocess all SubjectRecords and return a stacked dataset.

    Steps performed in order:
        1. Compute global max sequence length from the full record list.
        2. Build the high-pass filter once (reused for all records).
        3. For each record: record the valid length before padding, apply
           high-pass filter to acc channels, leave gyro channels unchanged,
           zero-pad to max_len.
        4. Stack into (N, max_len, 6) array.
        5. Extract aligned labels, subject_ids, step_ids, and valid_lengths
           arrays.
        6. Validate output shape and data integrity.

    No normalisation is applied here. Normalisation must be applied
    per fold in run_pipeline.py using training-fold statistics only.

    Parameters
    ----------
    records : list[SubjectRecord]
        Full output of load_pads(). Must contain the complete dataset
        (not a fold subset) so that max_len is globally consistent.
    sampling_rate_hz : float
        Sampling rate in Hz. Default: 100.0 (PADS Apple Watch Series 4).
    cutoff_hz : float
        High-pass filter cutoff frequency in Hz. Default: 0.5 Hz.
    filter_order : int
        Butterworth filter order. Default: 4.

    Returns
    -------
    PreprocessedDataset
        Stacked arrays ready for cross-validation splitting.
        PreprocessedDataset.valid_lengths contains the per-sample raw signal
        lengths recorded before zero-padding; pass it directly to
        extractor.extract_features() as the required valid_lengths argument.

    Raises
    ------
    ValueError
        If records is empty or filter parameters are invalid.
    AssertionError
        If output shape or data integrity checks fail.
    """
    if not records:
        raise ValueError("records list is empty — nothing to preprocess.")

    # Step 1: global max length — must use full dataset.
    max_len = _compute_max_length(records)
    logger.info(
        "Global max sequence length: %d samples (%.2f s at %.1f Hz).",
        max_len,
        max_len / sampling_rate_hz,
        sampling_rate_hz,
    )

    # Step 2: build filter once.
    sos = _build_highpass_filter(cutoff_hz, sampling_rate_hz, filter_order)
    logger.info(
        "High-pass filter: Butterworth order=%d, cutoff=%.2f Hz, fs=%.1f Hz.",
        filter_order,
        cutoff_hz,
        sampling_rate_hz,
    )

    # Steps 3–4: process and stack.
    processed_signals: list[np.ndarray] = []
    raw_valid_lengths: list[int] = []
    for i, record in enumerate(records):
        # Record the valid length from the raw signal before any padding.
        raw_valid_lengths.append(record.raw_signal.shape[0])
        filtered = _apply_highpass_to_signal(record.raw_signal, sos)
        padded = _zero_pad(filtered, max_len)
        processed_signals.append(padded)

        if (i + 1) % 500 == 0:
            logger.debug("Processed %d / %d records.", i + 1, len(records))

    signals_array = np.stack(processed_signals, axis=0).astype(np.float32)

    # Step 5: aligned metadata arrays.
    labels_array = np.array([r.label for r in records], dtype=int)
    subject_ids_array = np.array([r.subject_id for r in records], dtype=object)
    step_ids_array = np.array([r.step_id for r in records], dtype=object)
    valid_lengths_array = np.array(raw_valid_lengths, dtype=int)

    dataset = PreprocessedDataset(
        signals=signals_array,
        labels=labels_array,
        subject_ids=subject_ids_array,
        step_ids=step_ids_array,
        valid_lengths=valid_lengths_array,
        max_len=max_len,
        sampling_rate_hz=sampling_rate_hz,
    )

    # Step 6: validate.
    _validate_output(dataset)
    return dataset


def compute_max_length(records: list[SubjectRecord]) -> int:
    """Public wrapper: return the global max sequence length.

    Exposed separately so run_pipeline.py can log or store this value
    independently of the full preprocessing step.

    Parameters
    ----------
    records : list[SubjectRecord]
        Full list from load_pads().

    Returns
    -------
    int
        Maximum T across all raw_signal arrays.
    """
    return _compute_max_length(records)


# ---------------------------------------------------------------------------
# Inspection utility
# ---------------------------------------------------------------------------

def inspect_sampling_rate(records: list[SubjectRecord]) -> None:
    """Print observed sequence lengths to help verify the sampling rate.

    PADS documentation states 100 Hz. Confirm that 10.24 s steps have
    1024 samples and 20.48 s steps have 2048 samples.

    Parameters
    ----------
    records : list[SubjectRecord]
        Output of load_pads().
    """
    from collections import Counter

    lengths = Counter(r.raw_signal.shape[0] for r in records)
    print("=== Sequence Length Distribution ===")
    for length, count in sorted(lengths.items()):
        duration = length / SAMPLING_RATE_HZ
        print(f"  {length} samples → {duration:.2f} s at {SAMPLING_RATE_HZ} Hz  "
              f"({count} records)")
    print(f"\nGlobal max length: {max(lengths.keys())} samples")
    print("Expected: 1024 (10.24 s) and/or 2048 (20.48 s) for PADS at 100 Hz.")
