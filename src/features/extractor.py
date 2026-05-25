"""
src/features/extractor.py
--------------------------
Extracts hand-crafted time-domain and frequency-domain features from
preprocessed PADS IMU signals for use by Logistic Regression and Random Forest.

Input:  signals array of shape (N, max_len, 6), dtype float32.
Output: feature matrix of shape (N, F), dtype float64.

Feature set (computed per channel, 6 channels total):
    Time-domain  (12 features × 6 channels =  72):
        mean, std, min, max, range, rms, mad, skewness, kurtosis,
        zero_crossing_rate, p25, p75

    Frequency-domain (8 features × 6 channels = 48):
        dominant_freq, spectral_entropy,
        band_energy_low   (0.5 –  3 Hz),
        band_energy_mid   (3   – 10 Hz),
        band_energy_high  (10  – 20 Hz),
        spectral_mean, spectral_std, spectral_rolloff

    Total: 120 features per sample.

Padding awareness:
    Signals are zero-padded to the global max_len. Feature computation uses
    only the valid (non-padded) portion of each signal when `valid_lengths`
    is supplied. `valid_lengths` is a required argument; omitting it raises
    a ValueError. Obtain valid_lengths from the original SubjectRecord
    raw_signal lengths before stacking and pass them explicitly.

Normalisation:
    This module returns raw feature values only. Normalisation (z-score per
    feature, fit on the training split) is the responsibility of run_pipeline.py
    and must never be performed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sampling rate in Hz — must match preprocessor.py.
SAMPLING_RATE_HZ: float = 100.0

#: Frequency band boundaries in Hz for band energy features.
#: Lower bound matches the high-pass cutoff used in preprocessor.py.
FREQ_BANDS: tuple[tuple[float, float], ...] = (
    (0.5, 3.0),   # low:  slow postural sway / tremor fundamentals
    (3.0, 10.0),  # mid:  PD tremor range (4–6 Hz) and faster voluntary motion
    (10.0, 20.0), # high: rapid kinetic movements
)

#: Spectral rolloff threshold: frequency below which this fraction of total
#: spectral energy is contained.
SPECTRAL_ROLLOFF_THRESHOLD: float = 0.85

#: Total number of features per sample. Used for shape validation.
#: 12 time-domain + 8 frequency-domain = 20 per channel × 6 channels = 120.
N_FEATURES_PER_CHANNEL: int = 20
N_CHANNELS: int = 6
TOTAL_FEATURES: int = N_FEATURES_PER_CHANNEL * N_CHANNELS


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    """Output of extract_features().

    Attributes
    ----------
    X : np.ndarray
        Shape (N, F), dtype float64. Raw feature values, not normalised.
    feature_names : list[str]
        Length F. Human-readable name for each feature column, formatted as
        '<feature>_ch<channel_index>' (e.g. 'mean_ch0', 'dominant_freq_ch3').
        Aligned index-for-index with columns of X.
    n_samples : int
        Number of samples N.
    n_features : int
        Number of features F.
    """

    X: np.ndarray
    feature_names: list[str]
    n_samples: int = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples, self.n_features = self.X.shape


# ---------------------------------------------------------------------------
# Per-channel time-domain features
# ---------------------------------------------------------------------------

def _time_domain_features(x: np.ndarray) -> np.ndarray:
    """Compute 12 time-domain features from a 1-D signal array.

    Parameters
    ----------
    x : np.ndarray
        1-D float array of the valid signal portion for one channel.

    Returns
    -------
    np.ndarray
        Shape (12,), dtype float64.
        Order: mean, std, min, max, range, rms, mad, skewness, kurtosis,
               zero_crossing_rate, p25, p75.

    Notes
    -----
    skewness and kurtosis are set to 0.0 for constant or near-constant
    signals (std == 0) to avoid NaN propagation into the feature matrix.
    """
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_range = x_max - x_min
    rms = float(np.sqrt(np.mean(x ** 2)))
    mad = float(np.mean(np.abs(x - mean)))

    # skewness: return 0.0 for constant or near-constant signals (std == 0)
    # to avoid NaN. scipy_skew returns NaN when all values are identical.
    if len(x) > 2 and std > 0.0:
        skewness = float(scipy_skew(x))
        if np.isnan(skewness):
            skewness = 0.0
    else:
        skewness = 0.0

    # kurtosis: same guard as skewness.
    if len(x) > 3 and std > 0.0:
        kurt = float(scipy_kurtosis(x, fisher=True))
        if np.isnan(kurt):
            kurt = 0.0
    else:
        kurt = 0.0

    # Zero-crossing rate: fraction of consecutive pairs that cross zero.
    zcr = float(np.mean(np.diff(np.sign(x)) != 0)) if len(x) > 1 else 0.0
    p25 = float(np.percentile(x, 25))
    p75 = float(np.percentile(x, 75))

    return np.array(
        [mean, std, x_min, x_max, x_range, rms, mad, skewness, kurt, zcr, p25, p75],
        dtype=np.float64,
    )


_TIME_FEATURE_NAMES: list[str] = [
    "mean", "std", "min", "max", "range", "rms", "mad",
    "skewness", "kurtosis", "zero_crossing_rate", "p25", "p75",
]


# ---------------------------------------------------------------------------
# Per-channel frequency-domain features
# ---------------------------------------------------------------------------

def _frequency_domain_features(
    x: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Compute 8 frequency-domain features from a 1-D signal array.

    Uses a real-valued FFT. Only the positive-frequency half is used.

    Parameters
    ----------
    x : np.ndarray
        1-D float array of the valid signal portion for one channel.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Shape (8,), dtype float64.
        Order: dominant_freq, spectral_entropy,
               band_energy_low, band_energy_mid, band_energy_high,
               spectral_mean, spectral_std, spectral_rolloff.
    """
    n = len(x)
    if n < 2:
        return np.zeros(8, dtype=np.float64)

    fft_mag = np.abs(np.fft.rfft(x))              # shape: (n//2 + 1,)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)        # shape: (n//2 + 1,)
    power = fft_mag ** 2

    total_power = float(power.sum())

    # Dominant frequency: frequency bin with highest power.
    dominant_freq = float(freqs[np.argmax(power)])

    # Spectral entropy: normalised Shannon entropy of the power spectrum.
    if total_power > 0.0:
        p_norm = power / total_power
        # Avoid log(0): mask zero-power bins.
        nonzero = p_norm > 0.0
        spectral_entropy = float(
            -np.sum(p_norm[nonzero] * np.log2(p_norm[nonzero]))
            / np.log2(float(np.sum(nonzero)))
        ) if np.sum(nonzero) > 1 else 0.0
    else:
        spectral_entropy = 0.0

    # Band energies: fraction of total power within each frequency band.
    band_energies: list[float] = []
    for low, high in FREQ_BANDS:
        mask = (freqs >= low) & (freqs < high)
        band_power = float(power[mask].sum())
        band_energies.append(band_power / total_power if total_power > 0.0 else 0.0)

    # Spectral mean and std (frequency-weighted statistics).
    if total_power > 0.0:
        spectral_mean = float(np.sum(freqs * power) / total_power)
        spectral_std = float(
            np.sqrt(np.sum(((freqs - spectral_mean) ** 2) * power) / total_power)
        )
    else:
        spectral_mean = 0.0
        spectral_std = 0.0

    # Spectral rolloff: lowest frequency below which ROLLOFF_THRESHOLD of
    # total power is contained.
    if total_power > 0.0:
        cumulative = np.cumsum(power)
        rolloff_idx = np.searchsorted(cumulative, SPECTRAL_ROLLOFF_THRESHOLD * total_power)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)
        spectral_rolloff = float(freqs[rolloff_idx])
    else:
        spectral_rolloff = 0.0

    return np.array(
        [dominant_freq, spectral_entropy] + band_energies +
        [spectral_mean, spectral_std, spectral_rolloff],
        dtype=np.float64,
    )


_FREQ_FEATURE_NAMES: list[str] = [
    "dominant_freq",
    "spectral_entropy",
    "band_energy_low",
    "band_energy_mid",
    "band_energy_high",
    "spectral_mean",
    "spectral_std",
    "spectral_rolloff",
]


# ---------------------------------------------------------------------------
# Per-sample feature vector
# ---------------------------------------------------------------------------

def _extract_single_sample(
    signal: np.ndarray,
    valid_length: int,
    fs: float,
) -> np.ndarray:
    """Extract all features from one (max_len, 6) signal array.

    Parameters
    ----------
    signal : np.ndarray
        Shape (max_len, 6). Zero-padded preprocessed signal.
    valid_length : int
        Number of valid (non-padded) time steps. Must be >= 1 and
        <= signal.shape[0].
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Shape (TOTAL_FEATURES,), dtype float64. Channel-major ordering:
        all features for channel 0, then channel 1, ..., then channel 5.
    """
    features: list[np.ndarray] = []
    valid_signal = signal[:valid_length, :]  # strip padding

    for ch in range(N_CHANNELS):
        x = valid_signal[:, ch].astype(np.float64)
        td = _time_domain_features(x)
        fd = _frequency_domain_features(x, fs)
        features.append(np.concatenate([td, fd]))

    return np.concatenate(features)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    signals: np.ndarray,
    valid_lengths: np.ndarray,
) -> None:
    """Validate input shapes and dtypes. Raises on failure."""
    if signals.ndim != 3:
        raise ValueError(
            f"signals must be 3-D (N, max_len, 6), got shape {signals.shape}."
        )
    if signals.shape[2] != N_CHANNELS:
        raise ValueError(
            f"signals must have {N_CHANNELS} channels (axis 2), "
            f"got {signals.shape[2]}."
        )
    n, max_len, _ = signals.shape
    if valid_lengths.shape != (n,):
        raise ValueError(
            f"valid_lengths shape {valid_lengths.shape} != (N,) = ({n},)."
        )
    if int(valid_lengths.min()) < 1:
        raise ValueError(
            "valid_lengths contains values < 1. Every sample must have "
            "at least one valid time step."
        )
    if int(valid_lengths.max()) > max_len:
        raise ValueError(
            f"valid_lengths max ({valid_lengths.max()}) exceeds "
            f"max_len ({max_len})."
        )


def _validate_output(X: np.ndarray, n_samples: int) -> None:
    """Validate feature matrix shape and content. Raises on failure."""
    assert X.shape == (n_samples, TOTAL_FEATURES), (
        f"Feature matrix shape {X.shape} != expected ({n_samples}, {TOTAL_FEATURES})."
    )
    assert not np.any(np.isnan(X)), (
        "NaN values in feature matrix. Check for zero-length valid signals "
        "or all-zero channels."
    )
    assert not np.any(np.isinf(X)), (
        "Inf values in feature matrix. Check for degenerate signal values."
    )


# ---------------------------------------------------------------------------
# Feature name builder
# ---------------------------------------------------------------------------

def build_feature_names() -> list[str]:
    """Return the ordered list of feature names for all channels.

    Returns
    -------
    list[str]
        Length TOTAL_FEATURES. Format: '<feature_name>_ch<channel_index>'.
        Matches the column order of the feature matrix returned by
        extract_features().
    """
    names: list[str] = []
    all_feature_names = _TIME_FEATURE_NAMES + _FREQ_FEATURE_NAMES
    for ch in range(N_CHANNELS):
        for fname in all_feature_names:
            names.append(f"{fname}_ch{ch}")
    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    signals: np.ndarray,
    valid_lengths: np.ndarray,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> FeatureMatrix:
    """Extract hand-crafted features from preprocessed IMU signals.

    Computes 12 time-domain and 8 frequency-domain features per channel
    for each sample, yielding 120 features total (6 channels × 20 features).

    Feature computation is performed on the valid (non-padded) portion of
    each signal as specified by `valid_lengths`. `valid_lengths` is a
    required argument; callers must pass the per-record raw signal lengths
    stored before zero-padding in preprocessor.py. Omitting it raises a
    TypeError. Passing full padded lengths will bias statistics for records
    shorter than max_len.

    No normalisation is applied. The returned matrix contains raw feature
    values. Normalisation must be applied in run_pipeline.py using statistics
    fit on the training fold only.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N, max_len, 6), dtype float32 or float64.
        Output of preprocessor.preprocess_records().signals, or a fold
        subset thereof (signals[train_indices] or signals[test_indices]).
    valid_lengths : np.ndarray
        Shape (N,), dtype int. Number of valid time steps per sample before
        zero-padding. Required. Obtain from the original SubjectRecord
        raw_signal lengths before stacking (e.g. store
        [r.raw_signal.shape[0] for r in records] in PreprocessedDataset
        or alongside it, then index the same way as signals).
    sampling_rate_hz : float
        Sampling rate in Hz. Must match the value used in preprocessor.py.
        Default: 100.0.

    Returns
    -------
    FeatureMatrix
        Contains X of shape (N, 120), aligned feature_names of length 120,
        and convenience attributes n_samples and n_features.

    Raises
    ------
    TypeError
        If valid_lengths is not provided (missing required argument).
    ValueError
        If input shapes are invalid or valid_lengths is out of bounds.
    AssertionError
        If the output feature matrix contains NaN or Inf values.
    """
    _validate_inputs(signals, valid_lengths)

    n, max_len, _ = signals.shape
    lengths = valid_lengths.astype(int)

    feature_rows: list[np.ndarray] = []
    for i in range(n):
        row = _extract_single_sample(signals[i], int(lengths[i]), sampling_rate_hz)
        feature_rows.append(row)

        if (i + 1) % 500 == 0:
            logger.debug("Extracted features for %d / %d samples.", i + 1, n)

    X = np.stack(feature_rows, axis=0)  # (N, TOTAL_FEATURES)
    _validate_output(X, n)

    feature_names = build_feature_names()

    logger.info(
        "Feature extraction complete: %d samples × %d features.", n, TOTAL_FEATURES
    )

    return FeatureMatrix(X=X, feature_names=feature_names)


def get_valid_lengths(signals_raw: np.ndarray) -> np.ndarray:
    """Infer valid lengths from the raw stacked signals before padding.

    This is a diagnostic / recovery helper only. It detects the last
    non-zero row per sample as the valid length using a heuristic that
    fails silently if a signal genuinely ends with all-zero samples.

    WARNING: Do NOT use this function as a substitute for explicitly storing
    valid lengths during preprocessing. The canonical approach is to record
    [r.raw_signal.shape[0] for r in records] before calling
    preprocess_records() and pass those lengths directly to extract_features()
    as valid_lengths. This function exists solely for diagnostic use when
    explicit lengths are unavailable.

    Parameters
    ----------
    signals_raw : np.ndarray
        Shape (N, max_len, 6). Padded signal array from PreprocessedDataset.

    Returns
    -------
    np.ndarray
        Shape (N,), dtype int. Estimated valid length per sample.
    """
    # A time step is considered padding if all 6 channels are exactly zero.
    nonzero_mask = np.any(signals_raw != 0.0, axis=2)  # (N, max_len)
    lengths = np.array(
        [
            int(np.max(np.where(nonzero_mask[i])[0])) + 1
            if np.any(nonzero_mask[i]) else 1
            for i in range(signals_raw.shape[0])
        ],
        dtype=int,
    )
    return lengths
