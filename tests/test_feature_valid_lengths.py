"""
tests/test_feature_valid_lengths.py
------------------------------------
Tests that feature extraction respects valid_lengths and ignores zero-padding.
All data is synthetic; no real patient files are required.
"""

import numpy as np
import pytest

from src.features.extractor import extract_features, TOTAL_FEATURES


def _make_padded_signal(
    valid_len: int,
    max_len: int,
    n_channels: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a (1, max_len, 6) array with realistic values in [0:valid_len]
    and extreme sentinel values (1e9) in the padding region [valid_len:max_len].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    signal = np.zeros((1, max_len, n_channels), dtype=np.float32)
    signal[0, :valid_len, :] = rng.standard_normal((valid_len, n_channels)).astype(np.float32)
    signal[0, valid_len:, :] = 1e9  # extreme padding sentinel
    return signal


def test_valid_length_ignored_padding():
    """Features computed with correct valid_lengths must differ from features
    computed with full padded length when extreme sentinel padding is present."""
    valid_len = 200
    max_len = 1024
    signal = _make_padded_signal(valid_len, max_len)

    valid_lengths_correct = np.array([valid_len], dtype=int)
    valid_lengths_full = np.array([max_len], dtype=int)

    fm_correct = extract_features(signal, valid_lengths_correct)
    fm_full = extract_features(signal, valid_lengths_full)

    assert not np.allclose(fm_correct.X, fm_full.X), (
        "Features should differ when valid_lengths correctly excludes extreme padding."
    )


def test_extreme_padding_does_not_affect_mean():
    """The mean feature for each channel must reflect only valid samples."""
    valid_len = 100
    max_len = 500
    rng = np.random.default_rng(1)

    signal = np.zeros((1, max_len, 6), dtype=np.float32)
    signal[0, :valid_len, :] = 0.0  # valid region: all zeros → mean = 0
    signal[0, valid_len:, :] = 1e6  # padding: huge value

    valid_lengths = np.array([valid_len], dtype=int)
    fm = extract_features(signal, valid_lengths)

    # mean_ch0 is the first feature; should be ~0 since valid region is zeros
    mean_ch0 = fm.X[0, 0]
    assert abs(mean_ch0) < 1e-3, (
        f"mean_ch0={mean_ch0:.6f} — padding leaked into feature computation."
    )


def test_output_shape():
    """extract_features must return (N, TOTAL_FEATURES) for N samples."""
    n = 5
    valid_len = 300
    max_len = 1024
    rng = np.random.default_rng(2)

    signals = np.zeros((n, max_len, 6), dtype=np.float32)
    signals[:, :valid_len, :] = rng.standard_normal((n, valid_len, 6)).astype(np.float32)
    valid_lengths = np.full(n, valid_len, dtype=int)

    fm = extract_features(signals, valid_lengths)
    assert fm.X.shape == (n, TOTAL_FEATURES)


def test_no_nan_or_inf_in_output():
    """Feature matrix must contain no NaN or Inf values."""
    valid_len = 150
    max_len = 512
    rng = np.random.default_rng(3)

    signal = np.zeros((1, max_len, 6), dtype=np.float32)
    signal[0, :valid_len, :] = rng.standard_normal((valid_len, 6)).astype(np.float32)
    valid_lengths = np.array([valid_len], dtype=int)

    fm = extract_features(signal, valid_lengths)
    assert not np.any(np.isnan(fm.X)), "NaN values in feature matrix."
    assert not np.any(np.isinf(fm.X)), "Inf values in feature matrix."


def test_missing_valid_lengths_raises():
    """extract_features must raise TypeError if valid_lengths is omitted."""
    signal = np.zeros((1, 100, 6), dtype=np.float32)
    with pytest.raises(TypeError):
        extract_features(signal)  # type: ignore[call-arg]


def test_valid_length_one_sample():
    """extract_features must handle valid_length=1 without crashing."""
    max_len = 64
    signal = np.zeros((1, max_len, 6), dtype=np.float32)
    signal[0, 0, :] = 1.0
    valid_lengths = np.array([1], dtype=int)
    fm = extract_features(signal, valid_lengths)
    assert fm.X.shape == (1, TOTAL_FEATURES)
    assert not np.any(np.isnan(fm.X))
