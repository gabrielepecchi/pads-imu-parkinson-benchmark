"""
tests/test_cnn_smoke.py
-----------------------
Smoke test for the 1D-CNN network defined in src/models/cnn1d.py.
Uses tiny synthetic data only. No real dataset, no internet, no GPU required.
Forces CPU execution.
"""

import torch

from src.models.cnn1d import _CNN1DNetwork, N_CLASSES


BATCH_SIZE = 4
N_CHANNELS = 6
SEQ_LEN = 128


def test_cnn_forward_pass_output_shape():
    """Forward pass must return logits of shape (batch_size, 2)."""
    torch.manual_seed(0)
    net = _CNN1DNetwork(dropout=0.0).to("cpu")
    net.eval()

    x = torch.randn(BATCH_SIZE, N_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        logits = net(x)

    assert logits.shape == (BATCH_SIZE, N_CLASSES), (
        f"Expected output shape ({BATCH_SIZE}, {N_CLASSES}), got {tuple(logits.shape)}."
    )


def test_cnn_forward_pass_finite_values():
    """Forward pass output must contain only finite values."""
    torch.manual_seed(1)
    net = _CNN1DNetwork(dropout=0.0).to("cpu")
    net.eval()

    x = torch.randn(BATCH_SIZE, N_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        logits = net(x)

    assert torch.isfinite(logits).all(), "CNN output contains NaN or Inf values."


def test_cnn_softmax_sums_to_one():
    """Softmax of logits must sum to 1.0 per sample."""
    torch.manual_seed(2)
    net = _CNN1DNetwork(dropout=0.0).to("cpu")
    net.eval()

    x = torch.randn(BATCH_SIZE, N_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        logits = net(x)
        proba = torch.softmax(logits, dim=1)

    row_sums = proba.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(BATCH_SIZE), atol=1e-5), (
        f"Softmax rows do not sum to 1: {row_sums.tolist()}"
    )


def test_cnn_runs_on_cpu_explicitly():
    """Network must run on CPU without error even if CUDA is available."""
    net = _CNN1DNetwork(dropout=0.0).to(torch.device("cpu"))
    net.eval()

    x = torch.randn(2, N_CHANNELS, SEQ_LEN, device=torch.device("cpu"))
    with torch.no_grad():
        logits = net(x)

    assert logits.device.type == "cpu"
    assert logits.shape == (2, N_CLASSES)
