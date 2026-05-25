"""
src/models/cnn1d.py
--------------------
Compact 1D-CNN classifier for PD / HC binary classification on the PADS
benchmark.

Contract
--------
- Receives pre-normalised raw sequence arrays of shape (N, max_len, 6)
  (z-scored per fold in run_pipeline.py). No normalisation is performed here.
- Input layout matches PreprocessedDataset.signals: axis 0 = samples,
  axis 1 = time steps, axis 2 = channels (acc_x/y/z, gyr_x/y/z).
  PyTorch Conv1d expects (N, C, L); the model transposes internally.
- Class weights are computed exclusively from the training labels passed to
  fit(), never from test or full-dataset labels.
- predict_proba() returns probabilities for both classes; the positive-class
  (PD, label=1) column is column index 1 and is used by the metrics module.
- No CV logic, no feature extraction, no preprocessing, no metrics live here.

Architecture (compact, ~50k parameters):
    Input  : (N, 6, max_len)          ← transposed from (N, max_len, 6)
    Block 1: Conv1d(6  → 32, k=7, p=3) → BN → ReLU → MaxPool(2)
    Block 2: Conv1d(32 → 64, k=5, p=2) → BN → ReLU → MaxPool(2)
    Block 3: Conv1d(64 → 128, k=3, p=1) → BN → ReLU → AdaptiveAvgPool → (N, 128)
    Head   : Dropout(0.5) → Linear(128 → 2)
    Output : softmax probabilities  (N, 2)

Typical usage in run_pipeline.py:
    model = CNN1DModel(max_len=2048)
    model.fit(X_train_norm, y_train)          # X shape: (N_train, 2048, 6)
    proba  = model.predict_proba(X_test_norm) # shape: (N_test, 2)
    y_pred = model.predict(X_test_norm)        # shape: (N_test,)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Input channels — 6-channel IMU (acc_x/y/z + gyr_x/y/z).
N_INPUT_CHANNELS: int = 6

#: Number of output classes (HC=0, PD=1).
N_CLASSES: int = 2

#: Default training epochs.
DEFAULT_EPOCHS: int = 50

#: Default mini-batch size.
DEFAULT_BATCH_SIZE: int = 32

#: Default learning rate for Adam.
DEFAULT_LR: float = 1e-3

#: Default dropout probability in the classification head.
DEFAULT_DROPOUT: float = 0.5

#: Default random seed (weight initialisation + DataLoader shuffle).
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class CNN1DResult:
    """Inference output returned by CNN1DModel.get_result().

    Attributes
    ----------
    proba : np.ndarray
        Shape (N, 2), dtype float64. Predicted class probabilities.
        Column 0 = P(HC), column 1 = P(PD). Rows sum to 1.0.
    predicted_labels : np.ndarray
        Shape (N,), dtype int. Hard predictions (argmax of proba).
        Values in {0, 1}: 0 = HC, 1 = PD.
    n_samples : int
        Number of test samples N.
    """

    proba: np.ndarray
    predicted_labels: np.ndarray
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_samples = len(self.predicted_labels)


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU as a single reusable unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # BN subsumes bias
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _CNN1DNetwork(nn.Module):
    """Compact 1D convolutional network for binary IMU classification.

    Expects input of shape (N, C_in, L) where C_in=6 and L=max_len.
    Returns raw logits of shape (N, 2).
    """

    def __init__(self, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()

        # Three convolutional blocks with progressive channel widening
        # and spatial downsampling.
        self.block1 = nn.Sequential(
            _ConvBlock(in_channels=6, out_channels=32, kernel_size=7, padding=3),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            _ConvBlock(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            _ConvBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # Collapse temporal dimension to a fixed-size vector regardless
            # of input length. This makes the network length-agnostic and
            # avoids hard-coding max_len in the linear layer.
            nn.AdaptiveAvgPool1d(output_size=1),
        )

        # Classification head.
        self.head = nn.Sequential(
            nn.Flatten(),          # (N, 128, 1) → (N, 128)
            nn.Dropout(p=dropout),
            nn.Linear(128, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (N, 6, L). Pre-normalised IMU sequences.

        Returns
        -------
        torch.Tensor
            Shape (N, 2). Raw logits (not softmaxed).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class CNN1DModel:
    """1D-CNN classifier for PD / HC binary classification on the PADS benchmark.

    Wraps _CNN1DNetwork with training loop, class-weight-based loss weighting,
    fit/predict interface, and basic state validation. All hyperparameters are
    set at construction time and can be overridden for tuning runs.

    Class imbalance is handled by passing class weights to
    nn.CrossEntropyLoss(weight=...), computed exclusively from training labels.

    Parameters
    ----------
    max_len : int
        Sequence length (time steps) of each input sample. Must match
        PreprocessedDataset.max_len for the current fold's dataset. Used only
        for input validation; the network is length-agnostic via
        AdaptiveAvgPool1d.
    epochs : int
        Number of full passes over the training set. Default: 50.
    batch_size : int
        Mini-batch size for training and inference. Default: 32.
    lr : float
        Adam learning rate. Default: 1e-3.
    dropout : float
        Dropout probability in the classification head. Default: 0.5.
    random_state : int
        Seed for torch, numpy, and DataLoader shuffle reproducibility.
        Default: 42.
    device : str | None
        PyTorch device string ('cpu', 'cuda', 'mps'). If None, auto-detected:
        CUDA → MPS → CPU in order of preference. Default: None.

    Attributes
    ----------
    network_ : _CNN1DNetwork
        Fitted PyTorch model. Available after fit() is called.
    class_weight_ : dict[int, float]
        Per-class weights computed from training labels during fit().
    train_loss_history_ : list[float]
        Mean cross-entropy loss per epoch, recorded during fit().
    is_fitted_ : bool
        True after fit() has been called successfully.
    device_ : torch.device
        Resolved device used for training and inference.
    """

    def __init__(
        self,
        max_len: int,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        lr: float = DEFAULT_LR,
        dropout: float = DEFAULT_DROPOUT,
        random_state: int = DEFAULT_RANDOM_STATE,
        device: str | None = None,
    ) -> None:
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}.")
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state

        self.device_: torch.device = self._resolve_device(device)

        self.network_: _CNN1DNetwork | None = None
        self.class_weight_: dict[int, float] | None = None
        self.train_loss_history_: list[float] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        """Return the best available device if none is specified."""
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _set_seed(self) -> None:
        """Set global seeds for reproducibility."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _validate_X(self, X: np.ndarray, name: str = "X") -> None:
        """Validate that X is a 3-D finite float array of shape (N, max_len, 6)."""
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy ndarray, got {type(X).__name__}."
            )
        if X.ndim != 3:
            raise ValueError(
                f"{name} must be 3-D (N, max_len, 6), got shape {X.shape}."
            )
        if X.shape[1] != self.max_len:
            raise ValueError(
                f"{name} has time dimension {X.shape[1]} but "
                f"max_len={self.max_len}. Ensure the same max_len is used "
                "across training and inference, and that it matches "
                "PreprocessedDataset.max_len."
            )
        if X.shape[2] != N_INPUT_CHANNELS:
            raise ValueError(
                f"{name} must have {N_INPUT_CHANNELS} channels (axis 2), "
                f"got {X.shape[2]}."
            )
        if not np.isfinite(X).all():
            raise ValueError(
                f"{name} contains NaN or Inf values. "
                "Ensure normalisation was applied correctly in run_pipeline.py."
            )

    def _validate_y(self, y: np.ndarray, name: str = "y") -> None:
        """Validate that y is a 1-D integer array with values in {0, 1}."""
        if not isinstance(y, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy ndarray, got {type(y).__name__}."
            )
        if y.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D (N,), got shape {y.shape}."
            )
        unique_labels = set(y.tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"{name} must contain only values in {{0, 1}} (HC=0, PD=1), "
                f"got: {unique_labels}."
            )
        if unique_labels != {0, 1}:
            raise ValueError(
                f"{name} must contain both classes (0 and 1). "
                f"Found only: {unique_labels}. "
                "Check fold construction in cross_val.py."
            )

    def _compute_class_weights(self, y_train: np.ndarray) -> dict[int, float]:
        """Compute balanced class weights from training labels only.

        Uses the 'balanced' formula:
            weight_c = n_samples / (n_classes * count_c)

        Parameters
        ----------
        y_train : np.ndarray
            Shape (N_train,). Training labels, values in {0, 1}.

        Returns
        -------
        dict[int, float]
            Mapping {0: weight_hc, 1: weight_pd}.
        """
        n = len(y_train)
        classes = np.array([0, 1])
        weights = np.array([
            n / (2.0 * float(np.sum(y_train == c)))
            for c in classes
        ])
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.debug(
            "Class weights computed from training labels: HC=%.4f, PD=%.4f.",
            weight_dict[0],
            weight_dict[1],
        )
        return weight_dict

    def _to_tensor_X(self, X: np.ndarray) -> torch.Tensor:
        """Convert (N, max_len, 6) numpy array to (N, 6, max_len) float32 tensor.

        Conv1d expects (N, C, L); the input convention is (N, L, C) to match
        PreprocessedDataset.signals. The transpose is applied here, not in the
        network, to keep the network self-contained and testable.
        """
        # (N, L, C) → (N, C, L)
        return torch.from_numpy(
            X.transpose(0, 2, 1).astype(np.float32)
        )

    def _to_tensor_y(self, y: np.ndarray) -> torch.Tensor:
        """Convert (N,) integer label array to long tensor."""
        return torch.from_numpy(y.astype(np.int64))

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted."""
        if not self.is_fitted_ or self.network_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_proba() "
                "or predict()."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "CNN1DModel":
        """Train the 1D-CNN on pre-normalised sequence data.

        Builds and initialises a fresh _CNN1DNetwork, computes class weights
        from y_train only, then runs the training loop for self.epochs epochs
        using Adam and weighted cross-entropy loss. Loss per epoch is recorded
        in self.train_loss_history_.

        Parameters
        ----------
        X_train : np.ndarray
            Shape (N_train, max_len, 6), dtype float32 or float64.
            Pre-normalised IMU sequences for the training fold.
            Must contain only finite values.
        y_train : np.ndarray
            Shape (N_train,), dtype int. Binary training labels (0=HC, 1=PD).
            Must contain both classes.

        Returns
        -------
        CNN1DModel
            self, to allow method chaining.

        Raises
        ------
        TypeError
            If X_train or y_train are not numpy arrays.
        ValueError
            If shapes are invalid, values are non-finite, labels are outside
            {0, 1}, or a class is missing from y_train.
        """
        self._validate_X(X_train, name="X_train")
        self._validate_y(y_train, name="y_train")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}."
            )

        self._set_seed()

        # Class weights computed from training labels only.
        self.class_weight_ = self._compute_class_weights(y_train)
        loss_weights = torch.tensor(
            [self.class_weight_[0], self.class_weight_[1]],
            dtype=torch.float32,
            device=self.device_,
        )

        # Build a fresh network for this fold.
        self.network_ = _CNN1DNetwork(dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

        # DataLoader — shuffle uses the fixed seed set above.
        X_tensor = self._to_tensor_X(X_train)
        y_tensor = self._to_tensor_y(y_train)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(self.random_state),
        )

        self.train_loss_history_ = []
        self.network_.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                optimizer.zero_grad()
                logits = self.network_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(y_batch)

            mean_loss = epoch_loss / len(y_train)
            self.train_loss_history_.append(mean_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.debug(
                    "Epoch %d/%d — mean loss: %.4f.",
                    epoch + 1,
                    self.epochs,
                    mean_loss,
                )

        self.is_fitted_ = True
        logger.info(
            "CNN1DModel fitted: %d samples, max_len=%d, epochs=%d, "
            "batch_size=%d, lr=%.5f, device=%s, "
            "class_weight={0: %.4f, 1: %.4f}, "
            "final_loss=%.4f.",
            len(y_train),
            self.max_len,
            self.epochs,
            self.batch_size,
            self.lr,
            str(self.device_),
            self.class_weight_[0],
            self.class_weight_[1],
            self.train_loss_history_[-1],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for test samples.

        Runs a forward pass in evaluation mode (BatchNorm and Dropout
        behave deterministically), then applies softmax to convert logits
        to probabilities.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, max_len, 6), dtype float32 or float64.
            Pre-normalised IMU sequences. max_len must match the value
            passed to __init__ and used during fit().

        Returns
        -------
        np.ndarray
            Shape (N, 2), dtype float64. Column 0 = P(HC), column 1 = P(PD).
            Rows sum to 1.0. Pass column 1 to the metrics module as the
            positive-class probability.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If X shape is invalid or max_len mismatches.
        """
        self._check_is_fitted()
        self._validate_X(X, name="X")

        X_tensor = self._to_tensor_X(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.network_.eval()  # type: ignore[union-attr]
        proba_list: list[np.ndarray] = []

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device_)
                logits = self.network_(X_batch)  # type: ignore[union-attr]
                batch_proba = torch.softmax(logits, dim=1).cpu().numpy()
                proba_list.append(batch_proba)

        proba = np.concatenate(proba_list, axis=0).astype(np.float64)
        logger.debug(
            "predict_proba called: %d samples, output shape %s.",
            len(X),
            proba.shape,
        )
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions for test samples.

        Equivalent to argmax over predict_proba(). Provided for convenience;
        the primary metric (Balanced Accuracy from thresholded predictions)
        can also be computed from predict_proba() output.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, max_len, 6), dtype float32 or float64.

        Returns
        -------
        np.ndarray
            Shape (N,), dtype int. Predicted labels in {0, 1}.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If X shape is invalid.
        """
        proba = self.predict_proba(X)
        return proba.argmax(axis=1).astype(int)

    def get_result(self, X: np.ndarray) -> CNN1DResult:
        """Return a CNN1DResult containing probabilities and hard labels.

        Convenience wrapper that calls predict_proba() in a single pass and
        bundles the outputs into a typed dataclass.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, max_len, 6), dtype float32 or float64.

        Returns
        -------
        CNN1DResult
            Contains proba (N, 2), predicted_labels (N,), and n_samples.
        """
        proba = self.predict_proba(X)
        predicted_labels = proba.argmax(axis=1).astype(int)
        return CNN1DResult(proba=proba, predicted_labels=predicted_labels)
