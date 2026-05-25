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

Early stopping (v2 stability improvement)
------------------------------------------
fit() reserves a small stratified validation split from the training fold
(val_fraction=0.15 by default). Validation loss is evaluated at the end of
each epoch using the weighted cross-entropy criterion. Training stops when
validation loss has not improved for `patience` consecutive epochs (default
patience=8). The network weights from the epoch with the lowest validation
loss are restored before fit() returns. No test-fold data is involved at any
point. The full training fold (train + val) is never seen by the model
simultaneously during training — val samples are held out for monitoring only.

Typical usage in run_pipeline.py:
    model = CNN1DModel(max_len=2048)
    model.fit(X_train_norm, y_train)          # X shape: (N_train, 2048, 6)
    proba  = model.predict_proba(X_test_norm) # shape: (N_test, 2)
    y_pred = model.predict(X_test_norm)        # shape: (N_test,)
"""

from __future__ import annotations

import copy
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

#: Default training epochs (upper bound; early stopping may stop earlier).
DEFAULT_EPOCHS: int = 50

#: Default mini-batch size.
DEFAULT_BATCH_SIZE: int = 32

#: Default learning rate for Adam.
DEFAULT_LR: float = 1e-3

#: Default dropout probability in the classification head.
DEFAULT_DROPOUT: float = 0.5

#: Default random seed (weight initialisation + DataLoader shuffle).
DEFAULT_RANDOM_STATE: int = 42

#: Fraction of the training fold held out as the internal validation set.
#: Stratified by label. Kept small to maximise training data.
DEFAULT_VAL_FRACTION: float = 0.15

#: Number of consecutive epochs with no validation-loss improvement before
#: training is stopped and best weights are restored.
DEFAULT_PATIENCE: int = 8


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
    early stopping with best-weight restoration, fit/predict interface, and
    basic state validation. All hyperparameters are set at construction time
    and can be overridden for tuning runs.

    Class imbalance is handled by passing class weights to
    nn.CrossEntropyLoss(weight=...), computed exclusively from training labels.

    Early stopping monitors validation loss computed on a stratified holdout
    split (val_fraction) drawn from the training fold only. When validation
    loss has not improved for `patience` consecutive epochs the training loop
    is terminated and the weights from the best epoch are restored. No test
    data is used at any stage of this process.

    Parameters
    ----------
    max_len : int
        Sequence length (time steps) of each input sample. Must match
        PreprocessedDataset.max_len for the current fold's dataset. Used only
        for input validation; the network is length-agnostic via
        AdaptiveAvgPool1d.
    epochs : int
        Maximum number of full passes over the training set. Default: 50.
        Early stopping may halt training before this limit is reached.
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
    val_fraction : float
        Fraction of the training fold to reserve as an internal validation
        set for early stopping. Stratified by label. Default: 0.15.
    patience : int
        Number of consecutive epochs with no validation-loss improvement
        before early stopping is triggered. Default: 8.

    Attributes
    ----------
    network_ : _CNN1DNetwork
        Fitted PyTorch model (weights from the best validation-loss epoch).
        Available after fit() is called.
    class_weight_ : dict[int, float]
        Per-class weights computed from training labels during fit().
    train_loss_history_ : list[float]
        Mean training cross-entropy loss per epoch, recorded during fit().
    val_loss_history_ : list[float]
        Mean validation cross-entropy loss per epoch, recorded during fit().
    best_epoch_ : int
        Zero-based epoch index at which the best validation loss was achieved.
    best_val_loss_ : float
        Best (lowest) validation loss observed during training.
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
        val_fraction: float = DEFAULT_VAL_FRACTION,
        patience: int = DEFAULT_PATIENCE,
    ) -> None:
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}.")
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(
                f"val_fraction must be in (0, 1), got {val_fraction}."
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}.")

        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.random_state = random_state
        self.val_fraction = val_fraction
        self.patience = patience

        self.device_: torch.device = self._resolve_device(device)

        self.network_: _CNN1DNetwork | None = None
        self.class_weight_: dict[int, float] | None = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []
        self.best_epoch_: int = 0
        self.best_val_loss_: float = float("inf")
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

    def _stratified_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split X and y into train/val subsets, stratified by label.

        Each class is split independently at val_fraction so that the
        class ratio is approximately preserved in both subsets. The split
        is deterministic given self.random_state (set before this call).

        Parameters
        ----------
        X : np.ndarray
            Shape (N, max_len, 6).
        y : np.ndarray
            Shape (N,), values in {0, 1}.

        Returns
        -------
        X_tr, y_tr, X_val, y_val : np.ndarray
            Training and validation subsets. Both contain both classes.

        Raises
        ------
        ValueError
            If the validation split would leave fewer than 1 sample in either
            class in either subset.
        """
        rng = np.random.RandomState(self.random_state)
        train_idx: list[int] = []
        val_idx: list[int] = []

        for cls in (0, 1):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * self.val_fraction)))
            # Guard: always keep at least 1 sample in training for this class.
            n_val = min(n_val, len(cls_idx) - 1)
            if n_val < 1:
                raise ValueError(
                    f"Val split for class {cls} would leave 0 training samples "
                    f"(class count={len(cls_idx)}, val_fraction={self.val_fraction}). "
                    "Reduce val_fraction or increase fold size."
                )
            val_idx.extend(cls_idx[:n_val].tolist())
            train_idx.extend(cls_idx[n_val:].tolist())

        train_idx_arr = np.array(train_idx, dtype=int)
        val_idx_arr = np.array(val_idx, dtype=int)
        return X[train_idx_arr], y[train_idx_arr], X[val_idx_arr], y[val_idx_arr]

    def _eval_loss(
        self,
        loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        n_samples: int,
    ) -> float:
        """Compute mean weighted cross-entropy loss over a DataLoader in eval mode."""
        self.network_.eval()  # type: ignore[union-attr]
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)
                logits = self.network_(X_batch)  # type: ignore[union-attr]
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
        return total_loss / n_samples

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
        """Train the 1D-CNN on pre-normalised sequence data with early stopping.

        A stratified validation split (val_fraction of X_train) is held out
        from the training loop and used exclusively to monitor validation loss.
        Training stops when validation loss has not improved for `patience`
        consecutive epochs. The network weights from the epoch with the lowest
        validation loss are restored before this method returns.

        Class weights are computed from all labels in y_train (including val
        labels) so that the loss function reflects the full training-fold class
        distribution. No test data is used at any point.

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
            {0, 1}, a class is missing from y_train, or the val split cannot
            be constructed (too few samples per class).
        """
        self._validate_X(X_train, name="X_train")
        self._validate_y(y_train, name="y_train")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}."
            )

        self._set_seed()

        # --- Class weights from the full training-fold labels ---
        # Computed before the val split so the loss weighting reflects the
        # true training-fold class distribution.
        self.class_weight_ = self._compute_class_weights(y_train)
        loss_weights = torch.tensor(
            [self.class_weight_[0], self.class_weight_[1]],
            dtype=torch.float32,
            device=self.device_,
        )

        # --- Stratified train / val split (from training fold only) ---
        X_tr, y_tr, X_val, y_val = self._stratified_val_split(X_train, y_train)
        logger.info(
            "CNN1DModel: train split %d samples, val split %d samples "
            "(val_fraction=%.2f).",
            len(y_tr),
            len(y_val),
            self.val_fraction,
        )

        # --- Build network and optimiser ---
        self.network_ = _CNN1DNetwork(dropout=self.dropout).to(self.device_)
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=loss_weights)

        # --- DataLoaders ---
        train_loader = DataLoader(
            TensorDataset(self._to_tensor_X(X_tr), self._to_tensor_y(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        val_loader = DataLoader(
            TensorDataset(self._to_tensor_X(X_val), self._to_tensor_y(y_val)),
            batch_size=self.batch_size,
            shuffle=False,
        )

        # --- Training loop with early stopping ---
        self.train_loss_history_ = []
        self.val_loss_history_ = []
        self.best_val_loss_ = float("inf")
        self.best_epoch_ = 0
        best_weights: dict = copy.deepcopy(self.network_.state_dict())
        epochs_no_improve: int = 0

        for epoch in range(self.epochs):
            # -- Training pass --
            self.network_.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)
                optimizer.zero_grad()
                logits = self.network_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * len(y_batch)

            mean_train_loss = epoch_train_loss / len(y_tr)
            self.train_loss_history_.append(mean_train_loss)

            # -- Validation pass --
            mean_val_loss = self._eval_loss(val_loader, criterion, len(y_val))
            self.val_loss_history_.append(mean_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.debug(
                    "Epoch %d/%d — train_loss: %.4f, val_loss: %.4f.",
                    epoch + 1,
                    self.epochs,
                    mean_train_loss,
                    mean_val_loss,
                )

            # -- Early stopping check --
            if mean_val_loss < self.best_val_loss_:
                self.best_val_loss_ = mean_val_loss
                self.best_epoch_ = epoch
                best_weights = copy.deepcopy(self.network_.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d/%d "
                        "(no val_loss improvement for %d consecutive epochs).",
                        epoch + 1,
                        self.epochs,
                        self.patience,
                    )
                    break

        # --- Restore best weights ---
        self.network_.load_state_dict(best_weights)
        self.is_fitted_ = True

        logger.info(
            "CNN1DModel fitted: %d train samples (%d after val split), "
            "max_len=%d, best_epoch=%d, best_val_loss=%.4f, "
            "batch_size=%d, lr=%.5f, device=%s, "
            "class_weight={0: %.4f, 1: %.4f}.",
            len(y_train),
            len(y_tr),
            self.max_len,
            self.best_epoch_ + 1,
            self.best_val_loss_,
            self.batch_size,
            self.lr,
            str(self.device_),
            self.class_weight_[0],
            self.class_weight_[1],
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
