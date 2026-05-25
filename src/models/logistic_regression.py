"""
src/models/logistic_regression.py
----------------------------------
Logistic Regression classifier for PD / HC binary classification on the PADS
benchmark.

Contract
--------
- Receives pre-normalised feature matrices (z-scored per fold in run_pipeline.py).
  No normalisation is performed here.
- Class weights are computed exclusively from the training labels passed to
  fit(), never from test or full-dataset labels.
- predict_proba() returns probabilities for both classes; the positive-class
  (PD, label=1) column is column index 1 and is used by the metrics module.
- No CV logic, no feature extraction, no preprocessing, no metrics live here.

Typical usage in run_pipeline.py:
    model = LogisticRegressionModel()
    model.fit(X_train_norm, y_train)
    proba = model.predict_proba(X_test_norm)   # shape (N_test, 2)
    y_pred = model.predict(X_test_norm)        # shape (N_test,)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Solver chosen for compatibility with L2 penalty and binary problems.
#: 'lbfgs' converges well on medium-dimensional feature vectors (≤ a few
#: hundred features) and supports warm_start if needed later.
DEFAULT_SOLVER: str = "lbfgs"

#: Regularisation penalty. L2 is appropriate as a default; the pipeline can
#: tune C via cross-validation if needed.
DEFAULT_PENALTY: str = "l2"

#: Inverse regularisation strength. Smaller = stronger regularisation.
#: 1.0 is the sklearn default; tune in run_pipeline.py if desired.
DEFAULT_C: float = 1.0

#: Maximum number of solver iterations. Increase if convergence warnings appear.
DEFAULT_MAX_ITER: int = 1000

#: Random state for reproducibility (affects solver initialisation).
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class LogisticRegressionResult:
    """Inference output returned by LogisticRegressionModel.predict_proba().

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
# Model
# ---------------------------------------------------------------------------

class LogisticRegressionModel:
    """Wrapper around sklearn LogisticRegression for the PADS benchmark.

    Encapsulates class-weight computation from training labels, fit/predict
    interface, and basic state validation. All hyperparameters are set at
    construction time and can be overridden for tuning runs.

    Parameters
    ----------
    C : float
        Inverse regularisation strength. Default: 1.0.
    penalty : str
        Regularisation type ('l2' or 'none'). Default: 'l2'.
    solver : str
        Optimisation solver. Default: 'lbfgs'.
    max_iter : int
        Maximum solver iterations. Default: 1000.
    random_state : int
        Random seed for reproducibility. Default: 42.

    Attributes
    ----------
    model_ : sklearn.linear_model.LogisticRegression
        Fitted sklearn model. Available after fit() is called.
    classes_ : np.ndarray
        Class labels seen during fit(). Shape (2,), values [0, 1].
    class_weight_ : dict[int, float]
        Per-class weights computed from training labels during fit().
    is_fitted_ : bool
        True after fit() has been called successfully.
    """

    def __init__(
        self,
        C: float = DEFAULT_C,
        penalty: str = DEFAULT_PENALTY,
        solver: str = DEFAULT_SOLVER,
        max_iter: int = DEFAULT_MAX_ITER,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

        self.model_: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None
        self.class_weight_: dict[int, float] | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_X(self, X: np.ndarray, name: str = "X") -> None:
        """Validate that X is a 2-D finite float array."""
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy ndarray, got {type(X).__name__}."
            )
        if X.ndim != 2:
            raise ValueError(
                f"{name} must be 2-D (N, F), got shape {X.shape}."
            )
        if X.shape[1] == 0:
            raise ValueError(f"{name} has zero features (shape {X.shape}).")
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

        Uses sklearn's 'balanced' strategy:
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
        classes = np.array([0, 1])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.debug(
            "Class weights computed from training labels: HC=%.4f, PD=%.4f.",
            weight_dict[0],
            weight_dict[1],
        )
        return weight_dict

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_proba() "
                "or predict()."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionModel":
        """Fit the logistic regression model on training data.

        Class weights are derived exclusively from y_train using the
        'balanced' strategy. The fitted model is stored in self.model_.

        Parameters
        ----------
        X_train : np.ndarray
            Shape (N_train, F), dtype float64. Pre-normalised feature matrix
            for the training fold. Must contain only finite values.
        y_train : np.ndarray
            Shape (N_train,), dtype int. Binary training labels (0=HC, 1=PD).
            Must contain both classes.

        Returns
        -------
        LogisticRegressionModel
            self, to allow method chaining.

        Raises
        ------
        TypeError
            If X_train or y_train are not numpy arrays.
        ValueError
            If shapes are invalid, values are non-finite, or labels are
            outside {0, 1} or missing a class.
        """
        self._validate_X(X_train, name="X_train")
        self._validate_y(y_train, name="y_train")
        if len(X_train) != len(y_train):
            raise ValueError(
                f"X_train and y_train length mismatch: "
                f"{len(X_train)} vs {len(y_train)}."
            )

        # Compute class weights from training labels only.
        self.class_weight_ = self._compute_class_weights(y_train)
        self.classes_ = np.array([0, 1])

        self.model_ = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight_,
        )
        self.model_.fit(X_train, y_train)
        self.is_fitted_ = True

        logger.info(
            "LogisticRegressionModel fitted: %d samples, %d features, "
            "C=%.4f, penalty=%s, class_weight={0: %.4f, 1: %.4f}.",
            len(y_train),
            X_train.shape[1],
            self.C,
            self.penalty,
            self.class_weight_[0],
            self.class_weight_[1],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for test samples.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, F), dtype float64. Pre-normalised feature matrix.
            Must have the same number of features F as X_train passed to fit().

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
            If X is invalid or has a different number of features than seen
            during training.
        """
        self._check_is_fitted()
        self._validate_X(X, name="X")

        n_train_features = self.model_.coef_.shape[1]  # type: ignore[union-attr]
        if X.shape[1] != n_train_features:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted on "
                f"{n_train_features} features."
            )

        proba: np.ndarray = self.model_.predict_proba(X)  # type: ignore[union-attr]
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
            Shape (N, F), dtype float64. Pre-normalised feature matrix.

        Returns
        -------
        np.ndarray
            Shape (N,), dtype int. Predicted labels in {0, 1}.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If X is invalid or feature count mismatches the fitted model.
        """
        proba = self.predict_proba(X)
        return proba.argmax(axis=1).astype(int)

    def get_result(self, X: np.ndarray) -> LogisticRegressionResult:
        """Return a LogisticRegressionResult containing probabilities and hard labels.

        Convenience wrapper that calls predict_proba() and predict() in a
        single pass and bundles the outputs into a typed dataclass.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, F), dtype float64. Pre-normalised feature matrix.

        Returns
        -------
        LogisticRegressionResult
            Contains proba (N, 2), predicted_labels (N,), and n_samples.
        """
        proba = self.predict_proba(X)
        predicted_labels = proba.argmax(axis=1).astype(int)
        return LogisticRegressionResult(proba=proba, predicted_labels=predicted_labels)