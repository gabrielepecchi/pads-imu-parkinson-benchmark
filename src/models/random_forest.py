"""
src/models/random_forest.py
----------------------------
Random Forest classifier for PD / HC binary classification on the PADS
benchmark.

Contract
--------
- Receives pre-normalised feature matrices (z-scored per fold in run_pipeline.py).
  No normalisation is performed here.
- Class weights are computed exclusively from the training labels passed to
  fit(), never from test or full-dataset labels.
- predict_proba() returns probabilities for both classes; the positive-class
  (PD, label=1) column is column index 1 and is used by the metrics module.
- feature_importances() returns mean decrease in impurity (MDI) importances
  from the fitted forest, aligned with the feature columns of X_train.
- No CV logic, no feature extraction, no preprocessing, no metrics live here.

Typical usage in run_pipeline.py:
    model = RandomForestModel()
    model.fit(X_train_norm, y_train)
    proba       = model.predict_proba(X_test_norm)    # shape (N_test, 2)
    y_pred      = model.predict(X_test_norm)           # shape (N_test,)
    importances = model.feature_importances()          # FeatureImportanceResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of trees in the forest. 500 gives stable OOB estimates and
#: importances without excessive memory cost for F=120 features.
DEFAULT_N_ESTIMATORS: int = 500

#: Maximum features considered at each split. 'sqrt' is the standard choice
#: for classification and is the sklearn default.
DEFAULT_MAX_FEATURES: str = "sqrt"

#: Minimum samples required to split an internal node. 2 (sklearn default)
#: allows deep trees; increase to 5–10 to reduce variance if overfitting.
DEFAULT_MIN_SAMPLES_SPLIT: int = 2

#: Minimum samples required at a leaf node. 1 (sklearn default) allows
#: pure leaves; increase alongside min_samples_split if needed.
DEFAULT_MIN_SAMPLES_LEAF: int = 1

#: Whether to use bootstrap sampling. Required for OOB score estimation.
DEFAULT_BOOTSTRAP: bool = True

#: Expose out-of-bag score for a quick in-fold generalisation estimate.
#: Requires bootstrap=True.
DEFAULT_OOB_SCORE: bool = True

#: Number of parallel jobs. -1 uses all available CPU cores.
DEFAULT_N_JOBS: int = -1

#: Random state for reproducibility (tree construction and bootstrap sampling).
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class RandomForestResult:
    """Inference output returned by RandomForestModel.get_result().

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


@dataclass
class FeatureImportanceResult:
    """Feature importance output returned by RandomForestModel.feature_importances().

    Attributes
    ----------
    importances : np.ndarray
        Shape (F,), dtype float64. Mean decrease in impurity (MDI) per
        feature, normalised so that importances.sum() == 1.0.
    importances_std : np.ndarray
        Shape (F,), dtype float64. Standard deviation of MDI importances
        across trees. Useful for plotting error bars.
    feature_names : list[str] | None
        Length F. Feature names aligned with importances, if provided to
        feature_importances(). None if no names were supplied.
    n_features : int
        Total number of features F.
    """

    importances: np.ndarray
    importances_std: np.ndarray
    feature_names: list[str] | None
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features = len(self.importances)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RandomForestModel:
    """Wrapper around sklearn RandomForestClassifier for the PADS benchmark.

    Encapsulates class-weight computation from training labels, fit/predict
    interface, feature importance extraction, and basic state validation.
    All hyperparameters are set at construction time and can be overridden
    for tuning runs.

    Class imbalance is handled via per-sample weights passed to fit(),
    computed from training labels only using sklearn's 'balanced' strategy.
    This is the correct interface for RandomForestClassifier: unlike
    LogisticRegression, RF does not use a class_weight constructor parameter
    in the same way; explicit sample_weight at fit() time is the supported
    and documented approach.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest. Default: 500.
    max_features : str | int | float
        Number of features to consider at each split. Default: 'sqrt'.
    min_samples_split : int
        Minimum samples required to split an internal node. Default: 2.
    min_samples_leaf : int
        Minimum samples required at a leaf node. Default: 1.
    bootstrap : bool
        Whether to use bootstrap sampling. Default: True.
    oob_score : bool
        Whether to compute out-of-bag score. Requires bootstrap=True.
        Default: True.
    n_jobs : int
        Number of parallel jobs (-1 = all cores). Default: -1.
    random_state : int
        Random seed for reproducibility. Default: 42.

    Attributes
    ----------
    model_ : sklearn.ensemble.RandomForestClassifier
        Fitted sklearn model. Available after fit() is called.
    classes_ : np.ndarray
        Class labels seen during fit(). Shape (2,), values [0, 1].
    class_weight_ : dict[int, float]
        Per-class weights computed from training labels during fit().
    oob_score_ : float | None
        Out-of-bag accuracy estimate. Set after fit() when oob_score=True,
        otherwise None.
    is_fitted_ : bool
        True after fit() has been called successfully.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_features: str | int | float = DEFAULT_MAX_FEATURES,
        min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        bootstrap: bool = DEFAULT_BOOTSTRAP,
        oob_score: bool = DEFAULT_OOB_SCORE,
        n_jobs: int = DEFAULT_N_JOBS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model_: RandomForestClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.class_weight_: dict[int, float] | None = None
        self.oob_score_: float | None = None
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
        """Derive per-class weights from training labels only.

        Uses the 'balanced' formula:
            weight_c = n_samples / (n_classes * count_c)

        Stored on self for logging and downstream inspection. The actual
        imbalance correction for the forest is applied via per-sample weights
        at fit() time (see _compute_sample_weights).

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
        weights = np.array([
            len(y_train) / (2.0 * float(np.sum(y_train == c)))
            for c in classes
        ])
        weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        logger.debug(
            "Class weights computed from training labels: HC=%.4f, PD=%.4f.",
            weight_dict[0],
            weight_dict[1],
        )
        return weight_dict

    def _compute_sample_weights(self, y_train: np.ndarray) -> np.ndarray:
        """Derive per-sample weights from training labels only.

        sklearn's RandomForestClassifier does not honour the class_weight
        constructor parameter for per-sample reweighting during tree building
        in the same way as linear models. Passing explicit sample_weight to
        fit() is the correct and documented approach.

        Parameters
        ----------
        y_train : np.ndarray
            Shape (N_train,). Training labels, values in {0, 1}.

        Returns
        -------
        np.ndarray
            Shape (N_train,), dtype float64. Per-sample weights aligned with
            y_train. Samples from the minority class receive higher weight.
        """
        return compute_sample_weight(class_weight="balanced", y=y_train)

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted."""
        if not self.is_fitted_ or self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_proba(), "
                "predict(), or feature_importances()."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "RandomForestModel":
        """Fit the random forest on training data.

        Per-sample weights are derived exclusively from y_train using the
        'balanced' strategy and passed to the sklearn fit() call. The fitted
        model is stored in self.model_.

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
        RandomForestModel
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

        # Class weights stored for logging; sample weights used for fit().
        self.class_weight_ = self._compute_class_weights(y_train)
        sample_weights = self._compute_sample_weights(y_train)
        self.classes_ = np.array([0, 1])

        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model_.fit(X_train, y_train, sample_weight=sample_weights)
        self.is_fitted_ = True

        if self.oob_score and self.bootstrap:
            self.oob_score_ = float(self.model_.oob_score_)
            logger.info(
                "RandomForestModel fitted: %d samples, %d features, "
                "n_estimators=%d, max_features=%s, "
                "class_weight={0: %.4f, 1: %.4f}, OOB accuracy=%.4f.",
                len(y_train),
                X_train.shape[1],
                self.n_estimators,
                self.max_features,
                self.class_weight_[0],
                self.class_weight_[1],
                self.oob_score_,
            )
        else:
            self.oob_score_ = None
            logger.info(
                "RandomForestModel fitted: %d samples, %d features, "
                "n_estimators=%d, max_features=%s, "
                "class_weight={0: %.4f, 1: %.4f}.",
                len(y_train),
                X_train.shape[1],
                self.n_estimators,
                self.max_features,
                self.class_weight_[0],
                self.class_weight_[1],
            )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class probabilities for test samples.

        Probabilities are averaged over all trees in the forest (sklearn
        default: mean of per-tree class proportions at leaf nodes).

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

        n_train_features: int = self.model_.n_features_in_  # type: ignore[union-attr]
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

    def feature_importances(
        self,
        feature_names: list[str] | None = None,
    ) -> FeatureImportanceResult:
        """Return mean decrease in impurity (MDI) feature importances.

        Importances are extracted from the fitted forest and represent the
        total reduction in node impurity weighted by the probability of
        reaching each node, averaged across all trees. They are normalised
        to sum to 1.0 by sklearn convention.

        Standard deviations across individual trees are also returned to
        support error-bar plots and to flag features with high inter-tree
        variance in importance.

        Parameters
        ----------
        feature_names : list[str] | None
            Optional list of feature names of length F, aligned with columns
            of X_train. If provided, stored in the result for downstream use
            (e.g. plotting top-N features). Obtain from
            extractor.build_feature_names(). Default: None.

        Returns
        -------
        FeatureImportanceResult
            Contains importances (F,), importances_std (F,), feature_names,
            and n_features.

        Raises
        ------
        RuntimeError
            If called before fit().
        ValueError
            If feature_names is provided but its length does not match the
            number of features the model was fitted on.
        """
        self._check_is_fitted()

        n_features: int = self.model_.n_features_in_  # type: ignore[union-attr]

        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries but the "
                    f"model was fitted on {n_features} features."
                )

        importances: np.ndarray = (
            self.model_.feature_importances_  # type: ignore[union-attr]
        )

        # Per-tree importances: shape (n_estimators, n_features).
        per_tree = np.array([
            tree.feature_importances_
            for tree in self.model_.estimators_  # type: ignore[union-attr]
        ])
        importances_std: np.ndarray = per_tree.std(axis=0)

        logger.debug(
            "Feature importances extracted: %d features, "
            "top-5 indices by MDI: %s.",
            n_features,
            np.argsort(importances)[::-1][:5].tolist(),
        )

        return FeatureImportanceResult(
            importances=importances,
            importances_std=importances_std,
            feature_names=feature_names,
        )

    def get_result(self, X: np.ndarray) -> RandomForestResult:
        """Return a RandomForestResult containing probabilities and hard labels.

        Convenience wrapper that calls predict_proba() in a single pass and
        bundles the outputs into a typed dataclass.

        Parameters
        ----------
        X : np.ndarray
            Shape (N, F), dtype float64. Pre-normalised feature matrix.

        Returns
        -------
        RandomForestResult
            Contains proba (N, 2), predicted_labels (N,), and n_samples.
        """
        proba = self.predict_proba(X)
        predicted_labels = proba.argmax(axis=1).astype(int)
        return RandomForestResult(proba=proba, predicted_labels=predicted_labels)
