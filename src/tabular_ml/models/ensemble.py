"""Ensemble methods — stacking and blending for combining base learners."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


class StackingEnsemble:
    """Stacking ensemble with out-of-fold predictions and a meta-learner.

    Uses K-fold cross-validation on the training set to generate out-of-fold
    predicted probabilities from each base learner. These become features
    for a Logistic Regression meta-learner trained to combine them optimally.

    This avoids overfitting the meta-learner to the training predictions
    (which would be near-perfect for tree models).
    """

    def __init__(
        self,
        base_models: list[tuple[str, Any]],
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            base_models: List of (name, fitted_model) tuples.
            n_folds: Number of folds for out-of-fold prediction generation.
            random_state: Random seed for fold splitting and meta-learner.
        """
        self.base_models = base_models
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_learner = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced",
        )
        self._is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
    ) -> "StackingEnsemble":
        """Fit the meta-learner using out-of-fold predictions from base models.

        The base models are already fitted. We use K-fold to generate
        out-of-fold predictions (each sample is predicted by a model
        that didn't see it during this fold), then train the meta-learner
        on these predictions.

        Note: We use the already-fitted base models to predict on held-out
        folds. This is simpler than refitting base models per fold and is
        appropriate when base models were trained on the same training set.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            self
        """
        X = np.asarray(X_train)
        y = np.asarray(y_train)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((len(X), n_models))

        kf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_fold_val = (
                X_train.iloc[val_idx]
                if isinstance(X_train, pd.DataFrame)
                else X[val_idx]
            )

            for model_idx, (name, model) in enumerate(self.base_models):
                oof_predictions[val_idx, model_idx] = model.predict_proba(X_fold_val)[
                    :, 1
                ]

        # Train meta-learner on stacked out-of-fold predictions
        self.meta_learner.fit(oof_predictions, y)
        self._is_fitted = True

        # Store meta-learner coefficients for interpretability
        self.base_weights_ = dict(
            zip(
                [name for name, _ in self.base_models],
                self.meta_learner.coef_[0],
            )
        )

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate ensemble predictions by stacking base model probabilities.

        Args:
            X: Features to predict on.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        if not self._is_fitted:
            raise RuntimeError("StackingEnsemble is not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = np.column_stack(
            [model.predict_proba(X)[:, 1] for _, model in self.base_models]
        )

        # Meta-learner combines them
        return self.meta_learner.predict_proba(base_preds)

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Generate binary predictions.

        Args:
            X: Features to predict on.
            threshold: Classification threshold.

        Returns:
            Array of binary predictions.
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Serialize the ensemble to disk."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "StackingEnsemble":
        """Load a saved ensemble from disk."""
        return joblib.load(path)


class BlendingEnsemble:
    """Simple weighted average blending of base model probabilities.

    Optimizes weights by grid search on validation set PR-AUC.
    Serves as a simpler baseline to compare against stacking.
    """

    def __init__(self, base_models: list[tuple[str, Any]]):
        """
        Args:
            base_models: List of (name, fitted_model) tuples.
        """
        self.base_models = base_models
        self.weights_: np.ndarray | None = None
        self._is_fitted = False

    def fit(
        self,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        n_steps: int = 21,
    ) -> "BlendingEnsemble":
        """Find optimal blending weights via grid search on validation data.

        For 3 models, searches over a simplex of weights that sum to 1.

        Args:
            X_val: Validation features.
            y_val: Validation labels.
            n_steps: Grid resolution (higher = finer search).

        Returns:
            self
        """
        from sklearn.metrics import average_precision_score

        y = np.asarray(y_val)

        # Get base predictions on validation set
        base_preds = [model.predict_proba(X_val)[:, 1] for _, model in self.base_models]

        n_models = len(self.base_models)
        best_score = -1.0
        best_weights = np.ones(n_models) / n_models

        if n_models == 2:
            for w0 in np.linspace(0, 1, n_steps):
                weights = np.array([w0, 1 - w0])
                blended = sum(w * p for w, p in zip(weights, base_preds))
                score = average_precision_score(y, blended)
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
        elif n_models == 3:
            for w0 in np.linspace(0, 1, n_steps):
                for w1 in np.linspace(0, 1 - w0, n_steps):
                    w2 = 1 - w0 - w1
                    weights = np.array([w0, w1, w2])
                    blended = sum(w * p for w, p in zip(weights, base_preds))
                    score = average_precision_score(y, blended)
                    if score > best_score:
                        best_score = score
                        best_weights = weights.copy()
        else:
            # Fallback: equal weights
            best_weights = np.ones(n_models) / n_models

        self.weights_ = best_weights
        self._is_fitted = True

        weight_str = ", ".join(
            f"{name}: {w:.3f}" for (name, _), w in zip(self.base_models, self.weights_)
        )
        print(f"  Blending weights: {weight_str}")
        print(f"  Blending val PR-AUC: {best_score:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate blended probability predictions.

        Args:
            X: Features to predict on.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        if not self._is_fitted:
            raise RuntimeError("BlendingEnsemble is not fitted. Call fit() first.")

        base_preds = [model.predict_proba(X)[:, 1] for _, model in self.base_models]

        blended = sum(w * p for w, p in zip(self.weights_, base_preds))
        return np.column_stack([1 - blended, blended])

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Generate binary predictions."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Serialize the ensemble to disk."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BlendingEnsemble":
        """Load a saved ensemble from disk."""
        return joblib.load(path)
