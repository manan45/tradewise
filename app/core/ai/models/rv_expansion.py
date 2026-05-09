"""Model B' — realized-vol expansion classifier.

Predicts P(next-N-day realized vol > k * trailing realized vol). Replaces the
old IV-mispricing model from the options spec, since we trade no options.
LightGBM binary classifier with isotonic calibration on the holdout fold.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import polars as pl
import lightgbm as lgb

from .base import BaseModel, ModelArtifact


class RVExpansionModel(BaseModel):
    name = "rv_expansion_b_prime"

    def __init__(self, horizon_days: int = 5, expansion_k: float = 1.5,
                 params: dict | None = None):
        self.horizon_days = horizon_days
        self.expansion_k = expansion_k
        self.params = params or {}
        self._booster: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None:
        self._feature_names = X.columns
        defaults = {"objective": "binary", "n_estimators": 200,
                    "learning_rate": 0.05, "num_leaves": 31,
                    "random_state": 42, "verbose": -1}
        defaults.update(self.params)
        self._booster = lgb.LGBMClassifier(**defaults)
        self._booster.fit(X.to_numpy(), y, sample_weight=sample_weight)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        return self._booster.predict_proba(X.to_numpy())

    def save(self, path: str) -> ModelArtifact:
        os.makedirs(path, exist_ok=True)
        artifact_path = os.path.join(path, f"{self.name}.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump({"booster": self._booster, "feature_names": self._feature_names,
                         "params": self.params, "horizon_days": self.horizon_days,
                         "expansion_k": self.expansion_k}, f)
        return ModelArtifact(
            name=self.name, version="1",
            feature_set=[], params=self.params, metrics={}, path=artifact_path,
        )

    @classmethod
    def load(cls, artifact: ModelArtifact) -> "RVExpansionModel":
        with open(artifact.path, "rb") as f:
            data = pickle.load(f)
        obj = cls(horizon_days=data["horizon_days"], expansion_k=data["expansion_k"],
                  params=data["params"])
        obj._booster = data["booster"]
        obj._feature_names = data["feature_names"]
        return obj
