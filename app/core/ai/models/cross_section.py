"""Model D — cross-sectional LambdaRank for relative strength.

Per bar, ranks the universe and emits top-K longs / bottom-K shorts. Used to
weight Model C's per-asset signals so capital concentrates on the strongest
relative names. Trained with LightGBM `lambdarank` objective; group = bar.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.special import softmax

from .base import BaseModel, ModelArtifact


class CrossSectionRankModel(BaseModel):
    name = "cross_section_d"

    def __init__(self, top_k: int = 10, params: dict | None = None):
        self.top_k = top_k
        self.params = params or {}
        self._ranker: lgb.LGBMRanker | None = None
        self._feature_names: list[str] = []

    def fit(self, X: pl.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            groups: np.ndarray | None = None) -> None:
        self._feature_names = X.columns
        defaults = {"objective": "lambdarank", "n_estimators": 200,
                    "learning_rate": 0.05, "num_leaves": 31,
                    "random_state": 42, "verbose": -1}
        defaults.update(self.params)
        self._ranker = lgb.LGBMRanker(**defaults)
        X_np = X.to_numpy()
        if groups is None:
            groups = np.array([len(y)])
        self._ranker.fit(X_np, y, group=groups, sample_weight=sample_weight)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Returns rank scores normalised by softmax; top_k flagged as 1.0."""
        if self._ranker is None:
            raise RuntimeError("Model not fitted.")
        scores = self._ranker.predict(X.to_numpy())
        # Softmax normalise
        probs = softmax(scores)
        # Flag top_k
        result = np.zeros_like(probs)
        top_indices = np.argsort(probs)[-self.top_k:]
        result[top_indices] = 1.0
        return result

    def save(self, path: str) -> ModelArtifact:
        os.makedirs(path, exist_ok=True)
        artifact_path = os.path.join(path, f"{self.name}.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump({"ranker": self._ranker, "feature_names": self._feature_names,
                         "params": self.params, "top_k": self.top_k}, f)
        return ModelArtifact(
            name=self.name, version="1",
            feature_set=[], params=self.params, metrics={}, path=artifact_path,
        )

    @classmethod
    def load(cls, artifact: ModelArtifact) -> "CrossSectionRankModel":
        with open(artifact.path, "rb") as f:
            data = pickle.load(f)
        obj = cls(top_k=data["top_k"], params=data["params"])
        obj._ranker = data["ranker"]
        obj._feature_names = data["feature_names"]
        return obj
