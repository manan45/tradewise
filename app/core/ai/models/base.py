"""Base model protocol — single uniform interface for the training harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ModelArtifact:
    """Everything needed to deserialize and re-score a model in production."""
    name: str
    version: str
    feature_set: list[tuple[str, int]]   # (name, version) tuples
    params: dict[str, Any]
    metrics: dict[str, float]            # validation metrics frozen at train time
    path: str                            # on-disk artifact path (joblib/lgbm bin)


class BaseModel(Protocol):
    name: str

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        ...

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        ...

    def save(self, path: str) -> ModelArtifact:
        ...

    @classmethod
    def load(cls, artifact: ModelArtifact) -> "BaseModel":
        ...
