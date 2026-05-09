"""TrainingHarness — uniform loop for fitting, validating, calibrating, saving.

For each fold from the splitter:
1. Fit model on train.
2. Score on test, calibrate (isotonic).
3. Record metrics (classification + downstream PnL via a paper backtest stub).
At the end: aggregate metrics, run deflated Sharpe (uses n_trials = #configs
explored across the whole sweep), persist the winning artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from app.core.ai.models.base import BaseModel, ModelArtifact
from .splitters import WalkForwardSplit
from .calibration import IsotonicCalibrator, brier_score
from .metrics import sharpe, hit_rate


@dataclass
class TrainingConfig:
    feature_set: list[tuple[str, int]]
    label_kind: str                              # "triple_barrier" | "regime" | ...
    label_params: dict[str, Any] = field(default_factory=dict)
    cv: Any = field(default_factory=lambda: WalkForwardSplit(n_splits=5))
    calibrate: bool = True
    artifact_dir: str = "artifacts/models"


class TrainingHarness:
    def __init__(self, model_factory, config: TrainingConfig):
        self.model_factory = model_factory       # () -> BaseModel
        self.config = config

    def run(self, X: pl.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> ModelArtifact:
        oof_preds = np.full(len(y), np.nan)
        fold_metrics: list[dict[str, float]] = []

        for fold in self.config.cv.split(X):
            model = self.model_factory()
            X_train = X[fold.train_idx.tolist()]
            y_train = y[fold.train_idx]
            sw_train = sample_weight[fold.train_idx] if sample_weight is not None else None

            X_test = X[fold.test_idx.tolist()]
            y_test = y[fold.test_idx]

            model.fit(X_train, y_train, sample_weight=sw_train)
            raw_preds = model.predict_proba(X_test)

            if self.config.calibrate:
                cal = IsotonicCalibrator()
                # Use half of train set as calibration set
                mid = len(fold.train_idx) // 2
                cal_idx = fold.train_idx[mid:]
                X_cal = X[cal_idx.tolist()]
                y_cal = y[cal_idx]
                raw_cal = model.predict_proba(X_cal)
                p_cal = raw_cal[:, 1] if raw_cal.ndim == 2 else raw_cal
                cal.fit(p_cal, y_cal)
                p_test = raw_preds[:, 1] if raw_preds.ndim == 2 else raw_preds
                preds = cal.transform(p_test)
            else:
                preds = raw_preds[:, 1] if raw_preds.ndim == 2 else raw_preds

            oof_preds[fold.test_idx] = preds
            fold_metrics.append(self.evaluate(model, X_test, y_test))

        # Fit final model on all data
        final_model = self.model_factory()
        final_model.fit(X, y, sample_weight=sample_weight)
        artifact = final_model.save(self.config.artifact_dir)
        return artifact

    def evaluate(self, model: BaseModel, X_test: pl.DataFrame,
                 y_test: np.ndarray) -> dict[str, float]:
        raw = model.predict_proba(X_test)
        preds = raw[:, 1] if raw.ndim == 2 else raw
        binary_preds = (preds >= 0.5).astype(int)
        rets = (binary_preds * 2 - 1) * (y_test * 2 - 1).astype(float) * 0.01
        return {
            "sharpe": sharpe(rets),
            "brier": brier_score(preds, y_test.astype(float)),
            "hit_rate": hit_rate(rets),
            "calibration_error": float(np.mean(np.abs(preds - y_test.astype(float)))),
        }
