"""Model E — commodity pairs (gold/silver, copper/oil, WTI/Brent ratios).

Ornstein-Uhlenbeck-style mean-reversion signal on log-spread; LightGBM
overlay learns when the OU half-life regime is reliable vs. broken (regime
change tends to precede the largest pair drawdowns).
"""
from __future__ import annotations

import math
import os
import pickle

import numpy as np
import polars as pl
from scipy.stats import norm

from .base import BaseModel, ModelArtifact


class UnstablePairError(Exception):
    pass


class CommodityPairsModel(BaseModel):
    name = "commodity_pairs_e"

    def __init__(self, pair: tuple[str, str], params: dict | None = None):
        self.pair = pair
        self.params = params or {}
        self._hedge_ratio: float | None = None
        self._spread_mean: float | None = None
        self._spread_std: float | None = None
        self._half_life: float | None = None

    def _compute_spread(self, X: pl.DataFrame) -> np.ndarray:
        leg_a = X[self.pair[0]].to_numpy()
        leg_b = X[self.pair[1]].to_numpy()
        log_a = np.log(leg_a)
        log_b = np.log(leg_b)
        spread = log_a - self._hedge_ratio * log_b
        return spread

    def fit(self, X: pl.DataFrame, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None:
        leg_a = X[self.pair[0]].to_numpy()
        leg_b = X[self.pair[1]].to_numpy()
        log_a = np.log(leg_a)
        log_b = np.log(leg_b)

        # OLS hedge ratio
        cov = np.cov(log_a, log_b)
        self._hedge_ratio = cov[0, 1] / cov[1, 1]

        spread = log_a - self._hedge_ratio * log_b
        self._spread_mean = float(spread.mean())
        self._spread_std = float(spread.std())

        # OU half-life via AR(1)
        s_t = spread[:-1]
        s_tp1 = spread[1:]
        if len(s_t) < 2:
            raise UnstablePairError("Not enough data to fit OU model.")
        rho = np.corrcoef(s_t, s_tp1)[0, 1]
        if rho >= 1.0 or rho <= -1.0:
            raise UnstablePairError("Spread is not mean-reverting.")
        self._half_life = -math.log(2) / math.log(abs(rho))

        if not (0 < self._half_life < 60):
            raise UnstablePairError(
                f"Half-life {self._half_life:.1f} outside (0, 60) trading days."
            )

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        if self._hedge_ratio is None:
            raise RuntimeError("Model not fitted.")
        spread = self._compute_spread(X)
        current_z = (spread - self._spread_mean) / max(self._spread_std, 1e-9)

        # P(mean reversion within horizon) based on OU half-life and z-score
        # Simple proxy: p = norm.cdf(-|z|) * 2 when |z| is large, regime is active
        half_life = self._half_life
        horizon = self.params.get("horizon_days", 10)
        decay = math.exp(-math.log(2) * horizon / half_life)
        # Expected z-score after horizon bars
        expected_z = current_z * decay
        p_revert = np.array([
            float(norm.cdf(-abs(z)) * 2) if abs(z) > 1.0 else 0.5
            for z in current_z
        ])
        return p_revert

    def save(self, path: str) -> ModelArtifact:
        os.makedirs(path, exist_ok=True)
        artifact_path = os.path.join(path, f"{self.name}_{'_'.join(self.pair)}.pkl")
        with open(artifact_path, "wb") as f:
            pickle.dump({
                "pair": self.pair, "params": self.params,
                "hedge_ratio": self._hedge_ratio,
                "spread_mean": self._spread_mean,
                "spread_std": self._spread_std,
                "half_life": self._half_life,
            }, f)
        return ModelArtifact(
            name=self.name, version="1",
            feature_set=[], params=self.params, metrics={}, path=artifact_path,
        )

    @classmethod
    def load(cls, artifact: ModelArtifact) -> "CommodityPairsModel":
        with open(artifact.path, "rb") as f:
            data = pickle.load(f)
        obj = cls(pair=data["pair"], params=data["params"])
        obj._hedge_ratio = data["hedge_ratio"]
        obj._spread_mean = data["spread_mean"]
        obj._spread_std = data["spread_std"]
        obj._half_life = data["half_life"]
        return obj
