"""News-volume Z-score per symbol — surge detection."""
from __future__ import annotations

from datetime import datetime


class NewsVolumeScorer:
    def __init__(self, baseline_days: int = 30):
        self.baseline_days = baseline_days

    def zscore(self, symbol: str, ts: datetime,
               recent_count: int, baseline_mean: float,
               baseline_std: float) -> float:
        if baseline_std == 0:
            return float("nan")
        return (recent_count - baseline_mean) / baseline_std
