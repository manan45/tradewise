"""Pydantic schemas for every LLM tool I/O.

Single source of truth — input_schema fed to Anthropic comes from
``Model.model_json_schema()``, and our own pre/post validation uses the same
class. This way a tool can never be called with shapes the model can produce
but our handler can't parse.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# -------- Inputs --------
class GetMarketSnapshotIn(BaseModel):
    symbol: str
    as_of: datetime | None = None


class ScoreScenarioIn(BaseModel):
    symbol: str
    scenario: Literal["trend_continuation", "mean_reversion",
                      "vol_expansion", "breakout", "pead", "pairs"]
    horizon_days: int = Field(default=5, ge=1, le=30)


class ProposePlanIn(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    entry_hint: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class SearchSimilarSessionsIn(BaseModel):
    scenario: str
    top_k: int = Field(default=8, ge=1, le=50)
    embedding: list[float]


# -------- Outputs --------
class MarketSnapshotOut(BaseModel):
    symbol: str
    last: float
    regime: str
    rv_percentile: float
    rsi_14: float
    atr_14: float
    days_to_earnings: int | None = None


class ScenarioScoreOut(BaseModel):
    p_take: float
    rationale_features: dict[str, float]


class ProposedPlanOut(BaseModel):
    side: Literal["buy", "sell"]
    entry: float
    stop: float
    target: float
    horizon_days: int
    risk_per_share: float
    sizing_shares: int


class SimilarSessionsOut(BaseModel):
    sessions: list[dict]
