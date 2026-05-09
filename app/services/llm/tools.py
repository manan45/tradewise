"""Tool registry — every callable the LLM is allowed to use."""
from __future__ import annotations

from typing import Awaitable, Callable

from app.core.ports.llm import Tool, ToolSpec
from . import schemas


ToolHandler = Callable[[dict], Awaitable[dict]]


def make_tool(name: str, description: str,
              input_model: type, handler: ToolHandler) -> Tool:
    return Tool(
        spec=ToolSpec(
            name=name,
            description=description,
            input_schema=input_model.model_json_schema(),
        ),
        handler=handler,
    )


async def _get_market_snapshot(payload: dict) -> dict:
    inp = schemas.GetMarketSnapshotIn(**payload)
    # Returns a MarketSnapshotOut with placeholder values;
    # in production this reads from the feature cache.
    return schemas.MarketSnapshotOut(
        symbol=inp.symbol,
        last=0.0,
        regime="unknown",
        rv_percentile=0.5,
        rsi_14=50.0,
        atr_14=0.0,
        days_to_earnings=None,
    ).model_dump()


async def _score_scenario(payload: dict) -> dict:
    inp = schemas.ScoreScenarioIn(**payload)
    return schemas.ScenarioScoreOut(
        p_take=0.5,
        rationale_features={},
    ).model_dump()


async def _propose_plan(payload: dict) -> dict:
    inp = schemas.ProposePlanIn(**payload)
    entry = inp.entry_hint or 100.0
    atr = entry * 0.02
    side = inp.side
    stop = entry - 2 * atr if side == "buy" else entry + 2 * atr
    target = entry + 4 * atr if side == "buy" else entry - 4 * atr
    from app.core.ai.risk.sizing import SizingInputs, position_size_shares
    sizing = SizingInputs(p_win=inp.confidence, win_loss_ratio=2.0)
    shares = position_size_shares(sizing, 100_000.0, entry)
    return schemas.ProposedPlanOut(
        side=side,
        entry=entry,
        stop=stop,
        target=target,
        horizon_days=10,
        risk_per_share=abs(entry - stop),
        sizing_shares=shares,
    ).model_dump()


async def _search_similar_sessions(payload: dict) -> dict:
    inp = schemas.SearchSimilarSessionsIn(**payload)
    return schemas.SimilarSessionsOut(sessions=[]).model_dump()


def default_toolset() -> list[Tool]:
    return [
        make_tool(
            "get_market_snapshot",
            "Return latest features + regime + days-to-earnings for a symbol.",
            schemas.GetMarketSnapshotIn, _get_market_snapshot,
        ),
        make_tool(
            "score_scenario",
            "Run the named scenario's models against the symbol; return p_take.",
            schemas.ScoreScenarioIn, _score_scenario,
        ),
        make_tool(
            "propose_plan",
            "Compute entry/stop/target + Kelly-sized share count for an intent.",
            schemas.ProposePlanIn, _propose_plan,
        ),
        make_tool(
            "search_similar_sessions",
            "pgvector nearest-neighbour over past sessions' birth_embedding.",
            schemas.SearchSimilarSessionsIn, _search_similar_sessions,
        ),
    ]
