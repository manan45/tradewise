"""Walk-forward: train on rolling window, test on next, advance, repeat.

Returns the stitched out-of-sample equity curve plus per-fold metrics. This
is the only number we trust for go/no-go on a strategy.
"""
from __future__ import annotations

from datetime import date
from dateutil.relativedelta import relativedelta

from app.backtest.engine import BacktestConfig, BacktestResult
from app.backtest.policies.base import Policy
from app.backtest.runners.single import run_single


def run_walkforward(
    start: date,
    end: date,
    train_months: int,
    test_months: int,
    policy: Policy,
    universe: list[str],
    source_path: str = "data/bars",
) -> list[BacktestResult]:
    results: list[BacktestResult] = []
    fold_start = start
    while fold_start < end:
        train_end = fold_start + relativedelta(months=train_months)
        test_end = min(train_end + relativedelta(months=test_months), end)
        config = BacktestConfig(
            start=train_end,
            end=test_end,
            universe=universe,
        )
        result = run_single(config, policy, source_path=source_path)
        results.append(result)
        fold_start = train_end
        if fold_start >= end:
            break
    return results
