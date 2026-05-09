"""Policy sweep — runs many policies, applies deflated Sharpe to the survivors.

n_trials = total policies explored. Without this correction, sweeping any
non-trivial grid will surface false positives.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from app.backtest.engine import BacktestConfig, BacktestResult
from app.backtest.policies.base import Policy
from app.backtest.runners.single import run_single
from app.core.ai.training.metrics import deflated_sharpe


def run_sweep(
    start: date,
    end: date,
    policies: list[Policy],
    universe: list[str],
    apply_deflated_sharpe: bool = True,
    source_path: str = "data/bars",
) -> list[tuple[Policy, BacktestResult]]:
    results: list[tuple[Policy, BacktestResult, float]] = []
    n_trials = len(policies)

    for policy in policies:
        config = BacktestConfig(start=start, end=end, universe=universe)
        result = run_single(config, policy, source_path=source_path)
        sr = result.metrics.get("sharpe", 0.0)

        if apply_deflated_sharpe and n_trials > 1:
            eq_vals = np.array([e for _, e in result.equity_curve], dtype=float)
            rets = np.diff(eq_vals) / np.maximum(eq_vals[:-1], 1e-9)
            skew = float(np.mean((rets - rets.mean()) ** 3) / (rets.std() ** 3 + 1e-9))
            kurt = float(np.mean((rets - rets.mean()) ** 4) / (rets.std() ** 4 + 1e-9))
            dsr = deflated_sharpe(sr, n_trials, skew, kurt, max(len(rets), 2))
        else:
            dsr = sr

        results.append((policy, result, dsr))

    results.sort(key=lambda x: x[2], reverse=True)
    return [(p, r) for p, r, _ in results]
