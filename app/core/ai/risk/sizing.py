"""Position sizing — Kelly with calibrated probabilities.

f* = (p*b - (1-p)) / b   where p is calibrated win prob and b is the win/loss
ratio implied by the plan's PT/SL distance. We then take fractional Kelly
(default 0.25x) and cap at 1% of equity per trade.

Rationale: full Kelly is mathematically optimal under perfectly-calibrated
probs and stationary edge. We have neither, so the haircut is mandatory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingInputs:
    p_win: float                  # calibrated prob from meta-labeler
    win_loss_ratio: float         # |target - entry| / |entry - stop|
    fractional_kelly: float = 0.25
    max_fraction: float = 0.01    # 1% equity cap per trade


def kelly_fraction(p_win: float, win_loss_ratio: float) -> float:
    """Raw Kelly fraction; can be negative (=> don't take)."""
    b = win_loss_ratio
    if b <= 0:
        return 0.0
    f = (p_win * b - (1.0 - p_win)) / b
    return max(f, 0.0)


def position_size_shares(
    inputs: SizingInputs,
    account_equity: float,
    entry_price: float,
    contract_multiplier: float = 1.0,
) -> int:
    """Round-down integer shares/contracts after Kelly + cap."""
    if entry_price <= 0 or account_equity <= 0:
        return 0
    raw_kelly = kelly_fraction(inputs.p_win, inputs.win_loss_ratio)
    if raw_kelly <= 0:
        return 0
    fraction = raw_kelly * inputs.fractional_kelly
    fraction = min(fraction, inputs.max_fraction)
    dollar_risk = account_equity * fraction
    shares = dollar_risk / (entry_price * contract_multiplier)
    return max(0, math.floor(shares))
