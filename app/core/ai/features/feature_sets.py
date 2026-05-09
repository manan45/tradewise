"""Curated feature bundles used by each model.

Each bundle is a stable list of (feature_name, version) tuples that the
training harness pulls from the registry. Never inline feature names in model
code — always reference the bundle.
"""
from __future__ import annotations

# Model A — vol regime (per IMPLEMENTATION_PLAN §3.3 M1).
A_VOL_REGIME: list[tuple[str, int]] = [
    # ("realized_vol_20", 1), ("parkinson_vol_20", 1), ("vix_zscore_60", 1),
    # ("vix_term_slope", 1), ("atr_pct_close_14", 1),
]

# Model B' — RV expansion classifier (replaces options IV-mispricing).
B_PRIME_RV_EXPANSION: list[tuple[str, int]] = []

# Model C — per-asset direction over swing horizon (3-10d).
C_DIRECTION: list[tuple[str, int]] = []

# Model D — cross-sectional LambdaRank (relative strength across universe).
D_CROSS_SECTION: list[tuple[str, int]] = []

# Model E — commodity pairs (gold/silver, copper/oil ratios).
E_COMMODITY_PAIRS: list[tuple[str, int]] = []

# Meta-labeler — takes base model probs + regime + market state.
META: list[tuple[str, int]] = []
