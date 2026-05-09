"""Feature engineering layer.

Two hard rules baked into this module:

1. Point-in-time correctness — every feature MUST be computed via the helpers
   in ``asof.py``; raw indexing into pandas/polars frames is forbidden.
2. Determinism — feature definitions are registered by name in ``registry.py``
   and pinned by hash. The same feature_set version must produce identical
   numbers in research and in live (BACKTESTING.md §5 parity rule).
"""
from .registry import FeatureRegistry, FeatureSpec, get_registry
from .asof import asof_join, asof_lookup

__all__ = [
    "FeatureRegistry", "FeatureSpec", "get_registry",
    "asof_join", "asof_lookup",
]
