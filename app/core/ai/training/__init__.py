"""Training harness — same loop for every model.

Five primitives:
- labels.py: triple-barrier + meta-labels
- splitters.py: walk-forward + combinatorial purged k-fold
- calibration.py: isotonic + Platt + reliability
- metrics.py: deflated Sharpe, PSR, classification metrics
- harness.py: ties it all together; consumed by training_service entrypoint
"""
from .harness import TrainingHarness, TrainingConfig

__all__ = ["TrainingHarness", "TrainingConfig"]
