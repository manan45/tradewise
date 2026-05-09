"""Model zoo.

Five base models + a meta-labeler. Each model implements the BaseModel
protocol (fit/predict_proba/save/load) so the training harness can iterate
the same loop for all of them.

- A: vol regime classifier (low / range / trend / high_vol_expansion)
- B': realized-vol expansion next-N-days classifier (replaces IV mispricing)
- C: per-asset directional probability over swing horizon (3-10 days)
- D: cross-sectional LambdaRank (relative strength across the universe)
- E: commodity pairs spread mean-reversion (gold/silver, copper/oil, ...)
- meta: takes base outputs + market state → calibrated trade-take probability
"""
from .base import BaseModel, ModelArtifact

__all__ = ["BaseModel", "ModelArtifact"]
