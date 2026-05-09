"""Sentiment service.

Two heads: FinBERT (fine-tuned on financial text) for headlines/filings, and a
news-volume Z-score (raw publication rate vs 30d baseline). Outputs land on
the bus as `sentiment.{symbol}` snapshots that the meta-labeler consumes.
"""
from .finbert_scorer import FinBertScorer
from .news_volume import NewsVolumeScorer

__all__ = ["FinBertScorer", "NewsVolumeScorer"]
