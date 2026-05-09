"""FinBERT inference wrapper."""
from __future__ import annotations

from app.core.ports.news import NewsItem


class FinBertScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self.device,
                top_k=None,
            )
        return self._pipeline

    def score_batch(self, items: list[NewsItem]) -> list[dict[str, float]]:
        if not items:
            return []
        pipe = self._get_pipeline()
        texts = [item.headline[:512] for item in items]
        results = pipe(texts, batch_size=32)
        output = []
        for res in results:
            scores = {r["label"].lower(): r["score"] for r in res}
            output.append({
                "positive": scores.get("positive", 0.0),
                "negative": scores.get("negative", 0.0),
                "neutral": scores.get("neutral", 0.0),
            })
        return output
