"""News ingest — pulls per-symbol headlines, scores via FinBERT, publishes
sentiment snapshots on the bus."""
from __future__ import annotations

from datetime import datetime, timedelta

from app.core.ports.news import NewsProvider
from app.services.sentiment.finbert_scorer import FinBertScorer


async def run(
    asof: datetime,
    news: NewsProvider,
    scorer: FinBertScorer,
    symbols: list[str],
    db=None,
    bus=None,
) -> int:
    since = asof - timedelta(days=1)
    rows_written = 0
    all_items = []
    for symbol in symbols:
        items = await news.fetch_recent(symbol, since, limit=100)
        all_items.extend(items)

    if not all_items:
        return 0

    scores = scorer.score_batch(all_items)
    for item, score in zip(all_items, scores):
        if db is not None:
            await db.execute(
                """
                INSERT INTO news_items (id, symbol, ts, headline, source, sentiment_pos, sentiment_neg, sentiment_neu, ingested_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (id) DO UPDATE
                SET sentiment_pos=$6, sentiment_neg=$7, sentiment_neu=$8, ingested_at=NOW()
                """,
                item.id, item.symbol, item.ts, item.headline, item.source,
                score["positive"], score["negative"], score["neutral"],
            )
        rows_written += 1
    return rows_written
