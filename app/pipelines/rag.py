"""Background re-indexing of the RAG corpus (filings, prior post-mortems)."""
from __future__ import annotations

import os
from datetime import datetime

from app.services.rag.indexer import RagIndexer


async def run(asof: datetime, indexer: RagIndexer,
              source_paths: list[str]) -> int:
    total = 0
    for path in source_paths:
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            files = [path]
        else:
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith((".txt", ".md", ".json"))
            ]
        for fpath in files:
            with open(fpath) as f:
                text = f.read()
            doc = {
                "text": text,
                "source": fpath,
                "ts": asof.isoformat(),
            }
            n = await indexer.index_documents([doc])
            total += n
    return total
