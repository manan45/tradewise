"""BGE-base embedder."""
from __future__ import annotations

import asyncio


class BgeEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, normalize_embeddings=True, batch_size=32).tolist(),
        )
        return embeddings
