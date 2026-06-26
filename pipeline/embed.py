"""
Dense and sparse embedders wrapping fastembed.

Usage:
    dense = DenseEmbedder("BAAI/bge-base-en-v1.5")
    vectors = dense.embed(["text1", "text2"])  # list[list[float]]

    sparse = SparseEmbedder("Qdrant/bm25")
    sparse_vecs = sparse.embed(["text1", "text2"])  # list[dict[str, float]]
"""

import logging
from typing import Optional

from fastembed import SparseTextEmbedding, TextEmbedding

logger = logging.getLogger(__name__)


class DenseEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 256,
        max_length: int = 512,
    ):
        self.model = TextEmbedding(model_name=model_name, max_length=max_length)
        self.batch_size = batch_size

    def embed(self, texts: list[str], mode: str | None = None) -> list[list[float]]:
        results: list[list[float]] = []
        batches = list(self._batches(texts))
        total = len(batches)
        kwargs = {}
        if mode is not None:
            kwargs["mode"] = mode
        for i, batch in enumerate(batches, 1):
            logger.info("Dense embedding batch %d/%d", i, total)
            for vec in self.model.embed(batch, **kwargs):
                results.append(vec.tolist())
        return results

    def _batches(self, texts: list[str]):
        for i in range(0, len(texts), self.batch_size):
            yield texts[i : i + self.batch_size]


class SparseEmbedder:
    def __init__(self, model_name: str = "Qdrant/bm25", batch_size: int = 256):
        self.model = SparseTextEmbedding(model_name=model_name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        batches = list(self._batches(texts))
        total = len(batches)
        for i, batch in enumerate(batches, 1):
            logger.info("Sparse embedding batch %d/%d", i, total)
            for vec in self.model.embed(batch):
                results.append(vec.as_dict())
        return results

    def _batches(self, texts: list[str]):
        for i in range(0, len(texts), self.batch_size):
            yield texts[i : i + self.batch_size]
