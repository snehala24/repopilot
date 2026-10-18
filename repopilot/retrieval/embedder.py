"""Embedding interfaces and implementations (FastEmbed and test fakes)."""

from __future__ import annotations

import abc
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding


class Embedder(abc.ABC):
    """Abstract interface for text embedding models."""

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        """Return the embedding vector dimensionality."""

    @abc.abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks into vector representations."""

    @abc.abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query into a vector representation."""


class FastEmbedEmbedder(Embedder):
    """Local, CPU-friendly embedding via fastembed (ONNX runtime).

    Default model: BAAI/bge-small-en-v1.5 (384 dimensions).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 64,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model_instance: TextEmbedding | None = None
        self._dimensions: int = 384

    def _get_model(self) -> TextEmbedding:
        if self._model_instance is None:
            from fastembed import TextEmbedding

            kwargs = {}
            if self.cache_dir is not None:
                kwargs["cache_dir"] = str(self.cache_dir)
            self._model_instance = TextEmbedding(
                model_name=self.model_name,
                **kwargs,
            )
        return self._model_instance

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = list(model.embed(texts, batch_size=self.batch_size))
        return [
            e.tolist() if hasattr(e, "tolist") else list(e)  # type: ignore[union-attr]
            for e in embeddings
        ]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        embeddings = list(model.embed([text]))
        e = embeddings[0]
        return e.tolist() if hasattr(e, "tolist") else list(e)  # type: ignore[union-attr]


class FakeEmbedder(Embedder):
    """Deterministic, zero-network embedding generator for unit tests.

    Produces normalized L2 unit vectors seeded deterministically from text hashes.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_query(self, text: str) -> list[float]:
        # Hash text to an integer seed in [0, 2**32 - 1]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dimensions).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]
