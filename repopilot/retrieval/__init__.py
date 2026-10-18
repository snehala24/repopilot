"""RepoPilot retrieval package."""

from repopilot.retrieval.base import RetrievedChunk, Retriever
from repopilot.retrieval.dense import DenseRetriever
from repopilot.retrieval.embedder import Embedder, FakeEmbedder, FastEmbedEmbedder
from repopilot.retrieval.store import (
    QdrantVectorStore,
    SearchResult,
    chunk_id_to_uuid,
)

__all__ = [
    "DenseRetriever",
    "Embedder",
    "FakeEmbedder",
    "FastEmbedEmbedder",
    "QdrantVectorStore",
    "RetrievedChunk",
    "Retriever",
    "SearchResult",
    "chunk_id_to_uuid",
]
