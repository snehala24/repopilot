"""Dense vector retriever using embeddings and QdrantVectorStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from repopilot.retrieval.base import RetrievedChunk, Retriever
from repopilot.retrieval.embedder import Embedder, FastEmbedEmbedder
from repopilot.retrieval.store import QdrantVectorStore

if TYPE_CHECKING:
    from repopilot.config import Settings


class DenseRetriever(Retriever):
    """Dense vector retriever that embeds queries and queries Qdrant."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        collection_name: str,
    ) -> None:
        """Initialize with an embedder, vector store, and collection name.

        Accepts injected fakes for unit testing.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        collection_name: str,
        *,
        embedder: Embedder | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> DenseRetriever:
        """Build a DenseRetriever from application settings."""
        if embedder is None:
            embedder = FastEmbedEmbedder(
                model_name=settings.embedding.model,
                batch_size=settings.embedding.batch_size,
                cache_dir=settings.paths.cache_dir / "fastembed",
            )
        if vector_store is None:
            vector_store = QdrantVectorStore.from_settings(settings)
        return cls(
            embedder=embedder,
            vector_store=vector_store,
            collection_name=collection_name,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed *query*, search Qdrant, and map results to :class:`RetrievedChunk`."""
        if not query.strip():
            return []

        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            filters=filters,
        )

        chunks: list[RetrievedChunk] = []
        for res in results:
            meta = dict(res.payload)
            meta.update(
                {
                    "file_path": res.file_path,
                    "line_start": res.line_start,
                    "line_end": res.line_end,
                    "repository": res.repository,
                    "commit_sha": res.commit_sha,
                }
            )
            chunks.append(
                RetrievedChunk(
                    chunk_id=res.chunk_id,
                    text=res.content,
                    score=res.score,
                    metadata=meta,
                )
            )

        return chunks
