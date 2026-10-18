"""Qdrant vector store wrapper supporting local and server modes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from repopilot.ingestion.models import Chunk

if TYPE_CHECKING:
    from repopilot.config import Settings, VectorStoreSettings


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Map a stable chunk_id hash deterministically to a valid Qdrant UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


@dataclass(frozen=True)
class SearchResult:
    """A scored result returned from vector search."""

    chunk_id: str
    score: float
    repository: str
    file_path: str
    language: str
    line_start: int
    line_end: int
    content: str
    commit_sha: str
    payload: dict[str, Any]


class QdrantVectorStore:
    """Qdrant vector store wrapper supporting local (embedded) and server modes."""

    def __init__(
        self,
        mode: str = "local",
        *,
        path: Path | str | None = None,
        url: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            self.mode = mode
            return

        self.mode = mode
        if mode == "local":
            local_path = Path(path or "data/qdrant").resolve()
            local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(local_path))
        elif mode == "server":
            server_url = url or "http://localhost:6333"
            self.client = QdrantClient(url=server_url)
        else:
            raise ValueError(
                f"Unsupported Qdrant mode: {mode!r}. Must be 'local' or 'server'."
            )

    @classmethod
    def from_settings(
        cls, settings: Settings | VectorStoreSettings
    ) -> QdrantVectorStore:
        """Create a QdrantVectorStore instance from application settings."""
        cfg = getattr(settings, "vector_store", settings)
        return cls(mode=cfg.mode, path=cfg.path, url=cfg.url)

    def create_collection(
        self,
        collection_name: str,
        *,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> None:
        """Idempotently create a collection if it does not already exist."""
        if not self.client.collection_exists(collection_name):
            dist = (
                Distance.COSINE
                if distance.lower() == "cosine"
                else Distance[distance.upper()]
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=dist),
            )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        *,
        batch_size: int = 64,
    ) -> int:
        """Idempotently upsert code chunks and their embeddings in batches.

        Point IDs are deterministically derived from `chunk.chunk_id`, ensuring
        that repeated indexing updates existing chunks rather than duplicating them.

        Returns:
            The total number of chunks upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) does not match embeddings count ({len(embeddings)})"
            )
        if not chunks:
            return 0

        total_upserted = 0
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vectors = embeddings[i : i + batch_size]

            points: list[PointStruct] = []
            for chunk, vec in zip(batch_chunks, batch_vectors, strict=True):
                point_id = chunk_id_to_uuid(chunk.chunk_id)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "repository": chunk.repository,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "content": chunk.content,
                    "commit_sha": chunk.commit_sha,
                }
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)

        return total_upserted

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        filters: dict[str, Any] | Filter | None = None,
    ) -> list[SearchResult]:
        """Search nearest chunks with optional metadata filtering."""
        query_filter: Filter | None = None
        if isinstance(filters, Filter):
            query_filter = filters
        elif isinstance(filters, dict) and filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            query_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in response.points:
            p = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", str(hit.id)),
                    score=hit.score,
                    repository=p.get("repository", ""),
                    file_path=p.get("file_path", ""),
                    language=p.get("language", ""),
                    line_start=p.get("line_start", 0),
                    line_end=p.get("line_end", 0),
                    content=p.get("content", ""),
                    commit_sha=p.get("commit_sha", ""),
                    payload=p,
                )
            )

        return results

    def count(self, collection_name: str) -> int:
        """Return the total number of points in a collection."""
        res = self.client.count(collection_name=collection_name)
        return res.count

    def close(self) -> None:
        """Close client connection."""
        self.client.close()
