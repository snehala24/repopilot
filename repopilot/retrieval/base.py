"""Abstract retriever protocol and retrieved chunk representations."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedChunk:
    """A scored chunk retrieved from a vector store or search index.

    Attributes:
        chunk_id: Stable identifier of the chunk.
        text: Text or code content of the chunk.
        score: Relevance score (e.g. cosine similarity).
        metadata: Full metadata dictionary containing file path, line ranges,
            repository, commit SHA, and any custom attributes.
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def file_path(self) -> str:
        """POSIX relative path of the file containing this chunk."""
        return str(self.metadata.get("file_path", ""))

    @property
    def line_start(self) -> int:
        """1-indexed start line number."""
        return int(self.metadata.get("line_start", 0))

    @property
    def line_end(self) -> int:
        """1-indexed end line number."""
        return int(self.metadata.get("line_end", 0))

    @property
    def repository(self) -> str:
        """Repository name or slug."""
        return str(self.metadata.get("repository", ""))

    @property
    def commit_sha(self) -> str:
        """Git commit SHA the chunk was indexed at."""
        return str(self.metadata.get("commit_sha", ""))

    @property
    def citation(self) -> str:
        """Standard citation string in 'file_path:line_start-line_end' format."""
        if self.file_path and self.line_start and self.line_end:
            return f"{self.file_path}:{self.line_start}-{self.line_end}"
        return self.file_path or self.chunk_id


@runtime_checkable
class Retriever(Protocol):
    """Protocol for chunk retrieval backends."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks matching query, with optional metadata filters."""
        ...
