"""Data models for ingestion and chunking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """A single code or text chunk extracted from a repository source file.

    Attributes:
        chunk_id: Stable SHA-256 hash derived from repository, file_path, line_start,
            line_end, and content.
        repository: Name or identifier of the repository (e.g. 'repopilot' or 'owner/repo').
        file_path: POSIX relative file path within the repository.
        language: Programming or markup language name (e.g. 'python', 'markdown').
        line_start: 1-indexed starting line number.
        line_end: 1-indexed ending line number.
        content: Text content of the chunk.
        commit_sha: Git commit SHA the chunk was indexed at (metadata only, not in hash).
    """

    chunk_id: str
    repository: str
    file_path: str
    language: str
    line_start: int
    line_end: int
    content: str
    commit_sha: str

    @classmethod
    def create(
        cls,
        *,
        repository: str,
        file_path: str,
        language: str,
        line_start: int,
        line_end: int,
        content: str,
        commit_sha: str = "",
    ) -> Chunk:
        """Create a Chunk computing a stable chunk_id hash."""
        # Normalize file path to forward slashes for cross-platform stability
        norm_path = Path(file_path).as_posix()
        raw_key = f"{repository}:{norm_path}:{line_start}:{line_end}:{content}"
        chunk_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return cls(
            chunk_id=chunk_id,
            repository=repository,
            file_path=norm_path,
            language=language,
            line_start=line_start,
            line_end=line_end,
            content=content,
            commit_sha=commit_sha,
        )


@dataclass(frozen=True)
class SourceFile:
    """A source file discovered by the repository walker."""

    path: Path
    relative_path: str
    content: str
    language: str
    size_bytes: int
