"""Fixed-size character chunker with overlap and line tracking."""

from __future__ import annotations

from repopilot.ingestion.models import Chunk, SourceFile


def chunk_text(
    content: str,
    *,
    repository: str,
    file_path: str,
    language: str,
    commit_sha: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split text into fixed-size character chunks with overlap.

    Computes 1-indexed line start/end numbers and assigns a stable chunk_id
    hash based on repository, file_path, line_start, line_end, and content
    (commit_sha is stored as metadata only).

    Args:
        content: Text content to chunk.
        repository: Repository identifier.
        file_path: Relative file path within repository.
        language: Programming or markup language name.
        commit_sha: Git commit SHA (stored as metadata).
        chunk_size: Maximum characters per chunk (must be > 0).
        chunk_overlap: Number of overlapping characters (must be < chunk_size).

    Returns:
        List of :class:`Chunk` objects.
    """
    if not content:
        return []
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    total_len = len(content)
    step = chunk_size - chunk_overlap
    chunks: list[Chunk] = []

    for start_char in range(0, total_len, step):
        end_char = min(start_char + chunk_size, total_len)
        chunk_content = content[start_char:end_char]

        # 1-indexed line numbers
        line_start = content.count("\n", 0, start_char) + 1
        last_char_idx = max(start_char, end_char - 1)
        line_end = content.count("\n", 0, last_char_idx) + 1

        chunk = Chunk.create(
            repository=repository,
            file_path=file_path,
            language=language,
            line_start=line_start,
            line_end=line_end,
            content=chunk_content,
            commit_sha=commit_sha,
        )
        chunks.append(chunk)

        if end_char >= total_len:
            break

    return chunks


def chunk_file(
    source_file: SourceFile,
    *,
    repository: str,
    commit_sha: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Convenience helper to chunk a discovered :class:`SourceFile`."""
    return chunk_text(
        source_file.content,
        repository=repository,
        file_path=source_file.relative_path,
        language=source_file.language,
        commit_sha=commit_sha,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
