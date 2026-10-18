"""Unit tests for fixed-size character chunking and line tracking."""

import pytest

from repopilot.ingestion.chunker import chunk_text
from repopilot.ingestion.models import Chunk


def test_chunk_text_basic_overlap_and_lines():
    """Verify chunking splits text with specified overlap and correct line numbers."""
    # Build a 10-line text where each line is 10 chars: "line_01...\n"
    lines = [f"line_{i:02d}----" for i in range(10)]
    content = "\n".join(lines)  # total length 109 chars, 10 lines

    # chunk_size 50, overlap 10 -> step 40
    chunks = chunk_text(
        content,
        repository="test-repo",
        file_path="src/module.py",
        language="python",
        commit_sha="abc1234",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) == 3
    # First chunk: chars 0..50
    assert len(chunks[0].content) == 50
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 5
    assert chunks[0].language == "python"
    assert chunks[0].file_path == "src/module.py"

    # Second chunk: chars 40..90 (overlaps chars 40..50 of chunk 0)
    assert len(chunks[1].content) == 50
    assert chunks[1].line_start == 4
    assert chunks[1].line_end == 8
    assert chunks[0].content[40:50] == chunks[1].content[0:10]

    # Third chunk: chars 80..119 (tail)
    assert chunks[2].line_start == 7
    assert chunks[2].line_end == 10
    assert chunks[1].content[40:50] == chunks[2].content[0:10]



def test_chunk_id_stability():
    """Verify chunk_id is a deterministic hash excluding commit_sha."""
    c1 = Chunk.create(
        repository="org/repo",
        file_path="src/app.py",
        language="python",
        line_start=1,
        line_end=10,
        content="def run(): pass",
        commit_sha="commit-v1",
    )
    c2 = Chunk.create(
        repository="org/repo",
        file_path="src/app.py",
        language="python",
        line_start=1,
        line_end=10,
        content="def run(): pass",
        commit_sha="commit-v2",  # Different commit SHA
    )
    c3 = Chunk.create(
        repository="org/repo",
        file_path="src/app.py",
        language="python",
        line_start=1,
        line_end=10,
        content="def run(): pass modified",  # Different content
        commit_sha="commit-v1",
    )

    assert c1.chunk_id == c2.chunk_id
    assert c1.chunk_id != c3.chunk_id
    assert c1.commit_sha == "commit-v1"
    assert c2.commit_sha == "commit-v2"


def test_chunk_empty_and_short_text():
    """Empty text yields no chunks; short text yields exactly one chunk."""
    assert chunk_text("", repository="repo", file_path="f.py", language="python") == []

    short = "hello world"
    chunks = chunk_text(
        short,
        repository="repo",
        file_path="f.py",
        language="python",
        chunk_size=100,
        chunk_overlap=20,
    )
    assert len(chunks) == 1
    assert chunks[0].content == short
    assert chunks[0].line_start == 1
    assert chunks[0].line_end == 1


def test_chunk_validation_errors():
    """Invalid chunk_size or overlap must raise ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("test", repository="r", file_path="f", language="p", chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunk_text(
            "test",
            repository="r",
            file_path="f",
            language="p",
            chunk_size=10,
            chunk_overlap=-1,
        )

    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        chunk_text(
            "test",
            repository="r",
            file_path="f",
            language="p",
            chunk_size=50,
            chunk_overlap=50,
        )
