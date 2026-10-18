"""Unit tests for DenseRetriever."""

import math
from pathlib import Path

from repopilot.ingestion.models import Chunk
from repopilot.retrieval.base import RetrievedChunk, Retriever
from repopilot.retrieval.dense import DenseRetriever
from repopilot.retrieval.embedder import FakeEmbedder
from repopilot.retrieval.store import QdrantVectorStore


def test_dense_retriever_nearest_chunk_and_metadata(tmp_path: Path):
    """DenseRetriever maps query to nearest chunk with accurate metadata and citation."""
    store_dir = tmp_path / "qdrant"
    store = QdrantVectorStore(mode="local", path=store_dir)
    embedder = FakeEmbedder(dimensions=384)
    collection_name = "test_dense_col"

    try:
        store.create_collection(collection_name, vector_size=384, distance="Cosine")

        c1 = Chunk.create(
            repository="my-repo",
            file_path="src/calc.py",
            language="python",
            line_start=10,
            line_end=20,
            content="def add(a, b): return a + b",
            commit_sha="sha-1234",
        )
        c2 = Chunk.create(
            repository="my-repo",
            file_path="docs/guide.md",
            language="markdown",
            line_start=1,
            line_end=5,
            content="# Calculator Guide\nUse add() to sum numbers.",
            commit_sha="sha-1234",
        )

        vectors = embedder.embed_documents([c1.content, c2.content])
        store.upsert_chunks(collection_name, [c1, c2], vectors)

        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=store,
            collection_name=collection_name,
        )
        assert isinstance(retriever, Retriever)

        # Retrieve with exact query matching c1
        results = retriever.retrieve(c1.content, top_k=2)
        assert len(results) == 2
        top = results[0]
        assert isinstance(top, RetrievedChunk)
        assert top.chunk_id == c1.chunk_id
        assert top.text == c1.content
        assert math.isclose(top.score, 1.0, rel_tol=1e-4)

        # Verify property helpers
        assert top.file_path == "src/calc.py"
        assert top.line_start == 10
        assert top.line_end == 20
        assert top.repository == "my-repo"
        assert top.commit_sha == "sha-1234"
        assert top.citation == "src/calc.py:10-20"

        # Verify metadata filters
        md_results = retriever.retrieve(
            c1.content, top_k=2, filters={"language": "markdown"}
        )
        assert len(md_results) == 1
        assert md_results[0].chunk_id == c2.chunk_id
        assert md_results[0].file_path == "docs/guide.md"

        # Empty query returns empty list
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    finally:
        store.close()
