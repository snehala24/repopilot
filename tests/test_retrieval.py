"""Unit tests for FakeEmbedder and QdrantVectorStore (local mode, no network)."""

import math
from pathlib import Path

from repopilot.ingestion.models import Chunk
from repopilot.retrieval.embedder import FakeEmbedder
from repopilot.retrieval.store import QdrantVectorStore, chunk_id_to_uuid


def test_fake_embedder_deterministic_and_normalized():
    """Verify FakeEmbedder is deterministic, normalized to unit length, and matches dimensions."""
    embedder = FakeEmbedder(dimensions=384)
    assert embedder.dimensions == 384

    v1 = embedder.embed_query("def foo(): pass")
    v2 = embedder.embed_query("def foo(): pass")
    v3 = embedder.embed_query("different text")

    assert v1 == v2
    assert v1 != v3
    assert len(v1) == 384

    # Unit vector (L2 norm == 1.0)
    norm = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)

    docs = embedder.embed_documents(["doc1", "doc2"])
    assert len(docs) == 2
    assert docs[0] == embedder.embed_query("doc1")
    assert docs[1] == embedder.embed_query("doc2")


def test_qdrant_store_idempotent_upsert_and_search(tmp_path: Path):
    """Verify local QdrantVectorStore supports idempotent upsert and nearest search with filters."""
    qdrant_dir = tmp_path / "qdrant"
    store = QdrantVectorStore(mode="local", path=qdrant_dir)
    collection_name = "test_chunks"
    embedder = FakeEmbedder(dimensions=384)

    try:
        # 1. Create collection (idempotent)
        store.create_collection(collection_name, vector_size=384, distance="Cosine")
        store.create_collection(collection_name, vector_size=384, distance="Cosine")

        # 2. Prepare chunks across languages
        c1 = Chunk.create(
            repository="test-repo",
            file_path="src/main.py",
            language="python",
            line_start=1,
            line_end=5,
            content="def calculate_total(items): return sum(items)",
            commit_sha="commit-1",
        )
        c2 = Chunk.create(
            repository="test-repo",
            file_path="docs/guide.md",
            language="markdown",
            line_start=1,
            line_end=3,
            content="# Installation Guide\nRun pip install.",
            commit_sha="commit-1",
        )

        chunks = [c1, c2]
        vectors = embedder.embed_documents([c.content for c in chunks])

        # 3. Initial upsert
        upserted = store.upsert_chunks(collection_name, chunks, vectors)
        assert upserted == 2
        assert store.count(collection_name) == 2

        # 4. Idempotent re-upsert with same chunks (should not create duplicate points)
        upserted_again = store.upsert_chunks(collection_name, chunks, vectors)
        assert upserted_again == 2
        assert store.count(collection_name) == 2

        # 5. Re-index with updated commit_sha: chunk_id stays the same, payload updates
        c1_v2 = Chunk.create(
            repository="test-repo",
            file_path="src/main.py",
            language="python",
            line_start=1,
            line_end=5,
            content="def calculate_total(items): return sum(items)",
            commit_sha="commit-2-new-sha",
        )
        assert c1_v2.chunk_id == c1.chunk_id

        store.upsert_chunks(
            collection_name, [c1_v2], [embedder.embed_query(c1_v2.content)]
        )
        assert store.count(collection_name) == 2

        # 6. Nearest search for c1
        query_vec = embedder.embed_query(c1.content)
        results = store.search(collection_name, query_vec, limit=5)
        assert len(results) == 2
        assert results[0].chunk_id == c1.chunk_id
        assert results[0].commit_sha == "commit-2-new-sha"
        assert math.isclose(results[0].score, 1.0, rel_tol=1e-4)

        # 7. Metadata filtering (language == "markdown")
        md_results = store.search(
            collection_name,
            query_vec,
            limit=5,
            filters={"language": "markdown"},
        )
        assert len(md_results) == 1
        assert md_results[0].chunk_id == c2.chunk_id
        assert md_results[0].language == "markdown"

        # 8. Metadata filtering (no match)
        no_match = store.search(
            collection_name,
            query_vec,
            limit=5,
            filters={"language": "rust"},
        )
        assert len(no_match) == 0

    finally:
        store.close()


def test_chunk_id_to_uuid_deterministic():
    """Verify chunk_id_to_uuid generates deterministic UUIDs for Qdrant point IDs."""
    u1 = chunk_id_to_uuid("hash-123")
    u2 = chunk_id_to_uuid("hash-123")
    u3 = chunk_id_to_uuid("hash-456")

    assert u1 == u2
    assert u1 != u3
    assert len(u1) == 36
