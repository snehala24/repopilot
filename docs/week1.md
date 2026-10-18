# Week 1: Retrieval Foundations (Run 1a)

This milestone introduces the basic ingestion, embedding, and vector storage components for RepoPilot's V1 retrieval baseline.

## 1. Configuration Defaults (`configs/settings.yaml`)

### Ingestion (`ingestion`)
- `chunk_size: 1000`: Maximum characters per chunk.
- `chunk_overlap: 200`: Overlapping characters between consecutive chunks to preserve contextual boundary tokens.
- `max_file_size_bytes: 1000000`: Hard cutoff (1MB) to prevent ingesting oversized or generated artifacts.
- `skip_patterns`: Directories and patterns ignored by the file walker, including `.git`, `node_modules`, `vendor`, `dist`, `build`, `target`, `__pycache__`, `.venv`, lockfiles (`*.lock`, `package-lock.json`, `poetry.lock`, `Cargo.lock`, `yarn.lock`), minified assets (`*.min.js`, `*.min.css`), and source maps (`*.map`).

### Embeddings (`embedding`)
- `model: "BAAI/bge-small-en-v1.5"`: High-performance, lightweight CPU-friendly embedding model.
- `dimensions: 384`: Output embedding vector dimensionality.
- `batch_size: 64`: Batch size for embedding document chunks.

### Vector Store (`vector_store`)
- `mode: "local"`: Storage mode (`local` for embedded file-backed storage, `server` for Qdrant service).
- `path: "data/qdrant"`: Local directory used in embedded mode.
- `url: "http://localhost:6333"`: REST API endpoint for server mode.
- `collection_prefix: "repopilot"`: Prefix for generated Qdrant collections.

---

## 2. Design Decisions

### V1 Simple Character Chunker
A fixed-size character chunker with overlap and 1-indexed line tracking was chosen for the initial baseline. It provides predictable, deterministic boundaries across all programming and markup languages without external native parsing dependencies. Tree-sitter AST and language-aware chunking will be introduced in subsequent runs.

### Stable Idempotent `chunk_id` Hashing
The `chunk_id` is an SHA-256 hash computed over:
$$\text{SHA256}(\text{repository} : \text{file\_path} : \text{line\_start} : \text{line\_end} : \text{content})$$

- **Commit SHA Excluded from Hash**: `commit_sha` is stored as metadata on the chunk payload rather than inside the hash. When re-indexing after new commits, chunks with identical code positions update their metadata in-place rather than duplicating vectors in Qdrant.
- **Deterministic Qdrant UUIDs**: Because Qdrant requires point IDs to be UUIDs or unsigned integers, `chunk_id` is deterministically converted to a UUID via UUIDv5 (`uuid.uuid5(uuid.NAMESPACE_OID, chunk_id)`).

### FastEmbed & Test Fakes
- **Local CPU Inference**: `FastEmbedEmbedder` runs the ONNX-quantized `bge-small-en-v1.5` model locally via `fastembed`, requiring zero external API keys and no GPU.
- **Deterministic FakeEmbedder**: `FakeEmbedder` generates deterministic, normalized unit vectors from text hashes, allowing offline, fast, and repeatable unit tests without downloading neural network weights.

### Dual-Mode Qdrant Storage
- **Embedded Local Mode**: Runs directly within the process using local directory storage under `data/qdrant/`.
- **Docker Compose Service**: `docker-compose.yml` provides a pinned `qdrant/qdrant:v1.19.1` service with a 1GB memory limit and persistent volume storage (`qdrant_storage`) matching the `qdrant-client` 1.19.1 library version.
