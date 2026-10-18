# Week 1: Retrieval Foundations & V1 Pipeline

This document details the configuration defaults, architecture, CLI usage, and design decisions for RepoPilot's Week 1 milestones (Run 1a: Foundations & Run 1b: V1 Pipeline).

---

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
- `mode: "local"`: Storage mode (`local` for embedded file-backed storage in `data/qdrant`, `server` for Qdrant service at `http://localhost:6333`).
- `path: "data/qdrant"`: Local directory used in embedded mode.
- `url: "http://localhost:6333"`: REST API endpoint for server mode.
- `collection_prefix: "repopilot"`: Prefix for generated Qdrant collections.

---

## 2. CLI Usage (Windows PowerShell)

Ensure your virtual environment is active:
```powershell
.venv\Scripts\Activate.ps1
```

### Start Qdrant in Docker (Server Mode)
```powershell
docker compose up -d
```

### Indexing a Repository (`scripts/index_repo.py`)
Clones the repository into `data/repos/<name>` (reused if already present), walks source files, character chunks with overlap, computes embeddings locally with FastEmbed, and upserts into Qdrant collection `repopilot_<name>`:

```powershell
# Index into Qdrant Docker service (server mode)
python scripts\index_repo.py --repo https://github.com/pallets/click --name click --server

# Or index using local embedded storage
python scripts\index_repo.py --repo https://github.com/pallets/click --name click --local
```

### Querying the Repository (`scripts/ask_v1.py`)
Retrieves top relevant chunks via `DenseRetriever`, constructs a fortified QA prompt, queries the Gemini `strong` LLM tier, and outputs the answer, citations, and metrics:

```powershell
# Ask against Qdrant server mode
python scripts\ask_v1.py --repo click "How does Click define command options and arguments?" --server

# Ask with custom top-k
python scripts\ask_v1.py --repo click "Where is the main BaseCommand class implemented?" --top-k 3 --server
```

---

## 3. Design Decisions & V1 Pipeline Architecture

### V1 Simple Character Chunker
A fixed-size character chunker with overlap and 1-indexed line tracking was chosen for the initial baseline. It provides predictable, deterministic boundaries across all programming and markup languages without external native parsing dependencies. Tree-sitter AST and language-aware chunking will be introduced in subsequent runs.

### Stable Idempotent `chunk_id` Hashing
The `chunk_id` is an SHA-256 hash computed over:
$$\text{SHA256}(\text{repository} : \text{file\_path} : \text{line\_start} : \text{line\_end} : \text{content})$$

- **Commit SHA Excluded from Hash**: `commit_sha` is stored as metadata on the chunk payload rather than inside the hash. When re-indexing after new commits, chunks with identical code positions update their metadata in-place rather than duplicating vectors in Qdrant.
- **Deterministic Qdrant UUIDs**: Because Qdrant requires point IDs to be UUIDs or unsigned integers, `chunk_id` is deterministically converted to a UUID via UUIDv5 (`uuid.uuid5(uuid.NAMESPACE_OID, chunk_id)`).

### Prompt Security & Grounding Defense
The V1 QA pipeline wraps retrieved chunks inside explicit `<context id="..." file="..." lines="...">` blocks and enforces strict system rules:
1. **Context Grounding**: The LLM must answer strictly using information inside the `<context>` blocks.
2. **Mandatory Citations**: Every factual statement must cite file path and line numbers (e.g. `click/core.py:L10-L35`).
3. **Insufficient Context Refusal**: If the context does not contain the answer, the LLM must explicitly refuse rather than speculating.
4. **Data vs. Instruction Separation**: Repository content is explicitly declared as untrusted data, preventing prompt injection attacks hidden inside repository source code or markdown.

### Known Limitation: Stale Chunks on Re-indexing
> [!NOTE]
> In the V1 baseline, re-indexing an updated repository updates modified chunks in-place, but **does not remove stale chunks from deleted or renamed files**. Resolving stale chunk pruning will be addressed in future milestones with manifest tracking.

---

## 4. Live Indexing Benchmark (`pallets/click`)

Executed against Qdrant `v1.19.1` in Docker (server mode):
- **Repository**: `https://github.com/pallets/click` (HEAD: `2247b35ea1c47c727d7a06e51fa280e12a863ff6`)
- **Files Indexed**: 169 source files
- **Chunks Generated & Upserted**: 1,732 chunks (28 batches)
- **Qdrant Collection**: `repopilot_click` (1,732 points confirmed)
- **Elapsed Time**: 231.77s (including initial ONNX weight download)
