# RepoPilot

An agentic GitHub developer copilot built on **Advanced RAG + LangGraph + MCP**.

> **Status: Week 1 (Runs 1a, 1b & 1c).** LLM gateway + Week 1 foundations: repository
> ingestion, character chunking, local embeddings (FastEmbed), Qdrant vector store,
> V1 dense retrieval QA pipeline, and offline evaluation harness (Precision@K, Recall@K,
> MRR, NDCG@K, source-type breakdown, CLI runner, and candidate question drafter).
> Hybrid retrieval, reranking, and LangGraph agent graph come in subsequent weeks.

## Project overview

```
repopilot/
├── config.py          # pydantic-settings: .env (secrets) + configs/*.yaml (settings)
├── llm/               # Week 0: tiered LLM gateway
├── ingestion/         # Week 1: Git clone, file walker, character chunker
├── retrieval/         # Week 1: FastEmbed/FakeEmbedder, QdrantVectorStore, DenseRetriever
├── pipeline_v1.py     # Week 1: V1 indexing and QA pipeline
├── evaluation/        # Week 1: golden dataset schema, metrics, runner
├── mcp_servers/       # (later) MCP servers
├── graph/             # (later) LangGraph agent graph
├── evidence/          # (later) evidence pack
├── observability/     # (later) Langfuse tracing
├── api/               # (later) service API
└── ui/                # (later) user interface
configs/               # settings.yaml + llm.yaml
docker-compose.yml     # Qdrant vector database (pinned v1.19.1, 1GB memory limit)
docs/week1.md          # Week 1 design decisions, benchmarks and config reference
docs/eval.md           # Week 1 evaluation methodology, metrics and baseline table
eval_samples/          # verified sample golden questions (committed)
scripts/index_repo.py  # index a git repo into Qdrant
scripts/ask_v1.py      # ask grounded questions against an indexed repo
scripts/run_retrieval_eval.py # evaluate retrieval against golden datasets
scripts/draft_questions.py    # draft candidate evaluation questions
scripts/smoke_test.py  # live smoke test against the Gemini API
tests/                 # pytest suite — zero network, fakes + local Qdrant
```



### LLM gateway (`repopilot/llm/`)

- `get_llm(tier)` returns a client for the `fast`, `strong` or `judge` tier
  (defined in `configs/llm.yaml`) with `.generate()` and
  `.generate_structured()` methods.
- Every call returns **text + `latency_ms`, `input_tokens`, `output_tokens`,
  `cache_hit`**.
- Disk cache under `.cache/llm/` keyed on hash(model, system, prompt, params);
  cache hits cost **zero API calls** and zero budget.
- Retries with exponential backoff + jitter on 429 / transient 5xx;
  auth and validation errors fail fast.
- Per-run budget guard (calls + input/output tokens) from `settings.yaml`,
  raising `BudgetExceeded`.
- Structured output is validated with Pydantic, retrying up to 2 times with
  the validation error included in the prompt.
- Providers are swappable behind an abstract interface (`FakeProvider` ships
  for tests; only `GeminiProvider` is implemented in Week 0).

### Ingestion & Vector store (`repopilot/ingestion/` & `repopilot/retrieval/`)

- `clone_repo(url, dest)` clones git repositories via GitPython (shallow support).
- `walk_repo_files(repo_path)` walks repository files, skipping `.git`, binaries,
  vendored directories, lockfiles, and oversized files.
- `chunk_text(content)` splits text into character chunks with overlap, tracking
  1-indexed line start/end numbers and producing stable `chunk_id` hashes.
- `FastEmbedEmbedder` provides local CPU embeddings with `BAAI/bge-small-en-v1.5` (384-dim).
  `FakeEmbedder` provides deterministic, offline vectors for unit tests.
- `QdrantVectorStore` wraps Qdrant with idempotent upsert and metadata-filtered search,
  supporting local embedded storage (`data/qdrant`) and server mode (`docker-compose.yml`).
- `DenseRetriever` implements the `Retriever` protocol, embedding queries and mapping Qdrant results to `RetrievedChunk` objects.
- `pipeline_v1.py` exposes `index_repository()` (idempotent ingestion pipeline) and `answer_question()` (grounded QA with strict context boundary rules and line citations).
- See [docs/week1.md](docs/week1.md) for full design decisions, benchmarks and config reference.

### Evaluation harness (`repopilot/evaluation/`)

- `GoldenQuestion`: Pydantic schema for evaluation questions (`id`, `repo`, `repo_commit`, `question`, `category`, `expected_files`, `expected_answer_notes`, `answerable`).
- Pure Python retrieval metrics: Precision@K (chunks), Recall@K (unique files), MRR (unique files), and NDCG@K (unique files, binary relevance, ideal DCG based on $\min(K, |\text{expected\_files}|)$).
- Source-type classification: tracks chunk distribution across `code`, `docs`, and `tests`.
- Commit verification: validates dataset `repo_commit` against collection metadata to prevent evaluation on drifted code.
- `scripts/run_retrieval_eval.py`: CLI evaluation runner that outputs formatted summary tables and persists `results.json` and `summary.md`.
- `scripts/draft_questions.py`: bootstraps candidate evaluation questions via stratified chunk sampling and strong LLM generation (candidates are saved with `reviewed: false` and hints-only disclaimer for human curation).
- See [docs/eval.md](docs/eval.md) for complete metric definitions, guidelines, and V1 baseline tables.


## Setup (Windows)


Requires **Python 3.11+** (3.12 works; use `python` if it is on your PATH).

```powershell
# 1. Clone and enter the repo
git clone https://github.com/snehala24/repopilot.git
cd repopilot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell (use .venv\Scripts\activate for cmd)

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Create your local secrets file (never commit .env)
copy .env.example .env
#    -> edit .env and fill in GEMINI_API_KEY (from Google AI Studio)

# 5. Fill in the model IDs
#    -> edit configs/llm.yaml and replace each <PLACEHOLDER_MODEL_ID>
#       with a model ID from https://aistudio.google.com/app/apikey
```

All secrets live in `.env` (gitignored) and all non-secret settings live in
`configs/*.yaml`. Nothing sensitive is hard-coded in source.

## Run the smoke test

With your real key configured:

```powershell
python scripts\smoke_test.py
```

It calls the `fast` and `strong` tiers with a short prompt, then repeats the
`fast` call — the third call should print `cache_hit: True`.

## Run the tests

No API key and no network needed (all tests use `FakeProvider`):

```powershell
python -m pytest
```

## Index and query a repository (Windows)

Start Qdrant in Docker (or omit `--server` to use local embedded storage):

```powershell
# 1. Start Qdrant Docker service
docker compose up -d

# 2. Index a repository (e.g. pallets/click)
python scripts\index_repo.py --repo https://github.com/pallets/click --name click --server

# 3. Query the repository (requires real GEMINI_API_KEY in .env)
python scripts\ask_v1.py --repo click "How does Click define command options?" --server
```

## Run retrieval evaluation (Windows)

Run retrieval evaluation against golden datasets (zero LLM calls, deterministic metrics):

```powershell
# Evaluate DenseRetriever on Docker Qdrant (server mode)
python scripts\run_retrieval_eval.py --retriever dense --dataset eval_samples/golden_sample.jsonl --k 5 --server

# Or evaluate against local embedded storage
python scripts\run_retrieval_eval.py --retriever dense --dataset eval_samples/golden_sample.jsonl --k 5 --local
```

Detailed metrics (`Precision@K`, `Recall@K`, `MRR`, `NDCG@K`), source-type breakdowns, and the worst-performing queries are written to `results/<timestamp>_dense/summary.md` and `results.json`. See [docs/eval.md](docs/eval.md) for full benchmark documentation.

## Configuration reference


| File                 | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `.env`               | Secrets: `GEMINI_API_KEY`, `GITHUB_TOKEN`, Langfuse keys |
| `configs/settings.yaml` | App metadata, paths, budgets, ingestion, embedding, vector store |
| `configs/llm.yaml`   | LLM tiers (`fast`, `strong`, `judge`): provider, model, temperature, max output tokens |

