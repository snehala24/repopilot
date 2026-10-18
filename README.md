# RepoPilot

An agentic GitHub developer copilot built on **Advanced RAG + LangGraph + MCP**.

> **Status: Week 1 (Run 1a).** LLM gateway + Week 1 foundations: repository
> ingestion, character chunking, local embeddings (FastEmbed), and vector store
> (Qdrant local/server). Advanced retrieval pipelines, evaluation, and LangGraph
> agent graph come in subsequent runs.

## Project overview

```
repopilot/
├── config.py          # pydantic-settings: .env (secrets) + configs/*.yaml (settings)
├── llm/               # Week 0: tiered LLM gateway
├── ingestion/         # Week 1: Git clone, file walker, character chunker
├── retrieval/         # Week 1: FastEmbed/FakeEmbedder, QdrantVectorStore
├── mcp_servers/       # (later) MCP servers
├── graph/             # (later) LangGraph agent graph
├── evidence/          # (later) evidence pack
├── evaluation/        # (later) eval harness
├── observability/     # (later) Langfuse tracing
├── api/               # (later) service API
└── ui/                # (later) user interface
configs/               # settings.yaml + llm.yaml
docker-compose.yml     # Qdrant vector database (pinned v1.19.1, 1GB memory limit)
docs/week1.md          # Week 1 design decisions and config reference
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
- See [docs/week1.md](docs/week1.md) for full design decisions and config reference.

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

## Configuration reference

| File                 | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `.env`               | Secrets: `GEMINI_API_KEY`, `GITHUB_TOKEN`, Langfuse keys |
| `configs/settings.yaml` | App metadata, paths, budgets, ingestion, embedding, vector store |
| `configs/llm.yaml`   | LLM tiers (`fast`, `strong`, `judge`): provider, model, temperature, max output tokens |

