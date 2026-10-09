# RepoPilot

An agentic GitHub developer copilot built on **Advanced RAG + LangGraph + MCP**.

> **Status: Week 0.** This milestone contains only the project skeleton, typed
> configuration, and the Gemini LLM gateway. Retrieval, the LangGraph agent
> graph, and MCP servers come in later weeks.

## Project overview

```
repopilot/
├── config.py          # pydantic-settings: .env (secrets) + configs/*.yaml (settings)
├── llm/               # Week 0: tiered LLM gateway (see below)
├── ingestion/         # (later) repo/document ingestion
├── retrieval/         # (later) advanced RAG retrieval
├── mcp_servers/       # (later) MCP servers
├── graph/             # (later) LangGraph agent graph
├── evidence/          # (later) evidence pack
├── evaluation/        # (later) eval harness
├── observability/     # (later) Langfuse tracing
├── api/               # (later) service API
└── ui/                # (later) user interface
configs/               # settings.yaml (app, paths, budgets) + llm.yaml (LLM tiers)
scripts/smoke_test.py  # live smoke test against the Gemini API
tests/                 # pytest suite — FakeProvider only, zero network
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
| `configs/settings.yaml` | App metadata, paths, per-run LLM budgets          |
| `configs/llm.yaml`   | LLM tiers (`fast`, `strong`, `judge`): provider, model, temperature, max output tokens |
