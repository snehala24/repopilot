"""RepoPilot LLM gateway smoke test (needs a real GEMINI_API_KEY in .env).

Runs three calls and prints response, latency, tokens and cache_hit for each:

1. ``fast`` tier with a short prompt  (real API call)
2. ``strong`` tier with the same prompt (real API call)
3. ``fast`` tier again with the same prompt (expected: cache_hit=True)

Usage (from the project root, after ``pip install -r requirements.txt``):

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repopilot.config import load_settings  # noqa: E402
from repopilot.llm import BudgetGuard, LLMResponse, get_llm  # noqa: E402

PROMPT = "In one short sentence, explain what a monorepo is."


def show(label: str, response: LLMResponse) -> None:
    """Print one response with its usage metrics."""
    print(f"\n[{label}]")
    print(f"  cache_hit   : {response.cache_hit}")
    print(f"  latency_ms  : {response.latency_ms:.1f}")
    print(f"  input_tokens: {response.input_tokens}")
    print(f"  output_tokens: {response.output_tokens}")
    print(f"  model       : {response.model}")
    print(f"  response    : {response.text.strip()}")


def main() -> int:
    settings = load_settings()

    # Share one budget guard across the run (per-run limits from settings.yaml).
    budget = BudgetGuard(settings.budgets)
    fast = get_llm("fast", settings, budget=budget)
    strong = get_llm("strong", settings, budget=budget)

    print(f"Prompt: {PROMPT!r}")

    show("fast #1 (API call)", fast.generate(PROMPT))
    show("strong (API call)", strong.generate(PROMPT))

    repeat = fast.generate(PROMPT)
    show("fast #2 (same prompt — expect cache hit)", repeat)

    print(f"\nRun usage: {budget.usage()}")

    if not repeat.cache_hit:
        print(
            "\nFAILED: the repeated call did not hit the cache. "
            "Check that configs/llm.yaml and the gateway did not change "
            "model/params between identical calls.",
            file=sys.stderr,
        )
        return 1
    print("\nOK: second identical call was served from the disk cache.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # friendly, non-traceback failure for the CLI
        print(f"\nSMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
