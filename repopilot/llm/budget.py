"""Per-run budget guard: caps LLM calls and tokens from ``settings.yaml``.

Cache hits are free and do not consume budget. ``check()`` runs before every
provider call and raises :class:`BudgetExceeded` once any limit is reached;
``record_*()`` updates the counters after each call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import BudgetSettings


class BudgetExceeded(Exception):
    """A configured LLM budget limit from ``settings.yaml`` was hit."""


@dataclass
class BudgetGuard:
    """Counts API calls/tokens for one run."""

    budgets: BudgetSettings
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = field(default=0)

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if any limit has been reached."""
        if self.calls >= self.budgets.max_llm_calls:
            raise BudgetExceeded(
                f"LLM call budget exceeded: {self.calls}/"
                f"{self.budgets.max_llm_calls} calls used"
            )
        if self.input_tokens >= self.budgets.max_input_tokens:
            raise BudgetExceeded(
                f"LLM input-token budget exceeded: {self.input_tokens}/"
                f"{self.budgets.max_input_tokens} tokens used"
            )
        if self.output_tokens >= self.budgets.max_output_tokens:
            raise BudgetExceeded(
                f"LLM output-token budget exceeded: {self.output_tokens}/"
                f"{self.budgets.max_output_tokens} tokens used"
            )

    def record_call(self) -> None:
        """Count one provider API call (including attempts that failed)."""
        self.calls += 1

    def record_tokens(self, *, input_tokens: int, output_tokens: int) -> None:
        """Add token usage from one completed call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_cache_hit(self) -> None:
        """Count a free cache hit (does not affect budgets)."""
        self.cache_hits += 1

    def usage(self) -> dict[str, int]:
        """Snapshot of the counters (useful for logging/reporting)."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hits": self.cache_hits,
        }
