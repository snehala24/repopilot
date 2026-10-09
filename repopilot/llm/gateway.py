"""Tiered LLM gateway.

``get_llm(tier)`` returns a client exposing:

- ``generate(prompt, system=None) -> LLMResponse``
- ``generate_structured(prompt, schema, system=None) -> StructuredResponse``

Every call returns text plus ``latency_ms``, ``input_tokens``,
``output_tokens`` and ``cache_hit``. The gateway layers on top of a
:class:`~repopilot.llm.base.Provider`:

- disk cache (cache hits cost zero API calls and zero budget),
- exponential backoff + jitter on 429/5xx (tenacity); auth/validation errors
  fail fast,
- a per-run budget guard fed from ``settings.yaml``,
- structured output validated with Pydantic (up to 2 retries with the
  validation error appended to the prompt).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ..config import LLMTierSettings, Settings, load_settings
from .base import CompletionRequest, CompletionResult, Provider
from .budget import BudgetGuard
from .cache import CacheEntry, DiskCache
from .errors import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTransientError,
    StructuredOutputError,
)
from .gemini import GeminiProvider

T = TypeVar("T", bound=BaseModel)

#: How many times a structured response may be re-requested after a
#: validation failure (total attempts = 1 + retries).
DEFAULT_STRUCTURED_RETRIES = 2


@dataclass(frozen=True)
class RetryConfig:
    """Backoff settings for retryable provider errors (429 / transient 5xx)."""

    attempts: int = 4
    initial_wait: float = 0.5
    max_wait: float = 8.0
    jitter: float = 0.5


@dataclass(frozen=True)
class LLMResponse:
    """Result of a plain-text generation."""

    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    model: str
    tier: str


@dataclass(frozen=True)
class StructuredResponse(Generic[T]):
    """Result of a structured generation: parsed data plus usage metrics."""

    data: T
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    model: str
    tier: str


class LLMGateway:
    """Client for one LLM tier: cache + retries + budget + structured output."""

    def __init__(
        self,
        *,
        tier: str,
        tier_settings: LLMTierSettings,
        provider: Provider,
        cache: DiskCache,
        budget: BudgetGuard,
        retry: RetryConfig | None = None,
        max_structured_retries: int = DEFAULT_STRUCTURED_RETRIES,
    ) -> None:
        self.tier = tier
        self.tier_settings = tier_settings
        self.provider = provider
        self.cache = cache
        self.budget = budget
        self.retry = retry or RetryConfig()
        self.max_structured_retries = max_structured_retries

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Generate plain text for *prompt*."""
        request = self._request(prompt, system, response_schema=None)
        completion, latency_ms, cache_hit = self._complete(request)
        return LLMResponse(
            text=completion.text,
            latency_ms=latency_ms,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cache_hit=cache_hit,
            model=request.model,
            tier=self.tier,
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
    ) -> StructuredResponse[T]:
        """Generate a response validated against the Pydantic *schema*.

        On validation failure the call is repeated (up to
        ``max_structured_retries`` extra attempts) with the Pydantic
        validation error appended to the prompt.

        Raises:
            StructuredOutputError: if every attempt fails validation.
        """
        json_schema = schema.model_json_schema()
        attempts = 1 + self.max_structured_retries

        total_latency_ms = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        all_cached = True
        last_text = ""
        last_error: str | None = None

        for attempt in range(attempts):
            current_prompt = prompt
            if attempt > 0:
                current_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response failed validation with the "
                    "following error:\n"
                    f"{last_error}\n"
                    "Respond again with JSON only, matching the required "
                    "schema."
                )
            request = self._request(
                current_prompt, system, response_schema=json_schema
            )
            completion, latency_ms, cache_hit = self._complete(request)
            total_latency_ms += latency_ms
            total_input_tokens += completion.input_tokens
            total_output_tokens += completion.output_tokens
            all_cached = all_cached and cache_hit
            last_text = completion.text

            try:
                data = schema.model_validate_json(completion.text)
            except ValidationError as exc:
                last_error = str(exc)
                continue

            return StructuredResponse(
                data=data,
                text=completion.text,
                latency_ms=total_latency_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_hit=all_cached,
                model=request.model,
                tier=self.tier,
            )

        raise StructuredOutputError(
            f"Structured output for tier {self.tier!r} failed validation "
            f"after {attempts} attempts. Last validation error:\n{last_error}\n"
            f"Last response (truncated): {last_text[:500]}"
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _request(
        self,
        prompt: str,
        system: str | None,
        *,
        response_schema: dict | None,
    ) -> CompletionRequest:
        return CompletionRequest(
            model=self.tier_settings.model,
            prompt=prompt,
            system=system,
            temperature=self.tier_settings.temperature,
            max_output_tokens=self.tier_settings.max_output_tokens,
            response_schema=response_schema,
        )

    def _complete(
        self, request: CompletionRequest
    ) -> tuple[CompletionResult, float, bool]:
        """Run one completion through cache → budget → retrying provider."""
        start = time.perf_counter()
        entry = self.cache.get(request)
        if entry is not None:
            latency_ms = (time.perf_counter() - start) * 1000
            self.budget.record_cache_hit()
            return (
                CompletionResult(
                    text=entry.text,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                ),
                latency_ms,
                True,
            )

        retrying = Retrying(
            retry=retry_if_exception_type(
                (ProviderRateLimitError, ProviderTransientError)
            ),
            wait=wait_exponential_jitter(
                initial=self.retry.initial_wait,
                max=self.retry.max_wait,
                jitter=self.retry.jitter,
            ),
            stop=stop_after_attempt(self.retry.attempts),
            reraise=True,
        )
        result = retrying(self._attempt, request)
        latency_ms = (time.perf_counter() - start) * 1000

        self.cache.set(
            request,
            CacheEntry(
                text=result.text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                model=request.model,
                created_at=time.time(),
            ),
        )
        return result, latency_ms, False

    def _attempt(self, request: CompletionRequest) -> CompletionResult:
        """One budget-checked provider call (the unit retried by tenacity)."""
        self.budget.check()
        self.budget.record_call()
        result = self.provider.complete(request)
        self.budget.record_tokens(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return CompletionResult(
            text=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


def _build_provider(provider_name: str, settings: Settings) -> Provider:
    """Instantiate the provider named in ``llm.yaml``."""
    if provider_name == "gemini":
        api_key = settings.env.gemini_api_key
        key = api_key.get_secret_value().strip() if api_key else ""
        if not key:
            raise ProviderAuthError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and "
                "add your Google AI Studio API key."
            )
        return GeminiProvider(key)
    raise ValueError(
        f"Unknown LLM provider {provider_name!r} in configs/llm.yaml. "
        "Supported providers: gemini"
    )


def get_llm(
    tier: str,
    settings: Settings | None = None,
    *,
    provider: Provider | None = None,
    budget: BudgetGuard | None = None,
    cache_dir: Path | None = None,
    retry: RetryConfig | None = None,
) -> LLMGateway:
    """Return a client for the ``fast`` / ``strong`` / ``judge`` tier.

    Args:
        tier: Tier name from ``configs/llm.yaml``.
        settings: Pre-loaded configuration; loaded from ``.env`` + YAML when
            omitted.
        provider: Override the provider (tests inject :class:`FakeProvider`).
        budget: Share one :class:`BudgetGuard` across several tier clients of
            the same run; a fresh guard from ``settings.yaml`` otherwise.
        cache_dir: Override ``paths.llm_cache_dir`` from ``settings.yaml``.
        retry: Override the default backoff configuration.
    """
    settings = settings if settings is not None else load_settings()
    tier_settings = settings.llm.tier(tier)
    if provider is None:
        provider = _build_provider(tier_settings.provider, settings)
    cache = DiskCache(
        Path(cache_dir) if cache_dir is not None else settings.paths.llm_cache_dir
    )
    guard = budget if budget is not None else BudgetGuard(settings.budgets)
    return LLMGateway(
        tier=tier,
        tier_settings=tier_settings,
        provider=provider,
        cache=cache,
        budget=guard,
        retry=retry,
    )
