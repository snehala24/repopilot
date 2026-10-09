"""Exception hierarchy for the RepoPilot LLM gateway.

Retryable errors (rate limits, transient 5xx) are retried with exponential
backoff; auth and validation errors fail fast.
"""

from __future__ import annotations


class LLMGatewayError(Exception):
    """Base class for all LLM gateway errors."""


class ProviderError(LLMGatewayError):
    """Base class for errors raised by an LLM provider."""


class ProviderRateLimitError(ProviderError):
    """HTTP 429 — retryable with backoff and jitter."""


class ProviderTransientError(ProviderError):
    """Transient 5xx — retryable with backoff and jitter."""


class ProviderAuthError(ProviderError):
    """Authentication/permission failure — fail fast, never retried."""


class ProviderBadRequestError(ProviderError):
    """Invalid request (e.g. bad model ID) — fail fast, never retried."""


class StructuredOutputError(LLMGatewayError):
    """Structured output still failed validation after all retries."""
