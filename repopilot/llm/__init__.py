"""RepoPilot LLM gateway: tiered clients with cache, retries and budgets."""

from __future__ import annotations

from .base import CompletionRequest, CompletionResult, Provider
from .errors import (
    LLMGatewayError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
    StructuredOutputError,
)
from .fake import FakeProvider
from .gemini import GeminiProvider

__all__ = [
    "CompletionRequest",
    "CompletionResult",
    "FakeProvider",
    "GeminiProvider",
    "LLMGatewayError",
    "Provider",
    "ProviderAuthError",
    "ProviderBadRequestError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTransientError",
    "StructuredOutputError",
]
