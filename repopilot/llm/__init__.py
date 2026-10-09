"""RepoPilot LLM gateway: tiered clients with cache, retries and budgets."""

from __future__ import annotations

from .base import CompletionRequest, CompletionResult, Provider
from .budget import BudgetGuard, BudgetExceeded
from .cache import CacheEntry, DiskCache
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
from .gateway import (
    DEFAULT_STRUCTURED_RETRIES,
    LLMGateway,
    LLMResponse,
    RetryConfig,
    StructuredResponse,
    get_llm,
)

__all__ = [
    "DEFAULT_STRUCTURED_RETRIES",
    "BudgetExceeded",
    "BudgetGuard",
    "CacheEntry",
    "CompletionRequest",
    "CompletionResult",
    "DiskCache",
    "FakeProvider",
    "GeminiProvider",
    "LLMGateway",
    "LLMGatewayError",
    "LLMResponse",
    "Provider",
    "ProviderAuthError",
    "ProviderBadRequestError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTransientError",
    "RetryConfig",
    "StructuredOutputError",
    "StructuredResponse",
    "get_llm",
]
