"""Provider-agnostic interface for LLM backends.

The gateway only talks to this interface, so swapping Gemini for another
provider later means adding one class — no gateway changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class CompletionRequest:
    """Everything needed for one provider completion.

    ``response_schema`` is a JSON schema (dict) when the caller wants
    structured output; ``None`` for plain text generation.
    """

    model: str
    prompt: str
    system: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 1024
    response_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class CompletionResult:
    """Raw provider result: generated text plus token accounting."""

    text: str
    input_tokens: int
    output_tokens: int


class Provider(ABC):
    """Abstract LLM provider. Implementations must be synchronous."""

    name: ClassVar[str] = "provider"

    @abstractmethod
    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Execute one completion.

        Raises:
            ProviderRateLimitError: HTTP 429 (retryable).
            ProviderTransientError: transient 5xx (retryable).
            ProviderAuthError: authentication failure (fail fast).
            ProviderBadRequestError: invalid request (fail fast).
        """
        raise NotImplementedError
