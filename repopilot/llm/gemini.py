"""Gemini provider backed by the ``google-genai`` SDK (Google AI Studio)."""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import CompletionRequest, CompletionResult, Provider
from .errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
)

# HTTP status codes that are safe to retry with backoff.
_RETRYABLE_STATUS = frozenset({429})
_AUTH_STATUS = frozenset({401, 403})


def _map_api_error(exc: genai_errors.APIError) -> ProviderError:
    """Translate a google-genai API error into the gateway hierarchy."""
    code = int(exc.code or 0)
    message = str(exc)
    if code in _RETRYABLE_STATUS:
        return ProviderRateLimitError(message)
    if code in _AUTH_STATUS:
        return ProviderAuthError(message)
    if 500 <= code < 600:
        return ProviderTransientError(message)
    if 400 <= code < 500:
        return ProviderBadRequestError(message)
    # Unknown status: fail fast (never retried).
    return ProviderError(message)


class GeminiProvider(Provider):
    """Gemini via the Google AI Studio ``google-genai`` SDK."""

    name = "gemini"

    def __init__(self, api_key: str, *, client: genai.Client | None = None) -> None:
        self._client = client or genai.Client(api_key=api_key)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        config_kwargs: dict = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
        }
        if request.system:
            config_kwargs["system_instruction"] = request.system
        if request.response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = request.response_schema

        try:
            response = self._client.models.generate_content(
                model=request.model,
                contents=request.prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as exc:
            raise _map_api_error(exc) from exc

        try:
            text = response.text or ""
        except ValueError:
            # No text parts (e.g. safety block); callers see empty output.
            text = ""

        usage = response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        return CompletionResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
