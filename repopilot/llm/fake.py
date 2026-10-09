"""Deterministic fake provider for tests — zero network, zero API keys."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence

from .base import CompletionRequest, CompletionResult, Provider


def _approx_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)


class FakeProvider(Provider):
    """Canned, deterministic provider.

    Args:
        responses: Queue of texts returned by successive calls. When empty,
            ``default_response`` (or ``{}`` for schema requests) is returned.
        errors: Exceptions raised by the first calls, in order, before any
            response is consumed. Use this to simulate 429/5xx failures.
        default_response: Text used when ``responses`` is exhausted.
    """

    name = "fake"

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        errors: Sequence[Exception] | None = None,
        default_response: str = "This is a fake response.",
    ) -> None:
        self._responses: deque[str] = deque(responses or [])
        self._errors: deque[Exception] = deque(errors or [])
        self._default_response = default_response
        self.requests: list[CompletionRequest] = []

    @property
    def call_count(self) -> int:
        """Number of completed attempts (including ones that raised)."""
        return len(self.requests)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        if self._errors:
            raise self._errors.popleft()
        if self._responses:
            text = self._responses.popleft()
        elif request.response_schema is not None:
            text = json.dumps({})
        else:
            text = self._default_response
        return CompletionResult(
            text=text,
            input_tokens=_approx_tokens(request.prompt),
            output_tokens=_approx_tokens(text),
        )
