"""Retry behaviour: backoff on 429/5xx, fail-fast on auth/validation."""

from __future__ import annotations

import pytest

from repopilot.llm import (
    FakeProvider,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTransientError,
)


def test_retries_rate_limit_then_succeeds(make_gateway):
    provider = FakeProvider(
        errors=[ProviderRateLimitError("429 Too Many Requests")],
        responses=["recovered"],
    )
    llm = make_gateway(provider)

    response = llm.generate("hello")

    assert response.text == "recovered"
    assert provider.call_count == 2  # one 429 attempt + one success
    # Every attempt counts toward the call budget.
    assert llm.budget.usage()["calls"] == 2
    # Tokens are only recorded for calls that returned usage.
    assert llm.budget.usage()["input_tokens"] > 0


def test_retries_transient_5xx_with_backoff(make_gateway):
    provider = FakeProvider(
        errors=[ProviderTransientError("503")] * 3,
        responses=["eventually"],
    )
    llm = make_gateway(provider)

    response = llm.generate("hello")

    assert response.text == "eventually"
    assert provider.call_count == 4  # 3 failures + 1 success


def test_gives_up_after_max_attempts(make_gateway):
    provider = FakeProvider(errors=[ProviderRateLimitError("429")] * 10)
    llm = make_gateway(provider)  # FAST_RETRY.attempts == 4

    with pytest.raises(ProviderRateLimitError):
        llm.generate("hello")

    assert provider.call_count == 4


def test_auth_error_fails_fast_without_retry(make_gateway):
    provider = FakeProvider(errors=[ProviderAuthError("401 invalid key")])
    llm = make_gateway(provider)

    with pytest.raises(ProviderAuthError):
        llm.generate("hello")

    assert provider.call_count == 1  # never retried


def test_validation_error_fails_fast_without_retry(make_gateway):
    provider = FakeProvider(
        errors=[ProviderBadRequestError("400 model not found")]
    )
    llm = make_gateway(provider)

    with pytest.raises(ProviderBadRequestError):
        llm.generate("hello")

    assert provider.call_count == 1  # never retried
