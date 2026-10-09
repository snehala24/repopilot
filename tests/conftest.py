"""Shared test fixtures — FakeProvider only, zero network, zero API keys."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from repopilot.config import (
    AppSettings,
    BudgetSettings,
    Environment,
    LLMTierSettings,
    LLMSettings,
    PathsSettings,
    Settings,
)
from repopilot.llm import FakeProvider, RetryConfig, get_llm

#: Tier settings for tests: a fake provider with a harmless model ID.
TEST_TIER = LLMTierSettings(
    provider="fake",
    model="fake-model-for-tests",
    temperature=0.0,
    max_output_tokens=64,
)

#: Backoff so tiny that retry tests finish in milliseconds.
FAST_RETRY = RetryConfig(attempts=4, initial_wait=0.01, max_wait=0.05, jitter=0.01)


@pytest.fixture
def make_settings(
    tmp_path,
) -> Callable[..., Settings]:
    """Build a Settings object isolated under this test's tmp_path."""

    def _make(
        *,
        max_llm_calls: int = 10,
        max_input_tokens: int = 100_000,
        max_output_tokens: int = 100_000,
    ) -> Settings:
        return Settings(
            app=AppSettings(),
            paths=PathsSettings(
                data_dir=tmp_path / "data",
                cache_dir=tmp_path / ".cache",
                llm_cache_dir=tmp_path / ".cache" / "llm",
            ),
            budgets=BudgetSettings(
                max_llm_calls=max_llm_calls,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
            ),
            llm=LLMSettings(tiers={"fast": TEST_TIER}),
            env=Environment(_env_file=None),
        )

    return _make


@pytest.fixture
def make_gateway(
    make_settings,
) -> Callable[..., object]:
    """Build a 'fast'-tier gateway around an injected FakeProvider."""

    def _make(
        provider: FakeProvider | None = None,
        *,
        settings: Settings | None = None,
        budget=None,
        retry: RetryConfig | None = None,
    ):
        return get_llm(
            "fast",
            settings if settings is not None else make_settings(),
            provider=provider if provider is not None else FakeProvider(),
            budget=budget,
            retry=retry if retry is not None else FAST_RETRY,
        )

    return _make
