"""Budget guard: call and token limits from settings.yaml."""

from __future__ import annotations

import pytest

from repopilot.llm import BudgetExceeded, FakeProvider


def test_call_budget_exceeded_blocks_next_api_call(make_gateway, make_settings):
    settings = make_settings(max_llm_calls=1)
    provider = FakeProvider(responses=["first", "second"])
    llm = make_gateway(provider, settings=settings)

    first = llm.generate("prompt one")
    assert first.text == "first"

    with pytest.raises(BudgetExceeded):
        llm.generate("prompt two")  # different prompt -> cache miss -> check()

    assert provider.call_count == 1  # blocked before any API call
    assert llm.budget.usage()["calls"] == 1


def test_cache_hits_do_not_consume_budget(make_gateway, make_settings):
    settings = make_settings(max_llm_calls=1)
    provider = FakeProvider(responses=["only call"])
    llm = make_gateway(provider, settings=settings)

    llm.generate("same prompt")  # uses the single budgeted call
    cached = llm.generate("same prompt")  # served free from disk cache

    assert cached.cache_hit is True
    assert llm.budget.usage()["cache_hits"] == 1
    assert provider.call_count == 1
    assert llm.budget.usage()["calls"] == 1


def test_output_token_budget_exceeded(make_gateway, make_settings):
    settings = make_settings(max_output_tokens=1)
    provider = FakeProvider(responses=["a response with several words"])
    llm = make_gateway(provider, settings=settings)

    llm.generate("prompt one")  # records more than the 1-token budget

    with pytest.raises(BudgetExceeded) as excinfo:
        llm.generate("prompt two")

    assert "output-token" in str(excinfo.value)


def test_input_token_budget_exceeded(make_gateway, make_settings):
    settings = make_settings(max_input_tokens=1)
    provider = FakeProvider(responses=["fine", "fine again"])
    llm = make_gateway(provider, settings=settings)

    llm.generate("a prompt long enough to exceed one token")

    with pytest.raises(BudgetExceeded) as excinfo:
        llm.generate("a different prompt long enough to exceed one token")

    assert "input-token" in str(excinfo.value)
