"""Disk cache behaviour: hits, misses, and cross-client persistence."""

from __future__ import annotations

from repopilot.llm import FakeProvider


def test_cache_miss_then_hit_avoids_second_api_call(make_gateway):
    provider = FakeProvider(responses=["cached answer"])
    llm = make_gateway(provider)

    first = llm.generate("what is a monorepo?")
    second = llm.generate("what is a monorepo?")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == "cached answer"
    # The hit cost zero extra API calls and replayed the usage metrics.
    assert provider.call_count == 1
    assert second.input_tokens == first.input_tokens
    assert second.output_tokens == first.output_tokens


def test_cache_persists_on_disk_across_clients(make_gateway, tmp_path):
    llm1 = make_gateway(FakeProvider(responses=["persisted"]))
    llm1.generate("same prompt")

    # A brand-new gateway + provider instance: only a disk hit can serve this.
    provider2 = FakeProvider()  # no canned responses at all
    llm2 = make_gateway(provider2)
    second = llm2.generate("same prompt")

    assert second.cache_hit is True
    assert second.text == "persisted"
    assert provider2.call_count == 0

    cache_files = list((tmp_path / ".cache" / "llm").glob("*.json"))
    assert len(cache_files) == 1


def test_different_prompts_are_different_cache_keys(make_gateway):
    provider = FakeProvider(responses=["answer one", "answer two"])
    llm = make_gateway(provider)

    first = llm.generate("prompt one")
    second = llm.generate("prompt two")

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert provider.call_count == 2
