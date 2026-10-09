"""Structured output: Pydantic validation with retry on bad JSON."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from repopilot.llm import FakeProvider, StructuredOutputError


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


GOOD = '{"answer": "yes", "confidence": 0.8}'


def test_parses_valid_json_on_first_try(make_gateway):
    provider = FakeProvider(
        responses=['{"answer": "monorepos share tooling", "confidence": 0.95}']
    )
    llm = make_gateway(provider)

    result = llm.generate_structured("What is a monorepo?", Answer)

    assert isinstance(result.data, Answer)
    assert result.data.confidence == 0.95
    assert provider.call_count == 1
    assert result.cache_hit is False
    # The provider was asked for JSON matching the schema.
    assert provider.requests[0].response_schema is not None


def test_retries_with_validation_error_on_bad_json(make_gateway):
    provider = FakeProvider(responses=['{"unexpected": "shape"}', GOOD])
    llm = make_gateway(provider)

    result = llm.generate_structured("Is caching useful?", Answer)

    assert result.data.answer == "yes"
    assert provider.call_count == 2

    retry_prompt = provider.requests[1].prompt
    assert "failed validation" in retry_prompt
    assert "Field required" in retry_prompt  # the actual pydantic error


def test_retries_on_non_json_text(make_gateway):
    provider = FakeProvider(responses=["definitely not json", GOOD])
    llm = make_gateway(provider)

    result = llm.generate_structured("Is caching useful?", Answer)

    assert result.data.answer == "yes"
    assert provider.call_count == 2
    assert "failed validation" in provider.requests[1].prompt


def test_raises_after_exhausting_two_retries(make_gateway):
    provider = FakeProvider(responses=['{"bad": 1}'] * 3)
    llm = make_gateway(provider)

    with pytest.raises(StructuredOutputError):
        llm.generate_structured("Is caching useful?", Answer)

    assert provider.call_count == 3  # 1 initial attempt + 2 retries


def test_structured_response_tracks_usage(make_gateway):
    provider = FakeProvider(responses=[GOOD])
    llm = make_gateway(provider)

    result = llm.generate_structured("Is caching useful?", Answer)

    assert result.latency_ms >= 0
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.model == provider.requests[0].model
    assert result.tier == "fast"
