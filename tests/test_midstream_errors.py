"""A mid-stream provider failure must keep its own identity.

LiteLLM wraps anything that fails after the stream opened -- a 429 included --
in MidStreamFallbackError, which subclasses ServiceUnavailableError. Both loops
treat that as "could not reach the model provider", so a Gemini free-tier quota
(5 requests/minute) told the user to check a working network connection, and
chat's key rotation never fired because it only matches RateLimitError.
"""

import asyncio

import litellm
import pytest

from nero.llm.client import LLMClient
from nero.config.schema import LLMConfig


def test_the_wrapper_really_does_masquerade_as_unreachable():
    """Pins the upstream fact this whole fix rests on."""
    assert issubclass(
        litellm.exceptions.MidStreamFallbackError, litellm.exceptions.ServiceUnavailableError
    )
    assert not issubclass(
        litellm.exceptions.MidStreamFallbackError, litellm.exceptions.RateLimitError
    )


def _client():
    return LLMClient(
        config=LLMConfig(provider="gemini", model="gemini-2.5-flash"),
        assistant_name="Nero",
        registry=None,
        api_key="k",
    )


def _wrapped(original):
    return litellm.exceptions.MidStreamFallbackError(
        message="wrapped", model="m", llm_provider="gemini", original_exception=original
    )


def _drain(client, monkeypatch, error):
    class FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

    async def fake_acompletion(**_kwargs):
        return FailingStream()

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    async def run():
        async for _ in client._litellm_chat([{"role": "user", "content": "hi"}], []):
            pass

    return run


def test_a_mid_stream_rate_limit_surfaces_as_a_rate_limit(monkeypatch):
    original = litellm.exceptions.RateLimitError(
        "429 quota exceeded", model="gemini-2.5-flash", llm_provider="gemini"
    )
    run = _drain(_client(), monkeypatch, _wrapped(original))
    with pytest.raises(litellm.exceptions.RateLimitError):
        asyncio.run(run())


def test_a_mid_stream_auth_failure_surfaces_as_auth(monkeypatch):
    """Not rate-limit-specific: the wrapper hid every mid-stream cause alike."""
    original = litellm.exceptions.AuthenticationError(
        "bad key", model="gemini-2.5-flash", llm_provider="gemini"
    )
    run = _drain(_client(), monkeypatch, _wrapped(original))
    with pytest.raises(litellm.exceptions.AuthenticationError):
        asyncio.run(run())


def test_a_wrapper_with_no_original_still_propagates(monkeypatch):
    run = _drain(_client(), monkeypatch, _wrapped(None))
    with pytest.raises(litellm.exceptions.MidStreamFallbackError):
        asyncio.run(run())


def test_an_ordinary_stream_error_is_untouched(monkeypatch):
    error = litellm.exceptions.ServiceUnavailableError(
        "503", model="gemini-2.5-flash", llm_provider="gemini"
    )
    run = _drain(_client(), monkeypatch, error)
    with pytest.raises(litellm.exceptions.ServiceUnavailableError):
        asyncio.run(run())
