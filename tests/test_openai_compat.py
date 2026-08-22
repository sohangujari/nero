"""v1.5.4: listing models from an arbitrary OpenAI-compatible server."""
import httpx
import pytest

from nero.llm import openai_compat


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


@pytest.fixture
def fake_http(monkeypatch):
    """Routes by exact URL; anything unrouted 404s, like a real server."""
    calls = []
    routes = {}

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        if url in routes:
            return routes[url]
        return FakeResponse({}, status=404)

    monkeypatch.setattr("httpx.get", fake_get)
    return routes, calls


def _models(*ids):
    return FakeResponse({"data": [{"id": name} for name in ids]})


class TestFetchModels:
    def test_returns_models_and_the_url_that_answered(self, fake_http):
        routes, _ = fake_http
        routes["http://localhost:1234/v1/models"] = _models("llama-3.1-8b", "qwen3-8b")
        answered, models = openai_compat.fetch_models("http://localhost:1234/v1")
        assert answered == "http://localhost:1234/v1"
        assert models == ["llama-3.1-8b", "qwen3-8b"]

    def test_falls_back_to_v1_and_reports_the_corrected_url(self, fake_http):
        """LM Studio reports its address without the /v1 the API lives at."""
        routes, _ = fake_http
        routes["http://localhost:1234/v1/models"] = _models("llama-3.1-8b")
        answered, models = openai_compat.fetch_models("http://localhost:1234")
        assert answered == "http://localhost:1234/v1"
        assert models == ["llama-3.1-8b"]

    def test_never_probes_v1_twice(self, fake_http):
        _, calls = fake_http
        answered, models = openai_compat.fetch_models("http://localhost:1234/v1")
        assert [url for url, _headers in calls] == ["http://localhost:1234/v1/models"]
        assert (answered, models) == ("http://localhost:1234/v1", [])

    def test_ids_are_sorted(self, fake_http):
        routes, _ = fake_http
        routes["http://x/v1/models"] = _models("zeta", "alpha", "mid")
        assert openai_compat.fetch_models("http://x/v1")[1] == ["alpha", "mid", "zeta"]

    def test_a_connection_error_degrades_to_empty(self, monkeypatch):
        def refuse(url, headers=None, timeout=None):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr("httpx.get", refuse)
        assert openai_compat.fetch_models("http://down:1234") == ("http://down:1234", [])

    def test_an_unrecognised_shape_degrades_to_empty(self, fake_http):
        routes, _ = fake_http
        routes["http://x/models"] = FakeResponse({"models": ["not-openai-shaped"]})
        routes["http://x/v1/models"] = FakeResponse({"data": [{"name": "no-id-key"}]})
        assert openai_compat.fetch_models("http://x") == ("http://x", [])

    def test_a_non_object_json_body_degrades_to_empty(self, fake_http):
        """A bare array or null is valid JSON but has no .get — must not raise."""
        routes, _ = fake_http
        routes["http://x/v1/models"] = FakeResponse(["not", "an", "object"])
        assert openai_compat.fetch_models("http://x") == ("http://x", [])

    def test_a_null_json_body_degrades_to_empty(self, fake_http):
        routes, _ = fake_http
        routes["http://x/v1/models"] = FakeResponse(None)
        assert openai_compat.fetch_models("http://x") == ("http://x", [])

    def test_the_key_is_sent_only_when_given(self, fake_http):
        routes, calls = fake_http
        routes["http://x/v1/models"] = _models("m")
        openai_compat.fetch_models("http://x/v1", api_key="sk-test")
        assert calls[-1][1] == {"Authorization": "Bearer sk-test"}
        calls.clear()
        openai_compat.fetch_models("http://x/v1")
        assert calls[-1][1] == {}


class TestFetchModelsAnthropic:
    def test_a_clean_base_passes_through(self, fake_http):
        routes, _ = fake_http
        routes["https://api.moonshot.ai/anthropic/v1/models"] = _models("kimi-k2.5")
        answered, models = openai_compat.fetch_models_anthropic(
            "https://api.moonshot.ai/anthropic"
        )
        assert answered == "https://api.moonshot.ai/anthropic"
        assert models == ["kimi-k2.5"]

    def test_a_v1_suffix_is_corrected_by_stripping(self, fake_http):
        """The inverse of the LM Studio gap: LiteLLM appends /v1/messages
        itself, so a pasted OpenAI-style …/v1 base would double the path."""
        routes, _ = fake_http
        routes["http://localhost:8080/v1/models"] = _models("m1")
        answered, models = openai_compat.fetch_models_anthropic("http://localhost:8080/v1")
        assert answered == "http://localhost:8080"
        assert models == ["m1"]

    def test_the_given_base_is_tried_first(self, fake_http):
        """A server genuinely living under …/v1 must not be 'corrected'."""
        routes, _ = fake_http
        routes["http://h/v1/v1/models"] = _models("real")
        routes["http://h/v1/models"] = _models("stripped")
        answered, got = openai_compat.fetch_models_anthropic("http://h/v1")
        assert (answered, got) == ("http://h/v1", ["real"])

    def test_sends_the_anthropic_headers(self, fake_http):
        routes, calls = fake_http
        routes["https://api.moonshot.ai/anthropic/v1/models"] = _models("kimi-k2.5")
        openai_compat.fetch_models_anthropic(
            "https://api.moonshot.ai/anthropic", api_key="sk-test"
        )
        _url, headers = calls[0]
        assert headers["anthropic-version"] == openai_compat.ANTHROPIC_VERSION
        assert headers["x-api-key"] == "sk-test"
        assert "Authorization" not in headers

    def test_no_key_sends_no_auth_header(self, fake_http):
        routes, calls = fake_http
        routes["http://h/v1/models"] = _models("m")
        openai_compat.fetch_models_anthropic("http://h")
        _url, headers = calls[0]
        assert "x-api-key" not in headers

    def test_a_dead_server_returns_the_base_and_no_models(self, fake_http):
        assert openai_compat.fetch_models_anthropic("http://h") == ("http://h", [])

    def test_a_non_object_body_degrades_to_no_models(self, fake_http):
        """The d21c6a6 regression class: a body of [1,2,3] has no .get."""
        routes, _ = fake_http
        routes["http://h/v1/models"] = FakeResponse([1, 2, 3])
        assert openai_compat.fetch_models_anthropic("http://h") == ("http://h", [])
