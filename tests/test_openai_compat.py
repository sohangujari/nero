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

    def test_the_key_is_sent_only_when_given(self, fake_http):
        routes, calls = fake_http
        routes["http://x/v1/models"] = _models("m")
        openai_compat.fetch_models("http://x/v1", api_key="sk-test")
        assert calls[-1][1] == {"Authorization": "Bearer sk-test"}
        calls.clear()
        openai_compat.fetch_models("http://x/v1")
        assert calls[-1][1] == {}
