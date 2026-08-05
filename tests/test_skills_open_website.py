import asyncio

import pytest

from nero.skills.open_website.server import OpenWebsiteSkill, resolve


@pytest.fixture
def skill():
    return OpenWebsiteSkill()


def run(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


@pytest.fixture
def opened(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nero.skills.open_website.server.webbrowser.open",
        lambda url: calls.append(url) or True,
    )
    return calls


class TestResolve:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("youtube", "https://www.youtube.com"),
            ("YouTube", "https://www.youtube.com"),
            ("  github  ", "https://github.com"),
            ("open youtube", "https://www.youtube.com"),
            ("go to reddit", "https://www.reddit.com"),
        ],
    )
    def test_known_site_names(self, query, expected):
        assert resolve(query) == expected

    def test_full_url_passes_through(self):
        assert resolve("https://example.com/page") == "https://example.com/page"

    def test_http_url_passes_through(self):
        assert resolve("http://example.com") == "http://example.com"

    def test_bare_domain_gets_https(self):
        assert resolve("example.com") == "https://example.com"

    def test_domain_with_path_gets_https(self):
        assert resolve("example.com/docs") == "https://example.com/docs"

    def test_case_sensitive_path_is_preserved(self):
        # youtu.be video IDs are case-sensitive; lowercasing breaks the link.
        assert resolve("youtu.be/dQw4w9WgXcQ") == "https://youtu.be/dQw4w9WgXcQ"

    def test_mixed_case_domain_and_path_preserved(self):
        assert resolve("GitHub.com/Foo/Bar") == "https://GitHub.com/Foo/Bar"

    def test_known_site_lookup_is_still_case_insensitive(self):
        # The SITES table lookup itself stays case-insensitive; only the
        # pass-through domain path preserves original case.
        assert resolve("YouTube") == "https://www.youtube.com"

    @pytest.mark.parametrize("query", ["", "   ", "that site with the thing", "my bank"])
    def test_ambiguous_returns_none(self, query):
        assert resolve(query) is None


class TestMeta:
    def test_metadata(self, skill):
        assert skill.meta.name == "open_website"
        assert skill.meta.requires_network is True
        assert skill.meta.permission_tier == "state_changing"
        assert skill.meta.input_schema["required"] == ["site"]


class TestExecute:
    def test_opens_known_site(self, skill, opened):
        result = run(skill, site="youtube")
        assert opened == ["https://www.youtube.com"]
        assert "youtube.com" in result

    def test_opens_url(self, skill, opened):
        run(skill, site="https://example.com")
        assert opened == ["https://example.com"]

    def test_missing_site_returns_error(self, skill, opened):
        assert "Error" in run(skill)
        assert opened == []

    def test_ambiguous_asks_for_clarification(self, skill, opened):
        result = run(skill, site="that site with the thing")
        assert opened == []
        assert "not sure" in result.lower()
        assert "ask the user" in result.lower()

    def test_browser_failure_is_reported(self, skill, monkeypatch):
        monkeypatch.setattr(
            "nero.skills.open_website.server.webbrowser.open", lambda url: False
        )
        assert "couldn't open" in run(skill, site="youtube").lower()

    def test_browser_exception_is_caught(self, skill, monkeypatch):
        def boom(url):
            raise OSError("no display")

        monkeypatch.setattr("nero.skills.open_website.server.webbrowser.open", boom)
        result = run(skill, site="youtube")
        assert "Error" in result and "no display" in result
