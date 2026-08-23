import asyncio

import httpx

from nero.skills.web.server import FetchWebPageSkill, strip_html


def run(coro):
    return asyncio.run(coro)


def make_skill(html_or_exception):
    skill = FetchWebPageSkill()

    async def fake_fetch(url):
        if isinstance(html_or_exception, Exception):
            raise html_or_exception
        return html_or_exception

    skill._fetch = fake_fetch
    return skill


class TestMeta:
    def test_metadata(self):
        meta = FetchWebPageSkill.meta
        assert meta.name == "fetch_web_page"
        assert meta.requires_network is True
        assert meta.permission_tier == "read_only"
        assert meta.ingests_external_content is True
        assert "internet connection" in meta.offline_message
        assert meta.input_schema["required"] == ["url"]


class TestStripHtml:
    def test_strips_tags(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_drops_script_and_style_contents(self):
        html = "<style>.a{color:red}</style><p>Text</p><script>alert(1)</script>"
        assert strip_html(html) == "Text"


class TestExecute:
    def test_happy_path_wraps_in_envelope(self):
        skill = make_skill("<html><body><p>Hello world</p></body></html>")
        result = run(skill.execute(url="https://example.com"))
        assert "<untrusted_content" in result
        assert "Hello world" in result
        assert 'source="url:https://example.com"' in result

    def test_refuses_non_http_scheme(self):
        skill = FetchWebPageSkill()
        result = run(skill.execute(url="file:///etc/passwd"))
        assert "Error" in result

    def test_refuses_missing_url(self):
        skill = FetchWebPageSkill()
        result = run(skill.execute(url=""))
        assert "Error" in result

    def test_http_error_reported_gracefully(self):
        skill = make_skill(httpx.ConnectTimeout("timed out"))
        result = run(skill.execute(url="https://example.com"))
        assert "couldn't fetch" in result

    def test_truncates_to_max_chars(self):
        skill = make_skill("<p>" + ("x" * 100) + "</p>")
        result = run(skill.execute(url="https://example.com", max_chars=10))
        assert "truncated" in result
