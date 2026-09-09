"""web_search (nero/skills/search/server.py).

The skill's job is to turn a query into results the model can act on. What
matters here is that it never hands back a redirector URL, never lets a search
result look like an instruction, and degrades honestly when the engine is
unhappy rather than pretending there were no matches.
"""

import asyncio

import httpx
import pytest

from nero.skills.search.server import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_SNIPPET_CHARS,
    WebSearchSkill,
    clean_url,
    parse_results,
    render,
)

PAGE = """
<table border="0">
  <tr><td><a rel="nofollow" href="https://realpython.com/async-io-python/"
      class='result-link'>Python&#x27;s asyncio: A Walkthrough</a></td></tr>
  <tr><td class='result-snippet'>Explore how <b>asyncio</b> works
      and when to use it.</td></tr>
  <tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=x"
      class='result-link'>asyncio documentation</a></td></tr>
  <tr><td class='result-snippet'>The library reference.</td></tr>
</table>
"""


def run(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


def skill_returning(page, calls=None):
    """A skill whose fetch is scripted — no network in the test suite."""

    class Fake(WebSearchSkill):
        async def _fetch(self, query):
            if calls is not None:
                calls.append(query)
            if isinstance(page, Exception):
                raise page
            return page

    return Fake()


class TestMeta:
    def test_metadata(self):
        meta = WebSearchSkill().meta
        assert meta.name == "web_search"
        assert meta.requires_network is True
        assert meta.permission_tier == "read_only"
        assert meta.category == "Web"
        assert meta.offline_message

    def test_results_are_treated_as_untrusted(self):
        """A search result is attacker-authored text by definition: anyone can
        publish a page that says 'ignore your previous instructions'."""
        assert WebSearchSkill().meta.ingests_external_content is True


class TestParsing:
    def test_titles_urls_and_snippets_are_extracted(self):
        results = parse_results(PAGE, 10)
        assert len(results) == 2
        assert results[0]["title"] == "Python's asyncio: A Walkthrough"
        assert results[0]["url"] == "https://realpython.com/async-io-python/"
        assert results[0]["snippet"] == "Explore how asyncio works and when to use it."

    def test_a_redirector_url_is_unwrapped(self):
        """Handing the model /l/?uddg=... makes every follow-up fetch_web_page
        a wasted round trip."""
        assert parse_results(PAGE, 10)[1]["url"] == "https://docs.python.org/3/"

    @pytest.mark.parametrize(
        "href,expected",
        [
            ("https://example.com/a", "https://example.com/a"),
            ("//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.dev%2F", "https://x.dev/"),
            ("//example.com/direct", "https://example.com/direct"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_url_cleaning(self, href, expected):
        assert clean_url(href) == expected

    def test_the_limit_is_honoured(self):
        assert len(parse_results(PAGE, 1)) == 1

    def test_a_page_with_no_results_parses_to_nothing(self):
        assert parse_results("<html><body>nope</body></html>", 5) == []

    def test_a_result_with_no_snippet_still_counts(self):
        page = "<a href='https://x.dev' class='result-link'>X</a>"
        assert parse_results(page, 5) == [{"url": "https://x.dev", "title": "X", "snippet": ""}]

    def test_a_long_snippet_is_trimmed(self):
        page = (
            "<a href='https://x.dev' class='result-link'>X</a>"
            f"<td class='result-snippet'>{'word ' * 200}</td>"
        )
        assert len(parse_results(page, 5)[0]["snippet"]) <= MAX_SNIPPET_CHARS


class TestRendering:
    def test_results_are_numbered_with_url_and_snippet(self):
        text = render("asyncio", parse_results(PAGE, 10))
        assert "1. Python's asyncio: A Walkthrough" in text
        assert "https://realpython.com/async-io-python/" in text
        assert "2. asyncio documentation" in text

    def test_output_is_wrapped_as_untrusted_content(self):
        text = render("asyncio", parse_results(PAGE, 10))
        assert "search:asyncio" in text


class TestExecute:
    def test_a_search_returns_rendered_results(self):
        assert "realpython.com" in run(skill_returning(PAGE), query="asyncio")

    def test_an_empty_query_is_refused_without_a_request(self):
        calls = []
        assert "Error" in run(skill_returning(PAGE, calls), query="   ")
        assert calls == []

    def test_no_matches_says_so_and_names_rate_limiting(self):
        """An engine that is throttling this machine looks exactly like a query
        with no matches. Saying only 'no results' would send the model off to
        answer from memory instead."""
        result = run(skill_returning("<html></html>"), query="asyncio")
        assert "No results" in result and "rate-limit" in result

    def test_a_transport_failure_never_leaks_the_exception(self):
        result = run(skill_returning(httpx.ConnectError("no route to host")), query="asyncio")
        assert "no route" not in result
        assert "couldn't reach" in result

    @pytest.mark.parametrize(
        "given,expected", [(None, DEFAULT_LIMIT), (3, 3), (0, 1), (-5, 1), (999, MAX_LIMIT),
                           ("nonsense", DEFAULT_LIMIT)]
    )
    def test_the_limit_is_clamped_to_something_sane(self, given, expected):
        page = "".join(
            f"<a href='https://x.dev/{i}' class='result-link'>R{i}</a>" for i in range(50)
        )
        result = run(skill_returning(page), query="x", limit=given)
        assert result.count("https://x.dev/") == expected
