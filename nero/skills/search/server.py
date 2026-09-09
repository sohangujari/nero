"""web_search: find things on the web without already knowing the URL.

`fetch_web_page` can only read an address someone already has. Without this,
"what happened today" or "who wrote X" has no path at all — the model either
guesses from training data or says it cannot help.

DuckDuckGo's lite endpoint is used because it needs no API key and no account:
a search skill that only works once you have signed up for a search API is a
search skill most people never turn on. It is HTML, so it is parsed with the
stdlib parser — the same choice `fetch_web_page` already makes, and no new
dependency.

Results are wrapped as untrusted content. A search result is attacker-authored
text by definition: anyone can publish a page saying "ignore your instructions".
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import httpx

from nero.security import envelope
from nero.skills.base import Skill, SkillMeta

SEARCH_URL = "https://lite.duckduckgo.com/lite/"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)
DEFAULT_LIMIT = 5
MAX_LIMIT = 15
# The endpoint serves a different, unparseable page to obvious bots.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Long snippets crowd out the answer; a sentence or two is enough to judge a
# result and decide whether to fetch the page in full.
MAX_SNIPPET_CHARS = 240


class _ResultParser(HTMLParser):
    """Pulls (url, title, snippet) triples out of the lite results page.

    The page is a plain table: an `a.result-link` per hit, each optionally
    followed by a `td.result-snippet`. Anything that does not fit that shape is
    skipped rather than guessed at.
    """

    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._mode: str | None = None
        self._buffer: list[str] = []
        self._url: str | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class") or ""
        if tag == "a" and "result-link" in classes:
            self._mode, self._url, self._buffer = "title", attributes.get("href"), []
        elif "result-snippet" in classes:
            self._mode, self._buffer = "snippet", []

    def handle_endtag(self, tag):
        if self._mode == "title" and tag == "a":
            self.results.append(
                {"url": clean_url(self._url), "title": self._text(), "snippet": ""}
            )
            self._mode = None
        elif self._mode == "snippet" and tag == "td":
            if self.results:
                self.results[-1]["snippet"] = self._text()[:MAX_SNIPPET_CHARS]
            self._mode = None

    def handle_data(self, data):
        if self._mode:
            self._buffer.append(data)

    def _text(self) -> str:
        return " ".join("".join(self._buffer).split())


def clean_url(href: str | None) -> str:
    """The real destination, unwrapped from DuckDuckGo's redirector.

    Hits sometimes arrive as `//duckduckgo.com/l/?uddg=<real url>`. Handing the
    model the redirector would make every follow-up `fetch_web_page` a wasted
    round trip.
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlsplit(href)
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


def parse_results(html: str, limit: int) -> list[dict]:
    parser = _ResultParser()
    parser.feed(html)
    return [r for r in parser.results if r["url"]][:limit]


def render(query: str, results: list[dict]) -> str:
    lines = []
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}\n   {result['url']}")
        if result["snippet"]:
            lines.append(f"   {result['snippet']}")
    return envelope(f"search:{query}", "\n".join(lines))


class WebSearchSkill(Skill):
    meta = SkillMeta(
        name="web_search",
        description=(
            "Search the web and get back a list of results with titles, URLs and "
            "short snippets. Use this whenever you need current information, or "
            "when the user asks about something you do not know or cannot be sure "
            "is still true — news, prices, releases, people, events. Follow up "
            "with fetch_web_page on a result URL when you need the full text. "
            "Search operators work inside the query: 'site:x.com' to search one "
            "site, quotes for an exact phrase."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, as you would type it into a search box.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"How many results to return (default {DEFAULT_LIMIT}, max {MAX_LIMIT}).",
                },
            },
            "required": ["query"],
        },
        requires_network=True,
        permission_tier="read_only",
        category="Web",
        # Search results are written by whoever published the page. Anyone can
        # publish one saying "ignore your previous instructions".
        ingests_external_content=True,
        offline_message=(
            "Searching the web needs an internet connection, and you're in "
            "offline mode right now."
        ),
    )

    async def _fetch(self, query: str) -> str:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            response = await client.post(
                SEARCH_URL, data={"q": query}, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            return response.text

    async def execute(self, **kwargs) -> str:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return "Error: no search query provided."
        requested = kwargs.get("limit")
        try:
            # Not `or DEFAULT_LIMIT`: that quietly turns an explicit 0 into 5.
            # Absent means default; a number out of range gets clamped into it.
            limit = DEFAULT_LIMIT if requested is None else int(requested)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        limit = max(1, min(limit, MAX_LIMIT))
        try:
            html = await self._fetch(query)
        except httpx.HTTPError:
            # Never leak a raw transport error to the user.
            return "I couldn't reach the search engine just now. Try again in a moment."
        results = parse_results(html, limit)
        if not results:
            return (
                f"No results came back for {query!r}. Either nothing matched, or the "
                "search engine is rate-limiting this machine — try again shortly, or "
                "rephrase."
            )
        return render(query, results)


SKILL = WebSearchSkill()
