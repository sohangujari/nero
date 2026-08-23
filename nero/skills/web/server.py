"""Web skills: fetch_web_page.

HTML is stripped to text with the stdlib `html.parser` — no bs4, no new
dependency, per the design spec.
"""

from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx

from nero.security import envelope
from nero.skills.base import Skill, SkillMeta

TIMEOUT = httpx.Timeout(10.0, connect=5.0)
DEFAULT_MAX_CHARS = 20_000
_SKIP_TAGS = {"script", "style"}


class _TextExtractor(HTMLParser):
    """Collects visible text, dropping <script>/<style> contents."""

    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self._parts).split())


def strip_html(html_text: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html_text)
    return extractor.text()


class FetchWebPageSkill(Skill):
    meta = SkillMeta(
        name="fetch_web_page",
        description=(
            "Fetch a web page and return its text content, with HTML markup "
            "stripped out. Use this when the user asks you to look something up "
            "on a URL, read a web page, or summarize a link. Only http:// and "
            "https:// URLs are supported."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The http(s) URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "description": f"Maximum characters of text to return (default {DEFAULT_MAX_CHARS}).",
                },
            },
            "required": ["url"],
        },
        requires_network=True,
        permission_tier="read_only",
        ingests_external_content=True,
        offline_message=(
            "Fetching web pages needs an internet connection, and you're in "
            "offline mode right now."
        ),
    )

    async def _fetch(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def execute(self, **kwargs) -> str:
        url = str(kwargs.get("url") or "").strip()
        if not url:
            return "Error: no url provided."
        scheme = urlsplit(url).scheme.lower()
        if scheme not in ("http", "https"):
            return f"Error: only http:// and https:// URLs are supported, not {scheme or url!r}."
        max_chars = int(kwargs.get("max_chars") or DEFAULT_MAX_CHARS)
        try:
            body = await self._fetch(url)
        except httpx.HTTPError as exc:
            return f"I couldn't fetch {url}: {exc}"
        text = strip_html(body)
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        return envelope(f"url:{url}", text)


FETCH_WEB_PAGE = FetchWebPageSkill()
