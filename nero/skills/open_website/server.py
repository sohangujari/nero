import webbrowser

from nero.skills.base import Skill, SkillMeta

# Only unambiguous names belong here. Anything a reasonable person could mean
# two ways is better answered with a clarifying question than a wrong guess.
SITES = {
    "amazon": "https://www.amazon.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "drive": "https://drive.google.com",
    "github": "https://github.com",
    "gmail": "https://mail.google.com",
    "google": "https://www.google.com",
    "google drive": "https://drive.google.com",
    "google maps": "https://maps.google.com",
    "linkedin": "https://www.linkedin.com",
    "maps": "https://maps.google.com",
    "netflix": "https://www.netflix.com",
    "reddit": "https://www.reddit.com",
    "spotify": "https://open.spotify.com",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "twitter": "https://twitter.com",
    "wikipedia": "https://www.wikipedia.org",
    "x": "https://x.com",
    "youtube": "https://www.youtube.com",
}

# Models often echo the user's phrasing rather than extracting the site name.
_PREFIXES = ("open ", "go to ", "visit ", "launch ", "browse to ", "navigate to ")


def resolve(query: str) -> str | None:
    """A URL for `query`, or None when it's too ambiguous to guess.

    Returning None is a feature: the skill then asks for clarification rather
    than opening a wrong site, which is unrecoverable once the browser launches.
    """
    text = (query or "").strip()
    if text.lower().startswith(("http://", "https://")):
        return text
    # Lowercase only for the SITES lookup and prefix-stripping; `original` keeps
    # the caller's exact case for building the pass-through URL below. Domain
    # case is irrelevant to DNS, but path case is significant (e.g. youtu.be
    # video IDs), so the whole original string is preserved, not just the host.
    original = text.rstrip("/")
    lowered = original.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            original = original[len(prefix) :].strip()
            lowered = lowered[len(prefix) :].strip()
    if not lowered:
        return None
    if lowered in SITES:
        return SITES[lowered]
    # A dot and no spaces is domain-shaped; anything else is a guess.
    if "." in lowered and " " not in lowered:
        return f"https://{original}"
    return None


class OpenWebsiteSkill(Skill):
    meta = SkillMeta(
        name="open_website",
        description=(
            "Open a website in the user's default browser. Use this when the user "
            "asks to open, visit, or go to a website (e.g. 'open YouTube'). Accepts "
            "either a well-known site name or a full web address. Returns a "
            "confirmation, or asks for clarification if the site name is ambiguous."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "site": {
                    "type": "string",
                    "description": (
                        "A website name or address, e.g. 'YouTube' or "
                        "'https://example.com'."
                    ),
                }
            },
            "required": ["site"],
        },
        requires_network=True,
        permission_tier="state_changing",
        offline_message=(
            "Opening a website needs an internet connection, and you're in "
            "offline mode right now."
        ),
    )

    async def execute(self, **kwargs) -> str:
        query = str(kwargs.get("site") or "").strip()
        if not query:
            return "Error: no website was given."
        url = resolve(query)
        if url is None:
            return (
                f"I'm not sure which site {query!r} means. Ask the user for the full "
                "web address, then call this skill again with it."
            )
        try:
            opened = webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error opening {url}: {exc}"
        if not opened:
            return f"I couldn't open a browser for {url}."
        return f"Opened {url}."


SKILL = OpenWebsiteSkill()
