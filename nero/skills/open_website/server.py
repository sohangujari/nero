"""open_website: a site, in the browser the user actually meant.

`webbrowser.open` always uses the OS default, so "open YouTube in Chrome" on a
Mac whose default is Safari opened Safari. The browser has to be chosen
explicitly, and — like the music player — the choice is worth remembering
rather than asking for every single time.
"""

import platform
import shutil
import subprocess
import webbrowser
from collections.abc import Callable

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

# What people say -> what each platform needs. macOS wants the bundle name,
# Linux the executable. Aliases matter: nobody says "Brave Browser".
BROWSERS = {
    "safari": ("Safari", "safari"),
    "chrome": ("Google Chrome", "google-chrome"),
    "google chrome": ("Google Chrome", "google-chrome"),
    "chromium": ("Chromium", "chromium"),
    "firefox": ("Firefox", "firefox"),
    "mozilla firefox": ("Firefox", "firefox"),
    "edge": ("Microsoft Edge", "microsoft-edge"),
    "microsoft edge": ("Microsoft Edge", "microsoft-edge"),
    "brave": ("Brave Browser", "brave-browser"),
    "brave browser": ("Brave Browser", "brave-browser"),
    "arc": ("Arc", "arc"),
    "opera": ("Opera", "opera"),
    "vivaldi": ("Vivaldi", "vivaldi"),
    "zen": ("Zen Browser", "zen-browser"),
    "tor": ("Tor Browser", "tor-browser"),
}


def resolve_browser(name: str) -> tuple[str, str] | None:
    """(macOS app name, Linux executable) for a spoken browser name, or None."""
    wanted = (name or "").strip().lower().removesuffix(" browser").strip()
    if not wanted:
        return None
    if wanted in BROWSERS:
        return BROWSERS[wanted]
    for alias, pair in BROWSERS.items():
        if wanted in alias or alias in wanted:
            return pair
    return None


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
            "either a well-known site name or a full web address, and optionally "
            "which browser to use. Returns a confirmation, or asks for "
            "clarification if the site name is ambiguous."
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
                },
                "browser": {
                    "type": "string",
                    "description": (
                        "Which browser to open it in, e.g. 'Chrome' or 'Firefox'. "
                        "Pass this whenever the user names one ('open YouTube in "
                        "Chrome'); otherwise omit it and their usual browser is used."
                    ),
                },
            },
            "required": ["site"],
        },
        requires_network=True,
        permission_tier="state_changing",
        category="Apps",
        offline_message=(
            "Opening a website needs an internet connection, and you're in "
            "offline mode right now."
        ),
    )

    def __init__(
        self,
        preferred_browser: str | None = None,
        on_browser_chosen: Callable[[str], None] | None = None,
    ):
        self._preferred_browser = preferred_browser
        # Injected rather than importing ConfigManager, so the skill stays a
        # pure unit with no knowledge of how Nero persists anything.
        self._on_browser_chosen = on_browser_chosen

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
        requested = str(kwargs.get("browser") or "").strip()
        wanted = requested or (self._preferred_browser or "")
        try:
            if wanted:
                opened, detail = self._open_in(wanted, url)
                if opened:
                    # Only an explicit choice becomes the preference, and only
                    # once it worked — the same rule play_music follows.
                    if requested and self._on_browser_chosen is not None:
                        self._on_browser_chosen(requested)
                    return detail
                if requested:
                    # Asked for a browser they don't have: say so rather than
                    # silently opening a different one.
                    return detail
                # A stale preference must not strand every request. Fall through
                # to the default browser instead.
            opened = webbrowser.open(url)
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error opening {url}: {exc}"
        if not opened:
            return f"I couldn't open a browser for {url}."
        return f"Opened {url}."

    def _open_in(self, browser: str, url: str) -> tuple[bool, str]:
        """(opened, message). False means fall back or report, never crash."""
        pair = resolve_browser(browser)
        if pair is None:
            return False, (
                f"I don't recognise a browser called {browser!r}. "
                f"Known: {', '.join(sorted({app for app, _ in BROWSERS.values()}))}."
            )
        app_name, executable = pair
        system = platform.system()
        try:
            if system == "Darwin":
                result = subprocess.run(
                    ["open", "-a", app_name, url], capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    return True, f"Opened {url} in {app_name}."
                return False, f"I couldn't open {app_name} — it may not be installed."
            path = shutil.which(executable) or shutil.which(app_name.lower().replace(" ", "-"))
            if system == "Windows":
                result = subprocess.run(
                    ["cmd", "/c", "start", "", executable, url],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    return True, f"Opened {url} in {app_name}."
                return False, f"I couldn't open {app_name} — it may not be installed."
            if path is None:
                return False, f"{app_name} doesn't appear to be installed."
            subprocess.Popen(
                [path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"Opened {url} in {app_name}."
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"I couldn't open {app_name}: {exc}"


SKILL = OpenWebsiteSkill()
