"""play_music: control whichever music player the user actually has.

Three things this has to get right, and the old version got none of them:

- **Know what is installed.** Probing "is Spotify running?" on a Mac that has
  never had Spotify costs a subprocess to learn nothing. Looking for the app
  bundle is a stat() and answers the more useful question.
- **Ask once when the answer is genuinely ambiguous.** With Spotify *and* Music
  installed and neither playing, picking one silently is a coin flip. The skill
  asks, the answer comes back as `app`, and it is remembered — so it asks once
  ever, not once per request.
- **"Play" should start the music.** Refusing with "nothing is running, start it
  first" is the one answer nobody wants from an assistant.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from nero.skills.base import Skill, SkillMeta

logger = logging.getLogger("nero.skills.music")

ACTIONS = ("play", "pause", "next", "previous")

# Apple's catalogue search. No key, no account, no rate-limit registration —
# which is what makes "play God's Plan by Drake" work out of the box rather
# than behind a setup step.
ITUNES_SEARCH = "https://itunes.apple.com/search"
SEARCH_TIMEOUT = 8.0

# Spotify has no keyless search — the anonymous web-player token endpoint
# answers 403 — so starting a *named* song there needs an app registration.
# Play/pause/skip work without any of this.
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
KEYRING_CLIENT_ID = "spotify_client_id"
KEYRING_CLIENT_SECRET = "spotify_client_secret"
SPOTIFY_SETUP_HINT = (
    "Starting a specific song in Spotify needs API credentials, which are not "
    "set up. Tell the user to run `nero config spotify` (it takes a minute at "
    "developer.spotify.com), or offer to play it in another player instead."
)

Runner = Callable[..., subprocess.CompletedProcess]

# Long enough for a cold app launch on `play`, short enough that a wedged
# player cannot hold a turn open.
TIMEOUT_SECONDS = 10
# Starting a named song does more: it launches the app if it is closed, waits
# for a catalogue page to load, and waits for playback to actually begin. All
# of that is deliberate work the user asked for, so it gets a longer leash.
TRACK_TIMEOUT_SECONDS = 30
# How long to wait for a catalogue link to start playing before nudging it.
_PLAY_WAIT_TICKS = 20  # x 0.25s = 5s


def _default_runner(cmd: list[str], timeout: float = TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    # Argument list, never a shell string — same injection-safe pattern as open_app.
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@dataclass(frozen=True)
class Outcome:
    """Whether the player accepted the command, and what to tell the model.

    Separate flag rather than sniffing the message for the word "error": the
    preference is only learned from a choice that actually worked, and that
    decision should not rest on string matching.
    """

    ok: bool
    message: str


# Most macOS players speak iTunes' vocabulary; VLC is the common exception.
# MappingProxyType so a shared default cannot be edited through one player.
ITUNES_VERBS: Mapping[str, str] = MappingProxyType(
    {"play": "play", "pause": "pause", "next": "next track", "previous": "previous track"}
)
VLC_VERBS: Mapping[str, str] = MappingProxyType(
    {"play": "play", "pause": "pause", "next": "next", "previous": "previous"}
)


@dataclass(frozen=True)
class Player:
    name: str
    verbs: Mapping[str, str] = ITUNES_VERBS


# Ordered by how likely a given Mac has them. The name is both what the user
# says and what AppleScript needs, which is why there is no separate id.
MAC_PLAYERS = (
    Player("Music"),          # Apple Music, /System/Applications
    Player("Spotify"),
    Player("TIDAL"),
    Player("iTunes"),         # pre-Catalina
    Player("Swinsian"),
    Player("Doppler"),
    Player("VLC", VLC_VERBS),
)

APP_DIRS = (
    Path("/Applications"),
    Path("/System/Applications"),
    Path.home() / "Applications",
)


def _with_timeout(runner: Runner) -> Runner:
    """Let a runner be called with or without a timeout.

    Test doubles take just the command; production takes a timeout too, because
    starting a named song is allowed to take far longer than pressing pause.
    """

    def call(cmd, timeout=TIMEOUT_SECONDS):
        try:
            return runner(cmd, timeout=timeout)
        except TypeError:
            return runner(cmd)

    return call


def applescript_string(text: str) -> str:
    """`text` as an AppleScript string literal body.

    A song title can contain quotes ("Ain\'t It Fun", 12" mixes) and the query
    is interpolated into a script, so this is the same discipline as quoting
    SQL — not cosmetic.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"')


class MusicController(ABC):
    """Per-platform media control.

    pynput cannot report whether anything received a synthesised media key, so
    the two platforms that *can* answer "what is installed?" use their native
    interface instead.
    """

    @abstractmethod
    def available(self) -> list[str]:
        """Players this machine can actually drive, best guess first."""

    @abstractmethod
    def running(self) -> list[str]:
        """The subset of `available` currently running."""

    @abstractmethod
    def control(self, action: str, player: str) -> Outcome:
        """Perform `action` in `player`."""

    def play_local(self, query: str, player: str) -> Outcome:
        """Play `query` from what the player already has locally.

        Tried before anything on the network: a track in the library starts
        instantly and works on a plane. Not every player can be asked this, so
        the default is a clean miss rather than an error.
        """
        return Outcome(False, "")

    def open_track(self, url: str, player: str) -> Outcome:
        """Hand `url` (a catalogue link or a player URI) to `player`."""
        return Outcome(False, f"{player} cannot open a link.")


class MacOSController(MusicController):
    def __init__(self, runner: Runner | None = None, app_dirs=APP_DIRS, players=MAC_PLAYERS):
        self._run = _with_timeout(runner or _default_runner)
        self._app_dirs = tuple(app_dirs)
        self._players = tuple(players)

    def _player(self, name: str) -> Player | None:
        return next((p for p in self._players if p.name.lower() == name.lower()), None)

    def available(self) -> list[str]:
        """Installed players, found by app bundle rather than by asking each one.

        A filesystem check costs microseconds; an `osascript` probe costs ~35 ms
        each and, for an app that was never installed, spends it finding out
        nothing.
        """
        return [
            player.name
            for player in self._players
            if any((directory / f"{player.name}.app").exists() for directory in self._app_dirs)
        ]

    def running(self) -> list[str]:
        """One osascript call for every installed player, not one each."""
        installed = self.available()
        if not installed:
            return []
        script = "{" + ", ".join(f'application "{n}" is running' for n in installed) + "}"
        result = self._run(["osascript", "-e", script])
        if result.returncode != 0:
            return []
        flags = [part.strip() == "true" for part in (result.stdout or "").split(",")]
        return [name for name, on in zip(installed, flags, strict=False) if on]

    def control(self, action: str, player: str) -> Outcome:
        known = self._player(player)
        if known is None:
            return Outcome(False, f"I don't know how to control {player}.")
        verb = known.verbs[action]
        if action == "play":
            # `tell ... to play` launches the app if it is not running, which is
            # exactly what "play some music" means. No pre-check, so this is one
            # subprocess rather than two.
            script = f'tell application "{player}" to {verb}'
        else:
            # Pausing or skipping something that isn't running is a no-op worth
            # reporting rather than a launch. Still one call: the check and the
            # command travel together.
            script = (
                f'if application "{player}" is running then\n'
                f'  tell application "{player}" to {verb}\n'
                '  return "done"\n'
                "end if\n"
                'return "idle"'
            )
        result = self._run(["osascript", "-e", script])
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "unknown error"
            return Outcome(False, f"Could not {action} in {player}: {detail}")
        if (result.stdout or "").strip() == "idle":
            return Outcome(False, f"{player} isn't running, so there was nothing to {action}.")
        if action == "play":
            return Outcome(True, f"Playing in {player}.")
        return Outcome(True, f"{action.capitalize()} sent to {player}.")


    # Apple's players keep a searchable library and understand music.apple.com
    # links. Spotify has neither through AppleScript — see `open_track`.
    APPLE_PLAYERS = ("Music", "iTunes")

    def play_local(self, query: str, player: str) -> Outcome:
        if player not in self.APPLE_PLAYERS:
            return Outcome(False, "")
        wanted = applescript_string(query)
        # `search` is the same matcher as the app's own search field, so
        # "gods plan drake" finds it without the apostrophe or exact casing.
        script = (
            f'tell application "{player}"\n'
            f'  set hits to (search library playlist 1 for "{wanted}")\n'
            '  if hits is {} then return ""\n'
            "  set found to item 1 of hits\n"
            "  play found\n"
            '  return (get name of found) & " — " & (get artist of found)\n'
            "end tell"
        )
        result = self._run(["osascript", "-e", script], timeout=TRACK_TIMEOUT_SECONDS)
        if result.returncode != 0:
            return Outcome(False, "")
        played = (result.stdout or "").strip()
        if not played:
            return Outcome(False, "")
        return Outcome(True, f"Playing {played} in {player}.")

    def open_track(self, url: str, player: str) -> Outcome:
        if player in self.APPLE_PLAYERS:
            # Report what is actually coming out of the speaker rather than
            # assuming the link played: `open location` navigates, and whether
            # playback starts is the app's decision, not ours.
            # Polled rather than a fixed sleep: a catalogue page on a cold app
            # can take seconds to load, and a delay long enough to cover that
            # would be dead air on every fast case. `open location` usually
            # starts playback on its own; `play` is the nudge for when it does
            # not.
            script = (
                f'tell application "{player}"\n'
                f'  open location "{applescript_string(url)}"\n'
                f"  repeat {_PLAY_WAIT_TICKS} times\n"
                "    delay 0.25\n"
                "    if player state is playing then exit repeat\n"
                "  end repeat\n"
                "  if player state is not playing then\n"
                "    play\n"
                "    delay 0.75\n"
                "  end if\n"
                "  if player state is playing then\n"
                '    return (get name of current track) & " — " & (get artist of current track)\n'
                "  end if\n"
                '  return ""\n'
                "end tell"
            )
            result = self._run(["osascript", "-e", script], timeout=TRACK_TIMEOUT_SECONDS)
            if result.returncode != 0:
                detail = (result.stderr or "").strip() or "unknown error"
                return Outcome(False, f"Could not open that in {player}: {detail}")
            playing = (result.stdout or "").strip()
            if playing:
                return Outcome(True, f"Playing {playing} in {player}.")
            return Outcome(
                True, f"I opened it in {player}, but it didn't start playing by itself."
            )
        # Spotify speaks URIs: `play track "spotify:track:..."` starts it
        # immediately, which is what "play" means.
        script = (
            f'tell application "{player}"\n'
            f'  play track "{applescript_string(url)}"\n'
            "  delay 0.4\n"
            '  if player state is playing then return (get name of current track) '
            '& " — " & (get artist of current track)\n'
            '  return ""\n'
            "end tell"
        )
        result = self._run(["osascript", "-e", script], timeout=TRACK_TIMEOUT_SECONDS)
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "unknown error"
            return Outcome(False, f"Could not play that in {player}: {detail}")
        playing = (result.stdout or "").strip()
        if playing:
            return Outcome(True, f"Playing {playing} in {player}.")
        return Outcome(False, f"{player} accepted the track but didn't start playing.")


class LinuxController(MusicController):
    """MPRIS, via playerctl. `playerctl -l` is the same question as scanning
    /Applications on a Mac: which players are actually here."""

    def __init__(self, runner: Runner | None = None,
                 which: Callable[[str], str | None] = shutil.which):
        self._run = runner or _default_runner
        self._which = which

    def _names(self) -> list[str]:
        if self._which("playerctl") is None:
            return []
        result = self._run(["playerctl", "-l"])
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 or "No players found" in combined:
            return []
        # Bus names look like "spotify.instance123"; the leading segment is the
        # player, and it is what a person would say.
        seen: list[str] = []
        for line in (result.stdout or "").splitlines():
            name = line.strip().split(".")[0]
            if name and name not in seen:
                seen.append(name)
        return seen

    def available(self) -> list[str]:
        return self._names()

    def running(self) -> list[str]:
        # playerctl only lists players that are running, so the two coincide.
        return self._names()

    def control(self, action: str, player: str) -> Outcome:
        result = self._run(["playerctl", "-p", player, action])
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "playerctl failed"
            return Outcome(False, f"Could not {action} in {player}: {detail}")
        return Outcome(True, f"{action.capitalize()} sent to {player}.")

    def missing_tool_message(self) -> str:
        return (
            "I need `playerctl` to control music on Linux. "
            "Ask the user to install it with their package manager."
        )


class WindowsController(MusicController):
    """Media keys. Windows has no cheap way to enumerate players, so there is
    nothing to choose between and nothing to remember."""

    KEYS = {
        "play": "media_play_pause",
        "pause": "media_play_pause",
        "next": "media_next",
        "previous": "media_previous",
    }
    GENERIC = "the active player"

    def available(self) -> list[str]:
        return [self.GENERIC]

    def running(self) -> list[str]:
        return [self.GENERIC]

    def control(self, action: str, player: str) -> Outcome:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError:
            return Outcome(
                False,
                "Media control needs the pynput package, which isn't installed. "
                "Reinstalling Nero Agent should provide it.",
            )
        keyboard = Controller()
        key = getattr(Key, self.KEYS[action])
        keyboard.press(key)
        keyboard.release(key)
        # Say what was actually done rather than claiming a success we cannot see.
        return Outcome(
            True,
            f"I sent the {action} media key. If no player is running, "
            "nothing will have happened.",
        )


def match(wanted: str, names: list[str]) -> str | None:
    """`wanted` against known player names, forgiving about how it was typed.

    A model relaying "apple music" or "spotify " should not miss a player that
    is sitting right there.
    """
    wanted = wanted.strip().lower().removeprefix("apple ")
    for name in names:
        if name.lower() == wanted:
            return name
    for name in names:
        if wanted in name.lower() or name.lower() in wanted:
            return name
    return None


async def apple_catalog_url(query: str, client=None) -> tuple[str, str] | None:
    """(url, "Title — Artist") for the best catalogue match, or None.

    Apple's search endpoint needs no key, which is the whole reason it is the
    one used: a "play this song" that only works after registering for an API
    is a feature most people never switch on.
    """
    import httpx

    params = {"term": query, "entity": "song", "limit": 1}
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as fresh:
                response = await fresh.get(ITUNES_SEARCH, params=params)
        else:
            response = await client.get(ITUNES_SEARCH, params=params)
        response.raise_for_status()
        results = response.json().get("results") or []
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return None
    if not results:
        return None
    hit = results[0]
    url = hit.get("trackViewUrl")
    if not url:
        return None
    return url, f"{hit.get('trackName', query)} — {hit.get('artistName', '?')}"


async def spotify_track_uri(query: str, credentials, client=None) -> tuple[str, str] | None:
    """(spotify:track: URI, "Title — Artist") for the best match, or None.

    Client-credentials flow: the search endpoint is public data, so no user
    login is involved — just an app registration the user makes once.
    """
    import base64

    import httpx

    client_id, client_secret = credentials
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=SEARCH_TIMEOUT)
    try:
        token_response = await client.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {basic}"},
        )
        token_response.raise_for_status()
        token = token_response.json()["access_token"]
        found = await client.get(
            SPOTIFY_SEARCH_URL,
            params={"q": query, "type": "track", "limit": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        found.raise_for_status()
        items = found.json()["tracks"]["items"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return None
    finally:
        if owns_client:
            await client.aclose()
    if not items:
        return None
    hit = items[0]
    artists = ", ".join(a["name"] for a in hit.get("artists", []) if a.get("name"))
    return hit["uri"], f"{hit['name']} — {artists or '?'}"


class PlayMusicSkill(Skill):
    meta = SkillMeta(
        name="play_music",
        description=(
            "Play music on the user's computer: start a named song, or control "
            "what is already playing (play, pause, next track, previous track). "
            "When the user names a song — 'play God's Plan by Drake' — pass the "
            "whole thing as `track`, artist included, with action 'play'. Leave "
            "`track` out to just resume or skip. If they name a player ('on "
            "Spotify'), pass it as `app`; otherwise their usual player is used."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(ACTIONS),
                    "description": "One of: play, pause, next, previous.",
                },
                "track": {
                    "type": "string",
                    "description": (
                        "A specific song to play, e.g. \"God's Plan by Drake\". "
                        "Include the artist when the user says it. Only valid "
                        "with action 'play'."
                    ),
                },
                "app": {
                    "type": "string",
                    "description": (
                        "Which music player to use, e.g. 'Spotify' or 'Music'. "
                        "Omit unless the user named one, or you are answering a "
                        "question about which player to use."
                    ),
                },
            },
            "required": ["action"],
        },
        requires_network=False,
        permission_tier="state_changing",
        category="Apps",
    )

    def __init__(
        self,
        controller: MusicController | None = None,
        preferred_app: str | None = None,
        on_app_chosen: Callable[[str], None] | None = None,
        spotify_auth: Callable[[], tuple[str, str] | None] | None = None,
    ):
        self._controller = controller
        self._preferred_app = preferred_app
        # Read lazily and injected, so the skill never imports the keyring and
        # tests never touch it.
        self._spotify_auth = spotify_auth
        # Injected rather than importing ConfigManager, so the skill stays a
        # pure unit with no knowledge of how Nero persists anything.
        self._on_app_chosen = on_app_chosen

    def _resolve_controller(self) -> MusicController | None:
        if self._controller is not None:
            return self._controller
        system = platform.system()
        if system == "Darwin":
            return MacOSController()
        if system == "Linux":
            return LinuxController()
        if system == "Windows":
            return WindowsController()
        return None

    async def execute(self, **kwargs) -> str:
        action = str(kwargs.get("action") or "").strip().lower()
        if action not in ACTIONS:
            return f"I can only do these music actions: {', '.join(ACTIONS)}."
        controller = self._resolve_controller()
        if controller is None:
            return f"Error: music control isn't supported on {platform.system()!r}."

        available = controller.available()
        if not available:
            return self._nothing_available(controller)

        requested = str(kwargs.get("app") or "").strip()
        chosen = self._choose(controller, available, requested)
        if isinstance(chosen, str) and chosen not in available:
            return chosen  # a question or a complaint, not a player

        track = str(kwargs.get("track") or "").strip()
        if track and action != "play":
            # Naming a song only makes sense as "play this". Silently skipping
            # to the next track instead would be a confusing way to say no.
            return f"I can only start a specific song with 'play', not '{action}'."
        if track:
            outcome = await self._play_track(controller, track, chosen)
        else:
            outcome = controller.control(action, chosen)
        # Only learn from a choice that worked. Remembering a player that just
        # failed would make every later request fail the same way, silently.
        if requested and outcome.ok and self._on_app_chosen is not None:
            self._on_app_chosen(chosen)
        return outcome.message

    async def _play_track(self, controller: MusicController, track: str, player: str) -> Outcome:
        """Start one named song, cheapest route first.

        The library is tried before the network: a track the user already has
        starts instantly and works with no connection. Only on a miss does this
        go and look the song up.
        """
        # Run both at once. The library search is a subprocess that can take
        # seconds on a cold app; the lookup is a ~150 ms network call. Done in
        # sequence the lookup is pure added latency on every miss, and a miss is
        # the normal case for a song the user does not already own.
        # return_exceptions: the lookup is best effort. A dead network must not
        # be able to kill a local track that was about to play perfectly well.
        local, found = await asyncio.gather(
            asyncio.to_thread(controller.play_local, track, player),
            self._catalog(track, player),
            return_exceptions=True,
        )
        if isinstance(local, BaseException):
            logger.debug("library search failed", exc_info=local)
            local = Outcome(False, "")
        if isinstance(found, BaseException):
            logger.debug("catalogue lookup failed", exc_info=found)
            found = None
        if local.ok:
            return local
        if found is None:
            if player.lower().startswith("spotify") and not (
                self._spotify_auth and self._spotify_auth()
            ):
                # Never dress this up as success. The old wording ("I opened
                # that search — press play") was read back to the user as "I've
                # queued it up, enjoy", which is the opposite of what happened.
                return Outcome(False, SPOTIFY_SETUP_HINT)
            return Outcome(
                False,
                f"I couldn't find {track!r} in {player}. Ask the user to check the "
                "song and artist, or to play it themselves.",
            )
        url, label = found
        outcome = controller.open_track(url, player)
        # Name what was found only when the lookup actually learned something.
        # For a search hand-off the "label" is just the query echoed back, and
        # repeating it reads like a stutter.
        if outcome.ok and label != track and label not in outcome.message:
            return Outcome(True, f"{outcome.message.rstrip('.')} ({label}).")
        return outcome

    async def _catalog(self, track: str, player: str) -> tuple[str, str] | None:
        """Where to send the player for a song it does not already have."""
        if player.lower().startswith("spotify"):
            credentials = self._spotify_auth() if self._spotify_auth else None
            if not credentials:
                return None
            return await spotify_track_uri(track, credentials)
        return await apple_catalog_url(track)

    def _choose(self, controller: MusicController, available: list[str], requested: str) -> str:
        """The player to use, or the question to ask instead.

        Order matters: an explicit request wins, then a remembered preference,
        then the only option, then the only one playing. Asking is the last
        resort, not the first.
        """
        if requested:
            named = match(requested, available)
            if named is None:
                return (
                    f"I can't find a music player called {requested!r} on this computer. "
                    f"Available: {', '.join(available)}."
                )
            return named
        if self._preferred_app:
            remembered = match(self._preferred_app, available)
            if remembered is not None:
                return remembered
        if len(available) == 1:
            return available[0]
        playing = controller.running()
        if len(playing) == 1:
            return playing[0]
        return (
            f"The user has more than one music player: {', '.join(available)}. "
            "Ask them which one to use, then call play_music again with that "
            "name as `app`. Their answer will be remembered, so ask only once."
        )

    @staticmethod
    def _nothing_available(controller: MusicController) -> str:
        missing = getattr(controller, "missing_tool_message", None)
        if missing is not None:
            return missing()
        return (
            "I can't find a music player on this computer. Ask the user to "
            "install one, or to start the one they use."
        )


SKILL = PlayMusicSkill()
