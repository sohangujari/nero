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

import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from nero.skills.base import Skill, SkillMeta

ACTIONS = ("play", "pause", "next", "previous")

Runner = Callable[[list[str]], subprocess.CompletedProcess]

# Long enough for a cold app launch on `play`, short enough that a wedged
# player cannot hold a turn open.
TIMEOUT_SECONDS = 10


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess:
    # Argument list, never a shell string — same injection-safe pattern as open_app.
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)


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


class MacOSController(MusicController):
    def __init__(self, runner: Runner | None = None, app_dirs=APP_DIRS, players=MAC_PLAYERS):
        self._run = runner or _default_runner
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


class PlayMusicSkill(Skill):
    meta = SkillMeta(
        name="play_music",
        description=(
            "Control music that is already playing on the user's computer, or "
            "start their music player: play, pause, skip to the next track, or "
            "go back to the previous one. Use this when the user asks to play, "
            "pause, resume, or skip music. It cannot search for or choose "
            "specific songs. If the user names a player ('play on Spotify'), "
            "pass it as `app`; otherwise omit it and their usual player is used."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(ACTIONS),
                    "description": "One of: play, pause, next, previous.",
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
    ):
        self._controller = controller
        self._preferred_app = preferred_app
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

        outcome = controller.control(action, chosen)
        # Only learn from a choice that worked. Remembering a player that just
        # failed would make every later request fail the same way, silently.
        if requested and outcome.ok and self._on_app_chosen is not None:
            self._on_app_chosen(chosen)
        return outcome.message

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
