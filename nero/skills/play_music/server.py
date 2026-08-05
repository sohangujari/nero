import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable

from nero.skills.base import Skill, SkillMeta

ACTIONS = ("play", "pause", "next", "previous")

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess:
    # Argument list, never a shell string — same injection-safe pattern as open_app.
    return subprocess.run(cmd, capture_output=True, text=True, timeout=5)


class MusicController(ABC):
    """Per-platform media control (spec D3).

    pynput cannot report whether anything received a synthesised media key, so
    the two platforms that *can* answer "is anything playing?" use their native
    interface instead.
    """

    @abstractmethod
    def control(self, action: str) -> str:
        """Perform `action`, returning a result string for the model."""


class MacOSController(MusicController):
    APPS = ("Spotify", "Music")
    COMMANDS = {
        "play": "play",
        "pause": "pause",
        "next": "next track",
        "previous": "previous track",
    }

    def __init__(self, runner: Runner | None = None):
        self._run = runner or _default_runner

    def control(self, action: str) -> str:
        for app in self.APPS:
            if not self._is_running(app):
                continue
            result = self._run(
                ["osascript", "-e", f'tell application "{app}" to {self.COMMANDS[action]}']
            )
            if result.returncode != 0:
                detail = (result.stderr or "").strip() or "unknown error"
                return f"Could not {action} in {app}: {detail}"
            return f"{action.capitalize()} sent to {app}."
        return (
            "Nothing is playing — neither Spotify nor Music is running. "
            "Ask the user to start one first."
        )

    def _is_running(self, app: str) -> bool:
        result = self._run(["osascript", "-e", f'application "{app}" is running'])
        return result.returncode == 0 and (result.stdout or "").strip() == "true"


class LinuxController(MusicController):
    COMMANDS = {"play": "play", "pause": "pause", "next": "next", "previous": "previous"}

    def __init__(self, runner: Runner | None = None, which: Callable[[str], str | None] = shutil.which):
        self._run = runner or _default_runner
        self._which = which

    def control(self, action: str) -> str:
        if self._which("playerctl") is None:
            return (
                "I need `playerctl` to control music on Linux. "
                "Ask the user to install it with their package manager."
            )
        status = self._run(["playerctl", "status"])
        combined = (status.stdout or "") + (status.stderr or "")
        if status.returncode != 0 or "No players found" in combined:
            return (
                "Nothing is playing — I couldn't find an active media player. "
                "Ask the user to start one first."
            )
        result = self._run(["playerctl", self.COMMANDS[action]])
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "playerctl failed"
            return f"Could not {action}: {detail}"
        return f"{action.capitalize()} sent to the active media player."


class WindowsController(MusicController):
    KEYS = {
        "play": "media_play_pause",
        "pause": "media_play_pause",
        "next": "media_next",
        "previous": "media_previous",
    }

    def control(self, action: str) -> str:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError:
            return (
                "Media control needs the pynput package, which isn't installed. "
                "Reinstalling Nero should provide it."
            )
        keyboard = Controller()
        key = getattr(Key, self.KEYS[action])
        keyboard.press(key)
        keyboard.release(key)
        # Windows has no cheap way to ask whether a player exists, so say so
        # rather than claiming success we can't verify.
        return (
            f"I sent the {action} media key. If no player is running, "
            "nothing will have happened."
        )


class PlayMusicSkill(Skill):
    meta = SkillMeta(
        name="play_music",
        description=(
            "Control music that is already playing on the user's computer: play, "
            "pause, skip to the next track, or go back to the previous one. Use "
            "this when the user asks to play, pause, resume, or skip music. It "
            "cannot search for or choose specific songs."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(ACTIONS),
                    "description": "One of: play, pause, next, previous.",
                }
            },
            "required": ["action"],
        },
        requires_network=False,
        permission_tier="state_changing",
    )

    def __init__(self, controller: MusicController | None = None):
        self._controller = controller

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
        try:
            return controller.control(action)
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error controlling music: {exc}"


SKILL = PlayMusicSkill()
