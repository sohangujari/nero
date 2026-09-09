"""set_volume: system output volume, absolute or relative, plus mute.

Three ways to say the same thing, because people use all three: "set it to
30%", "turn it up a bit", "mute". The model maps the vague ones onto a number —
that is what it is good at — and this skill deals only in numbers, which is
what a computer is good at.

Reading the current level matters: "turn it up a bit" is meaningless without
knowing where it is now, and a skill that guessed would drift.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Callable

from nero.skills.base import Skill, SkillMeta

TIMEOUT_SECONDS = 5

# What the vaguer words are worth, in points of 100. Named here rather than
# left to the model so "a bit" means the same thing every time.
NUDGE = 10
STEP = 20

Runner = Callable[..., subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)


def clamp(value: float) -> int:
    return max(0, min(100, round(value)))


def describe(level: int, muted: bool) -> str:
    if muted:
        return "muted"
    if level == 0:
        return "silent"
    if level <= 25:
        return f"{level}% (low)"
    if level >= 85:
        return f"{level}% (loud)"
    return f"{level}%"


class VolumeController:
    """macOS/Linux/Windows behind one tiny interface, so the skill holds no
    platform knowledge at all."""

    def __init__(self, runner: Runner | None = None):
        self._run = runner or _run

    def read(self) -> tuple[int, bool] | None:
        """(level 0-100, muted), or None if it can't be read."""
        raise NotImplementedError

    def write(self, level: int, muted: bool) -> bool:
        raise NotImplementedError


class MacOSVolume(VolumeController):
    def read(self) -> tuple[int, bool] | None:
        result = self._run(
            [
                "osascript",
                "-e",
                "set v to get volume settings\n"
                'return (output volume of v as text) & "," & (output muted of v as text)',
            ]
        )
        if result.returncode != 0:
            return None
        parts = (result.stdout or "").strip().split(",")
        if len(parts) != 2:
            return None
        try:
            return clamp(float(parts[0])), parts[1].strip().lower() == "true"
        except ValueError:
            return None

    def write(self, level: int, muted: bool) -> bool:
        # Volume and mute in one call: two would let the speaker jump to the new
        # level for an instant before muting.
        muting = "with output muted" if muted else "without output muted"
        result = self._run(
            ["osascript", "-e", f"set volume output volume {level} {muting}"]
        )
        return result.returncode == 0


class LinuxVolume(VolumeController):
    SINK = "@DEFAULT_SINK@"

    def __init__(self, runner: Runner | None = None, which=shutil.which):
        super().__init__(runner)
        self._which = which

    def available(self) -> bool:
        return self._which("pactl") is not None

    def read(self) -> tuple[int, bool] | None:
        if not self.available():
            return None
        volume = self._run(["pactl", "get-sink-volume", self.SINK])
        mute = self._run(["pactl", "get-sink-mute", self.SINK])
        if volume.returncode != 0 or mute.returncode != 0:
            return None
        percent = None
        for token in (volume.stdout or "").split():
            if token.endswith("%"):
                try:
                    percent = clamp(float(token.rstrip("%")))
                except ValueError:
                    continue
                break
        if percent is None:
            return None
        return percent, "yes" in (mute.stdout or "").lower()

    def write(self, level: int, muted: bool) -> bool:
        if not self.available():
            return False
        first = self._run(["pactl", "set-sink-volume", self.SINK, f"{level}%"])
        second = self._run(["pactl", "set-sink-mute", self.SINK, "1" if muted else "0"])
        return first.returncode == 0 and second.returncode == 0


class WindowsVolume(VolumeController):
    """Media keys only. Windows has no scriptable way to read the level without
    a package Nero does not ship, so relative changes work and absolute ones
    say so rather than pretending."""

    def read(self) -> tuple[int, bool] | None:
        return None

    def write(self, level: int, muted: bool) -> bool:
        return False

    def nudge(self, steps: int, mute: bool | None) -> bool:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError:
            return False
        keyboard = Controller()
        if mute is not None:
            keyboard.press(Key.media_volume_mute)
            keyboard.release(Key.media_volume_mute)
            return True
        key = Key.media_volume_up if steps > 0 else Key.media_volume_down
        for _ in range(min(abs(steps), 50)):
            keyboard.press(key)
            keyboard.release(key)
        return True


class SetVolumeSkill(Skill):
    meta = SkillMeta(
        name="set_volume",
        description=(
            "Set or change the computer's output volume, or mute it. Use this "
            "whenever the user asks about volume or loudness. Pass `level` for an "
            "absolute value ('set it to 30%', 'make it quiet' -> 20, 'full "
            "volume' -> 100). Pass `change` for a relative one ('a bit louder' "
            f"-> {NUDGE}, 'louder' -> {STEP}, 'much louder' -> {STEP * 2}, "
            f"'quieter' -> -{STEP}). Pass `mute` true to mute and false to "
            "unmute. Give exactly one of the three."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Absolute volume, 0 (silent) to 100 (maximum).",
                },
                "change": {
                    "type": "integer",
                    "description": (
                        "Relative change in points, e.g. 10 for a bit louder, "
                        "-20 for quieter. Negative lowers it."
                    ),
                },
                "mute": {
                    "type": "boolean",
                    "description": "true to mute, false to unmute.",
                },
            },
            "required": [],
        },
        requires_network=False,
        permission_tier="state_changing",
        category="Apps",
    )

    def __init__(self, controller: VolumeController | None = None):
        self._controller = controller

    def _resolve(self) -> VolumeController | None:
        if self._controller is not None:
            return self._controller
        system = platform.system()
        if system == "Darwin":
            return MacOSVolume()
        if system == "Linux":
            return LinuxVolume()
        if system == "Windows":
            return WindowsVolume()
        return None

    async def execute(self, **kwargs) -> str:
        level = kwargs.get("level")
        change = kwargs.get("change")
        mute = kwargs.get("mute")
        if level is None and change is None and mute is None:
            return (
                "Say what to do with the volume: a level (0-100), a change "
                f"(+{NUDGE} for a bit louder, -{STEP} for quieter), or mute true/false."
            )
        controller = self._resolve()
        if controller is None:
            return f"Error: volume control isn't supported on {platform.system()!r}."

        try:
            current = controller.read()
        except Exception:  # noqa: BLE001 — must reach the model as a skill result
            current = None

        if current is None:
            return self._blind(controller, level, change, mute)

        now, was_muted = current
        # Unmute on any request to raise the volume. Turning it up while muted
        # and staying silent is technically obedient and practically useless.
        if level is not None:
            target, muted = clamp(level), False if level > 0 else was_muted
        elif change is not None:
            target, muted = clamp(now + change), was_muted and change <= 0
        else:
            target, muted = now, bool(mute)
            if not muted and now == 0:
                # Unmuting something already at zero leaves silence; give it
                # somewhere to go.
                target = STEP

        try:
            if not controller.write(target, muted):
                return "I couldn't change the volume just now."
        except Exception as exc:  # noqa: BLE001
            return f"I couldn't change the volume: {exc}"
        if muted:
            return "Muted."
        return f"Volume {describe(target, muted)}."

    @staticmethod
    def _blind(controller, level, change, mute) -> str:
        """Platforms that cannot report the current level. Relative changes and
        mute still work through media keys; absolute ones cannot."""
        nudge = getattr(controller, "nudge", None)
        if nudge is None:
            return "I couldn't read the current volume, so I left it alone."
        if mute is not None:
            return "Toggled mute." if nudge(0, mute) else "I couldn't reach the volume keys."
        if change is not None:
            steps = max(1, round(abs(change) / 2)) * (1 if change > 0 else -1)
            return (
                f"Nudged the volume {'up' if change > 0 else 'down'}."
                if nudge(steps, None)
                else "I couldn't reach the volume keys."
            )
        return (
            "I can only make the volume louder or quieter on this system, not set "
            "it to an exact level. Ask the user for 'louder' or 'quieter' instead."
        )


SKILL = SetVolumeSkill()
