"""App skills: open_app and close_app.

Both live here because both need the same thing — a way to find out what is
actually installed, so a name that misses can suggest one that would not.
"""

import difflib
import platform
import shutil
import subprocess
from pathlib import Path

from nero.skills.base import Skill, SkillMeta

# Long enough for an app that is slow to quit, short enough that a hung one
# cannot hold a turn open.
QUIT_TIMEOUT = 15

APP_DIRS = (
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)
# Close enough that "Sptify" finds Spotify, far enough that "Mail" does not
# confidently become "Maps".
SIMILARITY = 0.6


class AppSkill(Skill):
    """Shared app discovery. Neither skill scans until it has already failed,
    so the common case pays nothing for the suggestion machinery."""

    def __init__(self, app_dirs=APP_DIRS):
        self._app_dirs = tuple(app_dirs)

    def installed(self) -> list[str]:
        """App names on this machine."""
        names: list[str] = []
        for directory in self._app_dirs:
            try:
                names.extend(entry.stem for entry in directory.glob("*.app"))
            except OSError:
                continue
        return sorted(set(names))

    def resolve(self, app_name: str) -> str | None:
        """The installed app `app_name` confidently means, or None.

        Exact, then case-insensitive, then containment — "chrome" has to find
        "Google Chrome", or every request needs the full bundle name.

        Deliberately stops short of fuzzy matching. Opening the wrong app is a
        harmless annoyance; *quitting* the wrong one can lose work, so a name
        that only approximately matches is turned back into a question rather
        than acted on.
        """
        installed = self.installed()
        for name in installed:
            if name == app_name:
                return name
        wanted = app_name.strip().lower()
        if not wanted:
            return None
        for name in installed:
            if name.lower() == wanted:
                return name
        for name in installed:
            lowered = name.lower()
            # The reverse direction ("close the music app" -> Music) needs a
            # length floor, or a two-letter app name matches almost any phrase.
            if wanted in lowered or (len(lowered) >= 3 and lowered in wanted):
                return name
        return None

    def _did_you_mean(self, app_name: str) -> str:
        """A suggestion when the name was close, or a nudge when it wasn't.

        A bare "could not find it" is a dead end: the user has no way to know
        whether they typo'd, or the app is called something else, or it simply
        isn't installed.
        """
        installed = self.installed()
        if not installed:
            return ""
        close = difflib.get_close_matches(app_name, installed, n=3, cutoff=SIMILARITY)
        if close:
            return f" Did you mean {' or '.join(repr(name) for name in close)}?"
        return f" There are {len(installed)} apps installed; none look like that."


class OpenAppSkill(AppSkill):
    meta = SkillMeta(
        name="open_app",
        description=(
            "Open an application installed on the user's computer by name. "
            "Use this when the user asks to open, launch, or start an app "
            "(e.g. 'open Spotify'). Returns a confirmation or an error message "
            "if the app could not be found."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "The application's name, e.g. 'Safari' or 'Spotify'.",
                }
            },
            "required": ["app_name"],
        },
        requires_network=False,
        permission_tier="state_changing",
        category="Apps",
    )

    async def execute(self, **kwargs) -> str:
        app_name = str(kwargs.get("app_name") or "").strip()
        if not app_name:
            return "Error: no app name provided."
        try:
            system = platform.system()
            if system == "Darwin":
                return self._open_macos(app_name)
            if system == "Windows":
                return self._open_windows(app_name)
            if system == "Linux":
                return self._open_linux(app_name)
            return f"Error: unsupported platform {system!r}."
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error launching {app_name!r}: {exc}"

    def _open_macos(self, app_name: str) -> str:
        # `open -a` already does its own loose matching, so this is tried first
        # and the directory scan only happens once it has actually failed.
        result = subprocess.run(
            ["open", "-a", app_name], capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Opened {app_name}."
        detail = result.stderr.strip() or "unknown error"
        return f"Could not open an app called {app_name!r}: {detail}.{self._did_you_mean(app_name)}"

    def _open_windows(self, app_name: str) -> str:
        # `start` is a cmd built-in; the empty "" is its window-title slot. Passing
        # an argument list (shell=False) keeps user text out of shell parsing.
        result = subprocess.run(
            ["cmd", "/c", "start", "", app_name], capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Opened {app_name}."
        detail = result.stderr.strip() or "unknown error"
        return f"Could not open an app called {app_name!r}: {detail}"

    def _open_linux(self, app_name: str) -> str:
        desktop_id = self._find_desktop_id(app_name)
        if desktop_id and shutil.which("gtk-launch"):
            result = subprocess.run(
                ["gtk-launch", desktop_id], capture_output=True, text=True
            )
            if result.returncode == 0:
                return f"Opened {app_name}."
        executable = shutil.which(app_name) or shutil.which(app_name.lower())
        if executable:
            subprocess.Popen(
                [executable],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return f"Opened {app_name}."
        return f"Could not find an app called {app_name!r} on this system.{self._did_you_mean(app_name)}"

    @staticmethod
    def _find_desktop_id(app_name: str) -> str | None:
        """Match a .desktop entry by filename stem or Name= field, case-insensitively."""
        wanted = app_name.lower()
        search_dirs = [
            Path("/usr/share/applications"),
            Path.home() / ".local/share/applications",
        ]
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for entry in directory.glob("*.desktop"):
                if entry.stem.lower() == wanted:
                    return entry.stem
                try:
                    for line in entry.read_text(errors="ignore").splitlines():
                        if line.lower().startswith("name=") and line[5:].strip().lower() == wanted:
                            return entry.stem
                except OSError:
                    continue
        return None


class CloseAppSkill(AppSkill):
    """Quit a running app.

    Deliberately a *graceful* quit, not a kill: on macOS `tell application "X"
    to quit` is the same thing as pressing Cmd-Q, so an app with unsaved work
    puts up its own save dialog instead of losing it. That is also why this is
    state-changing rather than destructive — nothing is destroyed without the
    app itself asking first, and shipping it disabled would mean "close Chrome"
    fails by default, which is the complaint this exists to answer.
    """

    meta = SkillMeta(
        name="close_app",
        description=(
            "Close (quit) an application running on the user's computer by name. "
            "Use this when the user asks to close, quit, exit, or stop an app "
            "(e.g. 'close Chrome', 'quit Apple Music'). Quits gracefully, so an "
            "app with unsaved work will still prompt them."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "The application's name, e.g. 'Safari' or 'Spotify'.",
                }
            },
            "required": ["app_name"],
        },
        requires_network=False,
        permission_tier="state_changing",
        category="Apps",
    )

    async def execute(self, **kwargs) -> str:
        app_name = str(kwargs.get("app_name") or "").strip()
        if not app_name:
            return "Error: no app name provided."
        try:
            system = platform.system()
            if system == "Darwin":
                return self._close_macos(app_name)
            if system == "Windows":
                return self._close_windows(app_name)
            if system == "Linux":
                return self._close_linux(app_name)
            return f"Error: unsupported platform {system!r}."
        except subprocess.TimeoutExpired:
            return f"{app_name} did not respond to the quit request in time."
        except Exception as exc:  # noqa: BLE001 — must reach the model as a skill result
            return f"Error closing {app_name!r}: {exc}"

    def _close_macos(self, app_name: str) -> str:
        resolved = self.resolve(app_name)
        if resolved is None:
            return (
                f"I couldn't find an app called {app_name!r} on this computer."
                f"{self._did_you_mean(app_name)}"
            )
        # Guarded, and one subprocess rather than two: an unguarded `quit` would
        # *launch* an app that wasn't running, just to close it again.
        script = (
            f'if application "{resolved}" is running then\n'
            f'  tell application "{resolved}" to quit\n'
            '  return "done"\n'
            "end if\n"
            'return "idle"'
        )
        result = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=QUIT_TIMEOUT
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip() or "unknown error"
            return f"Could not close {resolved}: {detail}"
        if (result.stdout or "").strip() == "idle":
            return f"{resolved} isn't running."
        return f"Closed {resolved}."

    def _close_linux(self, app_name: str) -> str:
        if shutil.which("pkill") is None:
            return "I need `pkill` to close apps on Linux, and it isn't installed."
        # -x: match the process name exactly. Without it a request to close
        # "code" could match anything with "code" in its command line.
        for candidate in (app_name, app_name.lower(), app_name.lower().replace(" ", "-")):
            result = subprocess.run(
                ["pkill", "-TERM", "-x", candidate],
                capture_output=True,
                text=True,
                timeout=QUIT_TIMEOUT,
            )
            if result.returncode == 0:
                return f"Closed {app_name}."
        return f"Nothing named {app_name!r} appears to be running."

    def _close_windows(self, app_name: str) -> str:
        image = app_name if app_name.lower().endswith(".exe") else f"{app_name}.exe"
        # No /F: a graceful close, so unsaved work still prompts.
        result = subprocess.run(
            ["taskkill", "/IM", image], capture_output=True, text=True, timeout=QUIT_TIMEOUT
        )
        if result.returncode == 0:
            return f"Closed {app_name}."
        detail = (result.stdout or result.stderr or "").strip() or "unknown error"
        return f"Could not close {app_name!r}: {detail}"


# The seam a future FastMCP wrapper attaches to (spec D1) — wrap this object,
# don't touch the logic above it.
SKILL = OpenAppSkill()
CLOSE_SKILL = CloseAppSkill()
