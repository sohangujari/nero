import platform
import shutil
import subprocess
from pathlib import Path

from nero.tools.base import Tool


class OpenAppTool(Tool):
    name = "open_app"
    description = (
        "Open an application installed on the user's computer by name. "
        "Use this when the user asks to open, launch, or start an app "
        "(e.g. 'open Spotify'). Returns a confirmation or an error message "
        "if the app could not be found."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The application's name, e.g. 'Safari' or 'Spotify'.",
            }
        },
        "required": ["app_name"],
    }

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
        except Exception as exc:  # noqa: BLE001 — must reach Claude as a tool result
            return f"Error launching {app_name!r}: {exc}"

    def _open_macos(self, app_name: str) -> str:
        result = subprocess.run(
            ["open", "-a", app_name], capture_output=True, text=True
        )
        if result.returncode == 0:
            return f"Opened {app_name}."
        detail = result.stderr.strip() or "unknown error"
        return f"Could not open an app called {app_name!r}: {detail}"

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
        return f"Could not find an app called {app_name!r} on this system."

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
