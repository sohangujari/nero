"""Cron -> launchd translation and launchd agent install/uninstall.

launchd has no cron parser of its own — it takes `StartCalendarInterval`, a
dict of the fields that must match, with a missing key meaning "any value"
(cron's `*`). Only literal integers and `*` are supported: ranges, steps, and
lists would either need to be silently reinterpreted as something launchd can
express, or expanded into multiple StartCalendarInterval dicts, and either
one lets a schedule silently mean something other than what the user wrote.
Raising instead keeps "installed" == "runs when the user thinks it runs".
"""

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from platformdirs import user_log_dir

from nero.config.schema import RoutineConfig

# (cron field name, launchd key, min, max) in cron field order.
_FIELDS = [
    ("minute", "Minute", 0, 59),
    ("hour", "Hour", 0, 23),
    ("day", "Day", 1, 31),
    ("month", "Month", 1, 12),
    ("weekday", "Weekday", 0, 6),  # cron and launchd agree: Sun=0
]


class RoutineError(Exception):
    """A routine's schedule or launchd setup could not be resolved."""


def cron_to_calendar(schedule: str) -> dict:
    """Translate a 5-field cron string to a launchd StartCalendarInterval dict."""
    fields = schedule.split()
    if len(fields) != 5:
        raise RoutineError(
            f"Cron schedule {schedule!r} must have exactly 5 fields "
            f"(minute hour day month weekday), got {len(fields)}"
        )
    calendar: dict = {}
    for (field_name, key, low, high), raw in zip(_FIELDS, fields):
        if raw == "*":
            continue
        if "/" in raw:
            raise RoutineError(f"Step syntax {raw!r} in the {field_name} field is not supported")
        if "-" in raw:
            raise RoutineError(f"Range syntax {raw!r} in the {field_name} field is not supported")
        if "," in raw:
            raise RoutineError(f"List syntax {raw!r} in the {field_name} field is not supported")
        try:
            value = int(raw)
        except ValueError as exc:
            raise RoutineError(
                f"Invalid {field_name} value {raw!r}: must be an integer or '*'"
            ) from exc
        if not (low <= value <= high):
            raise RoutineError(f"{field_name} value {value} is out of range {low}-{high}")
        calendar[key] = value
    return calendar


def label_for(name: str) -> str:
    return f"com.neroagent.routine.{name}"


def default_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path(name: str, agents_dir: Path) -> Path:
    return Path(agents_dir) / f"{label_for(name)}.plist"


def is_installed(name: str, agents_dir: Path) -> bool:
    return plist_path(name, agents_dir).exists()


def resolve_executable() -> str:
    """An absolute path to the nero executable, for ProgramArguments.

    launchd has no shell PATH of its own — a plist pointing at a bare "nero"
    (sys.argv[0] when invoked via PATH) would silently never run.
    """
    resolved = Path(sys.argv[0]).resolve()
    if resolved.exists():
        return str(resolved)
    found = shutil.which("nero")
    if found:
        return found
    raise RoutineError(
        f"Could not resolve an absolute path to the nero executable "
        f"(sys.argv[0]={sys.argv[0]!r})"
    )


def _run_launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)


def install_routine(
    name: str, routine: RoutineConfig, executable: str, agents_dir: Path
) -> str:
    """Write the plist and load it with launchd. Only loads on darwin —
    elsewhere the plist is written and loading is reported as skipped."""
    agents_dir = Path(agents_dir)
    agents_dir.mkdir(parents=True, exist_ok=True)
    label = label_for(name)
    path = plist_path(name, agents_dir)
    log_dir = Path(user_log_dir("nero")) / "routines"
    log_dir.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": label,
        "ProgramArguments": [executable, "routine", "run", name],
        "StartCalendarInterval": cron_to_calendar(routine.schedule),
        "StandardOutPath": str(log_dir / f"{name}.out.log"),
        "StandardErrorPath": str(log_dir / f"{name}.err.log"),
    }
    with path.open("wb") as f:
        plistlib.dump(plist, f)

    if sys.platform != "darwin":
        return f"Wrote {path}. Loading skipped: launchd is darwin-only."

    result = _run_launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
    if result.returncode != 0:
        result = _run_launchctl("load", "-w", str(path))
        if result.returncode != 0:
            return (
                f"Wrote {path}, but launchctl could not load it: "
                f"{result.stderr.strip()}"
            )
    return f"Installed {label} ({path})."


def uninstall_routine(name: str, agents_dir: Path) -> str:
    """Unload from launchd and remove the plist. A missing plist is a no-op,
    not an error."""
    agents_dir = Path(agents_dir)
    label = label_for(name)
    path = plist_path(name, agents_dir)
    if not path.exists():
        return f"No installed routine named {name!r} — nothing to do."

    if sys.platform == "darwin":
        result = _run_launchctl("bootout", f"gui/{os.getuid()}/{label}")
        if result.returncode != 0:
            _run_launchctl("unload", str(path))
    path.unlink()
    return f"Uninstalled {label}."
