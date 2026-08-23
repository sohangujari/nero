"""Runtime environment guard for the pip/pipx install path.

Standalone binaries bundle Python 3.12, so this never fires there. It only
warns people who installed via pip/pipx against a mismatched interpreter — the
exact failure mode (system Python 3.13+) that motivated the pinned/bundled
distribution strategy.
"""

from __future__ import annotations

import sys

REQUIRED = (3, 12)


def check_python_version(current: tuple[int, ...] = sys.version_info[:3]) -> str | None:
    """Return a warning string if `current` isn't the supported 3.12, else None."""
    if tuple(current[:2]) == REQUIRED:
        return None
    running = ".".join(str(part) for part in current[:3])
    return (
        f"Nero Agent is tested and pinned to Python {REQUIRED[0]}.{REQUIRED[1]}; "
        f"you're running {running}. Some dependencies (voice/ML) may fail to "
        f"install or import. Prefer the standalone binary, or use a Python "
        f"{REQUIRED[0]}.{REQUIRED[1]} environment (e.g. `uv sync`)."
    )
