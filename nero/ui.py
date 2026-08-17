"""Terminal-aware selection for `nero config`.

Arrow keys when there is a real terminal, the numbered prompt everywhere else.
The fallback is not a nicety: every config-menu test drives stdin through a
pipe, and so does anyone scripting the menu. Losing it breaks both.
"""

import sys

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table


def pick(
    title: str,
    choices: list[tuple[str, str]],
    default: str | None = None,
    console: Console | None = None,
) -> str | None:
    """Choose one value from (value, label) pairs.

    Returns the chosen value, or None meaning "no change" — which covers a
    blank answer, an out-of-range answer, and Esc/Ctrl-C on the arrow picker.
    Callers must treat None as "leave the config alone".
    """
    if not choices:
        return None
    if _interactive():
        try:
            return _pick_arrows(title, choices, default)
        except Exception:  # noqa: BLE001 — no terminal failure may cost the user the task
            pass
    return _pick_numbered(title, choices, default, console or Console())


def _interactive() -> bool:
    """Both streams, not just stdin: a piped stdout means output is being
    captured, and a full-screen picker would corrupt it."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _pick_arrows(
    title: str, choices: list[tuple[str, str]], default: str | None
) -> str | None:
    # Imported here so a headless or piped run never loads prompt_toolkit.
    import questionary

    options = [questionary.Choice(title=label, value=value) for value, label in choices]
    # questionary matches `default` against Choice instances, not raw values.
    initial = next((option for option in options if option.value == default), None)
    return questionary.select(title, choices=options, default=initial, qmark="").ask()


def _pick_numbered(
    title: str, choices: list[tuple[str, str]], default: str | None, console: Console
) -> str | None:
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    for index, (value, label) in enumerate(choices, start=1):
        marker = "  [dim](current)[/dim]" if value == default else ""
        table.add_row(f"{index}.", label + marker)
    console.print(table)
    answer = Prompt.ask(
        "Number (Enter to keep current)", default="", show_default=False, console=console
    ).strip()
    if not answer.isdecimal() or not 1 <= int(answer) <= len(choices):
        if answer:
            console.print(f"[yellow]Pick 1–{len(choices)}.[/yellow]")
        return None
    return choices[int(answer) - 1][0]
