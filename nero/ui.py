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
    prompt: str = "Number (Enter to keep current)",
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
    return _pick_numbered(title, choices, default, console or Console(), prompt)


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
    title: str,
    choices: list[tuple[str, str]],
    default: str | None,
    console: Console,
    prompt: str,
) -> str | None:
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    for index, (value, label) in enumerate(choices, start=1):
        marker = "  [dim](current)[/dim]" if value == default else ""
        table.add_row(f"{index}.", label + marker)
    console.print(table)
    answer = Prompt.ask(prompt, default="", show_default=False, console=console).strip()
    index = _parse_selection(answer, len(choices), console)
    if index is None:
        return None
    return choices[index][0]


def _parse_selection(answer: str, count: int, console: Console) -> int | None:
    """Validate a numbered-prompt answer, printing the `Pick 1-N.` warning on
    a non-empty invalid answer. Returns the 0-based index, or None on a blank
    or out-of-range answer — shared by `_pick_numbered` and
    `_pick_many_numbered` so the range check can't drift between them."""
    if not answer.isdecimal() or not 1 <= int(answer) <= count:
        if answer:
            console.print(f"[yellow]Pick 1–{count}.[/yellow]")
        return None
    return int(answer) - 1


def pick_many(
    title: str,
    choices: list[tuple[str, str]],
    selected: set[str],
    console: Console | None = None,
) -> set[str] | None:
    """Choose any number of values from (value, label) pairs.

    Returns the chosen set, or None meaning "no change" — a blank answer, an
    out-of-range answer, and Esc/Ctrl-C on the checkbox. None is not the same
    answer as an empty set: None leaves the config alone, an empty set turns
    everything off. Callers must not collapse the two.
    """
    if not choices:
        return None
    if _interactive():
        try:
            return _pick_many_arrows(title, choices, selected)
        except Exception:  # noqa: BLE001 — no terminal failure may cost the user the task
            pass
    return _pick_many_numbered(title, choices, selected, console or Console())


def _pick_many_arrows(
    title: str, choices: list[tuple[str, str]], selected: set[str]
) -> set[str] | None:
    # Imported here so a headless or piped run never loads prompt_toolkit.
    import questionary

    options = [
        questionary.Choice(title=label, value=value, checked=value in selected)
        for value, label in choices
    ]
    answer = questionary.checkbox(title, choices=options, qmark="").ask()
    # ask() returns None on Esc/Ctrl-C and a list otherwise — including [].
    return None if answer is None else set(answer)


def _pick_many_numbered(
    title: str, choices: list[tuple[str, str]], selected: set[str], console: Console
) -> set[str] | None:
    """One toggle per call, matching the numbered menu this replaces: scripted
    runs and every existing test type a single number, not a list."""
    table = Table(title=title, show_header=False, box=None, padding=(0, 2))
    for index, (value, label) in enumerate(choices, start=1):
        # Escaped: rich would read a bare [x] as a style tag and fail on it.
        marker = "\\[x]" if value in selected else "\\[ ]"
        table.add_row(f"{index}.", marker, label)
    console.print(table)
    answer = Prompt.ask(
        "Toggle which number? (Enter to go back)",
        default="",
        show_default=False,
        console=console,
    ).strip()
    index = _parse_selection(answer, len(choices), console)
    if index is None:
        return None
    return selected ^ {choices[index][0]}
