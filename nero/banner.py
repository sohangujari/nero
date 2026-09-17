"""The start-up banner for a universal `nero` session.

One block, printed once, then out of the way. A terminal assistant is judged on
how fast you can start typing, so this says what is answering and what the
session can reach, and stops.

Narrow terminals get the wordmark instead of the block letters rather than a
wrapped, broken one — a banner that mangles itself is worse than no banner.
"""

from __future__ import annotations

from rich.console import Console

# The product is "Nero Agent", so the wordmark says so. Set as a lockup rather
# than one line: the same block font across "NERO AGENT" measures 80 columns,
# which wraps and mangles itself on the 80-column terminal it only just fits.
# NERO reads at 36, and AGENT sits under its right edge.
#
# This is the product name and does not follow `assistant.name` — renaming your
# assistant to "Jarvis" changes what it calls itself, not what it is.
WORDMARK = r"""
 ███╗   ██╗███████╗██████╗  ██████╗
 ████╗  ██║██╔════╝██╔══██╗██╔═══██╗
 ██╔██╗ ██║█████╗  ██████╔╝██║   ██║
 ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║
 ██║ ╚████║███████╗██║  ██║╚██████╔╝
 ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝
                       A  G  E  N  T
"""
MIN_WIDTH = 40


def render(
    console: Console,
    assistant_name: str,
    model: str,
    provider: str,
    mode: str,
    skills: int,
    channels: list[str],
    dashboard_url: str | None,
) -> None:
    """Print the banner and what this session can do.

    `channels` is what is actually answering, not what exists — a name here
    means a message sent there gets a reply now.
    """
    # "Nero Agent" is the product; `assistant_name` is what it answers to. They
    # are only the same until someone renames their assistant, and the banner
    # should keep saying which program this is.
    named = "" if assistant_name.strip().lower() == "nero" else f", answering as {assistant_name}"
    wide = console.width >= MIN_WIDTH
    if wide:
        console.print(f"[bold cyan]{WORDMARK.rstrip()}[/bold cyan]")
        console.print(f"[dim] your personal AI assistant{named}[/dim]\n")
    else:
        console.print(f"[bold cyan]Nero Agent[/bold cyan][dim]{named}[/dim]\n")

    offline = " [yellow](offline)[/yellow]" if mode == "offline" else ""
    # "terminal" is always true and always first: it is the thing the reader is
    # looking at, and leaving it out makes the list read as "instead of here".
    rows = [
        ("model", f"{model} [dim]via[/dim] {provider}{offline}"),
        ("skills", f"{skills} available"),
        ("answering", ", ".join(["terminal", *channels])),
    ]
    if dashboard_url:
        rows.append(("dashboard", dashboard_url))

    for label, value in rows:
        if wide:
            console.print(f"  [dim]{label:<9}[/dim]  {value}")
        else:
            # No column alignment and no indent: at this width the padding is
            # what pushes a URL onto a second line, and a wrapped URL cannot be
            # clicked or copied in one go.
            console.print(f"[dim]{label}[/dim] {value}", overflow="ignore", crop=False)

    if wide:
        console.print(
            "\n[dim]  /image <path>  send a picture   ·   /code <task>  use the coding model\n"
            "  exit or Ctrl+C to leave[/dim]\n"
        )
    else:
        console.print("\n[dim]exit or Ctrl+C to leave[/dim]\n")
