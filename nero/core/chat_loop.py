import logging
from collections.abc import Callable

import httpx
import litellm
from rich.console import Console
from rich.markup import escape

from nero.llm.ollama_adapter import OllamaModelError

logger = logging.getLogger("nero.chat")

EXIT_COMMANDS = {"exit", "quit"}


class ChatLoop:
    """The interactive REPL: reads input, streams Claude's replies, keeps history."""

    def __init__(
        self,
        client,
        console: Console,
        assistant_name: str,
        input_fn: Callable[[str], str] | None = None,
    ):
        self.client = client
        self.console = console
        self.assistant_name = assistant_name
        self.input_fn = input_fn or console.input
        # Session-only history; nothing is persisted across runs (Phase 1 scope).
        self.messages: list[dict] = []

    def run(self) -> None:
        self.console.print(
            f"[bold]{self.assistant_name}[/bold] is listening. "
            "Type [dim]exit[/dim] or press Ctrl+C to leave.\n"
        )
        while True:
            try:
                user_input = self.input_fn("[bold cyan]you>[/bold cyan] ")
            except (KeyboardInterrupt, EOFError):
                self._goodbye()
                return

            text = user_input.strip()
            if not text:
                continue
            if text.lower() in EXIT_COMMANDS:
                self._goodbye()
                return

            turn_start = len(self.messages)
            self.messages.append({"role": "user", "content": text})
            self.console.print(f"[bold magenta]{self.assistant_name}>[/bold magenta] ", end="")
            try:
                self.client.send(self.messages, on_text=self._print_chunk)
                self.console.print()
                # Inspection hook (visible under `nero --debug`): history must
                # contain only clean text turns and structured tool pairs.
                logger.debug("history after turn: %r", self.messages)
            except KeyboardInterrupt:
                del self.messages[turn_start:]
                self.console.print("\n[dim]Interrupted.[/dim]")
            except litellm.exceptions.AuthenticationError:
                del self.messages[turn_start:]
                self.console.print(
                    "\n[red]Your API key was rejected.[/red] "
                    "Update it with [bold]nero config[/bold]."
                )
            except OllamaModelError as exc:
                # Ordered before the connection branch on purpose: the server
                # answered, so "is Ollama running?" is the wrong question.
                del self.messages[turn_start:]
                self.console.print(f"\n[red]{escape(str(exc))}[/red]")
            except (
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.Timeout,
                httpx.HTTPError,  # direct-Ollama path talks httpx, not litellm
            ):
                del self.messages[turn_start:]
                hint = (
                    " If you're using Ollama, make sure it's running ([bold]ollama serve[/bold])."
                    if getattr(self.client, "provider", None) == "ollama"
                    else " Check your connection and try again."
                )
                self.console.print(f"\n[red]Could not reach the model provider.[/red]{hint}")
            except Exception as exc:  # noqa: BLE001 — the REPL must never crash on a turn
                del self.messages[turn_start:]
                self.console.print(
                    f"\n[red]Something went wrong with that turn:[/red] {exc}\n"
                    "You can retry, or type exit to quit."
                )

    def _print_chunk(self, text: str) -> None:
        # Raw print, not console.print: streamed chunks must not be markup-parsed
        # or line-wrapped mid-token.
        print(text, end="", flush=True, file=self.console.file)

    def _goodbye(self) -> None:
        self.console.print(f"\n[dim]{self.assistant_name} signing off. Bye![/dim]")
