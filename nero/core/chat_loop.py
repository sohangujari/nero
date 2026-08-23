import logging
from collections.abc import Callable

import httpx
import litellm
from rich.console import Console
from rich.markup import escape

from nero.llm.ollama_adapter import OllamaModelError

logger = logging.getLogger("nero.chat")

EXIT_COMMANDS = {"exit", "quit"}

# Errors worth one retry on a fallback model: the provider is unreachable or
# overloaded, not wrong about anything. Auth errors, 404/NotFound, and
# OllamaModelError are deliberately excluded — the server answered, so
# retrying elsewhere would hide a config bug rather than route around a blip.
TRANSIENT_ERRORS = (
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.Timeout,
    litellm.exceptions.RateLimitError,
    litellm.exceptions.InternalServerError,
    httpx.HTTPError,
)


class ChatLoop:
    """The interactive REPL: reads input, streams Claude's replies, keeps history."""

    def __init__(
        self,
        client,
        console: Console,
        assistant_name: str,
        input_fn: Callable[[str], str] | None = None,
        history=None,
        fallback_clients: list | None = None,
    ):
        self.client = client
        self.console = console
        self.assistant_name = assistant_name
        self.input_fn = input_fn or console.input
        self.history = history
        self.fallback_clients = fallback_clients or []
        # Seed from persisted history when memory is on; else a blank session.
        self.messages: list[dict] = history.recent() if history else []

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
                try:
                    self.client.send(self.messages, on_text=self._print_chunk)
                except TRANSIENT_ERRORS:
                    if not self.fallback_clients:
                        raise
                    # Walk the chain in priority order; first success wins. Each
                    # attempt rolls back first — partial tool/assistant turns
                    # from the failed stream must not leak into the retry, and a
                    # failed fallback attempt can leave its own partial state.
                    for i, fallback_client in enumerate(self.fallback_clients):
                        del self.messages[turn_start:]
                        self.messages.append({"role": "user", "content": text})
                        self.console.print(
                            "\n[yellow]Primary model unreachable — retrying with "
                            f"{fallback_client.model} via "
                            f"{fallback_client.provider}.[/yellow]"
                        )
                        self.console.print(
                            f"[bold magenta]{self.assistant_name}>[/bold magenta] ", end=""
                        )
                        try:
                            fallback_client.send(self.messages, on_text=self._print_chunk)
                            break
                        except TRANSIENT_ERRORS:
                            # Exceptions from the last attempt propagate to the
                            # existing outer handlers unchanged.
                            if i == len(self.fallback_clients) - 1:
                                raise
                self.console.print()
                if self.history is not None and self.messages[-1].get("role") == "assistant":
                    # Persist only on success — past every rollback branch below.
                    # messages[turn_start] is the user text; messages[-1] is the
                    # final assistant text turn — except when MAX_TOOL_ROUNDS is
                    # exhausted, which leaves a tool message last; the role check
                    # excludes that case.
                    self.history.append_turn(
                        self.messages[turn_start]["content"],
                        self.messages[-1]["content"],
                    )
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
            except litellm.exceptions.NotFoundError:
                # The most likely custom-endpoint failure: a model id the server
                # doesn't have, or a base URL with the wrong /v1 shape for its
                # dialect.
                del self.messages[turn_start:]
                model = getattr(self.client, "model", None) or "that model"
                self.console.print(
                    f"\n[red]The provider has no model called [bold]{model}[/bold].[/red] "
                    "Check the model name in [bold]nero config[/bold]."
                )
            except (
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.Timeout,
                httpx.HTTPError,  # direct-Ollama path talks httpx, not litellm
            ):
                del self.messages[turn_start:]
                provider = getattr(self.client, "provider", None)
                api_base = getattr(self.client, "api_base", None)
                if provider == "ollama":
                    hint = " If you're using Ollama, make sure it's running ([bold]ollama serve[/bold])."
                elif api_base:
                    hint = f" Check that a server is running at [bold]{api_base}[/bold]."
                else:
                    hint = " Check your connection and try again."
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
