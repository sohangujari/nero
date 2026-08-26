import asyncio
import base64
import logging
import shlex
from collections.abc import Callable
from pathlib import Path

import httpx
import litellm
from rich.console import Console
from rich.markup import escape

from nero.llm.ollama_adapter import OllamaModelError

logger = logging.getLogger("nero.chat")

EXIT_COMMANDS = {"exit", "quit"}

DEFAULT_IMAGE_QUESTION = "What's in this image?"
IMAGE_EXTENSIONS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".gif": "gif", ".webp": "webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024

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


def _validate_image(path_str: str) -> str | None:
    """Returns an error message if `path_str` isn't a sendable image, else None."""
    path = Path(path_str)
    if not path.is_file():
        return f"No such file: {path_str}"
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return f"Unsupported image type: {path.suffix or path_str}"
    if path.stat().st_size > MAX_IMAGE_BYTES:
        return f"Image too large (max {MAX_IMAGE_BYTES // (1024 * 1024)} MB): {path_str}"
    return None


def _build_image_message(path_str: str, question: str) -> dict:
    path = Path(path_str)
    fmt = IMAGE_EXTENSIONS[path.suffix.lower()]
    data = base64.b64encode(path.read_bytes()).decode()
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{data}"}},
        ],
    }


def find_compaction_cut(messages: list[dict]) -> int | None:
    """The largest `cut` such that dropping messages[:cut] leaves a clean
    boundary: messages[cut] is a user message and messages[cut - 1] is an
    assistant message carrying no tool_calls.

    Scanning from the end means a cut point that would split a tool-call
    sequence is skipped automatically: right after an assistant message with
    tool_calls comes a `tool` role message, never `user`, so that candidate
    fails and the search keeps walking left until it clears the whole
    tool_calls/tool group. Returns None if no valid boundary exists at all —
    per spec, compaction must be skipped rather than guess.
    """
    for cut in range(len(messages) - 1, 0, -1):
        before, after = messages[cut - 1], messages[cut]
        if after.get("role") == "user" and before.get("role") == "assistant" and not before.get("tool_calls"):
            return cut
    return None


def _transcript_text(messages: list[dict]) -> str:
    """Flatten messages to plain "role: content" lines for summarization.
    Only string content is included — tool-call payloads (content is always
    "" for those, by convention) carry nothing worth summarizing."""
    lines = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content:
            lines.append(f"{message.get('role')}: {content}")
    return "\n".join(lines)


def summarize_messages(client, messages: list[dict]) -> str | None:
    """One plain, tool-free LLM call on `client` to summarize `messages`.

    Never raises: compaction is a nice-to-have, and losing the turn over a
    failed summary call would be far worse than just skipping compaction.
    Returns None on any failure.
    """
    try:
        prompt = (
            "Summarize the following conversation concisely, preserving "
            "important facts, decisions, and context that will be needed "
            "later:\n\n" + _transcript_text(messages)
        )

        chunks: list[str] = []

        async def _run() -> None:
            async for text in client.stream_chat([{"role": "user", "content": prompt}], []):
                chunks.append(text)

        asyncio.run(_run())
        summary = "".join(chunks).strip()
        return summary or None
    except Exception as exc:  # noqa: BLE001 — never raise; see docstring
        logger.warning("Session compaction summary failed: %s", exc)
        return None


def compact_messages(messages: list[dict], client, threshold: int) -> tuple[list[dict], int] | None:
    """Replace the oldest span of `messages` with one summary message, if
    `messages` is over `threshold` and a clean cut boundary exists.

    Returns (new_messages, dropped_count), or None if compaction did not fire
    (disabled, under threshold, no valid boundary, or the summary call
    failed) — in every None case the caller must leave `messages` untouched.
    """
    if not threshold or len(messages) <= threshold:
        return None
    cut = find_compaction_cut(messages)
    if cut is None:
        return None
    summary = summarize_messages(client, messages[:cut])
    if summary is None:
        return None
    new_messages = [
        {"role": "user", "content": f"[Earlier conversation summary]\n{summary}"},
        *messages[cut:],
    ]
    return new_messages, cut


def _supports_vision(model: str) -> bool:
    """Never-raise: unknown/custom models assume capable, since custom rows
    have no catalog truth — let the provider error surface through the
    existing handlers instead."""
    try:
        return litellm.supports_vision(model=model)
    except Exception:  # noqa: BLE001
        return True


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
        registry=None,
        security=None,
        compact_after_messages: int = 0,
    ):
        self.client = client
        self.console = console
        self.assistant_name = assistant_name
        self.input_fn = input_fn or console.input
        self.history = history
        self.fallback_clients = fallback_clients or []
        # Optional: absent (voice loop, most tests) means no per-turn taint
        # reset and no session limits — behavior is unchanged from before
        # this feature existed.
        self.registry = registry
        self.security = security
        # 0 (the default) disables compaction entirely — unchanged behavior
        # for every caller that doesn't pass this.
        self.compact_after_messages = compact_after_messages
        self.turns_used = 0
        self.cost_usd = 0.0
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

            if self.registry is not None:
                self.registry.reset_turn()

            limit_message = self._limit_message()
            if limit_message is not None:
                self.console.print(f"[red]{limit_message}[/red]")
                continue

            if text.startswith("/image "):
                image_turn = self._image_turn(text)
                if image_turn is None:
                    continue
                user_message, history_text = image_turn
            else:
                user_message = {"role": "user", "content": text}
                history_text = text

            self.turns_used += 1
            used_client = self.client
            self.messages.append(user_message)
            self._maybe_compact()
            # Recomputed after compaction rather than captured before the
            # append: user_message is always the last element (compaction
            # only ever touches a prefix of self.messages), so this is valid
            # whether or not compaction actually fired this turn.
            turn_start = len(self.messages) - 1
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
                        self.messages.append(user_message)
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
                            used_client = fallback_client
                            break
                        except TRANSIENT_ERRORS:
                            # Exceptions from the last attempt propagate to the
                            # existing outer handlers unchanged.
                            if i == len(self.fallback_clients) - 1:
                                raise
                self.cost_usd += getattr(used_client, "last_turn_cost", 0.0) or 0.0
                self.console.print()
                if self.history is not None and self.messages[-1].get("role") == "assistant":
                    # Persist only on success — past every rollback branch below.
                    # history_text is the user text (plain, or "[image: ...] "
                    # for an image turn — history is text-only); messages[-1] is
                    # the final assistant text turn — except when MAX_TOOL_ROUNDS
                    # is exhausted, which leaves a tool message last; the role
                    # check excludes that case.
                    self.history.append_turn(
                        history_text,
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

    def _maybe_compact(self) -> None:
        result = compact_messages(self.messages, self.client, self.compact_after_messages)
        if result is None:
            return
        self.messages, dropped = result
        # escape(): the literal "[compacted ...]" text must not be parsed as
        # rich markup itself (only the surrounding [dim]...[/dim] tags should be).
        self.console.print(f"[dim]{escape(f'[compacted {dropped} earlier messages]')}[/dim]")

    def _limit_message(self) -> str | None:
        """None if the turn may proceed, else the message to show instead.

        Refuses the turn (the user can still type `exit`) rather than ending
        the session — a hard exit would be a surprising way to hit a budget
        cap. 0 means unlimited on both axes, so a user who never touched
        `security.*` sees no change at all.
        """
        if self.security is None:
            return None
        max_turns = self.security.max_turns_per_session
        if max_turns and self.turns_used >= max_turns:
            return f"Turn limit reached ({max_turns} per session). Type exit to leave."
        max_cost = self.security.max_cost_usd_per_session
        if max_cost and self.cost_usd >= max_cost:
            return f"Cost limit reached (${max_cost:g} per session). Type exit to leave."
        return None

    def _image_turn(self, text: str) -> tuple[dict, str] | None:
        """Parse, validate, and gate an `/image <path> [question]` command.

        Returns (user_message, history_text) ready to send, or None after
        printing an explanatory message — in which case no turn is consumed.
        """
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            self.console.print(f"[red]Could not parse that command: {escape(str(exc))}[/red]")
            return None
        if len(parts) < 2:
            self.console.print(escape("Usage: /image <path> [question]"), style="red")
            return None
        path_str = parts[1]
        question = " ".join(parts[2:]) or DEFAULT_IMAGE_QUESTION

        error = _validate_image(path_str)
        if error:
            self.console.print(f"[red]{escape(error)}[/red]")
            return None

        if self.client.provider == "ollama":
            self.console.print(
                "[red]Image input isn't supported on the local Ollama path yet.[/red]"
            )
            return None
        if not _supports_vision(self.client.litellm_model):
            self.console.print(
                "[yellow]This model may not support images — sending anyway.[/yellow]"
            )

        user_message = _build_image_message(path_str, question)
        history_text = f"[image: {Path(path_str).name}] {question}"
        return user_message, history_text

    def _print_chunk(self, text: str) -> None:
        # Raw print, not console.print: streamed chunks must not be markup-parsed
        # or line-wrapped mid-token.
        print(text, end="", flush=True, file=self.console.file)

    def _goodbye(self) -> None:
        self.console.print(f"\n[dim]{self.assistant_name} signing off. Bye![/dim]")
