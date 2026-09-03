import asyncio
import base64
import logging
import shlex
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import litellm
from rich.console import Console
from rich.markup import escape

from nero.llm.ollama_adapter import OllamaModelError
from nero.llm.routing import SessionStats, order_chain
from nero.spinner import Spinner

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


def _client_key(client) -> tuple[str, str]:
    """(provider, model) for stats, tolerating clients that expose neither.

    Every other client access in this file is a getattr with a default —
    a fake or adapter without these attributes must not turn a working turn
    into an exception the catch-all then rolls back.
    """
    return (
        getattr(client, "provider", None) or "unknown",
        getattr(client, "model", None) or "unknown",
    )


def find_compaction_cut(messages: list[dict], keep_recent: int) -> int | None:
    """The largest `cut` that both leaves a clean boundary and keeps at least
    `keep_recent` messages: messages[cut] is a user message and
    messages[cut - 1] is an assistant message carrying no tool_calls.

    `keep_recent` is what stops compaction from eating the whole conversation.
    Without it the largest valid cut is almost always "everything but the last
    message", so the model would lose the live thread it is mid-way through
    and see only a summary — technically correct, conversationally useless.

    Scanning from the end means a cut point that would split a tool-call
    sequence is skipped automatically: right after an assistant message with
    tool_calls comes a `tool` role message, never `user`, so that candidate
    fails and the search keeps walking left until it clears the whole
    tool_calls/tool group. Returns None if no valid boundary exists at all —
    compaction must be skipped rather than guess.
    """
    for cut in range(len(messages) - keep_recent, 0, -1):
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
    # Keep a recent window so compaction trims history without erasing the
    # conversation the user is actually having.
    cut = find_compaction_cut(messages, keep_recent=max(10, threshold // 2))
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
        route_by: str = "off",
        quality_rank: list[str] | None = None,
        health_check: bool = True,
        primary_api_keys: list[str] | None = None,
        coding_client=None,
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
        # v1.6.0 routing/health/key-rotation — all default to today's
        # behavior (off / all keys healthy / no rotation) for every caller
        # that doesn't pass these.
        self.route_by = route_by
        self.quality_rank = quality_rank or []
        self.health_check = health_check
        self.stats = SessionStats()
        # Set by _prefix() and stopped by _print_chunk() or the turn's finally.
        self._spinner: Spinner | None = None
        # Keys available for the PRIMARY provider only; rotation never
        # touches a /code-routed client. Slot 0 is whatever `client` already
        # holds, so rotation always advances from there.
        self.primary_api_keys = primary_api_keys or []
        self._key_slot = 0
        # Resolved once at startup (see cli._resolve_coding_client); None
        # means /code falls back to the primary rather than erroring.
        self.coding_client = coding_client
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

            is_code_turn = False
            if text == "/image" or text.startswith("/image "):
                image_turn = self._image_turn(text)
                if image_turn is None:
                    continue
                user_message, history_text = image_turn
            elif text == "/code" or text.startswith("/code "):
                code_turn = self._code_turn(text)
                if code_turn is None:
                    continue
                user_message, history_text = code_turn
                is_code_turn = True
            else:
                user_message = {"role": "user", "content": text}
                history_text = text

            self.turns_used += 1
            # /code routes this one turn to llm.coding_model; every other
            # turn (and /code with nothing resolved) uses the primary. Key
            # rotation is scoped to the real primary only — a one-off routed
            # client never rotates.
            primary_client = self.coding_client if (is_code_turn and self.coding_client) else self.client
            allow_rotation = primary_client is self.client
            if is_code_turn:
                if self.coding_client:
                    self.console.print(
                        f"[dim]Routing to {primary_client.model} via "
                        f"{primary_client.provider} for this turn.[/dim]"
                    )
                else:
                    self.console.print(
                        "[dim]No coding model configured — using the primary "
                        "model for this turn.[/dim]"
                    )
            used_client = primary_client
            self.messages.append(user_message)
            self._maybe_compact()
            # Recomputed after compaction rather than captured before the
            # append: user_message is always the last element (compaction
            # only ever touches a prefix of self.messages), so this is valid
            # whether or not compaction actually fired this turn.
            turn_start = len(self.messages) - 1
            self._prefix()
            try:
                try:
                    started = time.monotonic()
                    try:
                        primary_client.send(self.messages, on_text=self._print_chunk)
                    except litellm.exceptions.RateLimitError:
                        # One rotation per turn, primary only, before the
                        # fallback chain is even considered — see v1.6.0 spec.
                        rotated_key = self._next_key_for_rotation() if allow_rotation else None
                        if rotated_key is None:
                            raise
                        primary_client.api_key = rotated_key
                        del self.messages[turn_start:]
                        self.messages.append(user_message)
                        self.console.print(
                            "\n[yellow]Rate limited — retrying "
                            f"{primary_client.model} via {primary_client.provider} "
                            "with another key.[/yellow]"
                        )
                        self._prefix()
                        started = time.monotonic()
                        primary_client.send(self.messages, on_text=self._print_chunk)
                    provider_key, model_key = _client_key(primary_client)
                    self.stats.record_success(provider_key, model_key)
                    self.stats.record_latency(
                        provider_key, model_key, time.monotonic() - started
                    )
                except TRANSIENT_ERRORS:
                    self.stats.record_failure(*_client_key(primary_client))
                    if not self.fallback_clients:
                        raise
                    # Walk the chain in routed/health-filtered order; first
                    # success wins. Each attempt rolls back first — partial
                    # tool/assistant turns from the failed stream must not leak
                    # into the retry, and a failed fallback attempt can leave
                    # its own partial state.
                    candidates = self._ordered_fallback_candidates()
                    for i, fallback_client in enumerate(candidates):
                        del self.messages[turn_start:]
                        self.messages.append(user_message)
                        self.console.print(
                            "\n[yellow]Primary model unreachable — retrying with "
                            f"{fallback_client.model} via "
                            f"{fallback_client.provider}.[/yellow]"
                        )
                        self._prefix()
                        try:
                            started = time.monotonic()
                            fallback_client.send(self.messages, on_text=self._print_chunk)
                            used_client = fallback_client
                            self.stats.record_success(*_client_key(fallback_client))
                            self.stats.record_latency(
                                *_client_key(fallback_client), time.monotonic() - started
                            )
                            break
                        except TRANSIENT_ERRORS:
                            self.stats.record_failure(*_client_key(fallback_client))
                            # Exceptions from the last attempt propagate to the
                            # existing outer handlers unchanged.
                            if i == len(candidates) - 1:
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
                model = getattr(primary_client, "model", None) or "that model"
                self.console.print(
                    f"\n[red]The provider has no model called [bold]{model}[/bold].[/red] "
                    "Check the model name in [bold]nero config[/bold]."
                )
            except litellm.exceptions.RateLimitError:
                # Reached only once rotation and the fallback chain are spent.
                # Distinct from the connection branch below on purpose: the
                # provider answered, so "check your connection" is wrong advice.
                del self.messages[turn_start:]
                self.console.print(
                    "\n[yellow]The provider is rate-limiting you.[/yellow] Wait a "
                    "moment and try again, or switch models with [bold]nero config[/bold]."
                )
            except (
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.Timeout,
                httpx.HTTPError,  # direct-Ollama path talks httpx, not litellm
            ):
                logger.debug("provider unreachable", exc_info=True)
                del self.messages[turn_start:]
                provider = getattr(primary_client, "provider", None)
                api_base = getattr(primary_client, "api_base", None)
                if provider == "ollama":
                    hint = " If you're using Ollama, make sure it's running ([bold]ollama serve[/bold])."
                elif api_base:
                    hint = f" Check that a server is running at [bold]{api_base}[/bold]."
                else:
                    hint = " Check your connection and try again."
                self.console.print(f"\n[red]Could not reach the model provider.[/red]{hint}")
            except Exception as exc:  # noqa: BLE001 — the REPL must never crash on a turn
                logger.debug("turn failed", exc_info=True)
                del self.messages[turn_start:]
                self.console.print(
                    f"\n[red]Something went wrong with that turn:[/red] {exc}\n"
                    "You can retry, or type exit to quit."
                )
            finally:
                # Every branch above is reached without a single chunk having
                # arrived, so _print_chunk never got to stop the thread.
                if self._spinner is not None:
                    self._spinner.stop()
                    self._spinner = None

    def _maybe_compact(self) -> None:
        # Compaction is a full LLM round-trip of its own, run before the reply
        # prefix is even printed -- the silent half of a slow first turn.
        # Started unconditionally: compact_messages returns immediately when it
        # is under threshold, and the spinner draws nothing in its first 100 ms.
        spinner = Spinner(self.console.file)
        spinner.start()
        try:
            result = compact_messages(self.messages, self.client, self.compact_after_messages)
        finally:
            spinner.stop()
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

    def _code_turn(self, text: str) -> tuple[dict, str] | None:
        """Parse `/code <request>` — routes this one turn to llm.coding_model.

        Returns (user_message, history_text) ready to send, or None after
        printing a usage message — in which case no turn is consumed.
        """
        request = text.removeprefix("/code").strip()
        if not request:
            self.console.print(escape("Usage: /code <request>"), style="red")
            return None
        return {"role": "user", "content": request}, request

    def _next_key_for_rotation(self) -> str | None:
        """The next untried key for the primary provider, or None if there
        isn't one. Advances an internal slot pointer so a key already known
        to be rate-limited isn't retried on a later turn."""
        if self._key_slot + 1 >= len(self.primary_api_keys):
            return None
        self._key_slot += 1
        return self.primary_api_keys[self._key_slot]

    def _ordered_fallback_candidates(self) -> list:
        """The fallback chain, ordered by `route_by` and filtered by health —
        except an unhealthy entry is never dropped when it's the only
        candidate left (an unhealthy option beats no option)."""
        keyed = [(_client_key(client), client) for client in self.fallback_clients]
        order = order_chain([key for key, _ in keyed], self.route_by, self.stats,
                            self.quality_rank)
        # Index by position, not by key: two chain entries may share a
        # (provider, model) pair, and a dict would silently drop one.
        remaining = list(keyed)
        ordered_clients = []
        for key in order:
            for i, (candidate_key, client) in enumerate(remaining):
                if candidate_key == key:
                    ordered_clients.append(client)
                    del remaining[i]
                    break
        if not self.health_check:
            return ordered_clients
        healthy = [
            client for client in ordered_clients
            if not self.stats.is_unhealthy(*_client_key(client))
        ]
        return healthy or ordered_clients

    def _prefix(self) -> None:
        """Print the assistant prompt and spin until the first token lands."""
        self.console.print(f"[bold magenta]{self.assistant_name}>[/bold magenta] ", end="")
        self._spinner = Spinner(self.console.file)
        self._spinner.start()

    def _print_chunk(self, text: str) -> None:
        if self._spinner is not None:
            self._spinner.stop()
        # Raw print, not console.print: streamed chunks must not be markup-parsed
        # or line-wrapped mid-token.
        print(text, end="", flush=True, file=self.console.file)

    def _goodbye(self) -> None:
        self.console.print(f"\n[dim]{self.assistant_name} signing off. Bye![/dim]")
