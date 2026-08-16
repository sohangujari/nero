import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import litellm

from nero.config.schema import LLMConfig
from nero.llm import ollama, providers
from nero.llm.ollama_adapter import (
    ToolCallOutcome,
    ToolCallRequest,
    classify_tool_call,
    ollama_chat,
)
from nero.skills.registry import SkillRegistry

logger = logging.getLogger("nero.llm")

MAX_TOKENS = 8192
# Safety bound on tool-call round trips within a single user turn.
MAX_TOOL_ROUNDS = 10

APOLOGY = "I didn't quite catch that — could you rephrase?"

litellm.suppress_debug_info = True


@dataclass
class PendingToolCall:
    id: str
    name: str
    raw_arguments: str  # JSON string, for the assistant-turn echo in history
    arguments: dict | None  # parsed; None means the arguments were unparseable


@dataclass
class RoundResult:
    """Outcome of one completion round, set by stream_chat after iteration."""

    content: str | None
    tool_calls: list[PendingToolCall]


class _JsonGate:
    """Streams normal text live, but holds back output that might be a
    tool-call JSON blob until the round resolves.

    Raw tool-call JSON must never be shown to the user; legitimate replies
    that merely start with "{" get flushed once the round proves harmless.

    With `hold_all=True` (the ollama path), *everything* is buffered until the
    round resolves: small local models mix filler text with tool calls, and
    content accompanying a tool call must not be shown as if it were an answer.

    `streamed` records whether anything reached the user, so the caller knows
    whether it still needs to emit the resolved (possibly cleaned) text.
    """

    def __init__(self, on_text: Callable[[str], None], hold_all: bool = False):
        self._on_text = on_text
        self._buffer: list[str] = []
        self._state = "holding" if hold_all else "undecided"  # undecided | streaming | holding
        self._streamed = False

    def feed(self, text: str) -> None:
        if self._state == "streaming":
            self._emit(text)
            return
        self._buffer.append(text)
        if self._state == "undecided":
            head = "".join(self._buffer).lstrip()
            if head:
                if head.startswith(("{", "[")):
                    self._state = "holding"
                else:
                    self.flush()

    def _emit(self, text: str) -> None:
        self._streamed = True
        self._on_text(text)

    @property
    def streamed(self) -> bool:
        return self._streamed

    def flush(self) -> None:
        buffered = self._buffer
        self._buffer = []
        self._state = "streaming"
        for text in buffered:
            self._emit(text)

    def discard(self) -> None:
        self._buffer.clear()


class LLMClient:
    """Provider-agnostic chat client: cloud providers via LiteLLM,
    ollama via its native /api/chat endpoint (LiteLLM's ollama path routes to
    the legacy /api/generate endpoint and mis-maps tool_calls into content)."""

    def __init__(
        self,
        config: LLMConfig,
        assistant_name: str,
        registry: SkillRegistry,
        api_key: str | None = None,
        ollama_base_url: str = ollama.BASE_URL,
    ):
        self.config = config
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url
        self.registry = registry
        # Two separate concerns, stated separately on purpose: (1) Nero is a
        # general-purpose assistant, (2) it *additionally* has tools, gated to
        # explicit requests. An earlier tool-gating fix collapsed these into a
        # tool-only description, and small local models then refused everything
        # that wasn't an app request. Lead with the general capability.
        #
        # Kept deliberately short and non-nuanced: small local models (phi4-mini
        # et al) follow terse, concrete instructions far better than long ones.
        # No skill is named here — per-skill detail lives in SkillMeta.description,
        # which reaches the model as schema. That keeps this prompt a constant
        # length as skills are added, and avoids the over-description that made
        # the model fire tools on unrelated questions.
        self.system_prompt = (
            f"You are {assistant_name}, a helpful general-purpose AI assistant "
            "running in the user's terminal.\n\n"
            "Answer any question normally: maths, facts, explanations, jokes, "
            "casual conversation. You are a normal assistant and are never limited "
            "to one topic. Never say that you can only do one kind of task.\n\n"
            "You also have a few tools for doing things on the user's computer. "
            "Call a tool only when the user explicitly asks for that action — for "
            'example "open Calculator" or "what\'s the weather in Paris". For any '
            "other message, including ordinary questions, do not call any tool: "
            "just reply with text.\n\n"
            "When you are not calling a tool, reply with plain text only — never "
            "write tool-call JSON as text. Keep replies concise."
        )
        self._last_round: RoundResult | None = None

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def litellm_model(self) -> str:
        """The config's provider+model as a LiteLLM model string.

        Providers LiteLLM routes bare (claude, openai) carry prefix "". The
        startswith guard keeps a user who typed the fully-qualified name —
        "dashscope/qwen3-max" — from getting it applied twice, and leaves
        nested names like "anthropic/claude-sonnet-4.6" under openrouter/ intact.
        """
        prefix = providers.get(self.config.provider).prefix
        model = self.config.model
        if prefix and not model.startswith(prefix):
            return prefix + model
        return model

    def _tool_definitions(self) -> list[dict]:
        return self.registry.tool_definitions()

    async def stream_chat(self, messages: list, tools: list) -> AsyncIterator[str]:
        """One completion round: yields display-text deltas as they arrive.

        The round's full outcome (content + tool calls) lands in
        `self._last_round` for the caller to inspect after iteration.
        """
        if self.config.provider == "ollama":
            async for text in self._ollama_chat(messages, tools):
                yield text
        else:
            async for text in self._litellm_chat(messages, tools):
                yield text

    async def _litellm_chat(self, messages: list, tools: list) -> AsyncIterator[str]:
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = await litellm.acompletion(
            model=self.litellm_model,
            messages=[{"role": "system", "content": self.system_prompt}, *messages],
            tools=tools or None,
            max_tokens=MAX_TOKENS,
            stream=True,
            **kwargs,
        )
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is not None and getattr(delta, "content", None):
                yield delta.content
        full = litellm.stream_chunk_builder(chunks, messages=messages)
        message = full.choices[0].message if full and full.choices else None
        self._last_round = RoundResult(
            content=getattr(message, "content", None),
            tool_calls=self._pending_from_message(message),
        )

    @staticmethod
    def _pending_from_message(message) -> list[PendingToolCall]:
        if message is None:
            return []
        pending = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            raw = call.function.arguments or "{}"
            try:
                parsed = json.loads(raw)
                arguments = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                arguments = None
            pending.append(
                PendingToolCall(
                    id=call.id or f"call_{index}",
                    name=call.function.name,
                    raw_arguments=raw,
                    arguments=arguments,
                )
            )
        return pending

    async def _ollama_chat(self, messages: list, tools: list) -> AsyncIterator[str]:
        """Native Ollama path: /api/chat directly, no LiteLLM translation."""
        request_messages = [{"role": "system", "content": self.system_prompt}]
        request_messages += [self._to_ollama_message(m) for m in messages]
        model = self.config.model.removeprefix("ollama/")
        parts: list[str] = []
        calls = []
        async for response in ollama_chat(self.ollama_base_url, model, request_messages, tools):
            if response.tool_calls:
                calls.extend(response.tool_calls)
            if response.content:
                parts.append(response.content)
                yield response.content
        self._last_round = RoundResult(
            content="".join(parts) or None,
            tool_calls=[
                PendingToolCall(
                    id=f"call_{index}",
                    name=call.name,
                    raw_arguments=json.dumps(call.arguments),
                    arguments=call.arguments,
                )
                for index, call in enumerate(calls)
            ],
        )

    @staticmethod
    def _to_ollama_message(message: dict) -> dict:
        """History is stored OpenAI-style; Ollama wants dict arguments and
        plain tool messages."""
        if message.get("tool_calls"):
            return {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": _arguments_as_dict(call["function"]["arguments"]),
                        }
                    }
                    for call in message["tool_calls"]
                ],
            }
        if message.get("role") == "tool":
            return {"role": "tool", "content": message.get("content", "")}
        return {"role": message["role"], "content": message.get("content") or ""}

    def send(self, messages: list[dict], on_text: Callable[[str], None]) -> None:
        """Run one user turn to completion, mutating `messages` in place."""
        asyncio.run(self._run_turn(messages, on_text))

    async def _run_turn(self, messages: list[dict], on_text: Callable[[str], None]) -> None:
        tool_definitions = self._tool_definitions()
        # Ollama rounds are fully buffered: local models mix filler text with
        # tool calls, and accompanying content must not print as an answer.
        # Cloud rounds keep live streaming (their preamble text is intentional).
        hold_all = self.config.provider == "ollama"
        # Every known skill, not just the available ones: a call naming a
        # disabled skill must classify as a tool call so the registry can refuse
        # and audit it, rather than being discarded as MALFORMED.
        tool_names = self.registry.known_names()
        for _ in range(MAX_TOOL_ROUNDS):
            gate = _JsonGate(on_text, hold_all=hold_all)
            self._last_round = None
            async for text in self.stream_chat(messages, tool_definitions):
                gate.feed(text)
            round_result = self._last_round or RoundResult(content=None, tool_calls=[])
            content = round_result.content

            # Validate structured tool calls against each tool's schema — an
            # empty/missing-argument structured call is a malformed attempt,
            # not something to execute.
            valid_structured = [
                call
                for call in round_result.tool_calls
                if call.arguments is not None and self._valid_args(call.name, call.arguments)
            ]
            malformed_structured = bool(round_result.tool_calls) and not valid_structured

            # Three-way classify the text content for an embedded/whole blob.
            outcome, text_call, cleaned = classify_tool_call(
                content, tool_names, self._is_valid_request
            )

            # VALID: execute (structured wins; else the text-detected call).
            if valid_structured or outcome is ToolCallOutcome.VALID:
                gate.discard()  # tool-call JSON never reaches the user
                calls = valid_structured or [self._pending_from_request(text_call)]
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",  # never the raw blob — this is the history fix
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": call.raw_arguments},
                            }
                            for call in calls
                        ],
                    }
                )
                for call in calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": await self._execute_tool(call.name, call.arguments),
                        }
                    )
                continue

            # MALFORMED: discard the phantom tool call, but still show any real
            # conversational content the model produced alongside it.
            if malformed_structured or outcome is ToolCallOutcome.MALFORMED:
                gate.discard()
                logger.debug("Discarded malformed tool-call attempt: %r", content)
                self._emit_final(on_text, messages, cleaned, gate.streamed)
                return

            # Structural leak guard (ollama path only): if tools were offered and
            # the whole reply is JSON, it's tool-call protocol noise the model
            # narrated instead of calling — never an answer. Catches the open set
            # of fabricated shapes (function/result, status/message, escaped
            # quotes, leading `{}`) without chasing each one. Cloud replies stream
            # live and aren't buffered here, so this can't swallow a real answer.
            if hold_all and tool_definitions and _looks_like_json(content):
                gate.discard()
                logger.debug("Discarded all-JSON ollama reply as protocol noise: %r", content)
                self._emit_final(on_text, messages, None, gate.streamed)
                return

            # NONE: ordinary conversational content.
            gate.flush()
            self._emit_final(on_text, messages, content, gate.streamed)
            return
        on_text("\n[Stopped: too many tool calls in a single turn.]")

    @staticmethod
    def _pending_from_request(call: ToolCallRequest) -> PendingToolCall:
        return PendingToolCall(
            id="call_0",
            name=call.name,
            raw_arguments=json.dumps(call.arguments),
            arguments=call.arguments,
        )

    def _is_valid_request(self, call: ToolCallRequest) -> bool:
        return self._valid_args(call.name, call.arguments)

    def _valid_args(self, name: str, arguments) -> bool:
        return self.registry.validate(name, arguments)

    def _emit_final(
        self,
        on_text: Callable[[str], None],
        messages: list[dict],
        text: str | None,
        already_shown: bool,
    ) -> None:
        """Show and record the turn's final assistant text, apologizing if the
        model produced nothing usable. `already_shown` guards against
        re-printing content that was streamed live (cloud path)."""
        display = (text or "").strip()
        if not display:
            logger.debug("Model returned no usable content: %r", text)
            display = APOLOGY
        if not already_shown:
            on_text(display)
        messages.append({"role": "assistant", "content": display})

    async def _execute_tool(self, name: str, arguments: dict | None) -> str:
        return await self.registry.execute(name, arguments, self.provider)


def _looks_like_json(content: str | None) -> bool:
    """True if the reply leads with a JSON bracket.

    Only consulted on the ollama path when tools were offered — where a reply
    opening with `{`/`[` is a fabricated tool-call blob (often unbalanced or
    escaped, so no real parse is possible), never a genuine answer. The
    legitimate `{"city": "Tokyo"}` case only arises with no tools offered, which
    the caller already excludes.

    ponytail: leading-bracket heuristic, not a parse. Ceiling: on the ollama
    path a genuine prose answer that happens to start with `{` becomes an
    apology-and-retry. Vanishingly rare; tighten to a real parse only if it bites.
    """
    return (content or "").strip().replace('\\"', '"').startswith(("{", "["))


def _arguments_as_dict(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
