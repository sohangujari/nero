import asyncio
import json
import logging
import re
from datetime import datetime
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import litellm

from nero.config.schema import LLMConfig
from nero.llm import ollama, providers
from nero.memory.facts import facts_prompt_block
from nero.memory.facts import relevant as relevant_facts
from nero.llm.ollama_adapter import (
    ToolCallOutcome,
    ToolCallRequest,
    classify_tool_call,
    ollama_chat,
)
from nero.skills.registry import SkillRegistry

logger = logging.getLogger("nero.llm")

MAX_TOKENS = 8192

# Seconds the provider may go without sending a byte. LiteLLM hands this to
# httpx, whose read timeout is per-read, not per-request -- so a long
# generation is never cut off, only a stalled one. Generous because a queued
# free tier really can take 40 s to produce its first token; the point is that
# a dead connection surfaces as litellm.Timeout (already in both loops'
# transient-error tuples, so the fallback chain gets its turn) instead of
# hanging on LiteLLM's 600 s default.
REQUEST_TIMEOUT = 120.0
# Safety bound on tool-call round trips within a single user turn.
MAX_TOOL_ROUNDS = 10

APOLOGY = "I didn't quite catch that — could you rephrase?"


def current_time_line() -> str:
    """Today's date and the local clock, as one line for the system prompt.

    A model's only sense of "now" is its training cutoff, so without this it
    confidently answers with a date months or years in the past.

    Resolved per request, not once at startup: the Telegram bridge and a
    launchd-installed session can run for days, and a date captured at boot
    goes stale exactly as badly as no date at all.

    Rounded down to ten minutes on purpose, and that number was raised from one
    minute after measuring the real cost. The line sits in the system prompt,
    which is the *front* of the prompt, so changing it invalidates every token
    after it — including all twelve tool schemas. On llama3.2 that is a 2,000
    token re-evaluation:

        stable prefix    10,125 ms cold, then 265 ms, 287 ms
        clock ticking       312 ms, then 11,808 ms, 9,867 ms

    Minute precision therefore bought a ~10 s stall on the first turn of every
    new minute — the same prompt, the same tools, the same answer, ten seconds
    slower, for no reason a user could guess. Ten minutes cuts that to six
    possible stalls an hour instead of sixty.

    An earlier note here recorded minute precision as free (0.20 s either way).
    That was measured against a cloud provider before the tool payload reached
    1,740 tokens; it was true then and is not now.

    The obvious alternative — moving the clock to the end of the prompt, where
    it would not invalidate anything — was tried and does not work. As a
    trailing system message llama3.2's chat template answers it with a literal
    "assistant\n\n" header that lands in the reply; appended to the user's
    message the model treats it as the subject and answers "hi" with the time.
    """
    now = datetime.now().astimezone()
    now = now.replace(minute=now.minute // 10 * 10, second=0, microsecond=0)
    zone = now.strftime("%Z") or now.strftime("%z")
    # Day and year are interpolated rather than formatted: the dash-prefixed
    # strftime codes that strip leading zeros are a glibc extension, and Nero
    # ships a Windows binary where they raise.
    return (
        f"\n\nRight now it is {now:%A}, {now.day} {now:%B} {now.year}, "
        f"around {now:%H:%M} {zone}."
    )

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
        facts: list[tuple[str, str]] | None = None,
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
            "write tool-call JSON as text. Keep replies concise.\n\n"
            "A <memory> block on a message is your own recollection of earlier "
            "talks with this user — not something they just sent you. Answer as "
            "if you simply remember. Never say where an answer came from: no "
            '"based on what I have stored", "in my notes", "in my memory system", '
            '"you shared", "conversation history", "no facts found". Talk about '
            "what you know, never about how you know it."
        )
        # Carried on the user's message rather than appended here, and for a
        # measured reason. Tool schemas are rendered between the system prompt
        # and the question — about 2,000 tokens of them — and a 2B model
        # reading facts that far upstream stops using them: qwen3.5:2b answered
        # "who is my brother" correctly 1 time in 3 with the facts up here, and
        # 3 times in 3 with them next to the question. Nothing leaks into plain
        # chat either way (0/3 on "hi", "good morning", "what is 2+2").
        #
        # It also takes the one part of the prefix that changes as Nero learns
        # out of the prefix, which is the same lesson as `current_time_line`.
        self.facts = list(facts or [])
        self._last_round: RoundResult | None = None
        # Accumulated USD cost of the turn currently in progress (reset at the
        # top of _run_turn). Local/ollama rounds never touch this — no cost.
        # Never raises: an unpriced/custom model or a streaming response
        # litellm can't cost just leaves this at 0.0, so the ceiling simply
        # never trips rather than crashing the turn.
        self.last_turn_cost: float = 0.0

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def api_base(self) -> str | None:
        """The custom endpoint, or None.

        Guarded on the provider, not merely on the field being set: a base_url
        left over from an earlier custom endpoint persists inertly in config
        (nothing clears it on a provider switch), and must never leak into
        another provider's call.
        """
        if self.config.provider not in providers.CUSTOM_PROVIDERS:
            return None
        return self.config.base_url

    @property
    def aws_region(self) -> str | None:
        """The Bedrock region, or None.

        Guarded on the provider like api_base: an aws_region left in config
        persists inertly and must never leak into another provider's call.
        """
        if self.config.provider != "bedrock":
            return None
        return self.config.aws_region

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def litellm_model(self) -> str:
        """The config's provider+model as a LiteLLM model string.

        Providers LiteLLM routes bare (claude, openai) carry prefix "". The
        startswith guard keeps a user who typed the fully-qualified name —
        "dashscope/qwen3-max" — from getting it applied twice, and leaves
        nested names like "anthropic/claude-sonnet-4.6" under openrouter/ intact.
        """
        if self.config.provider in providers.CUSTOM_PROVIDERS:
            # Unconditional, unlike the startswith guard below: Together really
            # serves a model called "openai/gpt-oss-120b" (the groq shortlist
            # carries that exact id), and the guard would see the prefix already
            # present and pass it through — LiteLLM would then strip it and send
            # the bare "gpt-oss-120b" to the endpoint, which 404s. LiteLLM strips
            # exactly one prefix, so a doubled prefix is correct — for both
            # custom dialects; the prefix itself comes from the table.
            return providers.get(self.config.provider).prefix + self.config.model
        prefix = providers.get(self.config.provider).prefix
        model = self.config.model
        if prefix and not model.startswith(prefix):
            return prefix + model
        return model

    def _tool_definitions(self) -> list[dict]:
        return self.registry.tool_definitions()

    def system_message(self) -> str:
        """The system prompt as sent right now — the stable core plus today's
        date. Separate from `self.system_prompt` so the core stays a fixed,
        testable string and only this varies.

        Kept for callers that want the whole thing as one string; the wire
        format splits the two (see `outgoing`).
        """
        return self.system_prompt + current_time_line()

    def outgoing(self, messages: list[dict]) -> list[dict]:
        """The single place the outgoing message list is assembled.

        The clock has to live here, at the front, even though that is the worst
        place for caching — every alternative was worse (see
        `current_time_line`). What makes it affordable is its granularity: it
        only changes six times an hour, so the prefix survives in between.
        """
        outgoing = [{"role": "system", "content": self.system_message()}, *messages]
        if not self.facts:
            return outgoing
        # Only the facts this turn is about. Attaching all of them put a
        # seven-line block in front of "skip this track", and the model
        # answered the block instead of running the command.
        asked = _last_user_text(messages)
        block = facts_prompt_block(relevant_facts(self.facts, asked)).strip()
        if not block:
            return outgoing
        # Tagged, not headed, for the reason recorded in nero/memory/recall.py:
        # a bare list above a question reads to a small model like something the
        # user pasted, and gets answered instead of used.
        for index in range(len(outgoing) - 1, -1, -1):
            message = outgoing[index]
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                outgoing[index] = {
                    **message,
                    "content": f"<memory>\n{block}\n</memory>\n\n{message['content']}",
                }
                break
        return outgoing

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
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.aws_region:
            kwargs["aws_region_name"] = self.aws_region
        response = await litellm.acompletion(
            model=self.litellm_model,
            messages=self.outgoing(messages),
            tools=tools or None,
            max_tokens=MAX_TOKENS,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        chunks = []
        try:
            async for chunk in response:
                chunks.append(chunk)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is not None and getattr(delta, "content", None):
                    yield delta.content
        except litellm.exceptions.MidStreamFallbackError as exc:
            # LiteLLM wraps ANY failure that arrives after the stream opened --
            # a 429 included -- in this ServiceUnavailableError subclass. Left
            # alone it makes a rate limit read as "could not reach the model
            # provider" and skips key rotation, which only ever matched
            # RateLimitError. Unwrapping here fixes every caller at once
            # rather than teaching each handler about the wrapper.
            raise (exc.original_exception or exc) from exc
        full = litellm.stream_chunk_builder(chunks, messages=messages)
        message = full.choices[0].message if full and full.choices else None
        self._last_round = RoundResult(
            content=getattr(message, "content", None),
            tool_calls=self._pending_from_message(message),
        )
        try:
            self.last_turn_cost += litellm.completion_cost(completion_response=full) or 0.0
        except Exception:  # noqa: BLE001 — cost is a nice-to-have, never a turn-breaker
            pass

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
        outgoing = self.outgoing(messages)
        request_messages = [outgoing[0]]
        request_messages += [self._to_ollama_message(m) for m in outgoing[1:-1]]
        request_messages.append(outgoing[-1])
        model = self.config.model.removeprefix("ollama/")
        parts: list[str] = []
        calls = []
        async for response in ollama_chat(
            self.ollama_base_url,
            model,
            request_messages,
            tools,
            think=self.config.think,
            keep_alive=self.config.keep_alive,
        ):
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

    def _gated_tools(self, messages: list[dict], tool_definitions: list[dict]) -> list[dict]:
        """`tool_definitions`, or none of them when this turn is plainly just talk.

        Only for local models measured to call a tool on every message
        (`ollama.misfires_tools`). They cannot decide mid-generation whether to
        emit a tool call, but they answer "is this asking me to do something?"
        reliably — so the decision is taken *before* the turn, in a 145-token
        question, instead of being left to a 2,000-token prompt they will
        mishandle.

        Every other model, cloud or local, is untouched: it gates tools itself,
        and a second round trip to ask would be pure cost.
        """
        if not tool_definitions or self.config.provider != "ollama":
            return tool_definitions
        model = self.config.model.removeprefix("ollama/")
        if not ollama.misfires_tools(model):
            return tool_definitions
        asked = _last_user_text(messages)
        if not asked:
            return tool_definitions
        wants = ollama.wants_action(model, asked, self.ollama_base_url)
        if wants is False:
            logger.debug("gate: no tools offered for %r", asked[:60])
            return []
        return tool_definitions

    async def _run_turn(self, messages: list[dict], on_text: Callable[[str], None]) -> None:
        self.last_turn_cost = 0.0
        tool_definitions = self._gated_tools(messages, self._tool_definitions())
        # Ollama rounds are fully buffered: local models mix filler text with
        # tool calls, and accompanying content must not print as an answer.
        # Cloud rounds keep live streaming (their preamble text is intentional).
        hold_all = self.config.provider == "ollama"
        # Every known skill, not just the available ones: a call naming a
        # disabled skill must classify as a tool call so the registry can refuse
        # and audit it, rather than being discarded as MALFORMED.
        tool_names = self.registry.known_names()
        # A model too small to work the tool protocol answers *everything* with
        # tool-call JSON — llama3.2:1b replies to "hi" by reciting a schema back.
        # Suppressing that leaves an apology, which is not an answer. So the
        # turn is retried once with no tools offered, which is the only state
        # such a model can actually converse in. Capable models never reach it.
        retried_bare = False
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
                # ...unless what survived the strip is more of the same blob.
                # A model that mangles a tool call badly enough leaves an
                # unbalanced head behind — llama3.2:1b answers "hi" by echoing
                # a tool schema, and the stripper hands back
                # `{"type":"function","function":`. Printing that is worse than
                # printing nothing: this branch already knows the model was
                # attempting a tool call, so leading-bracket residue is protocol
                # noise on every provider, not just ollama.
                if _looks_like_json(cleaned):
                    cleaned = None
                if cleaned is None and not retried_bare and tool_definitions:
                    retried_bare = True
                    tool_definitions = []
                    logger.debug("Retrying the round with no tools offered")
                    continue
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
                if not retried_bare:
                    retried_bare = True
                    tool_definitions = []
                    continue
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


_CARRIED = re.compile(r"(?s).*</(?:memory|playbook)>\s*")


def _last_user_text(messages: list[dict]) -> str:
    """What the user actually typed on this turn.

    Recall and learned procedures ride on the front of the user message
    (nero/memory/recall.py), and handing a classifier a transcript of earlier
    conversation would have it classify the wrong thing entirely.
    """
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return _CARRIED.sub("", content).strip()
            return ""
    return ""


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
