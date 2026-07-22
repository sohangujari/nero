"""Direct adapter for Ollama's native /api/chat endpoint.

Bypasses LiteLLM entirely for the Ollama provider: LiteLLM's `ollama/{model}`
prefix routes to the legacy /api/generate endpoint (no structured tool-call
support), and its response adapter has a known upstream bug that leaks
`tool_calls` into `content` as raw JSON text. This module talks to Ollama's
own JSON structure directly, plus a fallback parser for small quantized models
that emit tool-call JSON as plain text even on the correct endpoint.
"""

import json
from collections.abc import AsyncIterator, Callable
from enum import Enum

import httpx
from pydantic import BaseModel


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict


# Spec alias: Ollama's native tool calls and fallback-parsed ones share a shape.
OllamaToolCall = ToolCallRequest


class OllamaChatResponse(BaseModel):
    content: str | None = None
    tool_calls: list[OllamaToolCall] | None = None


# Tool selection is a decision task, not a creative one — small local models
# are noticeably more consistent about whether to call a tool at low temperature.
OLLAMA_TEMPERATURE = 0.2


async def ollama_chat(
    base_url: str,
    model: str,
    messages: list,
    tools: list,
) -> AsyncIterator[OllamaChatResponse]:
    """Stream one chat completion from Ollama's native /api/chat endpoint.

    Yields one OllamaChatResponse per NDJSON chunk: content deltas as they
    arrive, and tool_calls parsed straight from Ollama's own response
    structure — no LiteLLM translation layer.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        async with client.stream("POST", f"{base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                message = chunk.get("message") or {}
                tool_calls = [
                    ToolCallRequest(
                        name=call["function"]["name"],
                        arguments=call["function"].get("arguments") or {},
                    )
                    for call in (message.get("tool_calls") or [])
                    if "function" in call
                ]
                yield OllamaChatResponse(
                    content=message.get("content") or None,
                    tool_calls=tool_calls or None,
                )


def extract_tool_call(response: OllamaChatResponse) -> ToolCallRequest | None:
    """Structured tool call if present, else fallback JSON detection on content."""
    if response.tool_calls:
        return response.tool_calls[0]
    return _try_parse_json_tool_call(response.content)


def _try_parse_json_tool_call(content: str | None) -> ToolCallRequest | None:
    """If content looks like a tool-call JSON blob instead of using the
    structured field, treat it as a tool call rather than chat output.

    Defense in depth for small quantized models; runs on every provider path.
    Accepts both a bare object and an array of objects — some models emit
    the whole tool_calls list as text (e.g. `[{"name": ..., "arguments": ...}]`).
    """
    if not content:
        return None
    stripped = content.strip()
    if not (
        stripped.startswith(("{", "["))
        and '"name"' in stripped
        and '"arguments"' in stripped
    ):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    candidates = parsed if isinstance(parsed, list) else [parsed]
    for candidate in candidates:
        call = _coerce_tool_call(candidate)
        if call is not None:
            return call
    return None


def _coerce_tool_call(candidate) -> ToolCallRequest | None:
    if not isinstance(candidate, dict):
        return None
    # OpenAI wire format: {"type": "function", "function": {"name", "arguments"}}.
    # Small models frequently emit this shape as plain text rather than using
    # the structured tool_calls field.
    inner = candidate.get("function")
    if isinstance(inner, dict) and "name" in inner:
        candidate = inner
    if "name" not in candidate or "arguments" not in candidate:
        return None
    arguments = candidate["arguments"]
    if isinstance(arguments, str):
        # Some models double-encode: {"arguments": "{\"app_name\": ...}"}
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(candidate["name"], str) or not isinstance(arguments, dict):
        return None
    return ToolCallRequest(name=candidate["name"], arguments=arguments)


class ToolCallOutcome(Enum):
    VALID = "valid"        # a well-formed tool call that passes validation
    MALFORMED = "malformed"  # tool-call-shaped attempt that fails validation
    NONE = "none"          # not a tool call at all


def classify_tool_call(
    content: str | None,
    tool_names: set[str],
    is_valid: Callable[[ToolCallRequest], bool],
) -> tuple[ToolCallOutcome, ToolCallRequest | None, str]:
    """Three-way classification of content that might be a tool-call blob.

    Returns (outcome, call, cleaned_content):

    - VALID: a coercible tool call that passes `is_valid` — execute it.
    - MALFORMED: content structurally resembles a tool-call attempt (references
      a known tool name, or carries an "arguments" object) but fails validation
      — e.g. empty required arguments, a junk `{}` array element, or a missing
      `arguments` key. `cleaned_content` is the conversational text with the
      blob stripped out; discard the blob, but still show that text.
    - NONE: no tool-call shape — show `content` unchanged.

    Handles blobs embedded *after* conversational text via bracket matching
    (respecting strings/escapes), single objects, and arrays of objects.
    """
    text = content or ""
    if not text:
        return ToolCallOutcome.NONE, None, text
    found = _extract_tool_blob(text)
    if found is not None:
        blob, remainder = found
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            # Small models emit subtly broken JSON (e.g. phi4-mini drops a brace:
            # `[{"type":"function","function":{...}}]}`). Rather than discard a
            # real intent, look for a well-formed tool-call object embedded in it.
            salvaged = _salvage_tool_call(blob, tool_names, is_valid)
            if salvaged is not None:
                return ToolCallOutcome.VALID, salvaged, remainder
            return ToolCallOutcome.MALFORMED, None, remainder
        candidates = parsed if isinstance(parsed, list) else [parsed]
        saw_attempt = False
        for candidate in candidates:
            if not _is_tool_attempt(candidate, tool_names):
                continue
            saw_attempt = True
            call = _coerce_tool_call(candidate)
            if call is not None and is_valid(call):
                return ToolCallOutcome.VALID, call, remainder
        if saw_attempt:
            return ToolCallOutcome.MALFORMED, None, remainder
        return ToolCallOutcome.NONE, None, text  # extracted region wasn't a tool call
    # No balanced JSON region — maybe a truncated tool blob referencing a tool.
    if _mentions_tool(text, tool_names) and ("{" in text or "[" in text):
        return ToolCallOutcome.MALFORMED, None, text[: _first_bracket(text)]
    return ToolCallOutcome.NONE, None, text


def _salvage_tool_call(
    blob: str, tool_names: set[str], is_valid: Callable[[ToolCallRequest], bool]
) -> ToolCallRequest | None:
    """Recover a valid tool call from JSON that doesn't parse as a whole.

    Scans every `{`-rooted, brace-balanced substring (ignoring square brackets,
    which are what small models typically mismatch) and returns the first that
    coerces to a known, valid tool call. Returns None if nothing valid is found,
    so genuinely malformed attempts still classify as MALFORMED.
    """
    for start, char in enumerate(blob):
        if char != "{":
            continue
        end = _matching_brace_end(blob, start)
        if end is None:
            continue
        try:
            candidate = json.loads(blob[start : end + 1])
        except json.JSONDecodeError:
            continue
        call = _coerce_tool_call(candidate)
        if call is not None and call.name in tool_names and is_valid(call):
            return call
    return None


def _matching_brace_end(text: str, start: int) -> int | None:
    """Index closing the `{` at `start`, counting braces only (string-aware)."""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
        elif char == "\\":
            escape = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index
    return None


def _is_tool_attempt(candidate, tool_names: set[str]) -> bool:
    """A dict that looks like it's trying to call a tool: it either names a
    known tool or carries an `arguments` object.

    Also unwraps the OpenAI wire shape, where those keys live one level down
    under "function" and the top level only has "type"/"function".
    """
    if not isinstance(candidate, dict):
        return False
    inner = candidate.get("function")
    if isinstance(inner, dict) and "name" in inner:
        candidate = inner
    return "arguments" in candidate or candidate.get("name") in tool_names


def _mentions_tool(content: str, tool_names: set[str]) -> bool:
    return any(f'"{name}"' in content for name in tool_names)


def _first_bracket(content: str) -> int:
    for index, char in enumerate(content):
        if char in "{[":
            return index
    return len(content)


def _extract_tool_blob(content: str) -> tuple[str, str] | None:
    """Find the first bracket-balanced JSON region carrying a tool marker.
    Returns (blob, remainder) where remainder is content with the blob removed."""
    for index, char in enumerate(content):
        if char not in "{[":
            continue
        end = _matching_bracket_end(content, index)
        if end is None:
            continue
        region = content[index : end + 1]
        if '"name"' in region or '"arguments"' in region:
            return region, content[:index] + content[end + 1 :]
    return None


def _matching_bracket_end(text: str, start: int) -> int | None:
    """Index of the bracket that closes the one at `start`, honoring JSON
    string literals and escapes; None if unbalanced."""
    close = "}" if text[start] == "{" else "]"  # noqa: F841 — clarity only
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
        elif char == "\\":
            escape = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    return index
    return None
