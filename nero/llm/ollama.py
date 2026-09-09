"""Helpers for the local Ollama server (no API key — it's all localhost)."""

import subprocess

import httpx

BASE_URL = "http://localhost:11434"


def reachable(base_url: str = BASE_URL, timeout: float = 1.0) -> bool:
    """Is the Ollama server up at its default local address?"""
    try:
        return httpx.get(f"{base_url}/api/version", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def list_models(base_url: str = BASE_URL) -> list[str]:
    """Names of locally pulled models (tags like 'qwen3:8b', 'phi4-mini:latest')."""
    response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
    response.raise_for_status()
    return [model["name"] for model in response.json().get("models", [])]


def has_model(name: str, base_url: str = BASE_URL) -> bool:
    """Whether Ollama would actually serve `name`, the way Ollama resolves it.

    A bare name means the `:latest` tag — it is not a wildcard over whatever
    happens to be pulled. Matching it as a prefix said yes to `llama3.2` on a
    machine holding only `llama3.2:1b`, while Ollama itself answered
    `model 'llama3.2' not found`; a fallback chain built on that answer looks
    like a safety net until the moment it is needed.
    """
    try:
        tags = list_models(base_url)
    except httpx.HTTPError:
        return False
    wanted = name if ":" in name else f"{name}:latest"
    return any(tag == name or tag == wanted for tag in tags)


def supports_tools(name: str, base_url: str = BASE_URL) -> bool | None:
    """Whether the model can do tool calling, per Ollama's own capability list.

    /api/show reports capabilities authoritatively (e.g. phi4-mini has 'tools',
    gemma3 does not), so there's no need to hand-maintain a list or send a probe
    chat. Returns None when it can't be determined — server down, or model not
    pulled — so callers can stay quiet rather than warn on a guess.
    """
    try:
        response = httpx.post(f"{base_url}/api/show", json={"model": name}, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    return "tools" in (response.json().get("capabilities") or [])


# Models Ollama reports as tool-capable that cannot decide when *not* to call
# a tool. This is a different question from the one /api/show answers, and
# Ollama has no field for it: llama3.2 lists "tools" in its capabilities and
# genuinely emits well-formed, schema-valid tool calls — for every message,
# including "hi" and "what is 2+2". Those calls pass validation and execute, so
# no amount of downstream guarding recovers the answer.
#
# Measured on llama3.2:latest (3.2B), temperature 0 and 0.7, offering 12 tools,
# 3 tools and 1 tool: a tool fired on 7 of 7 ordinary messages at every tool
# count ("what is 2+2" -> fetch_web_page; with one tool offered, open_app).
# With no tools offered the same model answers all 7 correctly in ~0.4 s.
#
# Entries are measured, never guessed — a wrong entry silently disables every
# skill for someone whose model works fine. Keyed on the family, before the
# tag, because the behaviour is the model's, not the quantisation's.
MISFIRES_TOOLS = frozenset({"llama3.2"})


def misfires_tools(name: str) -> bool:
    """Whether `name` is a model measured to call tools on ordinary chat.

    Purely a table lookup, so it stays true with the server down — unlike
    `supports_tools`, there is nothing to probe.
    """
    return name.split(":")[0] in MISFIRES_TOOLS


def pull_model(name: str) -> bool:
    """Run `ollama pull`, inheriting stdio so its progress bars render."""
    try:
        return subprocess.run(["ollama", "pull", name]).returncode == 0
    except OSError:
        return False
