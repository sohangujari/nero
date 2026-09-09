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


def pull_model(name: str) -> bool:
    """Run `ollama pull`, inheriting stdio so its progress bars render."""
    try:
        return subprocess.run(["ollama", "pull", name]).returncode == 0
    except OSError:
        return False
