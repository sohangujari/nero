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
    try:
        tags = list_models(base_url)
    except httpx.HTTPError:
        return False
    return any(tag == name or tag.startswith(f"{name}:") for tag in tags)


def pull_model(name: str) -> bool:
    """Run `ollama pull`, inheriting stdio so its progress bars render."""
    try:
        return subprocess.run(["ollama", "pull", name]).returncode == 0
    except OSError:
        return False
