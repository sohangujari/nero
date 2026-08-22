"""Helpers for an arbitrary OpenAI-compatible server (see llm.base_url).

Imports nothing from `nero`, so any module may import it without a cycle —
the same rule `nero.llm.providers` follows.
"""

import httpx


def fetch_models(
    base_url: str, api_key: str | None = None, timeout: float = 5.0
) -> tuple[str, list[str]]:
    """Model ids from an OpenAI-compatible server, and the base URL that answered.

    Tries `{base}/models`, then `{base}/v1/models`: LM Studio reports its
    address without the `/v1` the API actually lives at, and asking is cheaper
    than guessing. Skips the second candidate when the URL already ends in
    `/v1`, so a correct URL is never probed as `/v1/v1`.

    Returns `(base_url, [])` on anything going wrong — server down, auth
    refused, a response shape we don't recognise. Every caller degrades to
    free-text entry, because no failure here may cost the user the task.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    candidates = [base_url] if base_url.endswith("/v1") else [base_url, f"{base_url}/v1"]
    for candidate in candidates:
        try:
            response = httpx.get(f"{candidate}/models", headers=headers, timeout=timeout)
            response.raise_for_status()
            models = sorted(str(entry["id"]) for entry in response.json().get("data", []))
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            continue
        if models:
            return candidate, models
    return base_url, []
