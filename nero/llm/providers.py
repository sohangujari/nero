"""The one table of LLM provider facts.

Everything a provider needs — its LiteLLM prefix, its keyring entry, its
curated models — lives in one row here, so adding a provider is a single edit
rather than four spread across cli.py, manager.py and client.py (which is what
this module replaced).

Imports nothing from `nero`, so any module may import it without a cycle.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    """One provider's complete configuration surface.

    `name` is what the user writes in config and is the friendly name, not
    LiteLLM's internal key — the config already said "claude" rather than
    "anthropic" before this table existed. `prefix` and `catalog_key` are the
    LiteLLM-facing translation; `keyring_entry` is named after the upstream
    service so a key stored for `qwen` lands at `dashscope_api_key`.
    """

    name: str
    label: str
    prefix: str  # LiteLLM model prefix; "" for providers LiteLLM routes bare
    catalog_key: str  # matches litellm_provider in litellm.model_cost
    keyring_entry: str | None  # None means keyless (ollama)
    models: tuple[str, ...] = ()  # curated shortlist; models[0] is the default

    @property
    def default_model(self) -> str | None:
        return self.models[0] if self.models else None


# Curated shortlists are editorial, not measured: general-purpose chat
# flagships, "-latest" aliases preferred over dated snapshots because they rot
# slower, and no vision-only, code-only, or small-fast tiers. They are NOT a
# tool-reliability claim — Nero has real reliability data for exactly one model
# family (phi4-mini) and none for these. `nero history` is how any of them gets
# settled; a bad pick gets demoted from the shortlist and nothing else breaks.
PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        "claude", "Claude (Anthropic)", "", "anthropic", "anthropic_api_key",
        ("claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"),
    ),
    ProviderInfo(
        "openai", "OpenAI", "", "openai", "openai_api_key",
        ("gpt-5", "gpt-5.2", "gpt-5.1", "gpt-5-mini"),
    ),
    ProviderInfo(
        "gemini", "Gemini (Google)", "gemini/", "gemini", "gemini_api_key",
        ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview"),
    ),
    # Local and keyless; its model comes from hardware detection, not a list.
    ProviderInfo("ollama", "Ollama (local)", "ollama/", "ollama", None, ()),
    ProviderInfo(
        "mistral", "Mistral", "mistral/", "mistral", "mistral_api_key",
        ("mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
         "magistral-medium-latest"),
    ),
    ProviderInfo(
        "deepseek", "DeepSeek", "deepseek/", "deepseek", "deepseek_api_key",
        ("deepseek-chat", "deepseek-reasoner", "deepseek-v3.2"),
    ),
    ProviderInfo(
        "minimax", "MiniMax", "minimax/", "minimax", "minimax_api_key",
        ("MiniMax-M2.5", "MiniMax-M3", "MiniMax-M2.1"),
    ),
    ProviderInfo(
        "kimi", "Moonshot (Kimi)", "moonshot/", "moonshot", "moonshot_api_key",
        ("kimi-k2.6", "kimi-k2.5", "kimi-k2-thinking", "kimi-latest"),
    ),
    ProviderInfo(
        "qwen", "Qwen (Alibaba)", "dashscope/", "dashscope", "dashscope_api_key",
        ("qwen3-max", "qwen-plus", "qwen-flash", "qwen3-coder-plus"),
    ),
    ProviderInfo(
        "xai", "xAI (Grok)", "xai/", "xai", "xai_api_key",
        ("grok-4.6", "grok-4.5", "grok-4.3", "grok-4-fast-reasoning"),
    ),
    ProviderInfo(
        "glm", "Z.AI (GLM)", "zai/", "zai", "zai_api_key",
        ("glm-5", "glm-4.7", "glm-4.6", "glm-4.7-flash"),
    ),
    ProviderInfo(
        "openrouter", "OpenRouter", "openrouter/", "openrouter", "openrouter_api_key",
        # LiteLLM's router meta-model is "openrouter/openrouter/auto", whose bare form
        # begins with this provider's own prefix — the startswith guard in
        # LLMClient.litellm_model would then leave it half-qualified. Left off the
        # shortlist rather than branching the prefix path for one entry; it is still
        # reachable by typing the full name.
        ("anthropic/claude-sonnet-4.6", "openai/gpt-5.2", "deepseek/deepseek-v3.2"),
    ),
    ProviderInfo(
        "groq", "Groq", "groq/", "groq", "groq_api_key",
        ("llama-3.3-70b-versatile", "openai/gpt-oss-120b",
         "moonshotai/kimi-k2-instruct-0905"),
    ),
)

_BY_NAME = {info.name: info for info in PROVIDERS}


def get(name: str) -> ProviderInfo:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown provider: {name!r}") from None


def names() -> tuple[str, ...]:
    return tuple(info.name for info in PROVIDERS)


def catalog_models(name: str) -> list[str]:
    """Every tool-capable chat model LiteLLM knows for this provider, bare.

    The import is deliberately lazy and inside the function: it keeps this
    module free of a litellm dependency at import time, so `nero.llm.providers`
    stays importable and testable without litellm installed.

    Returns [] on anything going wrong — unknown provider, missing litellm, a
    catalog shape that changed under us. Every caller degrades to free-text
    entry, because no failure here may cost the user the task.
    """
    try:
        info = get(name)
    except KeyError:
        return []
    try:
        import litellm

        found = set()
        for model, meta in litellm.model_cost.items():
            if meta.get("litellm_provider") != info.catalog_key:
                continue
            # Both filters matter: the catalog carries TTS models
            # (minimax/speech-02-hd) and toolless safety classifiers
            # (groq/…llama-prompt-guard) that would be noise in a picker.
            if not meta.get("supports_function_calling"):
                continue
            if meta.get("mode") != "chat":
                continue
            bare = model.removeprefix(info.prefix) if info.prefix else model
            # Self-prefixed names (openrouter/auto) would keep their own provider
            # prefix as the bare form, and LLMClient.litellm_model's startswith
            # guard would then leave them half-qualified. Same reason openrouter/auto
            # is off the curated shortlist.
            if info.prefix and bare.startswith(info.prefix):
                continue
            found.add(bare)
        return sorted(found)
    except Exception:  # noqa: BLE001 — any catalog failure degrades to free text
        return []
