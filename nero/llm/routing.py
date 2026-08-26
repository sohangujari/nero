"""Cost/latency/quality routing of the fallback chain, and in-session health
tracking. See docs/superpowers/specs/2026-08-24-routing-design.md.

Pinned constraint: every dimension below has one concrete, testable
definition — no blended score, no learned weights. `route_by` picks exactly
ONE dimension and it orders the existing fallback chain; it never touches the
primary model the user configured.
"""

from collections import defaultdict
from statistics import median

from nero.llm import providers

# Consecutive transient failures (this session) before a provider/model is
# treated as unhealthy. One success clears it.
UNHEALTHY_THRESHOLD = 2

# How many recent latency samples `latency_median` considers.
LATENCY_WINDOW = 5


def _litellm_model_id(provider: str, model: str) -> str:
    """provider+model as the LiteLLM-facing model id, for a cost lookup.

    Mirrors LLMClient.litellm_model's prefixing rule without needing a live
    client instance — routing happens over (provider, model) pairs, most of
    which have no client yet.
    """
    try:
        info = providers.get(provider)
    except KeyError:
        return model
    if provider in providers.CUSTOM_PROVIDERS:
        return info.prefix + model
    if info.prefix and not model.startswith(info.prefix):
        return info.prefix + model
    return model


def _cost(provider: str, model: str) -> float | None:
    """Static catalog cost per token (input + output), or None when unpriced.

    Never raises: an unknown model, a missing litellm import, or a catalog
    shape change all degrade to None — the model simply sorts last rather
    than crashing routing.
    """
    try:
        import litellm

        meta = litellm.model_cost[_litellm_model_id(provider, model)]
        return meta["input_cost_per_token"] + meta["output_cost_per_token"]
    except Exception:  # noqa: BLE001 — unpriced/unknown degrades to None
        return None


class SessionStats:
    """In-memory, per-session measurements: latency samples and health
    counters. No persistence — a health verdict is only as good as this
    session; a stale one on disk would silently blacklist a recovered
    provider."""

    def __init__(self) -> None:
        self._latencies: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._consecutive_failures: dict[tuple[str, str], int] = defaultdict(int)

    def record_latency(self, provider: str, model: str, seconds: float) -> None:
        samples = self._latencies[(provider, model)]
        samples.append(seconds)
        del samples[:-LATENCY_WINDOW]

    def latency_median(self, provider: str, model: str) -> float | None:
        samples = self._latencies.get((provider, model))
        return median(samples) if samples else None

    def record_failure(self, provider: str, model: str) -> None:
        self._consecutive_failures[(provider, model)] += 1

    def record_success(self, provider: str, model: str) -> None:
        self._consecutive_failures[(provider, model)] = 0

    def is_unhealthy(self, provider: str, model: str) -> bool:
        return self._consecutive_failures.get((provider, model), 0) >= UNHEALTHY_THRESHOLD


def order_chain(
    entries: list[tuple[str, str]],
    route_by: str,
    stats: SessionStats,
    quality_rank: list[str],
) -> list[tuple[str, str]]:
    """Order `entries` — (provider, model) pairs — by `route_by`.

    "off" returns the input unchanged (regression lock: today's chain order).
    Every other dimension sorts unknown/unmeasured entries LAST — never
    promote something we know nothing about above something measured — and
    uses a stable sort, so ties keep the user's configured order.
    """
    if route_by == "cost":
        def key(entry: tuple[str, str]) -> tuple[bool, float]:
            cost = _cost(*entry)
            return (cost is None, cost or 0.0)
    elif route_by == "latency":
        def key(entry: tuple[str, str]) -> tuple[bool, float]:
            latency = stats.latency_median(*entry)
            return (latency is None, latency or 0.0)
    elif route_by == "quality":
        def key(entry: tuple[str, str]) -> tuple[bool, int]:
            _provider, model = entry
            try:
                return (False, quality_rank.index(model))
            except ValueError:
                return (True, 0)
    else:
        # "off", or anything unrecognized: leave the configured order alone.
        return list(entries)
    return sorted(entries, key=key)
