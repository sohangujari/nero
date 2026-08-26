"""v1.6.0 routing: cost/latency/quality ordering, health, /code, key rotation.

Pinned constraint from the design: every dimension has one concrete testable
definition. These tests are that definition made executable.
"""

import io

import litellm
import pytest
from rich.console import Console

from nero.core.chat_loop import ChatLoop
from nero.llm.routing import SessionStats, order_chain

CHAIN = [("openai", "gpt-5"), ("claude", "claude-haiku-4-5"), ("ollama", "llama3.2")]


def quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=100)


class FakeClient:
    """Duck-typed like LLMClient: provider/model/api_key plus send()."""

    def __init__(self, provider="claude", model="m", errors=None, api_key="key-1"):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.send_calls: list[list[dict]] = []
        self.keys_used: list[str] = []
        # errors: one entry per call; None means succeed.
        self._errors = list(errors or [])

    def send(self, messages, on_text):
        self.keys_used.append(self.api_key)
        error = self._errors.pop(0) if self._errors else None
        if error:
            raise error
        self.send_calls.append([dict(m) for m in messages])
        on_text("ok")
        messages.append({"role": "assistant", "content": "ok"})


def make_loop(inputs, client=None, **kwargs):
    client = client or FakeClient()
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    loop = ChatLoop(
        client, console=quiet_console(), assistant_name="Nero", input_fn=next_input, **kwargs
    )
    return loop, client


class TestOrderChainOff:
    def test_off_returns_the_configured_order_untouched(self):
        """Regression lock: routing must be invisible until switched on."""
        assert order_chain(CHAIN, "off", SessionStats(), []) == CHAIN

    def test_an_unrecognized_dimension_also_leaves_order_alone(self):
        assert order_chain(CHAIN, "nonsense", SessionStats(), []) == CHAIN


class TestCostOrdering:
    def test_cheaper_model_sorts_first_and_unpriced_sorts_last(self, monkeypatch):
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "gpt-5": {"input_cost_per_token": 5e-6, "output_cost_per_token": 5e-6},
                "claude-haiku-4-5": {
                    "input_cost_per_token": 1e-6,
                    "output_cost_per_token": 1e-6,
                },
            },
        )
        ordered = order_chain(CHAIN, "cost", SessionStats(), [])
        assert ordered[0] == ("claude", "claude-haiku-4-5")
        assert ordered[1] == ("openai", "gpt-5")
        # ollama/llama3.2 is unpriced — never promoted above a measured entry.
        assert ordered[2] == ("ollama", "llama3.2")

    def test_a_broken_catalog_never_raises(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {"gpt-5": {"wrong_shape": 1}})
        assert set(order_chain(CHAIN, "cost", SessionStats(), [])) == set(CHAIN)


class TestLatencyOrdering:
    def test_median_orders_and_unmeasured_sorts_last(self):
        stats = SessionStats()
        for seconds in (9.0, 10.0, 11.0):
            stats.record_latency("openai", "gpt-5", seconds)
        for seconds in (1.0, 2.0, 3.0):
            stats.record_latency("claude", "claude-haiku-4-5", seconds)
        ordered = order_chain(CHAIN, "latency", stats, [])
        assert ordered == [
            ("claude", "claude-haiku-4-5"),
            ("openai", "gpt-5"),
            ("ollama", "llama3.2"),
        ]

    def test_only_the_recent_window_counts(self):
        stats = SessionStats()
        for seconds in (100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0):
            stats.record_latency("openai", "gpt-5", seconds)
        # The five ancient 100s slid out of the window.
        assert stats.latency_median("openai", "gpt-5") == 1.0


class TestQualityOrdering:
    def test_rank_order_wins_and_unranked_sorts_last(self):
        ordered = order_chain(CHAIN, "quality", SessionStats(), ["llama3.2", "gpt-5"])
        assert ordered == [
            ("ollama", "llama3.2"),
            ("openai", "gpt-5"),
            ("claude", "claude-haiku-4-5"),
        ]

    def test_ties_keep_the_users_configured_order(self):
        """Stable sort: two unranked models must not be reshuffled."""
        ordered = order_chain(CHAIN, "quality", SessionStats(), [])
        assert ordered == CHAIN


class TestHealth:
    def test_two_consecutive_failures_mark_unhealthy_and_one_success_clears(self):
        stats = SessionStats()
        assert not stats.is_unhealthy("openai", "gpt-5")
        stats.record_failure("openai", "gpt-5")
        assert not stats.is_unhealthy("openai", "gpt-5")
        stats.record_failure("openai", "gpt-5")
        assert stats.is_unhealthy("openai", "gpt-5")
        stats.record_success("openai", "gpt-5")
        assert not stats.is_unhealthy("openai", "gpt-5")

    def test_the_chain_walk_skips_an_unhealthy_entry(self):
        sick = FakeClient(provider="openai", model="gpt-5")
        well = FakeClient(provider="claude", model="haiku")
        loop, _ = make_loop([], fallback_clients=[sick, well])
        loop.stats.record_failure("openai", "gpt-5")
        loop.stats.record_failure("openai", "gpt-5")
        assert loop._ordered_fallback_candidates() == [well]

    def test_an_unhealthy_entry_is_still_used_when_it_is_the_only_one(self):
        """An unhealthy option beats no option."""
        sick = FakeClient(provider="openai", model="gpt-5")
        loop, _ = make_loop([], fallback_clients=[sick])
        loop.stats.record_failure("openai", "gpt-5")
        loop.stats.record_failure("openai", "gpt-5")
        assert loop._ordered_fallback_candidates() == [sick]

    def test_health_check_off_keeps_every_candidate(self):
        sick = FakeClient(provider="openai", model="gpt-5")
        well = FakeClient(provider="claude", model="haiku")
        loop, _ = make_loop([], fallback_clients=[sick, well], health_check=False)
        loop.stats.record_failure("openai", "gpt-5")
        loop.stats.record_failure("openai", "gpt-5")
        assert loop._ordered_fallback_candidates() == [sick, well]

    def test_duplicate_provider_model_entries_are_both_kept(self):
        """Indexing by key would silently drop one of a duplicated pair."""
        first = FakeClient(provider="openai", model="gpt-5")
        second = FakeClient(provider="openai", model="gpt-5")
        loop, _ = make_loop([], fallback_clients=[first, second])
        assert loop._ordered_fallback_candidates() == [first, second]


class TestCodeCommand:
    def test_code_routes_the_turn_to_the_coding_client(self):
        coding = FakeClient(provider="openai", model="gpt-5-codex")
        loop, primary = make_loop(["/code write a parser", "exit"], coding_client=coding)
        loop.run()
        assert len(coding.send_calls) == 1
        assert coding.send_calls[0][-1] == {"role": "user", "content": "write a parser"}
        assert primary.send_calls == []

    def test_the_next_turn_returns_to_the_primary(self):
        coding = FakeClient(provider="openai", model="gpt-5-codex")
        loop, primary = make_loop(["/code fix it", "hello", "exit"], coding_client=coding)
        loop.run()
        assert len(coding.send_calls) == 1
        assert len(primary.send_calls) == 1

    def test_with_no_coding_model_the_primary_answers_anyway(self):
        loop, primary = make_loop(["/code do a thing", "exit"])
        loop.run()
        assert primary.send_calls[0][-1] == {"role": "user", "content": "do a thing"}

    def test_bare_code_command_shows_usage_and_consumes_no_turn(self):
        """A bare /code must show usage, not reach the model as literal text."""
        loop, primary = make_loop(["/code", "exit"])
        loop.run()
        assert primary.send_calls == []
        assert loop.turns_used == 0


class TestKeyRotation:
    def _rate_limit(self):
        return litellm.exceptions.RateLimitError(
            message="slow down", llm_provider="openai", model="gpt-5"
        )

    def test_a_rate_limit_retries_the_same_model_with_the_next_key(self):
        client = FakeClient(errors=[self._rate_limit()], api_key="key-1")
        fallback = FakeClient(provider="claude", model="haiku")
        loop, _ = make_loop(
            ["hello", "exit"],
            client=client,
            fallback_clients=[fallback],
            primary_api_keys=["key-1", "key-2"],
        )
        loop.run()
        assert client.keys_used == ["key-1", "key-2"]
        # Rotation happens BEFORE the chain is considered.
        assert fallback.send_calls == []
        assert len(client.send_calls) == 1

    def test_only_one_rotation_per_turn(self):
        client = FakeClient(errors=[self._rate_limit(), self._rate_limit()])
        loop, _ = make_loop(
            ["hello", "exit"], client=client, primary_api_keys=["key-1", "key-2"]
        )
        loop.run()
        # Two keys tried, then the error surfaces — never a loop.
        assert client.keys_used == ["key-1", "key-2"]

    def test_a_single_key_behaves_exactly_as_before(self):
        client = FakeClient(errors=[self._rate_limit()])
        loop, _ = make_loop(["hello", "exit"], client=client, primary_api_keys=["key-1"])
        loop.run()
        assert client.keys_used == ["key-1"]
        assert client.send_calls == []

    def test_a_rotated_key_is_not_retried_on_a_later_turn(self):
        client = FakeClient(errors=[self._rate_limit(), None, self._rate_limit()])
        loop, _ = make_loop(
            ["one", "two", "exit"], client=client, primary_api_keys=["key-1", "key-2", "key-3"]
        )
        loop.run()
        # key-1 rate-limited, rotate to key-2 (succeeds); next turn starts on
        # key-2, is rate-limited, rotates to key-3.
        assert client.keys_used == ["key-1", "key-2", "key-2", "key-3"]


class TestKeySlots:
    def test_get_api_keys_returns_base_first_and_ignores_gaps(self, tmp_path):
        from nero.config.manager import KEYRING_SERVICE, ConfigManager

        import keyring

        manager = ConfigManager(tmp_path / "config.yaml")
        keyring.set_password(KEYRING_SERVICE, "openai_api_key", "base")
        keyring.set_password(KEYRING_SERVICE, "openai_api_key_3", "third")
        assert manager.get_api_keys("openai") == ["base", "third"]
        # Regression lock: every existing caller still sees the base key.
        assert manager.get_api_key("openai") == "base"

    def test_set_api_key_writes_the_requested_slot(self, tmp_path):
        from nero.config.manager import ConfigManager

        manager = ConfigManager(tmp_path / "config.yaml")
        manager.set_api_key("openai", "base")
        manager.set_api_key("openai", "second", slot=2)
        assert manager.get_api_keys("openai") == ["base", "second"]

    def test_a_keyless_provider_has_no_slots(self, tmp_path):
        from nero.config.manager import ConfigManager

        assert ConfigManager(tmp_path / "config.yaml").get_api_keys("ollama") == []


class TestBareImageCommand:
    def test_bare_image_command_shows_usage_and_consumes_no_turn(self):
        loop, primary = make_loop(["/image", "exit"])
        loop.run()
        assert primary.send_calls == []
        assert loop.turns_used == 0
