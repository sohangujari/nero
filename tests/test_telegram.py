"""The Telegram bridge (nero/telegram.py).

The allowlist is the whole security model — a bot token is a URL anyone can
message, and Nero can open apps and read files on the machine. Most of what is
locked down here is that boundary.
"""

from datetime import timedelta

import time

import httpx
import pytest

from nero.telegram import (
    MAX_MESSAGE_CHARS,
    PAIRING_TTL_SECONDS,
    PairingStore,
    TelegramBot,
    TelegramError,
    incoming,
    serve,
    _split,
)


@pytest.fixture
def pairings(tmp_path):
    """Never the real ~/Library state file."""
    return PairingStore(tmp_path / "telegram.db")


def message(update_id, chat_id, text):
    return {"update_id": update_id, "message": {"chat": {"id": chat_id}, "text": text}}


class FakeBot:
    """Stands in for TelegramBot: scripted updates in, sent messages out."""

    def __init__(self, batches):
        self._batches = list(batches)
        self.sent = []
        self.offsets = []
        self.typing_for = []

    def updates(self, offset):
        self.offsets.append(offset)
        return self._batches.pop(0) if self._batches else []

    def send(self, chat_id, text):
        self.sent.append((chat_id, text))

    def typing(self, chat_id):
        self.typing_for.append(chat_id)


class TestAllowlist:
    def test_an_unpaired_chats_message_never_reaches_the_model(self, pairings):
        """The one invariant that cannot bend: a stranger can ask to pair, and
        that is all. Their text is not a turn."""
        bot = FakeBot([[message(1, 999, "open calculator")]])
        asked = []
        serve(bot, {42}, lambda t: asked.append(t) or "ok", once=True, pairings=pairings)
        assert asked == []
        assert len(bot.sent) == 1
        assert "pairing code" in bot.sent[0][1].lower()

    def test_the_code_goes_to_telegram_and_not_to_the_terminal(self, pairings):
        """Approving is meant to prove the approver is holding the phone. If
        the code were printed at the terminal too, it would prove nothing."""
        bot = FakeBot([[message(1, 999, "hi")]])
        events = []
        serve(bot, {42}, lambda t: "ok", once=True, pairings=pairings, on_event=events.append)
        code = pairings.pending() and bot.sent[0][1]
        digits = "".join(c for c in code if c.isdigit())
        assert digits, "no code was sent to Telegram"
        assert not any(digits[:6] in event for event in events)

    def test_a_paired_chat_is_answered(self, pairings):
        bot = FakeBot([[message(1, 42, "hello")]])
        serve(bot, {42}, lambda t: f"you said {t}", once=True, pairings=pairings)
        assert bot.sent == [(42, "you said hello")]

    def test_with_pairing_off_an_empty_allowlist_refuses_to_run(self, pairings):
        # The door-shut mode: no allowlist and no way in is a misconfiguration,
        # not a bot that answers everyone.
        with pytest.raises(TelegramError, match="No chat is allowed"):
            serve(FakeBot([]), set(), lambda t: "ok", once=True,
                  pairings=pairings, allow_pairing=False)

    def test_with_pairing_off_a_stranger_gets_silence(self, pairings):
        bot = FakeBot([[message(1, 999, "hi")]])
        serve(bot, {42}, lambda t: "ok", once=True, pairings=pairings, allow_pairing=False)
        assert bot.sent == []

    def test_a_refused_message_is_still_acknowledged_to_telegram(self, pairings):
        """Without advancing the offset past it, Telegram redelivers a refused
        message on every poll, forever."""

        class TwoPolls(FakeBot):
            def updates(self, offset):
                self.offsets.append(offset)
                if not self._batches:
                    raise KeyboardInterrupt  # end the loop after the second poll
                return self._batches.pop(0)

        bot = TwoPolls([[message(7, 999, "hi")]])
        with pytest.raises(KeyboardInterrupt):
            serve(bot, {42}, lambda t: "ok", pairings=pairings)
        assert bot.offsets == [None, 8], "the refused update was not skipped"


class TestPairingStore:
    def test_a_code_approves_its_own_chat(self, pairings):
        code = pairings.request(42)
        assert pairings.approve(code) == 42

    def test_a_code_is_single_use(self, pairings):
        code = pairings.request(42)
        pairings.approve(code)
        assert pairings.approve(code) is None

    def test_a_wrong_code_approves_nothing(self, pairings):
        pairings.request(42)
        assert pairings.approve("000000") is None

    def test_asking_again_invalidates_the_previous_code(self, pairings):
        """Otherwise a chat could farm codes and widen its guessing odds."""
        first = pairings.request(42)
        second = pairings.request(42)
        assert pairings.approve(first) is None
        assert pairings.approve(second) == 42

    def test_one_row_per_chat_however_often_it_asks(self, pairings):
        for _ in range(50):
            pairings.request(42)
        assert len(pairings.pending()) == 1

    def test_many_chats_probing_cannot_grow_the_queue_without_bound(self, pairings):
        for chat in range(60):
            pairings.request(chat)
        assert len(pairings.pending()) <= 20

    def test_an_expired_request_cannot_be_approved(self, pairings, monkeypatch):
        import nero.telegram as telegram_module

        code = pairings.request(42)
        real = telegram_module.datetime

        class Later(real):
            @classmethod
            def now(cls, tz=None):
                return real.now(tz) + timedelta(seconds=PAIRING_TTL_SECONDS + 60)

        monkeypatch.setattr(telegram_module, "datetime", Later)
        assert pairings.approve(code) is None
        assert pairings.pending() == []

    def test_codes_are_six_digits(self, pairings):
        code = pairings.request(42)
        assert len(code) == 6 and code.isdigit()

    def test_pending_reports_who_is_waiting(self, pairings):
        pairings.request(42)
        waiting = pairings.pending()
        assert [p.chat_id for p in waiting] == [42]
        assert waiting[0].age().endswith("ago")


class TestServing:
    def test_shows_typing_before_a_slow_turn(self, pairings):
        bot = FakeBot([[message(1, 42, "hi")]])
        serve(bot, {42}, lambda t: "ok", once=True, pairings=pairings)
        assert bot.typing_for == [42]

    def test_a_failing_turn_is_reported_not_crashed(self, pairings):
        """The bridge has to outlive a bad turn — it is a long-running server."""
        def boom(_text):
            raise RuntimeError("provider exploded")

        bot = FakeBot([[message(1, 42, "hi")]])
        serve(bot, {42}, boom, once=True, pairings=pairings)
        assert len(bot.sent) == 1
        assert "provider exploded" in bot.sent[0][1]

    def test_an_empty_reply_still_sends_something(self, pairings):
        bot = FakeBot([[message(1, 42, "hi")]])
        serve(bot, {42}, lambda t: None, once=True, pairings=pairings)
        assert bot.sent == [(42, "(no reply)")]

    def test_every_message_in_a_batch_is_answered_in_order(self, pairings):
        bot = FakeBot([[message(4, 42, "a"), message(5, 42, "b")]])
        serve(bot, {42}, lambda t: t.upper(), once=True, pairings=pairings)
        assert bot.sent == [(42, "A"), (42, "B")]

    def test_non_text_updates_are_skipped(self, pairings):
        sticker = {"update_id": 1, "message": {"chat": {"id": 42}, "sticker": {}}}
        bot = FakeBot([[sticker]])
        serve(bot, {42}, lambda t: "ok", once=True, pairings=pairings)
        assert bot.sent == []


class TestIncoming:
    def test_reads_chat_and_text(self):
        assert incoming(message(1, 42, "  hi  ")) == (42, "hi")

    def test_ignores_an_update_with_no_message(self):
        assert incoming({"update_id": 1, "edited_message": {}}) is None

    def test_ignores_an_empty_message(self):
        assert incoming(message(1, 42, "   ")) is None


class TestSplit:
    def test_short_text_is_one_message(self):
        assert _split("hello") == ["hello"]

    def test_long_text_is_split_not_truncated(self):
        parts = _split("x" * (MAX_MESSAGE_CHARS * 2 + 50))
        assert all(len(p) <= MAX_MESSAGE_CHARS for p in parts)
        assert sum(len(p) for p in parts) == MAX_MESSAGE_CHARS * 2 + 50

    def test_breaks_on_a_newline_when_it_can(self):
        text = "a" * (MAX_MESSAGE_CHARS - 10) + "\n" + "b" * 100
        assert _split(text)[0].endswith("a")

    def test_empty_text_is_never_sent_as_empty(self):
        assert _split("") == ["(no reply)"]


class TestBotApi:
    def bot(self, handler):
        transport = httpx.MockTransport(handler)
        return TelegramBot("token", client=httpx.Client(transport=transport))

    def test_a_rejected_token_says_so_plainly(self):
        bot = self.bot(lambda r: httpx.Response(401, json={"ok": False}))
        with pytest.raises(TelegramError, match="rejected the bot token"):
            bot.username()

    def test_an_api_error_carries_telegrams_description(self):
        bot = self.bot(
            lambda r: httpx.Response(400, json={"ok": False, "description": "chat not found"})
        )
        with pytest.raises(TelegramError, match="chat not found"):
            bot.send(1, "hi")

    def test_a_network_failure_is_wrapped(self):
        def boom(_request):
            raise httpx.ConnectError("no route")

        with pytest.raises(TelegramError, match="Could not reach Telegram"):
            self.bot(boom).username()

    def test_the_token_is_never_in_an_error_message(self):
        """Errors get printed and logged; the token must not ride along."""
        def boom(_request):
            raise httpx.ConnectError("no route")

        with pytest.raises(TelegramError) as caught:
            self.bot(boom).username()
        assert "token" not in str(caught.value).replace("bot token", "")

    def test_long_replies_go_out_as_several_messages(self):
        seen = []
        bot = self.bot(
            lambda r: seen.append(r) or httpx.Response(200, json={"ok": True, "result": {}})
        )
        bot.send(42, "y" * (MAX_MESSAGE_CHARS + 100))
        assert len(seen) == 2

    def test_a_typing_failure_never_breaks_the_turn(self):
        bot = self.bot(lambda r: httpx.Response(500, json={"ok": False}))
        bot.typing(42)  # must not raise


class TestEndToEnd:
    """The bridge against a mock Telegram, driving a real ChatLoop.ask."""

    def test_a_message_becomes_a_turn_and_the_reply_comes_back(self, pairings):
        import io

        from rich.console import Console

        from nero.core.chat_loop import ChatLoop

        class FakeClient:
            provider = "claude"

            def send(self, messages, on_text):
                on_text("blue")
                messages.append({"role": "assistant", "content": "Your colour is blue."})

        loop = ChatLoop(
            FakeClient(),
            console=Console(file=io.StringIO(), force_terminal=False),
            assistant_name="Nero",
        )

        sent = []
        polls = {"n": 0}

        def handler(request):
            body = request.read().decode()
            if "getUpdates" in str(request.url):
                polls["n"] += 1
                if polls["n"] > 1:
                    raise KeyboardInterrupt
                return httpx.Response(
                    200,
                    json={"ok": True, "result": [message(1, 42, "what colour do I like?")]},
                )
            if "sendMessage" in str(request.url):
                sent.append(body)
            return httpx.Response(200, json={"ok": True, "result": {}})

        bot = TelegramBot("t", client=httpx.Client(transport=httpx.MockTransport(handler)))
        with pytest.raises(KeyboardInterrupt):
            serve(bot, {42}, loop.ask, pairings=pairings)

        assert len(sent) == 1
        assert "Your colour is blue." in sent[0]
        # The turn really ran: it is in the conversation, not just echoed back.
        assert loop.messages[-1] == {"role": "assistant", "content": "Your colour is blue."}


class TestConfigMenu:
    """`nero config` is where most people will connect this, so the row has to
    be there, has to say what state the bridge is in, and must not disturb the
    ordinals the rest of the menu is addressed by."""

    def test_the_menu_offers_a_telegram_row(self, monkeypatch, tmp_path, isolate_audit_log):
        from tests.test_config_pickers import _manager, runner

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="\n")
        assert result.exit_code == 0
        assert "Telegram" in result.stdout

    def test_the_row_keeps_ordinal_17_so_nothing_else_shifts(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        # 16 is Endpoint URL / AWS Region and is addressed by number in
        # tests/test_bedrock.py; Telegram appends after it.
        from tests.test_config_pickers import _manager, runner

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        offered = []

        def fake_pick(title, choices, **kwargs):
            offered.append(choices)
            return ""

        monkeypatch.setattr(cli.ui, "pick", fake_pick)
        runner.invoke(cli.app, ["config"])
        rows = dict(offered[0])
        assert "17" in rows and "Telegram" in rows["17"]

    def test_the_row_says_what_still_needs_doing(self, tmp_path):
        from nero import cli
        from nero.config.schema import NeroConfig

        class NoToken:
            def get_telegram_token(self):
                return None

        class WithToken:
            def get_telegram_token(self):
                return "t"

        config = NeroConfig()
        assert "not set up" in cli._telegram_summary(NoToken(), config)
        assert "no chat paired" in cli._telegram_summary(WithToken(), config)
        config.telegram.allowed_chat_ids = [1, 2]
        assert "2 chats paired" in cli._telegram_summary(WithToken(), config)

    def test_setup_is_one_routine_for_both_entry_points(self):
        """The command and the menu must not drift into asking different
        things."""
        import inspect

        from nero import cli

        assert "_connect_telegram(" in inspect.getsource(cli.telegram_setup)
        assert "_connect_telegram(" in inspect.getsource(cli._interactive_menu)


class TestMenuReachability:
    def test_selecting_the_row_starts_setup(self, monkeypatch, tmp_path, isolate_audit_log):
        """The numbered fallback lists rows by position, not by row value, so
        the position the user types and the branch that runs must be checked
        together — not assumed from the ordinal."""
        from tests.test_config_pickers import _manager, runner

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        called = []
        monkeypatch.setattr(cli, "_connect_telegram", lambda m: called.append(True) or True)
        # Telegram is the last row, so its position depends on whether the
        # provider adds an Endpoint URL / AWS Region row.
        runner.invoke(cli.app, ["config"], input="16\n\n")
        assert called, "the Telegram row did not reach _connect_telegram"


def test_a_hand_set_voice_still_shows_a_gender():
    """The catalog is curated; voice.tts.voice_id is a free string. A voice
    outside the list must not render as '(?)'."""
    from nero import cli

    assert cli._voice_gender("am_adam") == "male"
    assert cli._voice_gender("af_heart") == "female"


class TestApproveCommand:
    def _cli(self, monkeypatch, tmp_path):
        from tests.test_config_pickers import _manager, runner

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        store = PairingStore(tmp_path / "telegram.db")
        monkeypatch.setattr(cli, "PairingStore", lambda: store)
        return cli, manager, store, runner

    def test_a_valid_code_writes_the_chat_into_the_allowlist(self, monkeypatch, tmp_path):
        cli, manager, store, runner = self._cli(monkeypatch, tmp_path)
        code = store.request(42)
        result = runner.invoke(cli.app, ["telegram", "approve", code])
        assert result.exit_code == 0
        assert manager.load().telegram.allowed_chat_ids == [42]
        assert manager.load().telegram.enabled is True

    def test_a_bad_code_changes_nothing_and_fails(self, monkeypatch, tmp_path):
        cli, manager, store, runner = self._cli(monkeypatch, tmp_path)
        store.request(42)
        result = runner.invoke(cli.app, ["telegram", "approve", "000000"])
        assert result.exit_code == 1
        assert manager.load().telegram.allowed_chat_ids == []

    def test_approving_a_second_device_keeps_the_first(self, monkeypatch, tmp_path):
        cli, manager, store, runner = self._cli(monkeypatch, tmp_path)
        runner.invoke(cli.app, ["telegram", "approve", store.request(42)])
        runner.invoke(cli.app, ["telegram", "approve", store.request(7)])
        assert manager.load().telegram.allowed_chat_ids == [7, 42]

    def test_pending_lists_who_is_waiting_without_leaking_codes(self, monkeypatch, tmp_path):
        cli, _manager_, store, runner = self._cli(monkeypatch, tmp_path)
        code = store.request(42)
        result = runner.invoke(cli.app, ["telegram", "pending"])
        assert "42" in result.stdout
        assert code not in result.stdout, "the code must only ever be readable in Telegram"

    def test_pending_says_so_when_nobody_is_waiting(self, monkeypatch, tmp_path):
        cli, _manager_, _store, runner = self._cli(monkeypatch, tmp_path)
        result = runner.invoke(cli.app, ["telegram", "pending"])
        assert result.exit_code == 0
        assert "No chats" in result.stdout


class TestApprovalTakesEffectWhileRunning:
    def test_a_chat_approved_mid_run_is_answered_without_a_restart(self, pairings):
        """The bug: the allowlist was read once at startup. You approved, the
        phone kept getting pairing codes, and nothing said why."""
        allowed = set()

        class TwoPolls(FakeBot):
            def updates(self, offset):
                self.offsets.append(offset)
                if not self._batches:
                    raise KeyboardInterrupt
                # Approval lands while the poll is open, as it would if you
                # ran `nero telegram approve` in another terminal.
                if len(self.offsets) > 1:
                    allowed.add(42)
                return self._batches.pop(0)

        bot = TwoPolls([[message(1, 42, "before")], [message(2, 42, "after")]])
        answered = []
        with pytest.raises(KeyboardInterrupt):
            serve(bot, allowed, lambda t: answered.append(t) or "ok",
                  pairings=pairings, refresh=lambda: set(allowed))
        assert answered == ["after"], "the mid-run approval was not picked up"

    def test_without_refresh_the_allowlist_is_still_honoured(self, pairings):
        bot = FakeBot([[message(1, 42, "hi")]])
        answered = []
        serve(bot, {42}, lambda t: answered.append(t) or "ok", once=True, pairings=pairings)
        assert answered == ["hi"]


class TestBridgeService:
    """`nero telegram install` — the answer to "how do I keep it running"."""

    def plist(self, tmp_path, monkeypatch):
        import plistlib

        from nero import routines

        # launchctl must never actually run in a test: a stray loaded agent
        # outlives the test run.
        monkeypatch.setattr(routines.sys, "platform", "linux")
        routines.install_bridge("/opt/nero/bin/nero", tmp_path)
        return plistlib.loads(routines.bridge_plist_path(tmp_path).read_bytes())

    def test_it_runs_the_bridge_at_login_and_keeps_it_alive(self, tmp_path, monkeypatch):
        plist = self.plist(tmp_path, monkeypatch)
        assert plist["ProgramArguments"] == ["/opt/nero/bin/nero", "telegram"]
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True

    def test_it_throttles_respawns(self, tmp_path, monkeypatch):
        # A process that dies instantly would otherwise respawn in a tight loop
        # and hammer Telegram.
        assert self.plist(tmp_path, monkeypatch)["ThrottleInterval"] >= 10

    def test_it_writes_logs_somewhere_findable(self, tmp_path, monkeypatch):
        plist = self.plist(tmp_path, monkeypatch)
        assert plist["StandardErrorPath"].endswith("telegram.err.log")

    def test_uninstall_removes_the_agent(self, tmp_path, monkeypatch):
        from nero import routines

        monkeypatch.setattr(routines.sys, "platform", "linux")
        routines.install_bridge("/opt/nero/bin/nero", tmp_path)
        assert routines.bridge_plist_path(tmp_path).exists()
        routines.uninstall_bridge(tmp_path)
        assert not routines.bridge_plist_path(tmp_path).exists()

    def test_uninstalling_what_was_never_installed_is_not_an_error(self, tmp_path, monkeypatch):
        from nero import routines

        monkeypatch.setattr(routines.sys, "platform", "linux")
        assert "not installed" in routines.uninstall_bridge(tmp_path)

    def test_it_does_not_collide_with_a_routine_agent(self, tmp_path, monkeypatch):
        from nero import routines

        assert routines.BRIDGE_LABEL != routines.label_for("telegram")

    def test_install_refuses_without_a_paired_chat(self, monkeypatch, tmp_path):
        """Starting a service with nobody to answer is a silent no-op the user
        would have to debug from logs."""
        from tests.test_config_pickers import _manager, runner

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(manager, "get_telegram_token", lambda: "t", raising=False)
        result = runner.invoke(cli.app, ["telegram", "install"])
        assert result.exit_code == 1
        assert "Nothing is paired" in result.stdout


class TestUniversalCommand:
    """`nero` runs everything; `nero chat` / `nero talk` / `nero telegram` are
    the single interfaces."""

    def test_bare_nero_asks_for_the_bridge(self):
        import inspect

        from nero import cli

        assert "_run_chat(with_telegram=True)" in inspect.getsource(cli.main)

    def test_nero_chat_does_not(self):
        import inspect

        from nero import cli

        assert "_run_chat(with_telegram=False)" in inspect.getsource(cli.chat)

    def test_the_bridge_is_silent_when_telegram_is_not_set_up(self, monkeypatch, tmp_path):
        """`nero` must behave exactly as it always has for anyone who never
        touched Telegram — no thread, no line of output."""
        from tests.test_config_pickers import _manager

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(manager, "get_telegram_token", lambda: None, raising=False)
        started = []
        monkeypatch.setattr(cli.threading, "Thread", lambda **k: started.append(k))
        assert cli._start_telegram_bridge(manager, manager.load(), object()) is None
        assert started == []

    def test_no_bridge_without_a_paired_chat(self, monkeypatch, tmp_path):
        from tests.test_config_pickers import _manager

        from nero import cli

        manager = _manager(tmp_path)
        monkeypatch.setattr(manager, "get_telegram_token", lambda: "t", raising=False)
        config = manager.load()
        assert config.telegram.allowed_chat_ids == []
        assert cli._start_telegram_bridge(manager, config, object()) is None

    def test_the_bridge_starts_when_configured(self, monkeypatch, tmp_path):
        from tests.test_config_pickers import _manager

        from nero import cli

        manager = _manager(tmp_path)
        manager.set_value("telegram.allowed_chat_ids", "42")
        monkeypatch.setattr(manager, "get_telegram_token", lambda: "t", raising=False)
        started = {}

        class FakeThread:
            def __init__(self, **kwargs):
                started.update(kwargs)

            def start(self):
                started["started"] = True

        monkeypatch.setattr(cli.threading, "Thread", FakeThread)
        stop = cli._start_telegram_bridge(manager, manager.load(), object())
        assert stop is not None and started.get("started")
        assert started["daemon"] is True, "must not hold the process open at exit"


class TestOneTurnAtATime:
    def test_two_interfaces_cannot_interleave_a_turn(self):
        """The terminal and the bridge share one ChatLoop. Without the lock,
        two turns append to self.messages at once and the provider is handed an
        interleaved conversation."""
        import io
        import threading

        from rich.console import Console

        from nero.core.chat_loop import ChatLoop

        overlaps = []
        inside = threading.Event()

        class SlowClient:
            provider = "claude"

            def send(self, messages, on_text):
                overlaps.append(len([m for m in messages if m.get("role") == "user"]))
                if not inside.is_set():
                    inside.set()
                    time.sleep(0.05)  # hold the turn open
                messages.append({"role": "assistant", "content": "ok"})

        loop = ChatLoop(
            SlowClient(),
            console=Console(file=io.StringIO(), force_terminal=False),
            assistant_name="Nero",
        )
        threads = [threading.Thread(target=loop.ask, args=(f"q{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each turn saw exactly one more user message than the last: no turn
        # started while another was mid-flight.
        assert overlaps == [1, 2, 3, 4], overlaps
        assert len(loop.messages) == 8
