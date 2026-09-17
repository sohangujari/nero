"""Reminders, and the half that makes them reminders: delivery.

Nero had routines before this and they cannot remind you of anything — an
installed routine writes its reply to a log file. The tests that matter here
are the ones about a reminder arriving.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from nero.reminders import (
    MAX_LATE,
    Reminder,
    ReminderStore,
    next_occurrence,
    parse_when,
)
from nero.skills.reminders.server import (
    CancelReminderSkill,
    ListRemindersSkill,
    RemindMeSkill,
)

NOW = datetime.fromisoformat("2026-09-11T14:30:00+05:30")  # a Friday


@pytest.fixture
def store(tmp_path):
    """Never the real ~/Library state file."""
    return ReminderStore(tmp_path / "reminders.db")


def run(skill, **kwargs):
    return asyncio.run(skill.execute(**kwargs))


class TestParsingATime:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2026-09-11T17:00:00", "2026-09-11 17:00"),
            ("5pm", "2026-09-11 17:00"),
            ("5:00 pm", "2026-09-11 17:00"),
            ("17:00", "2026-09-11 17:00"),
            ("at 10 pm", "2026-09-11 22:00"),
            ("23:59", "2026-09-11 23:59"),
            ("in 20 minutes", "2026-09-11 14:50"),
            ("in 2 hours", "2026-09-11 16:30"),
            ("in 3 days", "2026-09-14 14:30"),
            ("tomorrow at 9", "2026-09-12 09:00"),
            ("tomorrow 9am", "2026-09-12 09:00"),
        ],
    )
    def test_the_everyday_forms(self, text, expected):
        assert parse_when(text, now=NOW).strftime("%Y-%m-%d %H:%M") == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("5.25 pm", "2026-09-11 17:25"),
            ("5.25pm", "2026-09-11 17:25"),
            ("at 5.25 pm", "2026-09-11 17:25"),
            ("17.25", "2026-09-11 17:25"),
            ("9.05 a.m.", "2026-09-12 09:05"),
            ("tomorrow 8.45 am", "2026-09-12 08:45"),
        ],
    )
    def test_a_dot_separates_minutes_just_like_a_colon(self, text, expected):
        """"5.25 pm" is how most people write it. Reading only the "5" dropped
        the minutes *and* the pm, and silently set the reminder for 05:00 the
        next morning — the exact failure this parser exists to avoid."""
        assert parse_when(text, now=NOW).strftime("%Y-%m-%d %H:%M") == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("half past 5", "2026-09-11 17:30"),
            ("quarter past 6", "2026-09-11 18:15"),
            ("quarter to 7", "2026-09-11 18:45"),
            ("noon", "2026-09-12 12:00"),
        ],
    )
    def test_the_spoken_forms(self, text, expected):
        assert parse_when(text, now=NOW).strftime("%Y-%m-%d %H:%M") == expected

    @pytest.mark.parametrize("text", ["5 past 9", "around 5", "5-ish", "ten past the hour"])
    def test_a_time_phrase_it_half_understands_is_refused(self, text):
        """Each of these contains a number the clock pattern would happily read
        alone and turn into the wrong hour. Refusing beats a plausible guess."""
        assert parse_when(text, now=NOW) is None

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("today at 6.20 pm", "2026-09-11 18:20"),
            ("6.20 pm today", "2026-09-11 18:20"),
            ("tonight at 9", "2026-09-11 21:00"),
            ("this evening at 7", "2026-09-11 19:00"),
        ],
    )
    def test_a_day_the_user_named_is_honoured(self, text, expected):
        assert parse_when(text, now=NOW).strftime("%Y-%m-%d %H:%M") == expected

    def test_saying_today_never_moves_it_to_tomorrow(self):
        """"Today at 5 pm" said at half past six is a mistake worth pointing
        out. Silently making it tomorrow contradicts what the user said, and
        they only find out when it does not arrive."""
        evening = NOW.replace(hour=18, minute=30)
        due = parse_when("today at 5 pm", now=evening)
        assert due.strftime("%Y-%m-%d %H:%M") == "2026-09-11 17:00"
        assert due < evening  # the caller reports it as gone rather than moving it

    def test_an_evening_word_resolves_the_hour(self):
        """"Tonight at 9" is 21:00. Taking merely the soonest reading of an
        ambiguous hour gave 09:00 — this morning, and already gone."""
        evening = NOW.replace(hour=18, minute=10)
        assert parse_when("tonight at 9", now=evening).hour == 21

    def test_an_ambiguous_hour_prefers_a_time_still_to_come(self):
        evening = NOW.replace(hour=18, minute=10)
        assert parse_when("at 9", now=evening).strftime("%a %H:%M") == "Fri 21:00"

    def test_a_time_that_has_passed_today_means_tomorrow(self):
        """"Remind me at 9" at half past two means tomorrow morning, not a
        reminder that can never fire."""
        assert parse_when("9am", now=NOW).strftime("%Y-%m-%d %H:%M") == "2026-09-12 09:00"

    @pytest.mark.parametrize("text", ["sometime later", "banana", "", "   ", "when I get home"])
    def test_anything_else_is_refused_not_guessed(self, text):
        """A reminder set for the wrong time is worse than one that was not
        set, because the user stops watching for it."""
        assert parse_when(text, now=NOW) is None

    def test_an_impossible_clock_time_is_refused(self):
        assert parse_when("at 99:99", now=NOW) is None

    def test_a_naive_iso_timestamp_gets_the_local_zone(self):
        """Otherwise comparing it against `now` raises, and the reminder is
        lost at fire time rather than at set time."""
        parsed = parse_when("2026-09-11T17:00:00", now=NOW)
        assert parsed.tzinfo is not None


class TestRepeating:
    def test_daily_moves_a_day(self):
        assert next_occurrence(NOW, "daily").day == 12

    def test_weekly_moves_a_week(self):
        assert next_occurrence(NOW, "weekly").day == 18

    def test_weekdays_skips_the_weekend(self):
        """Friday's next weekday is Monday, not Saturday."""
        assert next_occurrence(NOW, "weekdays").strftime("%a %d") == "Mon 14"

    def test_a_one_off_does_not_come_back(self):
        assert next_occurrence(NOW, "") is None


class TestStore:
    def test_a_reminder_is_stored_and_listed(self, store):
        store.add("take your medicine", NOW + timedelta(hours=1))
        assert [r.text for r in store.pending()] == ["take your medicine"]

    def test_pending_is_soonest_first(self, store):
        store.add("later", NOW + timedelta(hours=5))
        store.add("sooner", NOW + timedelta(hours=1))
        assert [r.text for r in store.pending()] == ["sooner", "later"]

    def test_nothing_is_due_before_its_time(self, store):
        store.add("later", NOW + timedelta(hours=1))
        assert store.due(now=NOW) == []

    def test_something_is_due_once_its_time_comes(self, store):
        store.add("now", NOW)
        assert [r.text for r in store.due(now=NOW)] == ["now"]

    def test_a_missed_reminder_is_still_delivered(self, store):
        """A laptop asleep at 17:00 has no reminder at 17:00. Dropping a
        medicine reminder silently is the worst option available."""
        store.add("medicine", NOW)
        assert [r.text for r in store.due(now=NOW + timedelta(hours=3))] == ["medicine"]

    def test_a_reminder_missed_by_too_long_is_not_delivered(self, store):
        """A laptop shut for a week should not wake up and recite Monday."""
        store.add("medicine", NOW)
        assert store.due(now=NOW + MAX_LATE + timedelta(minutes=1)) == []

    def test_a_delivered_reminder_does_not_come_back(self, store):
        reminder = store.add("once", NOW)
        store.mark_delivered(reminder)
        assert store.due(now=NOW) == []
        assert store.pending() == []

    def test_a_repeating_reminder_re_arms_itself(self, store):
        reminder = store.add("standup", NOW, repeat="daily")
        following = store.mark_delivered(reminder)
        assert following is not None
        assert following.due_at.day == NOW.day + 1
        assert [r.text for r in store.pending()] == ["standup"]

    def test_cancelling_removes_it(self, store):
        reminder = store.add("nope", NOW + timedelta(hours=1))
        assert store.cancel(reminder.id) is True
        assert store.pending() == []

    def test_cancelling_something_that_is_not_there_says_so(self, store):
        assert store.cancel(999) is False

    def test_clear_removes_everything_pending(self, store):
        store.add("a", NOW + timedelta(hours=1))
        store.add("b", NOW + timedelta(hours=2))
        assert store.clear() == 2
        assert store.pending() == []


class TestTheMessage:
    def test_on_time_it_is_just_the_words(self):
        reminder = Reminder(id=1, text="take your medicine", due_at=NOW)
        assert reminder.message(now=NOW) == "⏰ take your medicine"

    def test_late_it_says_what_time_it_was_for(self):
        """Arriving at 16:30 without saying it was due at 14:30 is misleading
        about when the medicine was actually due."""
        reminder = Reminder(id=1, text="take your medicine", due_at=NOW)
        late = reminder.message(now=NOW + timedelta(hours=2))
        assert "14:30" in late
        assert "take your medicine" in late

    def test_a_couple_of_minutes_late_is_not_worth_mentioning(self):
        """The tick runs every minute, so almost every reminder is a few
        seconds late. Saying so on all of them is noise."""
        reminder = Reminder(id=1, text="medicine", due_at=NOW)
        assert reminder.message(now=NOW + timedelta(seconds=40)) == "⏰ medicine"

    def test_no_model_is_involved(self):
        """The user's own words, verbatim. A model at fire time could drift,
        cost a round trip, or apologise."""
        reminder = Reminder(id=1, text="call mum about the thing", due_at=NOW)
        assert "call mum about the thing" in reminder.message(now=NOW)


class TestTheSkill:
    def _skill(self, store, ensure=None):
        # `ensure` is stubbed in every test here. The real one writes a launchd
        # agent into ~/Library/LaunchAgents, and a test that forgot this once
        # installed a live agent running `pytest remind tick` every minute.
        return RemindMeSkill(store, now=lambda: NOW, ensure=ensure or (lambda: True))

    def test_setting_one_confirms_the_time(self, store):
        result = run(self._skill(store), text="take your medicine", when="5pm")
        assert "17:00" in result
        assert [r.text for r in store.pending()] == ["take your medicine"]

    def test_a_time_it_cannot_read_is_refused_with_a_question(self, store):
        result = run(self._skill(store), text="something", when="whenever")
        assert "couldn't work out" in result
        assert store.pending() == []

    def test_no_text_is_refused(self, store):
        assert "Error" in run(self._skill(store), text="  ", when="5pm")
        assert store.pending() == []

    def test_a_repeat_it_does_not_know_is_refused(self, store):
        result = run(self._skill(store), text="x", when="5pm", repeat="fortnightly")
        assert "Error" in result
        assert store.pending() == []

    def test_a_valid_repeat_is_kept(self, store):
        run(self._skill(store), text="standup", when="9am", repeat="weekdays")
        assert store.pending()[0].repeat == "weekdays"

    def test_listing_says_so_when_there_is_nothing(self, store):
        assert run(ListRemindersSkill(store)) == "No reminders set."

    def test_listing_shows_the_number_needed_to_cancel(self, store):
        reminder = store.add("medicine", NOW + timedelta(hours=1))
        assert str(reminder.id) in run(ListRemindersSkill(store))

    def test_cancelling_by_number(self, store):
        reminder = store.add("medicine", NOW + timedelta(hours=1))
        assert "Cancelled" in run(CancelReminderSkill(store), reminder_id=reminder.id)
        assert store.pending() == []

    def test_cancelling_a_number_that_is_not_there(self, store):
        assert "no pending reminder" in run(CancelReminderSkill(store), reminder_id=42)

    def test_a_non_numeric_id_is_refused(self, store):
        assert "Error" in run(CancelReminderSkill(store), reminder_id="the medicine one")


class TestDeliveryIsSetUpForYou:
    """Six reminders once sat correctly stored and correctly due while no agent
    existed to send them. Delivery is not something to leave opt-in."""

    def test_setting_a_reminder_installs_delivery(self, store):
        calls = []
        skill = RemindMeSkill(store, now=lambda: NOW, ensure=lambda: calls.append(1) or True)
        run(skill, text="medicine", when="5pm")
        assert calls == [1]

    def test_the_user_is_told_when_it_could_not_be_set_up(self, store):
        skill = RemindMeSkill(store, now=lambda: NOW, ensure=lambda: False)
        result = run(skill, text="medicine", when="5pm")
        assert "nero remind install" in result

    def test_the_reminder_is_still_stored_either_way(self, store):
        skill = RemindMeSkill(store, now=lambda: NOW, ensure=lambda: False)
        run(skill, text="medicine", when="5pm")
        assert [r.text for r in store.pending()] == ["medicine"]

    def test_an_executable_that_is_not_nero_is_refused(self, tmp_path):
        """sys.argv[0] is whatever is running. Under pytest that is pytest."""
        from nero.routines import RoutineError, install_reminders

        with pytest.raises(RoutineError, match="not the nero executable"):
            install_reminders("/usr/local/bin/pytest", tmp_path)
        assert not (tmp_path / "com.neroagent.reminders.plist").exists()

    def test_the_real_executable_is_accepted(self, tmp_path, monkeypatch):
        import nero.routines as routines_module
        from nero.routines import install_reminders

        monkeypatch.setattr(routines_module.sys, "platform", "linux")
        install_reminders("/opt/homebrew/bin/nero", tmp_path)
        assert (tmp_path / "com.neroagent.reminders.plist").exists()


class TestNoDuplicates:
    def test_asking_twice_for_the_same_thing_is_one_reminder(self, store):
        """A retried tool call, a repeated turn, or a user saying it twice all
        arrive here identically. Three copies is three notifications."""
        from datetime import datetime as dt

        due = NOW + timedelta(hours=1)
        first = store.add("drink water", due)
        assert store.add("drink water", due).id == first.id
        assert store.add("drink water", due).id == first.id
        assert len(store.pending()) == 1
        assert isinstance(first.due_at, dt)

    def test_the_same_words_at_a_different_time_are_two_reminders(self, store):
        store.add("drink water", NOW + timedelta(hours=1))
        store.add("drink water", NOW + timedelta(hours=2))
        assert len(store.pending()) == 2

    def test_a_delivered_one_does_not_block_setting_it_again(self, store):
        """Tomorrow's "drink water at 17:25" is a real new reminder."""
        reminder = store.add("drink water", NOW)
        store.mark_delivered(reminder)
        store.add("drink water", NOW)
        assert len(store.pending()) == 1


class TestItIsOfferedToTheModel:
    """The reason this feature did not exist: with no skill to call, the model
    wrote "reminder = Hello at 4:46 AM" into the fact store — a place nothing
    reads at 4:46 — and reported success."""

    def test_the_model_can_set_a_reminder(self):
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        offered = {t["function"]["name"] for t in build_registry(NeroConfig()).tool_definitions()}
        assert {"remind_me", "list_reminders", "cancel_reminder"} <= offered

    def test_setting_a_reminder_is_not_destructive(self):
        """It writes one row the user can see and cancel. Gating it behind a
        confirmation would make it unusable from a phone, where the registry
        fails closed."""
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        registry = build_registry(NeroConfig())
        assert registry.get("remind_me").meta.permission_tier != "destructive"


class TestCatchingAClaimThatIsNotTrue:
    """The worst failure this feature has. Once two "I've set a reminder"
    replies are in the transcript, qwen3.5:2b reproduces the wording instead of
    calling the tool — 0 calls in 4, claiming success every time. A system
    prompt rule changed nothing (0 in 4 with it). Asking again on its own, one
    tool and no history, was right 15 times out of 15."""

    @pytest.mark.parametrize(
        "reply",
        [
            'I\'ve set a reminder to remind you to "drink water" at 5:25 PM.',
            "I have set a reminder for 5pm.",
            "I will remind you at 5:40 PM.",
            "Your reminder is set for 5pm.",
            "Reminder set for today at 17:40: check the website",
        ],
    )
    def test_a_claim_is_recognised(self, reply):
        from nero.reminders import claims_a_reminder

        assert claims_a_reminder(reply) is True

    @pytest.mark.parametrize(
        "reply",
        [
            "Here are your reminders: 1. today at 17:00 — medicine",
            "What would you like me to remind you about?",
            "You have no reminders set.",
            "2 + 2 equals 4.",
            "",
        ],
    )
    def test_talking_about_reminders_is_not_a_claim(self, reply):
        """Otherwise "what reminders do I have" would trigger a salvage and
        invent one."""
        from nero.reminders import claims_a_reminder

        assert claims_a_reminder(reply) is False


class TestTheRegistryKnowsWhatRan:
    """The only reliable answer to "did it really do that?"."""

    def test_a_skill_that_ran_is_recorded(self):
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        registry = build_registry(NeroConfig())
        registry.reset_turn()
        asyncio.run(registry.execute("list_reminders", {}))
        assert "list_reminders" in registry.called

    def test_a_new_turn_starts_empty(self):
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        registry = build_registry(NeroConfig())
        asyncio.run(registry.execute("list_reminders", {}))
        registry.reset_turn()
        assert registry.called == set()

    def test_a_refused_skill_still_counts_as_attempted(self):
        """It reached the registry, so the turn did try. What matters for the
        claim check is whether the model reached for the tool at all."""
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        config = NeroConfig()
        config.skills.enabled.remind_me = False
        registry = build_registry(config)
        registry.reset_turn()
        asyncio.run(registry.execute("remind_me", {"text": "x", "when": "5pm"}))
        assert "remind_me" in registry.called


class TestEverySessionDelivers:
    """The first version wired the ticker into `nero` alone. `nero telegram` —
    the one people actually leave running — delivered nothing, which is exactly
    how a reminder set from a phone went missing."""

    def test_building_a_chat_loop_starts_the_ticker(self, monkeypatch, tmp_path):
        import threading

        from nero import cli
        from nero.config.manager import ConfigManager
        from nero.config.schema import NeroConfig

        started = []
        monkeypatch.setattr(
            cli, "_start_reminder_ticker",
            lambda manager, config: started.append(1) or threading.Event(),
        )
        monkeypatch.setattr(cli, "_build_fallback_clients", lambda *a, **k: [])
        monkeypatch.setattr(cli, "_resolve_coding_client", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_build_history", lambda _c: None)
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        cli._build_chat_loop(manager, manager.load(), None, SkillRegistry([]), [])
        assert started == [1], "a session was built that would never deliver a reminder"


from nero.skills.registry import SkillRegistry  # noqa: E402


class TestItWorksFromEveryInterface:
    """A reminder set from one place must arrive regardless of which place is
    running. Every one of these was broken at some point in this feature's
    life, and each was found only by someone noticing nothing arrived."""

    def test_every_long_running_session_ticks(self):
        import inspect

        from nero import cli

        via_loop = "_start_reminder_ticker" in inspect.getsource(cli._build_chat_loop)
        sessions = {
            "nero / nero chat": cli._run_chat,
            "nero telegram": cli.telegram_main,
            "discord / slack / googlechat": cli._run_channel,
            "nero dashboard": cli.dashboard,
            "nero talk": cli.talk,
        }
        missing = [
            name
            for name, fn in sessions.items()
            if "_start_reminder_ticker" not in inspect.getsource(fn)
            and not ("_build_chat_loop" in inspect.getsource(fn) and via_loop)
        ]
        assert not missing, f"these sessions would never deliver a reminder: {missing}"

    def test_the_skill_is_available_to_every_interface(self):
        """They all build one registry, so one check covers the lot."""
        from nero.config.schema import NeroConfig
        from nero.skills.registry import build_registry

        offered = {t["function"]["name"] for t in build_registry(NeroConfig()).tool_definitions()}
        assert "remind_me" in offered

    def test_only_one_ticker_runs_per_process(self):
        """`nero` builds a chat loop and starts bridges that build their own.
        Two tickers double every read for nothing."""
        import threading

        from nero import cli
        from nero.config.manager import ConfigManager

        manager = ConfigManager()
        for _ in range(3):
            cli._start_reminder_ticker(manager, manager.load())
        running = [t for t in threading.enumerate() if t.name == "nero-reminders"]
        assert len(running) == 1


class TestLocalDelivery:
    """Most people have paired nothing on day one. Without a local path they
    set a reminder, every part works, and nothing ever appears."""

    def test_it_is_attempted_before_any_chat_app(self, monkeypatch):
        import inspect

        from nero import cli

        source = inspect.getsource(cli.deliver)
        assert source.index("notify_locally") < source.index("get_telegram_token")

    def test_a_notification_that_cannot_be_posted_is_not_an_error(self, monkeypatch):
        import subprocess

        from nero import reminders

        def boom(*a, **k):
            raise OSError("no osascript here")

        monkeypatch.setattr(subprocess, "run", boom)
        assert reminders.notify_locally("anything") is False

    def test_it_is_skipped_off_macos(self, monkeypatch):
        from nero import reminders

        monkeypatch.setattr(reminders.sys, "platform", "linux")
        assert reminders.notify_locally("anything") is False

    def test_quotes_in_the_text_cannot_break_the_command(self, monkeypatch):
        """The text is the user's own words and goes into an AppleScript
        string."""
        from nero import reminders

        captured = {}
        monkeypatch.setattr(reminders.sys, "platform", "darwin")
        monkeypatch.setattr(
            reminders.subprocess, "run",
            lambda cmd, **k: captured.update(cmd=cmd),
        )
        reminders.notify_locally('say "hello" \\ now')
        script = captured["cmd"][-1]
        assert script.count('"') % 2 == 0
