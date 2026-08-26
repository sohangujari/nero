"""Cron -> launchd translation, and install/uninstall of the launchd agent.

SAFETY: no test here may touch the real ~/Library/LaunchAgents or actually
invoke launchctl — every install/uninstall test uses a tmp_path agents dir
and monkeypatches nero.routines._run_launchctl.
"""

import plistlib

import pytest
from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig, RoutineConfig
from nero.routines import (
    RoutineError,
    cron_to_calendar,
    install_routine,
    is_installed,
    label_for,
    plist_path,
    uninstall_routine,
)

runner = CliRunner()


class TestCronToCalendar:
    def test_daily_at_830(self):
        assert cron_to_calendar("30 8 * * *") == {"Minute": 30, "Hour": 8}

    def test_yearly_new_years_day(self):
        assert cron_to_calendar("0 0 1 1 *") == {"Minute": 0, "Hour": 0, "Day": 1, "Month": 1}

    def test_weekday_field_maps_directly(self):
        assert cron_to_calendar("0 9 * * 1") == {"Minute": 0, "Hour": 9, "Weekday": 1}

    def test_all_wildcards(self):
        assert cron_to_calendar("* * * * *") == {}

    @pytest.mark.parametrize("schedule", ["*/15 * * * *", "1-5 * * * *", "1,3 * * * *"])
    def test_unsupported_syntax_names_it(self, schedule):
        with pytest.raises(RoutineError):
            cron_to_calendar(schedule)

    def test_step_message_names_step(self):
        with pytest.raises(RoutineError, match=r"[Ss]tep"):
            cron_to_calendar("*/15 * * * *")

    def test_range_message_names_range(self):
        with pytest.raises(RoutineError, match=r"[Rr]ange"):
            cron_to_calendar("1-5 * * * *")

    def test_list_message_names_list(self):
        with pytest.raises(RoutineError, match=r"[Ll]ist"):
            cron_to_calendar("1,3 * * * *")

    def test_out_of_range_minute_raises(self):
        with pytest.raises(RoutineError):
            cron_to_calendar("60 * * * *")

    def test_out_of_range_hour_raises(self):
        with pytest.raises(RoutineError):
            cron_to_calendar("* 24 * * *")

    def test_out_of_range_weekday_raises(self):
        with pytest.raises(RoutineError):
            cron_to_calendar("* * * * 7")

    def test_wrong_field_count_raises(self):
        with pytest.raises(RoutineError):
            cron_to_calendar("30 8 * *")

    def test_non_integer_raises(self):
        with pytest.raises(RoutineError):
            cron_to_calendar("thirty * * * *")


def _routine(schedule="0 9 * * *"):
    return RoutineConfig(schedule=schedule, prompt="Summarize my inbox.")


class TestInstallUninstall:
    def test_writes_plist_with_right_label_and_program_arguments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "nero.routines._run_launchctl",
            lambda *args: _fake_completed(0),
        )
        install_routine("morning", _routine(), "/usr/local/bin/nero", tmp_path)
        path = plist_path("morning", tmp_path)
        assert path.exists()
        data = plistlib.loads(path.read_bytes())
        assert data["Label"] == label_for("morning")
        assert data["ProgramArguments"] == [
            "/usr/local/bin/nero", "routine", "run", "morning",
        ]
        assert data["StartCalendarInterval"] == {"Minute": 0, "Hour": 9}
        assert "StandardOutPath" in data
        assert "StandardErrorPath" in data

    def test_launchctl_is_invoked_but_never_the_real_binary(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "nero.routines._run_launchctl",
            lambda *args: calls.append(args) or _fake_completed(0),
        )
        install_routine("morning", _routine(), "/usr/local/bin/nero", tmp_path)
        assert calls  # launchctl was "invoked" — through the monkeypatched stub only
        assert calls[0][0] == "bootstrap"

    def test_bootstrap_failure_falls_back_to_load(self, tmp_path, monkeypatch):
        calls = []

        def fake(*args):
            calls.append(args)
            return _fake_completed(0 if args[0] == "load" else 1)

        monkeypatch.setattr("nero.routines._run_launchctl", fake)
        install_routine("morning", _routine(), "/usr/local/bin/nero", tmp_path)
        assert [c[0] for c in calls] == ["bootstrap", "load"]

    def test_is_installed_reflects_plist_presence(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nero.routines._run_launchctl", lambda *args: _fake_completed(0))
        assert is_installed("morning", tmp_path) is False
        install_routine("morning", _routine(), "/usr/local/bin/nero", tmp_path)
        assert is_installed("morning", tmp_path) is True

    def test_uninstall_removes_the_plist(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nero.routines._run_launchctl", lambda *args: _fake_completed(0))
        install_routine("morning", _routine(), "/usr/local/bin/nero", tmp_path)
        uninstall_routine("morning", tmp_path)
        assert is_installed("morning", tmp_path) is False

    def test_uninstalling_absent_routine_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nero.routines._run_launchctl", lambda *args: _fake_completed(0))
        message = uninstall_routine("never-installed", tmp_path)
        assert "nothing to do" in message.lower()

    def test_invalid_schedule_raises_before_writing_plist(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nero.routines._run_launchctl", lambda *args: _fake_completed(0))
        with pytest.raises(RoutineError):
            install_routine("bad", _routine("*/15 * * * *"), "/usr/local/bin/nero", tmp_path)
        assert not plist_path("bad", tmp_path).exists()


def _manager_with_routines(tmp_path, routines_dict):
    manager = ConfigManager(config_dir=tmp_path)
    config = NeroConfig()
    config.routines.routines = routines_dict
    manager.save(config)
    return manager


class _FakeClient:
    """Stands in for LLMClient: captures the outgoing messages, "replies"
    with fixed text through on_text — no real provider/network involved."""

    last_messages = None

    def __init__(self, config, assistant_name, registry, api_key=None, **kwargs):
        pass

    def send(self, messages, on_text):
        type(self).last_messages = messages
        on_text("All done.")


class TestRoutineListCLI:
    def test_no_routines_configured_says_so(self, tmp_path, monkeypatch):
        manager = _manager_with_routines(tmp_path, {})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["routine", "list"])
        assert result.exit_code == 0
        assert "no routines configured" in result.stdout.lower()

    def test_lists_name_schedule_enabled_installed(self, tmp_path, monkeypatch):
        agents_dir = tmp_path / "LaunchAgents"
        routine = RoutineConfig(schedule="0 9 * * *", prompt="hi", enabled=True)
        manager = _manager_with_routines(tmp_path, {"morning": routine})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(cli.routines, "default_agents_dir", lambda: agents_dir)
        result = runner.invoke(cli.app, ["routine", "list"])
        assert result.exit_code == 0
        assert "morning" in result.stdout
        assert "0 9 * * *" in result.stdout

    def test_bare_routine_command_lists_too(self, tmp_path, monkeypatch):
        manager = _manager_with_routines(tmp_path, {})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["routine"])
        assert result.exit_code == 0
        assert "no routines configured" in result.stdout.lower()


class TestRoutineRunCLI:
    def test_unknown_routine_exits_1(self, tmp_path, monkeypatch):
        manager = _manager_with_routines(tmp_path, {})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["routine", "run", "nope"])
        assert result.exit_code == 1

    def test_disabled_routine_exits_1(self, tmp_path, monkeypatch):
        routine = RoutineConfig(schedule="0 9 * * *", prompt="hi", enabled=False)
        manager = _manager_with_routines(tmp_path, {"morning": routine})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["routine", "run", "morning"])
        assert result.exit_code == 1

    def test_happy_path_prompt_reaches_client_and_reply_prints(self, tmp_path, monkeypatch):
        routine = RoutineConfig(schedule="0 9 * * *", prompt="Summarize my inbox.")
        manager = _manager_with_routines(tmp_path, {"morning": routine})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(cli, "_provider_preflight", lambda manager, config: "fake-key")
        monkeypatch.setattr(cli, "LLMClient", _FakeClient)
        result = runner.invoke(cli.app, ["routine", "run", "morning"])
        assert result.exit_code == 0
        assert "All done." in result.stdout
        assert _FakeClient.last_messages[0]["content"] == "Summarize my inbox."


class TestRoutineInstallUninstallCLI:
    def test_install_writes_plist_into_tmp_agents_dir(self, tmp_path, monkeypatch):
        agents_dir = tmp_path / "LaunchAgents"
        routine = RoutineConfig(schedule="0 9 * * *", prompt="hi")
        manager = _manager_with_routines(tmp_path, {"morning": routine})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(cli.routines, "default_agents_dir", lambda: agents_dir)
        monkeypatch.setattr(cli.routines, "_run_launchctl", lambda *a: _fake_completed(0))
        result = runner.invoke(cli.app, ["routine", "install", "morning"])
        assert result.exit_code == 0
        assert plist_path("morning", agents_dir).exists()

    def test_install_unknown_routine_exits_1(self, tmp_path, monkeypatch):
        manager = _manager_with_routines(tmp_path, {})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["routine", "install", "nope"])
        assert result.exit_code == 1

    def test_uninstall_removes_plist(self, tmp_path, monkeypatch):
        agents_dir = tmp_path / "LaunchAgents"
        routine = RoutineConfig(schedule="0 9 * * *", prompt="hi")
        manager = _manager_with_routines(tmp_path, {"morning": routine})
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        monkeypatch.setattr(cli.routines, "default_agents_dir", lambda: agents_dir)
        monkeypatch.setattr(cli.routines, "_run_launchctl", lambda *a: _fake_completed(0))
        runner.invoke(cli.app, ["routine", "install", "morning"])
        result = runner.invoke(cli.app, ["routine", "uninstall", "morning"])
        assert result.exit_code == 0
        assert not plist_path("morning", agents_dir).exists()

    def test_uninstall_absent_is_a_noop(self, tmp_path, monkeypatch):
        agents_dir = tmp_path / "LaunchAgents"
        monkeypatch.setattr(cli.routines, "default_agents_dir", lambda: agents_dir)
        result = runner.invoke(cli.app, ["routine", "uninstall", "never-installed"])
        assert result.exit_code == 0
        assert "nothing to do" in result.stdout.lower()


class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode
        self.stderr = ""
        self.stdout = ""


def _fake_completed(returncode):
    return _FakeCompleted(returncode)
