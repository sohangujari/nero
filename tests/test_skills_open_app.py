import asyncio
import subprocess

import pytest

from nero.skills.open_app.server import CloseAppSkill, OpenAppSkill


@pytest.fixture
def tool():
    return OpenAppSkill()


def run(tool, **kwargs):
    return asyncio.run(tool.execute(**kwargs))


class TestSkillInterface:
    def test_meta_shape(self, tool):
        assert tool.meta.name == "open_app"
        assert tool.meta.description
        assert tool.meta.input_schema["type"] == "object"
        assert tool.meta.input_schema["required"] == ["app_name"]
        assert "app_name" in tool.meta.input_schema["properties"]
        assert tool.meta.requires_network is False
        assert tool.meta.permission_tier == "state_changing"


class TestExecute:
    def test_missing_app_name_returns_error_string(self, tool):
        result = run(tool)
        assert result.startswith("Error")

    def test_blank_app_name_returns_error_string(self, tool):
        result = run(tool, app_name="   ")
        assert result.startswith("Error")

    def test_macos_success(self, tool, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("subprocess.run", fake_run)
        result = run(tool, app_name="Safari")
        assert "Safari" in result
        assert "Opened" in result
        # Argument list, never a shell string — injection-safe.
        assert calls == [["open", "-a", "Safari"]]

    def test_macos_app_not_found(self, tool, monkeypatch):
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="Unable to find application named 'NotAnApp'"
            )

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("subprocess.run", fake_run)
        result = run(tool, app_name="NotAnApp")
        assert "NotAnApp" in result
        assert "Opened" not in result

    def test_windows_uses_argument_list(self, tool, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs.get("shell", False)))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setattr("subprocess.run", fake_run)
        result = run(tool, app_name='notepad" & del C:\\')
        cmd, shell = calls[0]
        assert isinstance(cmd, list)
        assert shell is False
        assert cmd[-1] == 'notepad" & del C:\\'
        assert "Opened" in result

    def test_subprocess_exception_is_caught(self, tool, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("subprocess.run", fake_run)
        result = run(tool, app_name="Safari")
        assert "Error" in result
        assert "Safari" in result

    def test_unsupported_platform(self, tool, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Plan9")
        result = run(tool, app_name="Safari")
        assert "Error" in result


class TestSuggestions:
    """A bare "could not find it" is a dead end. The user cannot tell whether
    they typo'd, or the app is named something else, or it isn't installed."""

    def skill(self, tmp_path, *names):
        for name in names:
            (tmp_path / f"{name}.app").mkdir()
        return OpenAppSkill(app_dirs=(tmp_path,))

    def test_a_typo_is_matched_to_the_installed_app(self, tmp_path):
        assert "'Spotify'" in self.skill(tmp_path, "Spotify", "Safari")._did_you_mean("Sptify")

    def test_a_name_that_matches_nothing_says_so_without_guessing(self, tmp_path):
        """Suggesting 'Maps' for 'Mail' is worse than admitting no match."""
        hint = self.skill(tmp_path, "Spotify", "Safari")._did_you_mean("Blender")
        assert "Did you mean" not in hint
        assert "2 apps installed" in hint

    def test_no_installed_apps_adds_nothing(self, tmp_path):
        assert OpenAppSkill(app_dirs=(tmp_path,))._did_you_mean("Spotify") == ""

    def test_a_missing_directory_is_skipped_not_raised(self, tmp_path):
        skill = OpenAppSkill(app_dirs=(tmp_path / "nope", tmp_path))
        (tmp_path / "Music.app").mkdir()
        assert skill.installed() == ["Music"]

    def test_the_scan_does_not_run_when_the_app_opened(self, tmp_path, monkeypatch):
        """The happy path must not pay for the failure path."""
        scanned = []
        skill = OpenAppSkill(app_dirs=(tmp_path,))
        monkeypatch.setattr(skill, "installed", lambda: scanned.append(1) or [])
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", ""),
        )
        assert "Opened" in asyncio.run(skill.execute(app_name="Music"))
        assert scanned == []


class TestCloseApp:
    """Closing was simply missing — Nero would say it had no ability to quit an
    app and tell the user to do it themselves."""

    def skill(self, tmp_path, *names):
        for name in names:
            (tmp_path / f"{name}.app").mkdir(exist_ok=True)
        return CloseAppSkill(app_dirs=(tmp_path,))

    def macos(self, monkeypatch, returncode=0, stdout="done"):
        seen = []

        def fake(cmd, **kwargs):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode, stdout, "")

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("subprocess.run", fake)
        return seen

    def test_it_quits_a_running_app(self, tmp_path, monkeypatch):
        seen = self.macos(monkeypatch)
        result = run(self.skill(tmp_path, "Google Chrome"), app_name="chrome")
        assert result == "Closed Google Chrome."
        assert 'to quit' in seen[0][-1]

    def test_a_loose_name_finds_the_real_app(self, tmp_path, monkeypatch):
        """'close chrome' must reach 'Google Chrome' — nobody says the bundle
        name out loud."""
        self.macos(monkeypatch)
        assert "Google Chrome" in run(self.skill(tmp_path, "Google Chrome"), app_name="chrome")

    def test_an_app_that_is_not_running_is_not_launched_to_close_it(self, tmp_path, monkeypatch):
        """An unguarded `quit` starts the app first. The check and the quit
        travel in one script so that cannot happen."""
        seen = self.macos(monkeypatch, stdout="idle")
        result = run(self.skill(tmp_path, "Music"), app_name="Music")
        assert result == "Music isn't running."
        assert len(seen) == 1
        assert "is running" in seen[0][-1]

    def test_an_unknown_app_suggests_something_instead(self, tmp_path, monkeypatch):
        self.macos(monkeypatch)
        result = run(self.skill(tmp_path, "Music"), app_name="Musci")
        assert "Did you mean 'Music'?" in result

    def test_an_applescript_failure_is_reported(self, tmp_path, monkeypatch):
        self.macos(monkeypatch, returncode=1)
        assert "Could not close" in run(self.skill(tmp_path, "Music"), app_name="Music")

    def test_a_missing_app_name_is_refused(self, tmp_path):
        assert "Error" in run(self.skill(tmp_path), app_name="  ")

    def test_a_hung_app_does_not_hold_the_turn_open(self, tmp_path, monkeypatch):
        def hang(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 15)

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("subprocess.run", hang)
        assert "did not respond" in run(self.skill(tmp_path, "Music"), app_name="Music")

    def test_windows_closes_gracefully_rather_than_forcing(self, tmp_path, monkeypatch):
        """No /F: an app with unsaved work should still get to prompt."""
        seen = []
        monkeypatch.setattr("platform.system", lambda: "Windows")
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        run(self.skill(tmp_path), app_name="notepad")
        assert seen[0] == ["taskkill", "/IM", "notepad.exe"]

    def test_linux_matches_the_process_name_exactly(self, tmp_path, monkeypatch):
        """Without -x, closing 'code' could match anything with 'code' in its
        command line."""
        seen = []
        monkeypatch.setattr("platform.system", lambda: "Linux")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/pkill")
        monkeypatch.setattr(
            "subprocess.run",
            lambda cmd, **k: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""),
        )
        run(self.skill(tmp_path), app_name="firefox")
        assert seen[0] == ["pkill", "-TERM", "-x", "firefox"]

    def test_the_tier_lets_it_run_without_a_confirmation(self, tmp_path):
        """Shipping this destructive would mean 'close Chrome' fails by default,
        which is the complaint it exists to answer. A graceful quit still lets
        the app prompt about unsaved work."""
        assert CloseAppSkill().meta.permission_tier == "state_changing"
