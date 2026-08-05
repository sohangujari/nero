import asyncio
import subprocess

import pytest

from nero.skills.open_app.server import OpenAppSkill


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
