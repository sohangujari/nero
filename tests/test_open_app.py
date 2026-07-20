import asyncio
import subprocess

import pytest

from nero.tools.open_app import OpenAppTool


@pytest.fixture
def tool():
    return OpenAppTool()


def run(tool, **kwargs):
    return asyncio.run(tool.execute(**kwargs))


class TestToolInterface:
    def test_anthropic_definition_shape(self, tool):
        definition = tool.to_anthropic()
        assert definition["name"] == "open_app"
        assert definition["description"]
        assert definition["input_schema"]["type"] == "object"
        assert definition["input_schema"]["required"] == ["app_name"]
        assert "app_name" in definition["input_schema"]["properties"]


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


from nero.tools.base import validate_arguments

OPEN_APP_SCHEMA = {
    "type": "object",
    "properties": {"app_name": {"type": "string"}},
    "required": ["app_name"],
}


class TestValidateArguments:
    def test_valid(self):
        assert validate_arguments(OPEN_APP_SCHEMA, {"app_name": "Safari"}) is True

    def test_missing_required_field(self):
        assert validate_arguments(OPEN_APP_SCHEMA, {}) is False

    def test_empty_required_string(self):
        assert validate_arguments(OPEN_APP_SCHEMA, {"app_name": ""}) is False
        assert validate_arguments(OPEN_APP_SCHEMA, {"app_name": "   "}) is False

    def test_wrong_type(self):
        assert validate_arguments(OPEN_APP_SCHEMA, {"app_name": 5}) is False

    def test_non_dict_arguments(self):
        assert validate_arguments(OPEN_APP_SCHEMA, "nope") is False

    def test_extra_fields_tolerated(self):
        assert validate_arguments(OPEN_APP_SCHEMA, {"app_name": "Safari", "x": 1}) is True

    def test_generic_required_int(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        assert validate_arguments(schema, {"count": 3}) is True
        assert validate_arguments(schema, {"count": "3"}) is False
        assert validate_arguments(schema, {}) is False
