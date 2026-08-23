"""Execution skills (nero/skills/execution/server.py): run_shell, git_command,
run_python, run_javascript.

Only trivially safe commands (echo/true/python -c) run for real, and only
inside pytest's tmp_path or as throwaway subprocesses that touch nothing.
Where the assertion allows it, subprocess.run is mocked instead of run for
real (timeout kill, exit-code plumbing).
"""

import asyncio
import subprocess

from nero.config.schema import SecurityConfig
from nero.skills.execution.server import (
    MAX_OUTPUT_CHARS,
    GitCommandSkill,
    RunJavascriptSkill,
    RunPythonSkill,
    RunShellSkill,
)


def run(coro):
    return asyncio.run(coro)


class TestRunShell:
    def test_tier(self):
        assert RunShellSkill.meta.permission_tier == "destructive"

    def test_happy_path_reports_stdout_and_exit_code(self):
        result = run(RunShellSkill().execute(command="echo hello"))
        assert "hello" in result
        assert "Exit code: 0" in result

    def test_nonzero_exit_code_reported(self):
        result = run(RunShellSkill().execute(command="exit 7"))
        assert "Exit code: 7" in result

    def test_no_command_refused(self):
        result = run(RunShellSkill().execute(command=""))
        assert "Error" in result

    def test_allowlist_refusal_is_self_refused_without_prompting(self):
        security = SecurityConfig(command_allowlist=["echo hi"])
        result = run(RunShellSkill(security=security).execute(command="rm -rf /tmp/whatever"))
        assert "Error" in result
        assert "allowlist" in result

    def test_allowlisted_command_runs(self):
        security = SecurityConfig(command_allowlist=["echo hi"])
        result = run(RunShellSkill(security=security).execute(command="echo hi"))
        assert "hi" in result
        assert "Error" not in result

    def test_empty_allowlist_permits_everything(self):
        result = run(RunShellSkill(security=SecurityConfig()).execute(command="true"))
        assert "Exit code: 0" in result

    def test_denylisted_command_is_not_self_refused(self):
        # denylist enforcement lives in the CLI confirm prompt, not the skill.
        security = SecurityConfig(command_denylist=["true"])
        result = run(RunShellSkill(security=security).execute(command="true"))
        assert "Exit code: 0" in result

    def test_timeout_kills_and_reports(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="sleep 100", timeout=1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = run(RunShellSkill().execute(command="sleep 100", timeout_seconds=1))
        assert "timed out" in result

    def test_output_truncated_with_marker(self):
        result = run(
            RunShellSkill().execute(command=f"python3 -c \"print('x' * {MAX_OUTPUT_CHARS + 500})\"")
        )
        assert "...[truncated]" in result
        assert len(result) < MAX_OUTPUT_CHARS + 1000

    def test_timeout_capped_at_max(self, monkeypatch):
        seen = {}

        def fake_run(*args, timeout=None, **kwargs):
            seen["timeout"] = timeout
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        run(RunShellSkill().execute(command="echo hi", timeout_seconds=99999))
        assert seen["timeout"] == 300


class TestGitCommand:
    def test_tier(self):
        assert GitCommandSkill.meta.permission_tier == "destructive"

    def test_never_uses_shell_true(self, monkeypatch):
        seen = {}

        def fake_run(args, *, shell, **kwargs):
            seen["args"] = args
            seen["shell"] = shell
            return subprocess.CompletedProcess(args, 0, "status output", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = run(GitCommandSkill().execute(args=["status"]))
        assert seen["shell"] is False
        assert seen["args"] == ["git", "status"]
        assert "status output" in result

    def test_happy_path_real_git(self, tmp_path, monkeypatch):
        # A real, harmless, read-only git call in a freshly initialized repo —
        # cwd is switched to tmp_path first, so this never touches the actual
        # nero repo's own git state.
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        monkeypatch.chdir(tmp_path)
        result = run(GitCommandSkill().execute(args=["status", "-s"]))
        assert "Exit code:" in result

    def test_rejects_non_list_args(self):
        result = run(GitCommandSkill().execute(args="status"))
        assert "Error" in result

    def test_rejects_empty_args(self):
        result = run(GitCommandSkill().execute(args=[]))
        assert "Error" in result

    def test_rejects_non_string_items(self):
        result = run(GitCommandSkill().execute(args=["status", 1]))
        assert "Error" in result

    def test_push_refused_by_default(self):
        result = run(GitCommandSkill().execute(args=["push"]))
        assert "Error" in result
        assert "push" in result.lower()

    def test_push_refused_even_with_empty_allowlist(self):
        security = SecurityConfig(command_allowlist=[])
        result = run(GitCommandSkill(security=security).execute(args=["push"]))
        assert "Error" in result

    def test_push_allowed_when_explicitly_allowlisted(self, monkeypatch):
        seen = {}

        def fake_run(args, *, shell, **kwargs):
            seen["args"] = args
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        security = SecurityConfig(command_allowlist=["git push"])
        result = run(GitCommandSkill(security=security).execute(args=["push"]))
        assert "Error" not in result
        assert seen["args"] == ["git", "push"]

    def test_allowlist_refusal_for_non_push_command(self):
        security = SecurityConfig(command_allowlist=["git status"])
        result = run(GitCommandSkill(security=security).execute(args=["log"]))
        assert "Error" in result
        assert "allowlist" in result

    def test_denylisted_command_is_not_self_refused(self, monkeypatch):
        def fake_run(args, *, shell, **kwargs):
            return subprocess.CompletedProcess(args, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        security = SecurityConfig(command_denylist=["git reset --hard"])
        result = run(GitCommandSkill(security=security).execute(args=["reset", "--hard"]))
        assert "Error" not in result


class TestRunPython:
    def test_tier(self):
        assert RunPythonSkill.meta.permission_tier == "destructive"

    def test_happy_path_runs_in_subprocess(self):
        result = run(RunPythonSkill().execute(code="print(1 + 1)"))
        assert "2" in result
        assert "Exit code: 0" in result

    def test_never_execs_in_process(self, monkeypatch):
        # A poisoned in-process global must not appear if run_python truly
        # shells out rather than calling exec() here.
        monkeypatch.setattr("builtins.__nero_test_marker__", "poisoned", raising=False)
        result = run(RunPythonSkill().execute(code="print('__nero_test_marker__' in dir(__builtins__))"))
        assert "False" in result

    def test_no_code_refused(self):
        result = run(RunPythonSkill().execute(code=""))
        assert "Error" in result

    def test_syntax_error_reported_via_exit_code_and_stderr(self):
        result = run(RunPythonSkill().execute(code="def("))
        assert "Exit code: 1" in result or "SyntaxError" in result

    def test_timeout_kills_and_reports(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="python", timeout=1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = run(RunPythonSkill().execute(code="while True: pass", timeout=1))
        assert "timed out" in result


class TestRunJavascript:
    def test_tier(self):
        assert RunJavascriptSkill.meta.permission_tier == "destructive"

    def test_no_code_refused(self):
        result = run(RunJavascriptSkill().execute(code=""))
        assert "Error" in result

    def test_missing_node_reported_actionably(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        result = run(RunJavascriptSkill().execute(code="console.log(1)"))
        assert "Error" in result
        assert "Node" in result

    def test_happy_path_when_node_is_available(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node")

        def fake_run(args, *, shell, **kwargs):
            assert shell is False
            return subprocess.CompletedProcess(args, 0, "3\n", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = run(RunJavascriptSkill().execute(code="console.log(1+2)"))
        assert "3" in result
        assert "Exit code: 0" in result

    def test_timeout_kills_and_reports(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/node")

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="node", timeout=1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = run(RunJavascriptSkill().execute(code="while(true){}", timeout=1))
        assert "timed out" in result
