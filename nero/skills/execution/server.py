"""Execution skills: run_shell, git_command, run_python, run_javascript.

All four are permission_tier="destructive" — the registry's confirm gate
(SkillRegistry._dispatch) fires before execute() runs; with no confirm
callback they are refused (fail closed). That gate is not re-implemented
here.

run_shell and git_command additionally enforce the allowlist themselves,
per the design spec: `allowed(command, config.command_allowlist)` False
means the skill refuses ITSELF, before any prompting (an empty allowlist
permits everything, so this is inert until a user opts in). A denylist
match is deliberately NOT self-refused here — nero/cli.py's confirm prompt
already escalates a denylisted argument to a typed "yes" instead of a plain
y/N, and self-refusing here would make that escalation unreachable.

git push gets a stricter, separate rule: refused unless the reconstructed
command is explicitly present in a *non-empty* allowlist — pushing is the
repo's standing "user owns it" boundary, so it doesn't get the general
allowlist's "empty list permits everything" default.

run_python/run_javascript never exec() in-process: each writes to a temp
file and runs it with a fresh subprocess (sys.executable / node).
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from nero.config.schema import SecurityConfig
from nero.security import allowed
from nero.skills.base import Skill, SkillMeta

MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n...[truncated]"


def _clamp_timeout(value) -> int:
    try:
        timeout = int(value) if value else DEFAULT_TIMEOUT
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    return min(max(timeout, 1), MAX_TIMEOUT)


def _format_output(returncode: int, stdout: str, stderr: str) -> str:
    parts = [f"Exit code: {returncode}"]
    if stdout:
        parts.append(f"stdout:\n{_truncate(stdout)}")
    if stderr:
        parts.append(f"stderr:\n{_truncate(stderr)}")
    if not stdout and not stderr:
        parts.append("(no output)")
    return "\n".join(parts)


def _run(args, *, shell: bool, timeout: int, label: str) -> str:
    try:
        proc = subprocess.run(
            args, shell=shell, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"Error: {label} timed out after {timeout}s and was killed."
    except OSError as exc:
        return f"Error running {label}: {exc}"
    return _format_output(proc.returncode, proc.stdout, proc.stderr)


class RunShellSkill(Skill):
    meta = SkillMeta(
        name="run_shell",
        description=(
            "Run a shell command in the user's current working directory and "
            "return its stdout, stderr, and exit code. Use this when the user "
            "explicitly asks you to run a shell/terminal command."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "timeout_seconds": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, capped at {MAX_TIMEOUT}).",
                },
            },
            "required": ["command"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    def __init__(self, security: SecurityConfig | None = None):
        self._security = security or SecurityConfig()

    async def execute(self, **kwargs) -> str:
        command = str(kwargs.get("command") or "").strip()
        if not command:
            return "Error: no command provided."
        if not allowed(command, self._security.command_allowlist):
            return f"Error: {command!r} is not on the command allowlist; refusing to run it."
        timeout = _clamp_timeout(kwargs.get("timeout_seconds"))
        return _run(command, shell=True, timeout=timeout, label=f"`{command}`")


class GitCommandSkill(Skill):
    meta = SkillMeta(
        name="git_command",
        description=(
            'Run a git command in the user\'s current working directory. Pass '
            '`args` as a list of strings, e.g. ["status"] or '
            '["commit", "-m", "message"]. `push` is refused unless explicitly '
            "allowlisted."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Git subcommand and arguments, e.g. ["status"].',
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, capped at {MAX_TIMEOUT}).",
                },
            },
            "required": ["args"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    def __init__(self, security: SecurityConfig | None = None):
        self._security = security or SecurityConfig()

    async def execute(self, **kwargs) -> str:
        args = kwargs.get("args")
        if not isinstance(args, list) or not args or not all(isinstance(a, str) for a in args):
            return "Error: args must be a non-empty list of strings."
        command_string = "git " + " ".join(args)
        allowlist = self._security.command_allowlist
        if args[0] == "push" and not (allowlist and allowed(command_string, allowlist)):
            return (
                "Error: git push is refused unless the exact command is "
                "explicitly allowlisted in security.command_allowlist."
            )
        if not allowed(command_string, allowlist):
            return f"Error: {command_string!r} is not on the command allowlist; refusing to run it."
        timeout = _clamp_timeout(kwargs.get("timeout"))
        return _run(["git", *args], shell=False, timeout=timeout, label=command_string)


class RunPythonSkill(Skill):
    meta = SkillMeta(
        name="run_python",
        description=(
            "Run a Python snippet in a fresh subprocess and return its stdout, "
            "stderr, and exit code. Use this for calculations, data processing, "
            "or scripting tasks the user asks for in Python."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code to run."},
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, capped at {MAX_TIMEOUT}).",
                },
            },
            "required": ["code"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        code = str(kwargs.get("code") or "")
        if not code.strip():
            return "Error: no code provided."
        timeout = _clamp_timeout(kwargs.get("timeout"))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            path = handle.name
        try:
            return _run([sys.executable, path], shell=False, timeout=timeout, label="python code")
        finally:
            Path(path).unlink(missing_ok=True)


class RunJavascriptSkill(Skill):
    meta = SkillMeta(
        name="run_javascript",
        description=(
            "Run a JavaScript snippet with Node.js in a fresh subprocess and "
            "return its stdout, stderr, and exit code. Use this for "
            "calculations or scripting tasks the user asks for in JavaScript. "
            "Requires Node.js to be installed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "JavaScript source code to run."},
                "timeout": {
                    "type": "integer",
                    "description": f"Timeout in seconds (default {DEFAULT_TIMEOUT}, capped at {MAX_TIMEOUT}).",
                },
            },
            "required": ["code"],
        },
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        code = str(kwargs.get("code") or "")
        if not code.strip():
            return "Error: no code provided."
        node = shutil.which("node")
        if node is None:
            return "Error: Node.js (`node`) is not installed or not on PATH; can't run JavaScript."
        timeout = _clamp_timeout(kwargs.get("timeout"))
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(code)
            path = handle.name
        try:
            return _run([node, path], shell=False, timeout=timeout, label="javascript code")
        finally:
            Path(path).unlink(missing_ok=True)


RUN_SHELL = RunShellSkill()
GIT_COMMAND = GitCommandSkill()
RUN_PYTHON = RunPythonSkill()
RUN_JAVASCRIPT = RunJavascriptSkill()
