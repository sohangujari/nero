"""Minimal MCP client over stdio: JSON-RPC 2.0 on a child process's pipes.

Hand-rolled on purpose. The official SDK resolves to a dozen packages
(starlette, uvicorn, sse-starlette, cryptography...) that exist to *serve*
MCP, while a client needs exactly three methods — initialize, tools/list,
tools/call. stdlib only, so the binary doesn't grow for machinery we never
call.
"""

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from string import Template

from nero import __version__

logger = logging.getLogger("nero.mcp")

# Sent as our preferred revision; whatever the server answers with is what we
# speak. tools/* is stable across the published revisions, and hanging up on a
# version mismatch would break more servers than it protects.
PROTOCOL_VERSION = "2025-06-18"
HANDSHAKE_TIMEOUT = 10.0
LIST_TIMEOUT = 10.0
MAX_TOOL_PAGES = 10
STDERR_LINES = 20
TERMINATE_GRACE = 3.0


class MCPError(RuntimeError):
    """Anything that makes a server unusable: spawn, handshake, or transport."""


def expand_env(env: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR} against the parent environment.

    Secrets live in the user's shell, never in Nero Agent's config file. An
    unset variable is an error naming it, not a silently empty value that
    reaches the server as a broken credential.
    """
    expanded = {}
    for key, value in env.items():
        template = Template(value)
        missing = [name for name in template.get_identifiers() if name not in os.environ]
        if missing:
            raise MCPError(f"environment variable ${{{missing[0]}}} is not set")
        expanded[key] = template.substitute(os.environ)
    return expanded


def render_content(blocks: list[dict]) -> str:
    """Flatten MCP content blocks to text the model can read."""
    parts = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(f"[{block.get('type', 'unknown')} content omitted]")
    return "\n".join(parts)


class MCPConnection:
    """One stdio MCP server: a child process plus its two reader threads."""

    def __init__(self, name, command, args=(), env=None, timeout=60.0):
        self.name = name
        self.timeout = timeout
        self.protocol_version = None
        self.server_info: dict = {}
        self._command = [command, *args]
        self._env = dict(env or {})
        self._process = None
        self._messages: queue.Queue = queue.Queue()
        self._stderr: deque = deque(maxlen=STDERR_LINES)
        self._next_id = 0
        self._closed = False

    def start(self) -> None:
        """Spawn the server and complete the initialize handshake."""
        child_env = {**os.environ, **expand_env(self._env)}
        try:
            self._process = subprocess.Popen(  # noqa: S603 — the user configured this command
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=child_env,
            )
        except OSError as exc:
            raise MCPError(str(exc)) from exc
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self._handshake()

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor = None
        for _ in range(MAX_TOOL_PAGES):
            result = self._request("tools/list", {"cursor": cursor} if cursor else {}, LIST_TIMEOUT)
            tools.extend(result.get("tools", []))
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
        logger.warning("mcp %s: stopped after %d pages of tools", self.name, MAX_TOOL_PAGES)
        return tools

    def call_tool(self, tool: str, arguments: dict | None) -> str:
        result = self._request(
            "tools/call", {"name": tool, "arguments": arguments or {}}, self.timeout
        )
        text = render_content(result.get("content", []))
        if result.get("isError"):
            return f"Error from {self.name}: {text}"
        return text

    def close(self) -> None:
        """Reap the child. Safe to call twice — sessions must not leak processes."""
        if self._process is None or self._closed:
            return
        self._closed = True
        try:
            if self._process.stdin:
                self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=TERMINATE_GRACE)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=TERMINATE_GRACE)
        except OSError:
            pass

    # -- transport ---------------------------------------------------------

    def _handshake(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "nero-agent", "version": __version__},
            },
            HANDSHAKE_TIMEOUT,
        )
        self.protocol_version = result.get("protocolVersion")
        self.server_info = result.get("serverInfo", {})
        # Servers may refuse tool calls until this lands.
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(f"timed out after {timeout:.0f}s waiting for {method}")
            try:
                message = self._messages.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(f"timed out after {timeout:.0f}s waiting for {method}") from None
            if message is None:
                # Put EOF back so every later caller sees it too, not just the
                # first one to notice.
                self._messages.put(None)
                raise MCPError(f"server exited{self._stderr_tail()}")
            # Servers emit log notifications whenever they like; only the
            # message carrying our id answers this request.
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"] or {}
                raise MCPError(str(error.get("message", "unknown error")))
            return message.get("result", {})

    def _send(self, message: dict) -> None:
        if self._process is None or self._process.stdin is None:
            raise MCPError("not connected")
        try:
            self._process.stdin.write(json.dumps(message) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise MCPError(f"server is gone ({exc})") from exc

    def _read_stdout(self) -> None:
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._messages.put(json.loads(line))
            except ValueError:
                logger.debug("mcp %s: non-JSON line %r", self.name, line[:200])
        self._messages.put(None)

    def _read_stderr(self) -> None:
        # Drained in its own thread: MCP servers log here freely, and a full
        # pipe would eventually block the child mid-conversation.
        for line in self._process.stderr:
            line = line.strip()
            if line:
                self._stderr.append(line)

    def _stderr_tail(self) -> str:
        lines = list(self._stderr)
        return ": " + " / ".join(lines[-3:]) if lines else ""
