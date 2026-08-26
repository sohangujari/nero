"""MCP client tests.

These drive a real subprocess speaking real JSON-RPC over real pipes: the
threading, queueing, and id-matching are the parts most likely to be subtly
wrong, and a mocked transport would exercise none of them.
"""

import asyncio
import sys
import textwrap

import pytest

from nero.config.schema import MCPConfig, MCPServerConfig
from nero.mcp.client import MCPConnection, MCPError, expand_env, render_content
from nero.mcp.skill import MCPSkill, load_servers, skill_name

# A fake server is a python script that reads JSON-RPC lines and answers them.
# `BODY` is spliced in to customize behaviour per test.
SERVER_TEMPLATE = '''
import json, sys, os

def send(message):
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()

{body}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    method = request.get("method")
    if method == "notifications/initialized":
        continue
    handle(request, send)
'''

BASIC_BODY = '''
TOOLS = [
    {"name": "echo", "description": "Echo text back",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"]}},
    {"name": "env_peek", "description": "Report an env var",
     "inputSchema": {"type": "object", "properties": {}}},
]

def handle(request, send):
    method = request["method"]
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request["id"], "result": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "serverInfo": {"name": "fake", "version": "1.0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": request["id"], "result": {"tools": TOOLS}})
    elif method == "tools/call":
        name = request["params"]["name"]
        if name == "echo":
            text = request["params"]["arguments"].get("text", "")
            send({"jsonrpc": "2.0", "id": request["id"],
                  "result": {"content": [{"type": "text", "text": f"echo: {text}"}]}})
        elif name == "env_peek":
            send({"jsonrpc": "2.0", "id": request["id"],
                  "result": {"content": [{"type": "text",
                                          "text": os.environ.get("NERO_TEST_TOKEN", "<unset>")}]}})
'''


def write_server(tmp_path, body, name="server.py"):
    path = tmp_path / name
    path.write_text(SERVER_TEMPLATE.format(body=textwrap.dedent(body)))
    return path


@pytest.fixture
def connect(tmp_path):
    """Start a fake server; guarantees the child is reaped after the test."""
    connections = []

    def _connect(body, env=None, timeout=10.0, name="fake"):
        script = write_server(tmp_path, body, name=f"{name}.py")
        connection = MCPConnection(
            name=name, command=sys.executable, args=[str(script)], env=env, timeout=timeout
        )
        connections.append(connection)
        connection.start()
        return connection

    yield _connect
    for connection in connections:
        connection.close()


class TestHandshakeAndDiscovery:
    def test_handshake_records_the_servers_own_version(self, connect):
        connection = connect(BASIC_BODY)
        assert connection.protocol_version == "2025-06-18"
        assert connection.server_info["name"] == "fake"

    def test_tools_list_returns_names_and_schemas_intact(self, connect):
        tools = connect(BASIC_BODY).list_tools()
        assert [tool["name"] for tool in tools] == ["echo", "env_peek"]
        assert tools[0]["inputSchema"]["required"] == ["text"]

    def test_pagination_follows_next_cursor(self, connect):
        tools = connect('''
            def handle(request, send):
                method = request["method"]
                if method == "initialize":
                    send({"jsonrpc": "2.0", "id": request["id"], "result": {}})
                elif method == "tools/list":
                    if not request["params"].get("cursor"):
                        send({"jsonrpc": "2.0", "id": request["id"], "result": {
                            "tools": [{"name": "one", "inputSchema": {}}], "nextCursor": "p2"}})
                    else:
                        send({"jsonrpc": "2.0", "id": request["id"], "result": {
                            "tools": [{"name": "two", "inputSchema": {}}]}})
        ''').list_tools()
        assert [tool["name"] for tool in tools] == ["one", "two"]


class TestToolCalls:
    def test_round_trip_returns_joined_text(self, connect):
        assert connect(BASIC_BODY).call_tool("echo", {"text": "hi"}) == "echo: hi"

    def test_is_error_result_is_prefixed_not_raised(self, connect):
        result = connect('''
            def handle(request, send):
                if request["method"] == "initialize":
                    send({"jsonrpc": "2.0", "id": request["id"], "result": {}})
                else:
                    send({"jsonrpc": "2.0", "id": request["id"], "result": {
                        "content": [{"type": "text", "text": "disk full"}], "isError": True}})
        ''').call_tool("x", {})
        assert result == "Error from fake: disk full"

    def test_json_rpc_error_object_becomes_a_readable_message(self, connect):
        connection = connect('''
            def handle(request, send):
                if request["method"] == "initialize":
                    send({"jsonrpc": "2.0", "id": request["id"], "result": {}})
                else:
                    send({"jsonrpc": "2.0", "id": request["id"],
                          "error": {"code": -32602, "message": "no such tool"}})
        ''')
        with pytest.raises(MCPError, match="no such tool"):
            connection.call_tool("nope", {})

    def test_interleaved_notifications_are_discarded(self, connect):
        """A log notification arriving first must not be mistaken for the response."""
        result = connect('''
            def handle(request, send):
                if request["method"] == "initialize":
                    send({"jsonrpc": "2.0", "id": request["id"], "result": {}})
                    return
                send({"jsonrpc": "2.0", "method": "notifications/message",
                      "params": {"level": "info", "data": "working"}})
                send({"jsonrpc": "2.0", "id": 999, "result": {"content": []}})
                send({"jsonrpc": "2.0", "id": request["id"],
                      "result": {"content": [{"type": "text", "text": "real answer"}]}})
        ''').call_tool("x", {})
        assert result == "real answer"

    def test_non_text_blocks_render_a_placeholder(self):
        rendered = render_content([{"type": "text", "text": "a"}, {"type": "image", "data": "..."}])
        assert rendered == "a\n[image content omitted]"


class TestFailureModes:
    def test_a_server_that_exits_reports_its_stderr(self, tmp_path):
        script = tmp_path / "broken.py"
        script.write_text('import sys; sys.stderr.write("boom: missing token\\n"); sys.exit(1)')
        connection = MCPConnection("broken", sys.executable, [str(script)])
        with pytest.raises(MCPError) as excinfo:
            connection.start()
        assert "boom: missing token" in str(excinfo.value)
        connection.close()

    def test_a_missing_command_is_an_error_not_a_crash(self):
        connection = MCPConnection("ghost", "/nonexistent/definitely-not-here")
        with pytest.raises(MCPError):
            connection.start()

    def test_a_silent_server_times_out_within_budget(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nero.mcp.client.HANDSHAKE_TIMEOUT", 0.5)
        script = tmp_path / "mute.py"
        script.write_text("import sys\nfor line in sys.stdin:\n    pass\n")
        connection = MCPConnection("mute", sys.executable, [str(script)])
        with pytest.raises(MCPError, match="timed out"):
            connection.start()
        connection.close()

    def test_close_reaps_the_child_and_is_safe_twice(self, connect):
        connection = connect(BASIC_BODY)
        connection.close()
        assert connection._process.poll() is not None
        connection.close()  # must not raise


class TestEnvExpansion:
    def test_expansion_reaches_the_child(self, connect, monkeypatch):
        monkeypatch.setenv("NERO_OUTER_TOKEN", "s3cret")
        connection = connect(BASIC_BODY, env={"NERO_TEST_TOKEN": "${NERO_OUTER_TOKEN}"})
        assert connection.call_tool("env_peek", {}) == "s3cret"

    def test_a_missing_variable_names_itself(self, monkeypatch):
        monkeypatch.delenv("NERO_ABSENT", raising=False)
        with pytest.raises(MCPError, match="NERO_ABSENT"):
            expand_env({"TOKEN": "${NERO_ABSENT}"})

    def test_literal_values_pass_through(self):
        assert expand_env({"MODE": "readonly"}) == {"MODE": "readonly"}


class TestSkillBridge:
    def test_untrusted_servers_are_destructive_trusted_are_state_changing(self, connect):
        connection = connect(BASIC_BODY)
        tool = connection.list_tools()[0]
        assert MCPSkill(connection, tool).meta.permission_tier == "destructive"
        assert MCPSkill(connection, tool, trusted=True).meta.permission_tier == "state_changing"

    def test_results_are_enveloped_as_untrusted_content(self, connect):
        connection = connect(BASIC_BODY)
        skill = MCPSkill(connection, connection.list_tools()[0])
        result = asyncio.run(skill.execute(text="hi"))
        assert "<untrusted_content" in result
        assert "mcp:fake/echo" in result
        assert "echo: hi" in result
        assert skill.meta.ingests_external_content is True

    def test_names_are_prefixed_and_sanitized(self):
        assert skill_name("git-hub", "create issue") == "git_hub_create_issue"


class TestLoadServers:
    def _config(self, tmp_path, body, **overrides):
        script = write_server(tmp_path, body)
        return MCPConfig(servers={"fake": MCPServerConfig(
            command=sys.executable, args=[str(script)], **overrides)})

    def test_loads_tools_and_returns_closable_connections(self, tmp_path):
        skills, connections, warnings = load_servers(self._config(tmp_path, BASIC_BODY), set())
        try:
            assert [skill.meta.name for skill in skills] == ["fake_echo", "fake_env_peek"]
            assert warnings == []
        finally:
            for connection in connections:
                connection.close()

    def test_a_name_collision_skips_the_remote_tool(self, tmp_path):
        skills, connections, warnings = load_servers(
            self._config(tmp_path, BASIC_BODY), {"fake_echo"}
        )
        try:
            assert [skill.meta.name for skill in skills] == ["fake_env_peek"]
            assert any("already a skill" in warning for warning in warnings)
        finally:
            for connection in connections:
                connection.close()

    def test_a_broken_server_warns_and_the_others_still_load(self, tmp_path):
        broken = tmp_path / "broken.py"
        broken.write_text("import sys; sys.exit(3)")
        good = write_server(tmp_path, BASIC_BODY, name="good.py")
        config = MCPConfig(servers={
            "broken": MCPServerConfig(command=sys.executable, args=[str(broken)]),
            "good": MCPServerConfig(command=sys.executable, args=[str(good)]),
        })
        skills, connections, warnings = load_servers(config, set())
        try:
            assert any("broken" in warning for warning in warnings)
            assert [skill.meta.name for skill in skills] == ["good_echo", "good_env_peek"]
        finally:
            for connection in connections:
                connection.close()

    def test_disabled_servers_are_never_spawned(self, tmp_path):
        config = self._config(tmp_path, BASIC_BODY, enabled=False)
        skills, connections, warnings = load_servers(config, set())
        assert (skills, connections, warnings) == ([], [], [])
