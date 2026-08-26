"""Bridge remote MCP tools into the local skill registry.

One `MCPSkill` per remote tool means MCP tools inherit everything the registry
already does: the audit trail, enable/disable, offline gating, and the
destructive-tier confirmation gate. No parallel dispatch path.
"""

import asyncio
import logging
import re

from nero.mcp.client import MCPConnection, MCPError
from nero.security import envelope
from nero.skills.base import Skill, SkillMeta

logger = logging.getLogger("nero.mcp")


def skill_name(server: str, tool: str) -> str:
    """`server_tool`, sanitized — tool names reach the model as function names."""
    return re.sub(r"[^A-Za-z0-9_]", "_", f"{server}_{tool}")


class MCPSkill(Skill):
    """One remote MCP tool.

    Everything a third-party server returns is untrusted content: it goes out
    through `envelope()` and marks the turn tainted, exactly like a fetched web
    page. An untrusted server's tools sit on the `destructive` tier, which
    routes every call through the existing confirmation gate — including its
    fail-closed behaviour when no gate is wired.
    """

    def __init__(self, connection: MCPConnection, tool: dict, trusted=False, requires_network=True):
        self._connection = connection
        self._tool = tool["name"]
        description = (tool.get("description") or "").strip()
        self.meta = SkillMeta(
            name=skill_name(connection.name, self._tool),
            description=f"[MCP: {connection.name}] {description}".strip(),
            input_schema=tool.get("inputSchema") or {"type": "object", "properties": {}},
            requires_network=requires_network,
            permission_tier="state_changing" if trusted else "destructive",
            ingests_external_content=True,
            offline_message=(
                f"The {connection.name} MCP server needs an internet connection, "
                "and you're in offline mode right now."
            )
            if requires_network
            else None,
        )

    async def execute(self, **kwargs) -> str:
        # to_thread: the transport is blocking pipe I/O, and the event loop is
        # driving the rest of the turn.
        text = await asyncio.to_thread(self._connection.call_tool, self._tool, kwargs)
        return envelope(f"mcp:{self._connection.name}/{self._tool}", text)


def load_servers(mcp_config, taken_names) -> tuple[list[Skill], list[MCPConnection], list[str]]:
    """Spawn every enabled server and wrap its tools.

    Returns (skills, connections, warnings). A server that fails to start,
    handshake, or list its tools is skipped with a warning — a flaky
    third-party server must never take the assistant down. Callers own the
    returned connections and must `close()` them.
    """
    skills: list[Skill] = []
    connections: list[MCPConnection] = []
    warnings: list[str] = []
    names = set(taken_names)

    for name, server in mcp_config.servers.items():
        if not server.enabled:
            continue
        connection = MCPConnection(
            name=name,
            command=server.command,
            args=server.args,
            env=server.env,
            timeout=server.timeout_seconds,
        )
        try:
            connection.start()
            tools = connection.list_tools()
        except MCPError as exc:
            connection.close()
            warnings.append(f"MCP server {name!r} unavailable: {exc}")
            continue
        connections.append(connection)
        for tool in tools:
            if not tool.get("name"):
                continue
            skill = MCPSkill(
                connection,
                tool,
                trusted=server.trusted,
                requires_network=server.requires_network,
            )
            if skill.meta.name in names:
                # Never shadow a built-in: a remote server could otherwise
                # redefine what `get_weather` means.
                warnings.append(
                    f"MCP tool {name}/{tool['name']} skipped: "
                    f"{skill.meta.name} is already a skill"
                )
                continue
            names.add(skill.meta.name)
            skills.append(skill)
    return skills, connections, warnings
