"""What the dashboard shows, minus the showing.

The page itself is the React/shadcn app in web/ served by nero/webui.py; this
module is the read side it asks for — history, the skill audit, and a
whitelisted slice of config. Kept separate because the whitelist is the point:
this is the one place that decides what a browser is allowed to see, and it
must never reach the keyring.
"""

import logging

from typing import get_args

from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import LLMConfig, Mode, NeroConfig
from nero.llm import providers
from nero.core.audit_log import AuditLog, default_audit_path
from nero.memory.history_store import HistoryStore, default_history_path

logger = logging.getLogger("nero.dashboard")

AUDIT_LIMIT = 100


def load_config() -> NeroConfig:
    """A missing or invalid config file must never crash the dashboard."""
    try:
        return ConfigManager().load()
    except ConfigError as exc:
        logger.warning("Could not load config for dashboard: %s", exc)
        return NeroConfig()


# Every section here is safe to show a browser and safe to edit from one.
# `mcp` is the section that is not: MCPServerConfig.env holds literal
# environment values, which are commonly API keys. It has its own page, which
# reports env by key name only. `routines` has its own page too.
CONFIG_SECTIONS = (
    "assistant", "mode", "llm", "skills", "memory", "security",
    "telegram", "discord", "slack", "googlechat", "voice",
)


def config_payload(config: NeroConfig | None = None) -> dict:
    """The whitelisted sections — never anything from the keyring, and never
    the one config section that stores secrets in plain text."""
    config = config or load_config()
    dumped = config.model_dump(mode="json")
    return {section: dumped[section] for section in CONFIG_SECTIONS}


def audit_payload(limit: int = AUDIT_LIMIT) -> list[dict]:
    entries = AuditLog(default_audit_path()).recent(limit=limit)
    return [entry.model_dump(mode="json") for entry in entries]


def history_payload() -> list[dict]:
    return HistoryStore(default_history_path(), session_id="dashboard").recent()


def sessions_payload() -> list[dict]:
    return HistoryStore(default_history_path(), session_id="dashboard").sessions()


def models_payload(config: NeroConfig | None = None) -> dict:
    """The routing picture: what answers now, and what answers if that fails."""
    config = config or load_config()
    llm = config.llm
    return {
        "provider": llm.provider,
        "model": llm.model,
        "base_url": llm.base_url,
        "fallback_chain": llm.fallback_chain
        or ([f"{llm.fallback_provider}/{llm.fallback_model}"] if llm.fallback_provider else []),
        "route_by": llm.route_by,
        "quality_rank": llm.quality_rank,
        "health_check": llm.health_check,
        "coding_model": llm.coding_model,
        "whitelist": llm.model_whitelist,
        "blacklist": llm.model_blacklist,
        "mode": config.mode,
        # Offered as the options in the dashboard's dropdowns, so the page
        # cannot drift from the Literals the schema actually accepts.
        "providers": list(providers.names()),
        "modes": list(get_args(Mode)),
        "route_by_options": list(get_args(LLMConfig.model_fields["route_by"].annotation)),
    }


def channels_payload(config: NeroConfig | None = None) -> dict:
    """Every way a person can reach Nero, and whether that way is open.

    The dashboard is one of them, so it says so — a reader looking at this
    page should not have to wonder which door they came through.
    """
    config = config or load_config()

    def installed(channel: str) -> bool:
        try:
            from nero.routines import bridge_plist_path, default_agents_dir

            return bridge_plist_path(default_agents_dir(), channel).exists()
        except Exception:  # noqa: BLE001 — a missing launchd dir is not an error here
            return False

    def chat_app(channel: str, peers: list, noun: str) -> dict:
        bridge = installed(channel)
        return {
            "enabled": getattr(config, channel).enabled,
            "detail": f"{len(peers)} paired {noun}(s)"
            + (", runs at login" if bridge else ""),
            "paired": len(peers),
            "chat_ids": [str(peer) for peer in peers],
            "bridge_installed": bridge,
        }

    return {
        "terminal": {"enabled": True, "detail": "nero chat / nero talk"},
        "dashboard": {"enabled": True, "detail": "this page, on 127.0.0.1"},
        "voice": {
            "enabled": config.voice.enabled,
            "detail": f"{config.voice.stt.engine} in, {config.voice.tts.engine} out"
            f" ({config.voice.tts.voice_id})",
        },
        "telegram": chat_app("telegram", list(config.telegram.allowed_chat_ids), "chat"),
        "discord": chat_app("discord", list(config.discord.allowed_channel_ids), "channel"),
        "slack": chat_app("slack", list(config.slack.allowed_channel_ids), "channel"),
        "googlechat": chat_app(
            "googlechat", list(config.googlechat.allowed_channel_ids), "space"
        ),
    }


def routines_payload(config: NeroConfig | None = None) -> list[dict]:
    """Scheduled prompts, with whether launchd actually has each one loaded.

    Config and launchd can disagree — a routine added but never installed is
    the most common surprise, so `installed` is reported separately from
    `enabled` rather than merged into one status.
    """
    config = config or load_config()
    try:
        from nero.routines import default_agents_dir, is_installed

        agents_dir = default_agents_dir()
    except Exception:  # noqa: BLE001
        agents_dir = is_installed = None
    rows = []
    for name, routine in config.routines.routines.items():
        rows.append(
            {
                "name": name,
                "schedule": routine.schedule,
                "prompt": routine.prompt,
                "enabled": routine.enabled,
                "installed": bool(is_installed and is_installed(name, agents_dir)),
            }
        )
    return sorted(rows, key=lambda r: r["name"])


def skills_payload(registry=None) -> list[dict]:
    """What the model may call, and why anything missing is missing.

    Taken from the live registry rather than the config toggles, because
    `enabled` and `available` are different answers: offline mode withdraws a
    network skill that is still switched on.
    """
    if registry is None:
        return []
    rows = []
    for name in sorted(registry.known_names()):
        skill = registry.get(name)
        if skill is None:
            continue
        rows.append(
            {
                "name": name,
                "description": skill.meta.description,
                "category": skill.meta.category,
                "tier": skill.meta.permission_tier,
                "requires_network": skill.meta.requires_network,
                "enabled": registry.is_enabled(name),
                "available": registry.is_available(name),
            }
        )
    return rows


def mcp_payload(config: NeroConfig | None = None) -> list[dict]:
    """Configured MCP servers. `env` is deliberately reduced to its key names —
    the values are commonly secrets."""
    config = config or load_config()
    return [
        {
            "name": name,
            "command": server.command,
            "args": server.args,
            "env_keys": sorted(server.env),
            "enabled": server.enabled,
            "trusted": server.trusted,
            "requires_network": server.requires_network,
        }
        for name, server in sorted(config.mcp.servers.items())
    ]


def memory_payload(config: NeroConfig | None = None) -> dict:
    config = config or load_config()
    try:
        from nero.memory.facts import FactStore, default_facts_path

        facts = [
            {"key": f.key, "value": f.value, "source": f.source, "updated_at": f.updated_at}
            for f in FactStore(default_facts_path()).all()
        ]
    except Exception:  # noqa: BLE001 — a stats line must never break the page
        facts = []
    try:
        from nero.memory.playbooks import PlaybookStore

        playbooks = [
            {
                "name": book.name,
                "task": book.task,
                "steps": book.steps,
                "avoid": book.avoid,
                "version": book.version,
                "uses": book.uses,
                "updated_at": book.updated_at,
            }
            for book in PlaybookStore().all()
        ]
    except Exception:  # noqa: BLE001 — a stats line must never break the page
        playbooks = []
    memory = config.memory
    return {
        "enabled": memory.enabled,
        "facts": len(facts),
        "fact_list": facts,
        "max_history_turns": memory.max_history_turns,
        "compact_after_messages": memory.compact_after_messages,
        "semantic_recall": memory.semantic_recall,
        "notes_dir": memory.notes_dir,
        "learning": memory.learning,
        "learn_after": memory.learn_after,
        "playbooks": len(playbooks),
        "playbook_list": playbooks,
    }


class EditError(Exception):
    """An edit the dashboard asked for that Nero will not make."""


def apply_edit(action: str, key: str, value: str | None = None) -> None:
    """Perform one edit from the dashboard.

    Five verbs, one door. `set` and `remove` go through ConfigManager — the
    same validate-then-save path `nero config set` uses, so a value the CLI
    would reject is rejected here too and a half-written config is never
    persisted. The other two delete stored data rather than settings.

    The audit log is deliberately not editable. It is the record of what Nero
    actually did, and a record you can quietly edit from a browser is not one.
    """
    if not key:
        raise EditError("Nothing to edit.")
    logger.warning("dashboard edit: %s %s", action, key)
    if action == "set":
        try:
            ConfigManager().set_value(key, "" if value is None else str(value))
        except ConfigError as exc:
            raise EditError(str(exc)) from exc
    elif action == "remove":
        try:
            ConfigManager().remove_value(key)
        except ConfigError as exc:
            raise EditError(str(exc)) from exc
    elif action == "forget_session":
        HistoryStore(default_history_path(), session_id="dashboard").forget_session(key)
    elif action == "forget_fact":
        from nero.memory.facts import FactStore, default_facts_path

        FactStore(default_facts_path()).forget(key)
    elif action == "forget_playbook":
        from nero.memory.playbooks import PlaybookStore

        PlaybookStore().forget(key)
    else:
        raise EditError(f"Unknown action: {action!r}")


def state_payload(registry=None) -> dict:
    """Everything the sidebar's pages read, in one round trip.

    One endpoint rather than seven: the payload is small, the pages are read
    together, and a single fetch keeps the whole view from showing a different
    config on each page while the user clicks around.
    """
    config = load_config()
    sessions = sessions_payload()
    skills = skills_payload(registry)
    memory = memory_payload(config)
    return {
        "assistant": config.assistant.name,
        "mode": config.mode,
        "channels": channels_payload(config),
        "models": models_payload(config),
        "skills": skills,
        "routines": routines_payload(config),
        "sessions": sessions,
        "mcp": mcp_payload(config),
        "memory": memory,
        "counts": {
            "skills_available": sum(1 for s in skills if s["available"]),
            "skills_total": len(skills),
            "sessions": len(sessions),
            "turns": sum(s["turns"] for s in sessions),
            "routines": len(config.routines.routines),
            "playbooks": memory["playbooks"],
            "mcp": len(config.mcp.servers),
        },
    }
