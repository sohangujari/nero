"""What the dashboard shows, minus the showing.

The page itself is the React/shadcn app in web/ served by nero/webui.py; this
module is the read side it asks for — history, the skill audit, and a
whitelisted slice of config. Kept separate because the whitelist is the point:
this is the one place that decides what a browser is allowed to see, and it
must never reach the keyring.
"""

import logging

from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import NeroConfig
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


def config_payload(config: NeroConfig | None = None) -> dict:
    """Only the whitelisted sections — never anything from the keyring."""
    config = config or load_config()
    return {
        "mode": config.mode,
        "llm": config.llm.model_dump(),
        "skills": {"enabled": config.skills.enabled.model_dump()},
        "voice": {"enabled": config.voice.enabled},
    }


def audit_payload(limit: int = AUDIT_LIMIT) -> list[dict]:
    entries = AuditLog(default_audit_path()).recent(limit=limit)
    return [entry.model_dump(mode="json") for entry in entries]


def history_payload() -> list[dict]:
    return HistoryStore(default_history_path(), session_id="dashboard").recent()
