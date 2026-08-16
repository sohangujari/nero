from pathlib import Path

import keyring
import keyring.errors
import yaml
from platformdirs import user_config_dir
from pydantic import ValidationError

from nero.config.schema import NeroConfig
from nero.llm import providers

KEYRING_SERVICE = "nero"


class ConfigError(Exception):
    """A configuration file or value is invalid."""


class ConfigManager:
    """Loads, saves, and edits Nero's YAML config; holds the API key in the OS keyring."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = Path(config_dir) if config_dir else Path(user_config_dir("nero"))
        self.config_path = self.config_dir / "config.yaml"

    def exists(self) -> bool:
        return self.config_path.exists()

    def load(self) -> NeroConfig:
        if not self.exists():
            return NeroConfig()
        try:
            data = yaml.safe_load(self.config_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {self.config_path}: {exc}") from exc
        try:
            return NeroConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"Invalid config in {self.config_path}: {exc}") from exc

    def save(self, config: NeroConfig) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))

    def set_value(self, key_path: str, value: str) -> NeroConfig:
        """Set a dotted key path (e.g. "assistant.name") and persist, validating first."""
        data = self.load().model_dump()
        parts = key_path.split(".")
        node = data
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"Unknown config key: {key_path!r}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise ConfigError(f"Unknown config key: {key_path!r}")
        node[parts[-1]] = value
        try:
            config = NeroConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(f"Invalid value for {key_path!r}: {exc}") from exc
        self.save(config)
        return config

    @staticmethod
    def _entry(provider: str) -> str | None:
        """The keyring entry for a provider, or None if it needs no key.

        Unknown providers answer None rather than raising: `config show` and the
        menu both render through here, and a config holding a name this build
        doesn't know must not crash the display.
        """
        try:
            return providers.get(provider).keyring_entry
        except KeyError:
            return None

    @staticmethod
    def provider_needs_key(provider: str) -> bool:
        return ConfigManager._entry(provider) is not None

    def get_api_key(self, provider: str = "claude") -> str | None:
        entry = self._entry(provider)
        if entry is None:
            return None
        try:
            return keyring.get_password(KEYRING_SERVICE, entry)
        except keyring.errors.KeyringError:
            # No backend (Linux CI, headless boxes) or a locked/denied keychain.
            # Every caller already handles "no key stored", and `config`/`config
            # show` render through here — letting this escape kills the menu.
            # Deliberately read-only: set_api_key must still raise, or a failed
            # write would look like success and lose the key.
            return None

    def set_api_key(self, provider: str, value: str) -> None:
        entry = self._entry(provider)
        if entry is None:
            raise ConfigError(f"Provider {provider!r} does not use an API key.")
        keyring.set_password(KEYRING_SERVICE, entry, value)

    @staticmethod
    def mask_api_key(key: str) -> str:
        if len(key) < 8:
            return "****"
        if key.startswith("sk-ant-"):
            return f"sk-ant-...{key[-4:]}"
        return f"...{key[-4:]}"
