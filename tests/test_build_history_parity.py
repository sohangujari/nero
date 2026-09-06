import inspect

from nero import cli


def _source(fn):
    return inspect.getsource(fn)


class TestSharedConstruction:
    def test_run_chat_builds_registry_and_history_via_helpers(self):
        # History moved into _build_chat_loop when Telegram started sharing it;
        # what matters is that the text path still reaches it through the
        # helpers rather than constructing its own.
        src = _source(cli._run_chat) + _source(cli._build_chat_loop)
        assert "_build_registry(" in src
        assert "_build_history(" in src

    def test_telegram_and_chat_share_one_loop_builder(self):
        """A turn from a phone must be the same turn as a turn in the terminal
        — same fallback chain, key rotation, memory and skills."""
        for fn in (cli._run_chat, cli.telegram_main):
            assert "_build_chat_loop(" in _source(fn)

    def test_talk_builds_registry_and_history_via_helpers(self):
        src = _source(cli.talk)
        assert "_build_registry(" in src
        assert "_build_history(" in src

    def test_build_history_disabled_returns_none(self):
        from nero.config.schema import NeroConfig

        config = NeroConfig()
        config.memory.enabled = False
        assert cli._build_history(config) is None

    def test_build_history_passes_max_turns_through(self, isolate_audit_log):
        from nero.config.schema import NeroConfig

        config = NeroConfig()
        config.memory.max_history_turns = 7
        assert cli._build_history(config).max_turns == 7
