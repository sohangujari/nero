"""Phase 7: the known-set config rows are arrow-key pickers, with the numbered
prompt still driving every piped run — which is every test in this file."""
from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig

runner = CliRunner()


def _manager(tmp_path):
    m = ConfigManager(config_dir=tmp_path)
    m.save(NeroConfig())
    return m


class TestSTTModelPicker:
    def test_every_hardware_recommendation_appears_in_the_picker(self):
        """Hardware detection must never write a model the picker cannot show
        as the current selection."""
        from nero.hardware.tiers import DEFAULT_TIER, TIERS
        from nero.voice.stt import STT_MODELS

        offered = {model for model, _label in STT_MODELS}
        recommended = {stt for _ram, _local, stt, _tts in TIERS} | {DEFAULT_TIER[1]}
        assert recommended <= offered

    def test_picking_a_row_writes_the_model(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 6, then numbered choice 1 (tiny), then blank to finish.
        runner.invoke(cli.app, ["config"], input="6\n1\n\n")
        assert manager.load().voice.stt.model == "tiny"

    def test_the_last_row_escapes_to_free_text(self, monkeypatch, tmp_path, isolate_audit_log):
        from nero.voice.stt import STT_MODELS

        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        escape_row = len(STT_MODELS) + 1
        result = runner.invoke(
            cli.app, ["config"], input=f"6\n{escape_row}\ndistil-large-v3\n\n"
        )
        assert result.exit_code == 0
        assert manager.load().voice.stt.model == "distil-large-v3"

    def test_enter_leaves_the_model_alone(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        set_calls = []
        monkeypatch.setattr(manager, "set_value", lambda key, *a, **kw: set_calls.append(key))
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 6, then Enter at the picker, then blank to finish.
        runner.invoke(cli.app, ["config"], input="6\n\n\n")
        assert "voice.stt.model" not in set_calls


class TestTTSEnginePicker:
    def test_rows_match_the_schema_literal(self):
        """The picker table is presentation text; the Literal is the source of
        truth. Adding an engine to either side alone must fail this."""
        from typing import get_args

        from nero.config.schema import TTSConfig

        assert [engine for engine, _label in cli.TTS_ENGINES] == list(
            get_args(TTSConfig.model_fields["engine"].annotation)
        )

    def test_picking_a_row_writes_the_engine(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        assert manager.load().voice.tts.engine == "kokoro"
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 7, then numbered choice 2 (chatterbox), then blank to finish.
        runner.invoke(cli.app, ["config"], input="7\n2\n\n")
        assert manager.load().voice.tts.engine == "chatterbox"

    def test_enter_leaves_the_engine_alone(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        set_calls = []
        monkeypatch.setattr(manager, "set_value", lambda key, *a, **kw: set_calls.append(key))
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 7, then Enter at the picker, then blank to finish.
        runner.invoke(cli.app, ["config"], input="7\n\n\n")
        assert "voice.tts.engine" not in set_calls
