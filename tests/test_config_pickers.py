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


class TestVoicePicker:
    def test_enter_does_not_silently_reassign_the_voice(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        """The old numbered table defaulted to row 1, so Enter overwrote a
        non-default voice with Bella."""
        manager = _manager(tmp_path)
        manager.set_value("voice.tts.voice_id", "bm_george")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 8, then Enter at the picker, then blank to finish.
        runner.invoke(cli.app, ["config"], input="8\n\n\n")
        assert manager.load().voice.tts.voice_id == "bm_george"

    def test_the_picker_labels_carry_the_gender(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="8\n\n\n")
        assert "Bella (female)" in result.stdout


class TestSkillsCheckbox:
    def test_toggling_off_writes_only_the_changed_key(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        """A broken rewrite of the loop could write every toggle back with
        correct values and still pass a state-only assertion. Spy on
        set_value so a regression that rewrites all four keys — not just the
        one that changed — fails here even though the resulting state would
        look right."""
        manager = _manager(tmp_path)
        assert manager.load().skills.enabled.open_app is True
        assert manager.load().skills.enabled.play_music is True
        set_calls = []
        original_set_value = manager.set_value

        def spy_set_value(key, *a, **kw):
            set_calls.append(key)
            return original_set_value(key, *a, **kw)

        monkeypatch.setattr(manager, "set_value", spy_set_value)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        # Row 10, then numbered choice 1 (open_app), then Enter to leave the
        # submenu, then blank to finish.
        runner.invoke(cli.app, ["config"], input="10\n1\n\n")
        assert set_calls == ["skills.enabled.open_app"]
        after = manager.load().skills.enabled
        assert after.open_app is False
        assert after.play_music is True

    def test_enter_at_the_submenu_changes_nothing(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        """Guards the most dangerous failure mode in this task: collapsing
        `None` (leave the config alone) into an empty set (disable every
        skill). If a future edit dropped the `if picked is None: return`
        guard, Enter at the submenu would silently disable all four skills
        instead of changing nothing — this is what would catch that."""
        manager = _manager(tmp_path)
        before = manager.load().skills.enabled.model_dump()
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="10\n\n\n")
        assert result.exit_code == 0
        assert manager.load().skills.enabled.model_dump() == before

    def test_the_submenu_renders_checkbox_markers(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        """The discriminating test for this task. The old submenu rendered a
        four-column table with yes/no cells under an "enabled" header; the
        pick_many fallback renders [x]/[ ] markers instead."""
        manager = _manager(tmp_path)
        manager.set_value("skills.enabled.open_app", "false")
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="10\n\n\n")
        assert "[x]" in result.stdout
        assert "[ ]" in result.stdout

    def test_network_needing_skills_are_labelled_in_plain_text(
        self, monkeypatch, tmp_path, isolate_audit_log
    ):
        """questionary renders rich markup literally, so the annotation must
        not be wrapped in [dim]...[/dim]."""
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="10\n\n\n")
        assert "get_weather (needs network)" in result.stdout
        assert "[dim]" not in result.stdout


class TestOuterMenuPicker:
    """v1.5.3: the top-level row list is a picker too. Piped runs still type a
    number — the numbered fallback is what every other menu test drives."""

    def test_the_outer_menu_goes_through_ui_pick(self, monkeypatch, tmp_path, isolate_audit_log):
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        offered = []
        answers = iter(["12", None])  # toggle Memory, then Esc out of the menu

        def fake_pick(title, choices, **kwargs):
            offered.append(choices)
            return next(answers)

        monkeypatch.setattr(cli.ui, "pick", fake_pick)
        result = runner.invoke(cli.app, ["config"])

        assert result.exit_code == 0
        # Row values stay "1".."15" so the picker feeds the same dispatch chain.
        assert [value for value, _label in offered[0][:15]] == [str(n) for n in range(1, 16)]
        # A falsy last row is the whole exit contract: it and Esc share `not choice`.
        assert offered[0][-1] == ("", "Done")
        assert manager.load().memory.enabled is False

    def test_a_piped_run_still_gets_the_numbered_rows(self, monkeypatch, tmp_path, isolate_audit_log):
        """The arrow picker is invisible to scripts; the fallback is the whole
        menu for them, so it must list every row and say how to leave."""
        manager = _manager(tmp_path)
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["config"], input="\n")

        assert result.exit_code == 0
        assert "Assistant Name" in result.stdout
        assert "VAD Auto-Stop" in result.stdout
        assert "Enter to finish" in result.stdout
