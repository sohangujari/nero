import pytest
from pydantic import ValidationError

from nero.config.schema import NeroConfig, VoiceConfig


def test_voice_defaults():
    cfg = NeroConfig()
    assert cfg.voice.enabled is True
    assert cfg.voice.input_mode == "press_to_talk"
    assert cfg.voice.stt.engine == "faster-whisper"
    assert cfg.voice.stt.model == "base"
    assert cfg.voice.tts.engine == "kokoro"
    assert cfg.voice.tts.voice_id == "af_bella"


def test_voice_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"enabled": True, "bogus": 1})


def test_voice_rejects_bad_tts_engine():
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"tts": {"engine": "espeak"}})


def test_voice_roundtrips_through_full_config():
    data = NeroConfig().model_dump()
    assert data["voice"]["tts"]["voice_id"] == "af_bella"
    assert NeroConfig.model_validate(data).voice.stt.model == "base"


from typer.testing import CliRunner

from nero import cli

runner = CliRunner()


def _fake_manager(tmp_path, config: NeroConfig):
    class M:
        def exists(self):
            return True

        def load(self):
            return config

        def get_api_key(self, provider="claude"):
            return "sk-ant-test"

        def mask_api_key(self, key):
            return "sk-ant-...test"

    return M()


def test_talk_disabled_points_to_config(monkeypatch, tmp_path):
    config = NeroConfig()
    config.voice.enabled = False
    monkeypatch.setattr(cli, "ConfigManager", lambda: _fake_manager(tmp_path, config))
    result = runner.invoke(cli.app, ["talk"])
    assert result.exit_code == 0
    assert "nero config" in result.stdout


def test_talk_missing_voice_deps_shows_install_hint(monkeypatch, tmp_path):
    from nero.voice.errors import VoiceDependencyError

    config = NeroConfig()  # claude provider, voice enabled
    monkeypatch.setattr(cli, "ConfigManager", lambda: _fake_manager(tmp_path, config))

    def boom(*a, **k):
        raise VoiceDependencyError('Install them with: pip install "nero[voice]"')

    monkeypatch.setattr(cli, "FasterWhisperSTT", boom)
    result = runner.invoke(cli.app, ["talk"])
    assert "nero[voice]" in result.stdout


def test_talk_once_runs_voice_loop(monkeypatch, tmp_path):
    config = NeroConfig()
    monkeypatch.setattr(cli, "ConfigManager", lambda: _fake_manager(tmp_path, config))
    monkeypatch.setattr(cli, "FasterWhisperSTT", lambda model: object())
    monkeypatch.setattr(cli, "_build_registry", lambda manager, config: object())
    # Must be stubbed: the real pre-flight downloads ~300 MB of model weights.
    preflight_calls = []
    monkeypatch.setattr(
        cli, "_preflight_voice_models", lambda engine: preflight_calls.append(engine)
    )
    tts_builds = []

    class FakeTTS:
        SAMPLE_RATE = 24000

    def fake_build_tts(engine, voice_id):
        tts_builds.append(engine)
        return FakeTTS()

    monkeypatch.setattr(cli, "build_tts", fake_build_tts)

    captured = {}

    class FakeLoop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(cli, "VoiceLoop", FakeLoop)
    result = runner.invoke(cli.app, ["talk", "--once"])
    assert result.exit_code == 0
    assert captured["ran"] is True
    assert captured["once"] is True
    # Models are fetched and the TTS engine built ONCE up front — not lazily
    # per turn inside make_player (that caused a silent mid-turn 300 MB download).
    assert preflight_calls == ["kokoro"]
    assert tts_builds == ["kokoro"]
    # make_player must not rebuild the engine; calling it twice builds no more TTS.
    captured["make_player"]()
    captured["make_player"]()
    assert tts_builds == ["kokoro"]


from nero.hardware.detector import HardwareSpecs


def test_apply_detection_populates_voice_defaults(monkeypatch, tmp_path):
    from nero.config.manager import ConfigManager

    manager = ConfigManager(config_dir=tmp_path)
    manager.save(NeroConfig())
    monkeypatch.setattr(
        cli, "detect_hardware",
        lambda: HardwareSpecs(ram_gb=4.0, cpu_cores=8, os="Darwin", has_ollama=False),
    )
    config, _specs, _rec = cli._apply_detection(manager)
    assert config.voice.stt.model == "tiny"  # 4 GB tier
    assert config.voice.tts.engine == "kokoro"


def test_apply_detection_preserves_user_voice_choice(monkeypatch, tmp_path):
    from nero.config.manager import ConfigManager

    manager = ConfigManager(config_dir=tmp_path)
    cfg = NeroConfig()
    cfg.voice.stt.model = "large-v3-turbo"  # explicit user choice
    manager.save(cfg)
    monkeypatch.setattr(
        cli, "detect_hardware",
        lambda: HardwareSpecs(ram_gb=4.0, cpu_cores=8, os="Darwin", has_ollama=False),
    )
    config, _specs, _rec = cli._apply_detection(manager)
    assert config.voice.stt.model == "large-v3-turbo"  # not clobbered


def test_config_show_includes_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ConfigManager", lambda: _fake_manager(tmp_path, NeroConfig()))
    result = runner.invoke(cli.app, ["config", "show"])
    assert "af_bella" in result.stdout
    assert "online" in result.stdout


def test_ignore_further_interrupts_installs_sig_ign():
    """Once exiting, a second Ctrl+C must not interrupt interpreter teardown."""
    import signal

    previous = signal.getsignal(signal.SIGINT)
    try:
        cli._ignore_further_interrupts()
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_talk_ignores_interrupts_after_loop_exits(monkeypatch, tmp_path):
    """The guard is installed even if the voice loop raises."""
    import signal

    config = NeroConfig()
    monkeypatch.setattr(cli, "ConfigManager", lambda: _fake_manager(tmp_path, config))
    monkeypatch.setattr(cli, "FasterWhisperSTT", lambda model: object())
    monkeypatch.setattr(cli, "_build_registry", lambda manager, config: object())
    monkeypatch.setattr(cli, "_preflight_voice_models", lambda engine: None)

    class FakeTTS:
        SAMPLE_RATE = 24000

    monkeypatch.setattr(cli, "build_tts", lambda engine, voice_id: FakeTTS())
    installed = []
    monkeypatch.setattr(
        cli, "_ignore_further_interrupts", lambda: installed.append(True)
    )

    class ExplodingLoop:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise RuntimeError("loop blew up")

    monkeypatch.setattr(cli, "VoiceLoop", ExplodingLoop)
    runner.invoke(cli.app, ["talk", "--once"])
    assert installed == [True]


class TestVADConfig:
    def test_defaults(self):
        from nero.config.schema import NeroConfig

        vad = NeroConfig().voice.vad
        assert vad.enabled is True
        assert vad.silence_ms == 800
        assert vad.threshold == 0.5
        assert vad.max_utterance_seconds == 180
        assert vad.wait_for_speech_seconds == 30

    def test_barge_in_defaults_on(self):
        from nero.config.schema import NeroConfig

        assert NeroConfig().voice.barge_in is True

    def test_rejects_unknown_keys(self):
        from pydantic import ValidationError
        from nero.config.schema import NeroConfig

        with pytest.raises(ValidationError):
            NeroConfig.model_validate({"voice": {"vad": {"nonsense": 1}}})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("silence_ms", 199),
            ("threshold", -0.1),
            ("threshold", 1.1),
            ("max_utterance_seconds", 0),
            ("wait_for_speech_seconds", 0),
        ],
    )
    def test_out_of_range_values_rejected(self, field, value):
        from pydantic import ValidationError
        from nero.config.schema import NeroConfig

        with pytest.raises(ValidationError):
            NeroConfig.model_validate({"voice": {"vad": {field: value}}})


class TestBargeInActive:
    """barge_in depends on VAD; the two settings are not independent."""

    @pytest.mark.parametrize(
        ("vad_enabled", "barge_in", "expected"),
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),   # inert: no detector exists
            (False, False, False),
        ],
    )
    def test_truth_table(self, vad_enabled, barge_in, expected):
        from nero.config.schema import NeroConfig

        config = NeroConfig.model_validate(
            {"voice": {"barge_in": barge_in, "vad": {"enabled": vad_enabled}}}
        )
        assert config.voice.barge_in_active is expected


class TestBuildVad:
    def test_returns_none_and_warns_when_the_model_will_not_load(self, monkeypatch, capsys):
        from nero import cli
        from nero.config.schema import NeroConfig
        from nero.voice.errors import VADUnavailableError
        from rich.console import Console

        monkeypatch.setattr(cli, "ensure_vad_model", lambda on_progress=None: "/nope.onnx")

        def boom(*a, **k):
            raise VADUnavailableError("corrupt")

        monkeypatch.setattr(cli, "VoiceActivityDetector", boom)
        console = Console()
        assert cli._build_vad(NeroConfig(), console) is None
        assert "press Enter" in capsys.readouterr().out

    def test_returns_none_without_warning_when_vad_is_disabled(self, monkeypatch, capsys):
        from nero import cli
        from nero.config.schema import NeroConfig
        from rich.console import Console

        config = NeroConfig()
        config.voice.vad.enabled = False
        assert cli._build_vad(config, Console()) is None
        assert capsys.readouterr().out == ""

    def test_download_failure_also_degrades(self, monkeypatch, capsys):
        import httpx
        from nero import cli
        from nero.config.schema import NeroConfig
        from rich.console import Console

        def boom(on_progress=None):
            raise httpx.ConnectError("offline")

        monkeypatch.setattr(cli, "ensure_vad_model", boom)
        assert cli._build_vad(NeroConfig(), Console()) is None
        assert "press Enter" in capsys.readouterr().out
