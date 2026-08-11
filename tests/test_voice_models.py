from pathlib import Path

from nero.voice import models


class TestVadModel:
    def test_ensure_vad_model_downloads_into_the_voice_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(models, "voice_cache_dir", lambda: tmp_path)
        seen = {}

        def fake_ensure_file(cache, name, url, on_progress=None):
            seen["cache"], seen["name"], seen["url"] = cache, name, url
            dest = Path(cache) / name
            dest.write_bytes(b"onnx")
            return dest

        monkeypatch.setattr(models, "_ensure_file", fake_ensure_file)
        path = models.ensure_vad_model()

        assert path == tmp_path / "silero_vad.onnx"
        assert seen["name"] == "silero_vad.onnx"
        assert seen["url"].endswith(".onnx")

    def test_vad_model_present_is_false_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(models, "voice_cache_dir", lambda: tmp_path)
        assert models.vad_model_present() is False

    def test_vad_model_present_is_false_for_a_zero_byte_file(self, tmp_path, monkeypatch):
        """A truncated/interrupted download must not count as present."""
        monkeypatch.setattr(models, "voice_cache_dir", lambda: tmp_path)
        (tmp_path / "silero_vad.onnx").write_bytes(b"")
        assert models.vad_model_present() is False

    def test_vad_model_present_is_true_for_a_real_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(models, "voice_cache_dir", lambda: tmp_path)
        (tmp_path / "silero_vad.onnx").write_bytes(b"onnx")
        assert models.vad_model_present() is True
