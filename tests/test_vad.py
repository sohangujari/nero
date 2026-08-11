import numpy as np
import pytest

from nero.voice.errors import VADUnavailableError
from nero.voice.vad import VoiceActivityDetector


class FakeSession:
    """Mimics onnxruntime.InferenceSession for silero VAD.

    Returns scripted speech probabilities so tests never load a real model.
    """

    def __init__(self, probabilities):
        self._probs = list(probabilities)
        self.states_seen = []

    def run(self, _outputs, feeds):
        self.states_seen.append(feeds["state"].copy())
        prob = self._probs.pop(0) if self._probs else 0.0
        new_state = feeds["state"] + 1.0  # visibly different, to prove it is carried
        return [np.array([[prob]], dtype=np.float32), new_state]


def frame():
    return np.zeros(VoiceActivityDetector.FRAME_SAMPLES, dtype=np.float32)


class TestIsSpeech:
    def test_probability_above_threshold_is_speech(self):
        vad = VoiceActivityDetector(model_path=None, threshold=0.5, _session=FakeSession([0.9]))
        assert vad.is_speech(frame()) is True

    def test_probability_below_threshold_is_not_speech(self):
        vad = VoiceActivityDetector(model_path=None, threshold=0.5, _session=FakeSession([0.1]))
        assert vad.is_speech(frame()) is False

    def test_threshold_is_configurable(self):
        vad = VoiceActivityDetector(model_path=None, threshold=0.05, _session=FakeSession([0.1]))
        assert vad.is_speech(frame()) is True

    def test_wrong_frame_size_is_rejected(self):
        """Silero silently misbehaves on non-512 frames at 16 kHz; fail loudly."""
        vad = VoiceActivityDetector(model_path=None, _session=FakeSession([0.9]))
        with pytest.raises(ValueError):
            vad.is_speech(np.zeros(160, dtype=np.float32))


class TestRecurrentState:
    def test_state_is_carried_between_frames(self):
        session = FakeSession([0.1, 0.1, 0.1])
        vad = VoiceActivityDetector(model_path=None, _session=session)
        for _ in range(3):
            vad.is_speech(frame())
        first, second, third = session.states_seen
        assert not np.array_equal(first, second)
        assert not np.array_equal(second, third)

    def test_reset_clears_state(self):
        session = FakeSession([0.1, 0.1])
        vad = VoiceActivityDetector(model_path=None, _session=session)
        vad.is_speech(frame())
        vad.reset()
        vad.is_speech(frame())
        first, after_reset = session.states_seen
        assert np.array_equal(first, after_reset)


class TestLoadFailure:
    def test_unloadable_model_raises_vad_unavailable(self, tmp_path):
        """A corrupt or missing model must raise our own error, not an ORT one,
        so the caller can fall back without importing onnxruntime."""
        bad = tmp_path / "broken.onnx"
        bad.write_bytes(b"not really onnx")
        with pytest.raises(VADUnavailableError):
            VoiceActivityDetector(model_path=bad)


class TestRealModel:
    """Opt-in check against the actual silero model.

    Skipped unless the model is already cached, so CI stays offline and the
    suite never downloads. This is the only test that touches real onnxruntime;
    it exists because every other test fakes the session, which would happily
    keep passing if the real tensor names ever changed.
    """

    def test_real_model_scores_silence_low(self):
        from nero.voice.models import vad_model_present, ensure_vad_model

        if not vad_model_present():
            pytest.skip("silero model not cached; run `nero talk` once to fetch it")
        vad = VoiceActivityDetector(ensure_vad_model(), threshold=0.5)
        assert vad.is_speech(np.zeros(512, dtype=np.float32)) is False
