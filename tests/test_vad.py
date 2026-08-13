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
        self.inputs_seen = []

    def run(self, _outputs, feeds):
        self.states_seen.append(feeds["state"].copy())
        self.inputs_seen.append(feeds["input"].copy())
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


class TestContextWindow:
    """Pins the exact bug that shipped: silero needs 576 samples (64 samples of
    carried-over context + the 512-sample frame), not a bare 512. The dynamic
    ONNX input shape means feeding 512 raises nothing — it just silently
    returns near-zero probability forever. This test catches that with no
    audio at all: it only inspects what tensor width and content reach the
    (fake) model.
    """

    def test_model_is_fed_576_samples_with_carried_context(self):
        session = FakeSession([0.1, 0.1])
        vad = VoiceActivityDetector(model_path=None, _session=session)
        frame1 = np.arange(512, dtype=np.float32)
        frame2 = np.arange(512, 1024, dtype=np.float32)

        vad.is_speech(frame1)
        vad.is_speech(frame2)

        input1, input2 = session.inputs_seen
        assert input1.shape == (1, 576)
        assert input2.shape == (1, 576)
        # First call: context starts at zero.
        assert np.array_equal(input1[0, :64], np.zeros(64, dtype=np.float32))
        assert np.array_equal(input1[0, 64:], frame1)
        # Second call: context is the last 64 samples of the PREVIOUS frame.
        assert np.array_equal(input2[0, :64], frame1[-64:])
        assert np.array_equal(input2[0, 64:], frame2)


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

    def test_real_model_input_width_materially_changes_the_probability(self):
        """Positive assertion against the real model (not just a negative one).

        This does not synthesize actual speech — a Kokoro-generated fixture
        would work too, but it would need ~300 MB of extra cached weights on
        top of silero's, and would only be able to skip-clean, not actually
        run, on most dev/CI machines. What we actually need to prove is
        narrower and doesn't need real speech at all: that feeding the model
        the correct 576-wide (64-sample context + 512-sample frame) input
        produces a materially different probability than feeding it a bare
        512-sample frame, for identical audio content. That's precisely the
        property a width bug like the one just fixed breaks — with the bug,
        the model returns near-zero regardless of the input; the difference
        this test measures is the entire bug.
        """
        import onnxruntime as ort

        from nero.voice.models import vad_model_present, ensure_vad_model

        if not vad_model_present():
            pytest.skip("silero model not cached; run `nero talk` once to fetch it")

        session = ort.InferenceSession(
            str(ensure_vad_model()), providers=["CPUExecutionProvider"]
        )
        state = np.zeros(VoiceActivityDetector._STATE_SHAPE, dtype=np.float32)
        sr = np.array(VoiceActivityDetector.SAMPLE_RATE, dtype=np.int64)

        # Deterministic speech-like content (a couple of harmonics in typical
        # voice pitch range) — not real speech, but not silence either, so it
        # actually exercises the model instead of comparing two near-zeros.
        t = np.arange(576, dtype=np.float32) / VoiceActivityDetector.SAMPLE_RATE
        content = (
            0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 420 * t)
        ).astype(np.float32)

        bare_512 = content[-512:].reshape(1, -1)
        with_context = content.reshape(1, -1)

        prob_512, _ = session.run(
            None, {"input": bare_512, "state": state.copy(), "sr": sr}
        )
        prob_576, _ = session.run(
            None, {"input": with_context, "state": state.copy(), "sr": sr}
        )

        assert abs(float(prob_576[0][0]) - float(prob_512[0][0])) > 0.3
