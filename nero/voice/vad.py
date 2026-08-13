"""Voice activity detection via silero's raw ONNX model.

The PyPI `silero-vad` package unconditionally pulls torch + torchaudio, which
would undo the kokoro-onnx work that removed torch from the bundle. Running the
~2.3 MB .onnx directly on the onnxruntime we already ship costs one small
wrapper and keeps the binary at its current size.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nero.voice.errors import VADUnavailableError


class VoiceActivityDetector:
    """Answers one question: is this 32 ms frame speech?

    Owns silero's recurrent state, so frames MUST be fed in order and `reset()`
    called between separate utterances. Nothing above this class sees a tensor.
    """

    FRAME_SAMPLES = 512  # 32 ms at 16 kHz; silero rejects other sizes at this rate
    SAMPLE_RATE = 16000
    _STATE_SHAPE = (2, 1, 128)

    # Silero v5 actually wants 64 samples of carried-over audio prepended to
    # each 512-sample frame (576 total) — this is what the upstream silero-vad
    # Python wrapper feeds it. The ONNX graph declares its `input` tensor with
    # dynamic shape [None, None], so passing a bare 512-sample frame raises
    # nothing; the model just silently returns a near-zero probability for
    # every frame, forever. This constant is an internal implementation
    # detail — callers still pass exactly FRAME_SAMPLES samples.
    _CONTEXT_SAMPLES = 64

    def __init__(self, model_path: Path | None, threshold: float = 0.5, _session=None):
        self.threshold = threshold
        self._session = _session if _session is not None else self._load(model_path)
        self._state = np.zeros(self._STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT_SAMPLES), dtype=np.float32)
        self._sr = np.array(self.SAMPLE_RATE, dtype=np.int64)

    @staticmethod
    def _load(model_path: Path | None):
        try:
            import onnxruntime as ort

            return ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 — any load failure must degrade, not crash
            raise VADUnavailableError(
                f"Could not load the voice-activity model: {exc}"
            ) from exc

    def reset(self) -> None:
        self._state = np.zeros(self._STATE_SHAPE, dtype=np.float32)
        # Context is per-utterance carry-over, same as recurrent state — leaking
        # the tail of one utterance's audio into the next would be the same
        # class of bug we're fixing here.
        self._context = np.zeros((1, self._CONTEXT_SAMPLES), dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size != self.FRAME_SAMPLES:
            raise ValueError(
                f"VAD needs exactly {self.FRAME_SAMPLES} samples, got {frame.size}"
            )
        frame_2d = frame.reshape(1, -1)
        model_input = np.concatenate([self._context, frame_2d], axis=1)
        probability, self._state = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": self._sr,
            },
        )
        self._context = frame_2d[:, -self._CONTEXT_SAMPLES :]
        return float(probability[0][0]) >= self.threshold
