"""First-launch model fetching for voice engines.

Model weights are NOT bundled into the shipped binary (that would bloat it and
tie a release to specific weights). Instead they're downloaded once into a
per-user cache the first time voice is used, then reused.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from platformdirs import user_cache_dir

# kokoro-onnx v1.0 model files (onnxruntime weights + packed voice styles).
_KOKORO_MODEL = (
    "kokoro-v1.0.onnx",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
)
_KOKORO_VOICES = (
    "voices-v1.0.bin",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
)

# silero VAD v5, ~2.3 MB. Downloaded like the Kokoro weights rather than bundled,
# so the shipped binary stays the same size.
_SILERO_VAD = (
    "silero_vad.onnx",
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
)


def voice_cache_dir() -> Path:
    path = Path(user_cache_dir("nero")) / "voice-models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_present() -> bool:
    """True if both Kokoro files are already cached (no download needed)."""
    cache = voice_cache_dir()
    return all(
        (cache / name).exists() and (cache / name).stat().st_size > 0
        for name, _url in (_KOKORO_MODEL, _KOKORO_VOICES)
    )


def ensure_kokoro_model(on_progress=None) -> tuple[Path, Path]:
    """Return (model_path, voices_path), downloading them on first use.

    `on_progress(name, downloaded_bytes, total_bytes)` is called as bytes arrive
    so callers can render a progress bar — these files are ~300 MB and a silent
    download is indistinguishable from a hang.
    """
    cache = voice_cache_dir()
    model_path = _ensure_file(cache, *_KOKORO_MODEL, on_progress=on_progress)
    voices_path = _ensure_file(cache, *_KOKORO_VOICES, on_progress=on_progress)
    return model_path, voices_path


def vad_model_present() -> bool:
    """True if the VAD model is already cached (no download needed)."""
    dest = voice_cache_dir() / _SILERO_VAD[0]
    return dest.exists() and dest.stat().st_size > 0


def ensure_vad_model(on_progress=None) -> Path:
    """Return the cached silero VAD model path, downloading it on first use."""
    return _ensure_file(voice_cache_dir(), *_SILERO_VAD, on_progress=on_progress)


def _ensure_file(cache: Path, name: str, url: str, on_progress=None) -> Path:
    dest = cache / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")  # truncated on retry, so partials never accumulate
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=httpx.Timeout(None, connect=10.0)
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        downloaded = 0
        if on_progress is not None:
            on_progress(name, 0, total)
        with tmp.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(name, downloaded, total)
    tmp.replace(dest)
    return dest
