from __future__ import annotations


class VoiceError(Exception):
    """Base class for all voice-mode failures."""


class VoiceDependencyError(VoiceError):
    """The nero[voice] extras aren't installed."""


class MicUnavailableError(VoiceError):
    """No microphone is available or the input device could not be opened."""


class MicPermissionError(VoiceError):
    """The OS denied microphone access (common first-run on macOS)."""


class PlaybackError(VoiceError):
    """Audio playback failed."""


class TTSLoadError(VoiceError):
    """A TTS engine's model files could not be loaded."""


class VADUnavailableError(VoiceError):
    """The voice-activity model could not be downloaded or loaded."""


class BargeIn(Exception):
    """The user spoke over Nero. Not an error — a control-flow signal.

    Deliberately not a VoiceError: it is raised out of `tap()` to unwind a
    normal turn, and must not be caught by generic voice error handling.
    """
