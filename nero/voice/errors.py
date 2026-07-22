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
