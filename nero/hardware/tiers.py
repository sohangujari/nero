"""RAM → recommended local model, STT model, and TTS engine, as an ordered lookup.

This table lives in its own file on purpose: edit it freely as models improve.
Each row is (max_ram_gb_exclusive, local_model, stt_model, tts_engine); the first
row whose bound the detected RAM falls under wins, and DEFAULT_TIER covers
everything above.
"""

TIERS: list[tuple[float, str, str, str]] = [
    (6, "gemma3:2b", "tiny", "kokoro"),
    (8, "llama3.2:3b", "base", "kokoro"),
    (16, "phi4-mini", "small", "kokoro"),
]

DEFAULT_TIER: tuple[str, str, str] = ("qwen3:8b", "large-v3-turbo", "kokoro")  # 16 GB+
