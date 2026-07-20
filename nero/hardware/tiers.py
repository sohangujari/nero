"""RAM → recommended local model, as an ordered lookup.

This table lives in its own file on purpose: edit it freely as local models
improve. Each row is (max_ram_gb_exclusive, model); the first row whose bound
the detected RAM falls under wins, and DEFAULT_TIER covers everything above.
"""

TIERS: list[tuple[float, str]] = [
    (6, "gemma3:2b"),
    (8, "llama3.2:3b"),
    (16, "phi4-mini"),
]

DEFAULT_TIER = "qwen3:8b"  # 16 GB and up
