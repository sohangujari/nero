import platform

import psutil
from pydantic import BaseModel

from nero.hardware.tiers import DEFAULT_TIER, TIERS
from nero.llm import ollama


class HardwareSpecs(BaseModel):
    ram_gb: float
    cpu_cores: int
    os: str  # Darwin, Windows, Linux
    has_ollama: bool  # Ollama server reachable at localhost:11434


def detect_hardware() -> HardwareSpecs:
    return HardwareSpecs(
        ram_gb=round(psutil.virtual_memory().total / (1024**3), 1),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        os=platform.system(),
        has_ollama=ollama.reachable(),
    )


def recommend_model(specs: HardwareSpecs) -> str:
    for max_ram_exclusive, model in TIERS:
        if specs.ram_gb < max_ram_exclusive:
            return model
    return DEFAULT_TIER
