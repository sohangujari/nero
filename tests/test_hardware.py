from types import SimpleNamespace

import httpx
import pytest

from nero.hardware.detector import HardwareSpecs, detect_hardware, recommend_model
from nero.llm import ollama


def specs(ram_gb):
    return HardwareSpecs(ram_gb=ram_gb, cpu_cores=8, os="Darwin", has_ollama=False)


class TestRecommendModel:
    @pytest.mark.parametrize(
        ("ram_gb", "expected"),
        [
            (2, "gemma3:2b"),
            (4, "gemma3:2b"),
            (5.9, "gemma3:2b"),
            (6, "llama3.2:3b"),
            (7.9, "llama3.2:3b"),
            (8, "phi4-mini"),
            (15.9, "phi4-mini"),
            (16, "qwen3:8b"),
            (32, "qwen3:8b"),
        ],
    )
    def test_tiers(self, ram_gb, expected):
        assert recommend_model(specs(ram_gb)) == expected


class TestDetectHardware:
    def test_reads_psutil_and_ollama(self, monkeypatch):
        monkeypatch.setattr(
            "psutil.virtual_memory", lambda: SimpleNamespace(total=17_179_869_184)
        )
        monkeypatch.setattr("psutil.cpu_count", lambda logical=True: 8)
        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("nero.llm.ollama.reachable", lambda: True)
        result = detect_hardware()
        assert result.ram_gb == 16.0
        assert result.cpu_cores == 8
        assert result.os == "Darwin"
        assert result.has_ollama is True


class TestOllamaHelpers:
    def test_reachable_true_on_200(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.get", lambda url, timeout: SimpleNamespace(status_code=200)
        )
        assert ollama.reachable() is True

    def test_reachable_false_on_connection_error(self, monkeypatch):
        def raise_error(url, timeout):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr("httpx.get", raise_error)
        assert ollama.reachable() is False

    def test_has_model_matching(self, monkeypatch):
        monkeypatch.setattr(
            ollama, "list_models", lambda base_url=None: ["phi4-mini:latest", "qwen3:8b"]
        )
        assert ollama.has_model("qwen3:8b") is True
        assert ollama.has_model("phi4-mini") is True  # bare name matches any tag
        assert ollama.has_model("phi4") is False
        assert ollama.has_model("llama3.2:3b") is False

    def test_pull_model_runs_ollama_cli(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)
        assert ollama.pull_model("qwen3:8b") is True
        assert calls == [["ollama", "pull", "qwen3:8b"]]
