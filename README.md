# Nero

A cross-platform CLI personal AI assistant: a streaming text conversation with
an LLM in your terminal, where the model can call one real tool — opening an
application on your machine.

**Phase 2** adds multi-provider support: Nero talks to Claude, GPT, Gemini, or
a local Ollama model, switchable via config with zero code changes. On first
run it detects your hardware and recommends a local model tier.

**Phase 3** adds voice: `nero talk` lets you speak to Nero and hear it speak
back, using the same LLM providers and tools as text mode. More tools and
persistent memory are later phases.

## Providers

| Provider | Default model | API key |
| --- | --- | --- |
| `claude` | `claude-sonnet-5` | keyring `nero/anthropic_api_key` |
| `openai` | `gpt-5` | keyring `nero/openai_api_key` |
| `gemini` | `gemini-2.5-pro` | keyring `nero/gemini_api_key` |
| `ollama` | hardware recommendation | none — fully local/offline |

Switch with `nero config` (interactive) or `nero config set llm.provider
ollama`. Choosing ollama auto-fills the model from the hardware recommendation
and never asks for a key. For ollama, Nero checks that the server is running
(`ollama serve`) and offers to `ollama pull` the model if it isn't downloaded.

## Install

Nero ships two ways. Most people want the binary.

### End users — standalone binary (no Python needed)

Download the file for your OS from the [Releases](https://github.com/sohangujari/nero/releases)
page, make it executable, and run it. There's no Python to install, no `pip`, no
dependency resolver — the binary bundles a fixed Python 3.12 and everything else.

```sh
# macOS / Linux
chmod +x nero-macos        # or nero-linux
./nero-macos

# Windows
nero-windows.exe
```

The binaries are **unsigned**, so the OS may warn on first launch:

- **macOS** — Gatekeeper blocks unsigned downloads. **Double-clicking is a dead
  end**: it shows *"Apple could not verify…"* with only *Done* / *Move to Bin* —
  no *Open* button. Instead **right-click (or Control-click) the file → Open →
  Open** — that path offers the *Open* button the double-click dialog doesn't.
  If your macOS hides it there, open **System Settings → Privacy & Security** and
  click **Open Anyway**, or clear the quarantine flag from a terminal:
  `xattr -d com.apple.quarantine nero-macos`.
- **Windows** — SmartScreen may prompt; choose *More info* → *Run anyway*.

The first launch runs a short setup: Nero detects your hardware, recommends a local
model, and asks for your provider and (for cloud providers) an API key. Voice model
weights (whisper, kokoro-onnx) download once into a per-user cache the first time you
run `nero talk`. Nothing else needs installing — no Python, no `pip`.

### Developers / contributors — uv

Nero is pinned to **Python 3.12** and uses [uv](https://docs.astral.sh/uv/),
which manages its own isolated 3.12 — independent of whatever Python is on your
`PATH`. This is what makes installs reproducible: `uv.lock` pins every dependency
(including transitive ones) to the exact tested version.

```sh
uv sync --extra voice --group dev   # installs the locked deps into .venv on py3.12
uv run nero                         # chat (text)
uv run nero talk                    # voice
uv run pytest -q                    # tests
```

`uv sync` resolves against the committed `uv.lock`, so you get the identical
versions CI tested — the Python-3.13 install failure that motivated this setup
cannot recur. If you install via bare `pip`/`pipx` against a non-3.12
interpreter instead, Nero prints a one-line warning pointing you here.

## Voice (Phase 3)

```sh
nero talk          # speak; Nero transcribes, answers, and speaks back
nero talk --once   # a single exchange, then exit
```

Press Enter to start recording, Enter again to stop. What Nero heard is printed
before it replies. Say "stop" (or "exit") to leave, or press Ctrl+C.

The default voice is `af_bella` (female); the male equivalent is `am_michael`.
Change it — along with the STT model and TTS engine — in `nero config` (the
Voice rows) or with `nero config set voice.tts.voice_id am_michael`. Disable
voice entirely with `nero config set voice.enabled false`. On first run the STT
model and TTS engine are auto-selected from your detected hardware tier.

## First run

```sh
nero
```

On the very first run, Nero walks you through setup: it asks for your Anthropic
API key (input is masked), stores it in your OS keychain — never in a file —
and writes a default config. Then you're chatting:

```
you> open Spotify for me
Nero> Done — Spotify is opening now.
```

Type `exit`, `quit`, or press Ctrl+C to leave. Conversation history lives only
in memory for the current session.

## Commands

| Command | What it does |
| --- | --- |
| `nero` | Start the chat REPL (first run triggers setup) |
| `nero config` | Interactive menu: assistant name, model, API key |
| `nero config set <key> <value>` | Scriptable edit, e.g. `nero config set llm.provider ollama` |
| `nero config show` | Print current config (API key masked) |
| `nero detect` | Re-run hardware detection, refresh the local-model recommendation |
| `nero --debug` | Chat with verbose stderr logging (tool-call plumbing, per-turn history) |
| `nero --version` | Print the installed version |

Valid config keys: `assistant.name`, `llm.provider`
(claude|openai|gemini|ollama), `llm.model`. The `hardware.*` block is
auto-populated by detection. The RAM → local-model table lives in
`nero/hardware/tiers.py` — edit it as models improve.

## Config file location

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/nero/config.yaml` |
| Windows | `%APPDATA%\nero\config.yaml` |
| Linux | `~/.config/nero/config.yaml` |

The API key is never written there — it lives in the OS keyring under the
service name `nero`.

## Development

```sh
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

Layout: `nero/config` (schema + manager), `nero/tools` (tool interface +
`open_app`), `nero/llm` (Claude client: streaming + tool loop), `nero/core`
(REPL), `nero/cli.py` (typer entry point).
