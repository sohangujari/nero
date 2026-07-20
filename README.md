# Nero

A cross-platform CLI personal AI assistant: a streaming text conversation with
an LLM in your terminal, where the model can call one real tool — opening an
application on your machine.

**Phase 2** adds multi-provider support: Nero talks to Claude, GPT, Gemini, or
a local Ollama model, switchable via config with zero code changes. On first
run it detects your hardware and recommends a local model tier. Voice, more
tools, and persistent memory are later phases.

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

Requires Python 3.12+.

```sh
pip install -e .
```

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
