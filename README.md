# Nero Agent

A cross-platform CLI personal AI assistant: a streaming text conversation with
an LLM in your terminal, where the model can call one real tool — opening an
application on your machine.

**Phase 2** adds multi-provider support: Nero Agent talks to Claude, GPT, Gemini, or
a local Ollama model, switchable via config with zero code changes. On first
run it detects your hardware and recommends a local model tier.

**Phase 3** adds voice: `nero talk` lets you speak to Nero Agent and hear it speak
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
and never asks for a key. For ollama, Nero Agent checks that the server is running
(`ollama serve`) and offers to `ollama pull` the model if it isn't downloaded.

## Install

Nero Agent ships two ways. Most people want the binary.

### End users — standalone binary (no Python needed)

Download the file for your OS from the [Releases](https://github.com/sohangujari/nero/releases)
page, make it executable, and run it. There's no Python to install, no `pip`, no
dependency resolver — the binary bundles a fixed Python 3.12 and everything else.

Asset names carry their release version — substitute the file you actually
downloaded in the commands below.

```sh
# macOS / Linux
chmod +x nero-macos-v0.1.1        # or nero-linux-v0.1.1
./nero-macos-v0.1.1

# Windows
nero-windows-v0.1.1.exe
```

The binaries are **unsigned**, so the OS may warn on first launch:

- **macOS** — Gatekeeper blocks unsigned downloads. **Double-clicking is a dead
  end**: it shows *"Apple could not verify…"* with only *Done* / *Move to Bin* —
  no *Open* button. Instead **right-click (or Control-click) the file → Open →
  Open** — that path offers the *Open* button the double-click dialog doesn't.
  If your macOS hides it there, open **System Settings → Privacy & Security** and
  click **Open Anyway**, or clear the quarantine flag from a terminal:
  `xattr -d com.apple.quarantine nero-macos-v0.1.1`.
- **Windows** — SmartScreen may prompt; choose *More info* → *Run anyway*.

The first launch runs a short setup: Nero Agent detects your hardware, recommends a local
model, and asks for your provider and (for cloud providers) an API key. Voice model
weights (whisper, kokoro-onnx) download once into a per-user cache the first time you
run `nero talk`. Nothing else needs installing — no Python, no `pip`.

### Developers / contributors — uv

Nero Agent is pinned to **Python 3.12** and uses [uv](https://docs.astral.sh/uv/),
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
interpreter instead, Nero Agent prints a one-line warning pointing you here.

## Voice (Phase 3)

```sh
nero talk          # speak; Nero Agent transcribes, answers, and speaks back
nero talk --once   # a single exchange, then exit
```

`nero talk` is hands-free. It starts listening immediately, stops recording on
its own when you stop talking, answers, and then goes straight back to
listening — there is no key to press between turns. Talk over Nero Agent while
it's replying to interrupt it; it stops, keeps what it managed to say in the
conversation, and answers your new question instead, so you can change the
subject mid-reply.

If nobody speaks for a minute (`voice.vad.wait_for_speech_seconds`) the session
falls asleep: it releases the microphone and stops running speech detection.
Press Enter to wake it, or Ctrl+C to leave.

```
● Listening — I'll stop when you do
🗣  heard: what's the weather in mumbai
Nero> It's 31°C and humid right now.

● Listening — I'll stop when you do

💤 Asleep — press Enter to wake me, or Ctrl+C to leave.
```

With `voice.vad.enabled` set to `false` there is no endpointing, so that mode
keeps its press-Enter-to-start, press-Enter-to-stop prompt and never sleeps.

Barge-in needs headphones in practice: on built-in speakers, Nero Agent hears its
own reply through the mic and interrupts itself almost every time, so Nero Agent
auto-disables barge-in when the default output device looks like built-in
speakers and prints one line explaining why. Headphones re-enable it
automatically (no acoustic path from speaker to mic). If your speakers sit
far enough from the mic that self-hearing genuinely isn't a problem, force it
back on with `nero config set voice.force_barge_in true`. To turn barge-in
off entirely instead, use `nero config set voice.barge_in false`.

Tuning (all optional):

| Setting | Default | What it does |
| --- | --- | --- |
| `voice.vad.enabled` | `true` | Master switch. `false` restores press-Enter-to-stop. |
| `voice.barge_in` | `true` | Interrupt Nero Agent by talking. Inert when `vad.enabled` is false. |
| `voice.force_barge_in` | `false` | Bypass the built-in-speaker auto-suppression above. |
| `voice.vad.silence_ms` | `800` | Silence that ends your turn. Raise it if you're cut off mid-thought. |
| `voice.vad.threshold` | `0.5` | Speech sensitivity. Raise it in a noisy room. |
| `voice.vad.max_utterance_seconds` | `180` | Hard cap on one recording. |
| `voice.vad.wait_for_speech_seconds` | `60` | Silence before a hands-free session sleeps and releases the mic. |
| `voice.stt.language` | `"en"` | Language pinned for transcription. `null` auto-detects, costing ~0.5s per turn. |

Nero speaks the reply, it doesn't read it out. Markdown is stripped before
synthesis — `**bold**`, headings, list markers, and links are formatting, and
pronouncing them costs real time (measured against Kokoro: +2.7 s for one bold
pair, +4.8 s for a link, because the URL itself gets spoken). Your terminal
still shows the original text.

What Nero Agent heard is printed before it replies. Say "stop" (or "exit") to leave,
or press Ctrl+C.

### Choosing a voice

`nero config` offers Kokoro voices ordered best-first by Kokoro's own published
per-voice grades. Anything below C+ is left out — the list used to offer
`am_adam`, graded **F+** and the least intelligible voice the model ships,
while omitting `af_heart` (A). If your replies sound muddy, the voice is the
first thing to change:

```sh
nero config set voice.tts.voice_id af_heart    # A     (female)
nero config set voice.tts.voice_id af_bella    # A-    (female, default)
nero config set voice.tts.voice_id am_michael  # C+    (male)
```

### How the reply reaches your speakers

Nero synthesizes ahead of what it is saying. The model's text is split into
sentences as it streams, sentence one starts playing while sentence two is
still being synthesized, and everything goes to a single audio stream that
stays open for the turn. The first segment is cut short — at a clause, or at
60 characters — because it is the only one you wait on in silence.

Concretely, on a four-sentence reply on an M-series laptop: dead air between
sentences dropped from 4.8s to 0.9s, and the wait for the first word from
2.0s to 0.7s.

How long a segment may be is not fixed — it grows with how much audio the
speaker already has banked. Capping only the *first* segment was not enough: a
27-character opener ("Sure, I can help with that.") gives 1.49s of speech,
and the 73-character sentence after it took 1.87s to synthesize, so the
speaker ran dry in the middle of the reply. Every reply, at the same place.
Each segment is now limited to what synthesis can finish in time
(`catch_up_limit`), and when that bites, the break lands on a clause rather
than the last space. Measured, same replies, first word unchanged:

| reply | dead air before | after |
| --- | --- | --- |
| weather forecast | 0.70 s | **0.00 s** |
| casual greeting | 1.96 s | **0.49 s** |
| dense explanation | 0.00 s | 0.00 s |

`nero talk --debug` prints one latency line per turn:

```
voice turn: stt=0.67 ttft=0.42 speech=0.91 done=5.18 (4 sentences)
```

`stt` is transcription, `ttft` the model's first token, `speech` the first
sentence reaching the synthesizer, `done` the end of playback — all in seconds
from the moment you stopped talking.

The default voice is `af_bella` (female); the male equivalent is `am_michael`.
Change it — along with the STT model and TTS engine — in `nero config` (the
Voice rows) or with `nero config set voice.tts.voice_id am_michael`. Disable
voice entirely with `nero config set voice.enabled false`. On first run the STT
model and TTS engine are auto-selected from your detected hardware tier.

## First run

```sh
nero
```

On the very first run, Nero Agent walks you through setup: it asks for your Anthropic
API key (input is masked), stores it in your OS keychain — never in a file —
and writes a default config. Then you're chatting:

```
you> open Spotify for me
Nero> Done — Spotify is opening now.
```

Type `exit`, `quit`, or press Ctrl+C to leave. Conversation history persists
across restarts — see [Memory](#memory).

### One command, or one interface

`nero` on its own is the universal session: the terminal chat, and the Telegram
bridge alongside it if you have paired a phone. Both share one conversation, so
a question asked from your phone and one typed here continue each other — turns
are serialized, never interleaved. If Telegram isn't set up, `nero` behaves
exactly as it always has.

| Command | What runs |
| --- | --- |
| `nero` | terminal chat + Telegram |
| `nero chat` | terminal chat alone |
| `nero talk` | voice alone |
| `nero telegram` | the Telegram bridge alone |

A reply arriving from your phone prints into the terminal while `nero` is
running, which can land mid-prompt. Use `nero chat` if you would rather the
terminal stayed yours alone.

## Commands

| Command | What it does |
| --- | --- |
| `nero` | Everything at once: terminal chat, plus Telegram if paired (first run triggers setup) |
| `nero chat` | Terminal chat only |
| `nero config` | Interactive menu: assistant name, model, API key |
| `nero config set <key> <value>` | Scriptable edit, e.g. `nero config set llm.provider ollama` |
| `nero config show` | Print current config (API key masked) |
| `nero detect` | Re-run hardware detection, refresh the local-model recommendation |
| `nero history` | Log of recent skill invocations (the audit trail) |
| `nero mcp` | List configured MCP servers and the tools they expose |
| `nero facts` | Show / forget the structured facts remembered about you |
| `nero notes index` \| `nero notes search <q>` | Build and query the local FTS5 index over your notes |
| `nero routine list` \| `run` \| `install` \| `uninstall` | Manage scheduled routines (launchd) |
| `nero approvals` | Review destructive actions a routine queued for you |
| `nero dashboard` | Read-only localhost viewer: history, skill audit, config |
| `nero telegram setup` \| `approve <code>` \| `pending` | Connect a Telegram bot and pair a phone |
| `nero telegram` | Answer Telegram messages from paired chats |
| `nero telegram install` \| `uninstall` | Run the bridge in the background, from login |
| `nero config set-key <provider> --slot N` | Store an extra API key for rotation |
| `nero --debug` | Chat with verbose stderr logging (tool-call plumbing, per-turn history) |
| `nero --version` | Print the installed version |

In chat, `/image <path> [question]` sends a photo or screenshot to a
vision-capable model, and `/code <request>` routes one turn to
`llm.coding_model`.

Valid config keys include `assistant.name`, `llm.provider`, `llm.model`,
`llm.fallback_chain`, `llm.route_by`, `llm.coding_model`,
`skills.enabled.*`, `security.command_denylist`, `memory.notes_dir`, and the
`mcp.servers` / `routines.routines` blocks above. `nero config show` prints
the current values. The `hardware.*` block is
auto-populated by detection. The RAM → local-model table lives in
`nero/hardware/tiers.py` — edit it as models improve.

## Memory

Every turn re-sends what Nero is given, and a provider spends most of its
time-to-first-token *prefilling* that prompt. Sending the whole conversation
means it grows forever and gets slower forever. Sending a fixed window of the
last N messages is constant, but it pays full price for N messages that are
usually irrelevant — and still forgets anything older.

Nero does neither. Each turn it sends:

| | what it is | cost |
| --- | --- | --- |
| system prompt + facts | stable across the session, so it caches | fixed |
| recalled exchanges | top-4, retrieved *for this message* | ~300 tokens |
| the last 16 messages | recency, so "it" and "that one" resolve | ~500 tokens |

Everything else is retrieved, not carried. The same 120-turn conversation,
asking a question whose answer is 100 turns back (llama3.2 on an M3):

| policy | messages sent | ~tokens | time to first token |
| --- | --- | --- | --- |
| whole conversation | 242 | 7,285 | 141.7 s |
| fixed 60-message window | 32 | 936 | 12.9 s |
| **window + retrieval** | **14** | **588** | **6.8 s** |

All three answer the question. The last one does it with 12× fewer tokens.

### How retrieval works

Two rankings, fused with [Reciprocal Rank Fusion](https://mem0.ai/blog/ai-memory-benchmarks-in-2026):

- **keyword** — SQLite FTS5 with Porter stemming, so "running" finds "run".
  About 86% recall on its own.
- **semantic** — a local embedding model (`nomic-embed-text` via ollama).
  Optional; fusing it in is worth about +9 points, the largest single gain
  measured on LongMemEval.

RRF reads only ranks, never scores, so a bm25 score and a cosine similarity
never have to be made comparable. With no embedder the semantic list is empty
and recall degrades cleanly to keyword-only.

Both streams have a floor, because "nothing is relevant" is the common answer
and neither ranker gives it for free — vector search always has a nearest
neighbour, and bm25 ranks hits against each other with no absolute cutoff.
Semantic hits must clear 0.55 cosine (measured: relevant queries score
0.60–0.81, irrelevant ones 0.40–0.47); keyword-only hits must share more than
one word with the query, or *"write a python function"* recalls an old turn
purely because *"functionality"* stems to *"function"*. Asked seven questions
against a real transcript — three that should recall, four that shouldn't —
this gets all seven right.

The difference is not academic. Against a real transcript, asking *"what
colour do I like"* when the stored turn says *"favorite color"*:

```
keyword only        (nothing recalled)
keyword + semantic  you: What is my favorite color?
                    Nero: Your favorite color is blue!
```

Lookup costs **31 ms** over 600 indexed turns.

### Recalled memory must not sound recalled

Retrieved context rides on the user's message, so how it is framed decides
whether the model *talks about it*. Rendered as a headed transcript, models
answer it as a document the user pasted — *"I see you're sharing a previous
conversation snippet!"*. Two things fix that, both measured against a 3B local
model because that is the hardest case:

- the block is **tagged** `<memory>`, not headed, and the system prompt says
  what the tag is — your own recollection, not something they just sent you
- speakers are labelled `user:` / `assistant:`. The friendlier `they said:` /
  `you said:` made the model lose track of whose preference it was reading
  (*"blue is my favourite colour"*) — 2 slips in 6 against 0.

The system prompt also forbids narrating provenance *by name* — `"in my
notes"`, `"you shared"`, `"memory system"`, `"no facts found"` — because an
abstract rule was not enough for a small model. Talk about what you know,
never about how you know it.

Ten questions against a real transcript: 8/10 clean on llama3.2 (3B), 5/5 on
glm-4.5-flash. The residual is model size, not framing — a 3B model also
contradicts memory it was handed.

### Turning it on

Semantic recall needs a running ollama with `nomic-embed-text` (274 MB), plus
numpy — it is probed once per session and silently stays off otherwise.

```sh
ollama pull nomic-embed-text
nero config set memory.semantic_recall false   # to opt out
```

If your *chat* model also runs on ollama and the machine is short on VRAM, the
two models will evict each other; the probe times out after 5 s and Nero falls
back to keyword-only rather than making you wait.

Nothing is ever deleted. Every exchange stays in `history.db`; the window only
governs what gets re-sent. `memory.compact_after_messages` (default 16) is the
window, `memory.max_history_turns` (default 8) is how much of the last session
is restored at startup, and `0` for the window sends the whole transcript
again. Durable knowledge ("I live in Mumbai") belongs in `nero facts`, which
goes into the system prompt every turn.

## Knowing what day it is

A model's only sense of "now" is its training cutoff, so asked for the date it
answers confidently and months out of date. Nero puts the local date, weekday
and clock into the system prompt on **every request** — not once at startup,
because the Telegram bridge and a launchd-installed session can run for days.

```
Right now it is Sunday, 6 September 2026, 18:42 IST.
```

Truncated to the minute on purpose: the string then stays identical across a
quick back-and-forth, so the prefix a provider (or ollama's KV cache) has
already processed still matches. Measured on llama3.2 — minute precision costs
nothing, an ISO timestamp with microseconds costs nearly double:

| timestamp in the system prompt | time to first token |
| --- | --- |
| none | 0.20 s |
| **to the minute** | **0.20 s** |
| to the microsecond | 0.37 s |

## Skills, and what asks permission

Skills reach the model through one registry, so every call is audited and
individually switchable. They sit in three tiers:

- **read-only** (`get_weather`, `read_file`, `fetch_web_page`, `search_notes`,
  `recall_facts`) — run silently.
- **state-changing** (`open_app`, `open_website`, `play_music`,
  `remember_fact`) — run silently.
- **destructive** (`write_file`, `edit_file`, `delete_path`, `move_path`,
  `run_shell`, `git_command`, `run_python`, `run_javascript`, `forget_fact`)
  — **ship disabled**, and once enabled each call shows you the exact
  arguments and waits for a yes. A command matching `security.command_denylist`
  makes you type `yes` in full. With no way to ask (a pipe, a scheduled
  routine) a destructive call is refused, never assumed.

Anything fetched from the web or read off disk comes back wrapped as
untrusted data, and marks the turn — the confirmation prompt then says so,
because an injected instruction could be behind the call that follows.

Enable one with `nero config set skills.enabled.run_shell true`.

## Telegram

Talk to Nero from your phone. It uses long polling, so nothing is hosted and no
port is opened — Nero connects out to Telegram, and your machine stays where it
is. Built on `httpx`, which was already a dependency; there is no bot framework.

**1. Create the bot.** Message [@BotFather](https://t.me/BotFather), send
`/newbot`, copy the token it gives you.

**2. Store the token.** Either route — both run the same routine, so they
cannot drift into asking for different things:

```sh
nero config           # pick the "Telegram" row
nero telegram setup   # or straight from the command line
```

**3. Pair and verify.** Open your bot's link, press **Start**. It replies with
a six-digit pairing code. Type that code where Nero is running:

```sh
nero telegram approve 483920
```

`setup` prompts for it right there; if the bridge is already running, use the
command. `nero telegram pending` lists chats waiting to be approved.

**4. Run it.** Messages are only answered while something is serving them —
either `nero` (which answers Telegram alongside the terminal chat) or the
bridge on its own:

```sh
nero                  # terminal chat + Telegram
nero telegram         # the bridge alone, until you close it (Ctrl+C)
```

That stops when you close the terminal. To have it always on — starting at
login and restarting if it ever dies:

```sh
nero telegram install     # a launchd agent; running now, and after every reboot
nero telegram uninstall   # stop it again
```

Logs land in `~/Library/Logs/nero/telegram.err.log`. Approving a new chat takes
effect on the running bridge within a few seconds — no restart.

A turn from your phone is the *same* turn as a turn in the terminal — both go
through `ChatLoop.ask`, so the fallback chain, key rotation, memory recall,
skills and every error message behave identically.

### Why a code, and not just "the first chat wins"

A bot token is a URL anyone can message, and Nero can open apps and read files
on your machine. So an unpaired chat gets exactly one thing back: a code, which
grants nothing on its own.

The code is shown **only in Telegram** — never printed at the terminal. That is
the whole point of the two channels: typing it into the terminal proves that
whoever is approving is also holding the phone. Pairing depends on possession,
not on being the first to find the bot.

- an unpaired chat's message is **never run as a turn**, only answered with a code
- codes expire after 10 minutes, are single-use, and one chat holds at most one
  — asking again invalidates the last, so a stranger cannot farm guesses
- the waiting queue is capped, so spam cannot grow it without bound
- codes are compared with `secrets.compare_digest`, so a wrong guess leaks no
  prefix through timing
- destructive skills stay refused: the registry fails closed with no confirm
  callback (the rule voice mode already follows), and there is no safe way to
  approve `rm -rf` from a phone keyboard

Pair another device by messaging the bot from it and approving its code.
Approved chats live in `telegram.allowed_chat_ids`, which stays the single
source of truth:

```sh
nero config set telegram.allowed_chat_ids 12345678,87654321
```

## MCP servers

Nero Agent is an MCP client: point it at third-party servers over stdio and
their tools join the same registry, with the same audit trail and gates.

```yaml
mcp:
  servers:
    github:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"   # read from your shell, never stored here
      trusted: false        # every call asks first
    notes:
      command: npx
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/notes"]
      trusted: true         # local and trusted: no per-call prompt
      requires_network: false
```

`nero mcp` verifies a server starts and lists its tools. A server that fails
to start warns and is skipped — it never blocks the assistant.

## Fallbacks and routing

`llm.fallback_chain` (`["openai/gpt-5", "ollama/llama3.2"]`) is tried in
order when the primary model fails with a connection, timeout, rate-limit, or
5xx error — never on an auth error or an unknown model, where retrying
elsewhere would just hide the real problem. Set `llm.route_by` to `cost`,
`latency`, or `quality` to order that chain; `off` (the default) keeps your
configured order. Entries that fail twice in a session are skipped, unless
one is all that's left.

## Scheduled routines

```bash
nero config set routines.routines.morning.schedule "30 8 * * *"
nero config set routines.routines.morning.prompt "Summarise today's weather"
nero routine install morning
```

This writes a launchd agent that runs `nero routine run morning`. Routines
run unattended, so a destructive skill call from one is never executed on the
spot — it lands in a queue, and your next interactive session tells you it's
waiting for `nero approvals`.

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
