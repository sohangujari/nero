from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import traceback
from collections.abc import Callable

import httpx
import litellm
from rich.markup import escape

from nero.llm.ollama_adapter import OllamaModelError
from nero.voice.audio_io import RECORD_SAMPLE_RATE, listen_for_barge_in
from nero.voice.errors import (
    BargeIn,
    MicPermissionError,
    MicUnavailableError,
    PlaybackError,
    TTSLoadError,
)
from nero.voice.sentence_buffer import SentenceBuffer
from nero.spinner import Spinner

logger = logging.getLogger("nero.voice")

EXIT_WORDS = {"stop", "exit", "quit", "goodbye"}


def _debug_dump_interrupt(where: str) -> None:
    """Under --debug only: show where a Ctrl+C landed, for diagnosing stalls.

    Silent unless `nero --debug` (or `nero talk --debug`) raised the `nero`
    logger to DEBUG, so an ordinary Ctrl+C exit stays clean.

    `print_exc()` points at the frame that was blocked when the interrupt
    arrived; the thread dump is what catches a stall inside the background
    synthesis/playback thread, which the main thread's traceback never shows.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    print(f"\n--- debug: {where} ---", file=sys.stderr)
    traceback.print_exc()
    print("--- all thread stacks ---", file=sys.stderr)
    frames = sys._current_frames()
    for thread in threading.enumerate():
        frame = frames.get(thread.ident)
        print(f"\n[thread {thread.name} alive={thread.is_alive()}]", file=sys.stderr)
        if frame is not None:
            traceback.print_stack(frame, file=sys.stderr)
    print("--- end thread stacks ---", file=sys.stderr)


class TurnTimer:
    """Per-turn latency trace, emitted as one line under `--debug`.

    Replaces a scatter of ad-hoc STAGE log lines left over from a hang hunt.
    The individual numbers say very little on their own — what diagnoses a slow
    turn is seeing them beside each other, so they are collected and printed
    once, at the end of the turn:

        voice turn: stt=0.67 ttft=0.42 speech=0.91 done=5.18 (4 sentences)

    `stt` is transcription, `ttft` the model's first token, `speech` the first
    sentence reaching the TTS queue — the moment the user stops waiting in
    silence — and `done` the end of playback. All offsets are seconds from the
    end of recording.
    """

    def __init__(self):
        self._start = time.monotonic()
        self._marks: dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record the first time `name` happened. Later calls are ignored."""
        self._marks.setdefault(name, time.monotonic() - self._start)

    def log(self, sentences: int) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        self.mark("done")
        trace = " ".join(f"{name}={at:.2f}" for name, at in self._marks.items())
        logger.debug("voice turn: %s (%d sentences)", trace, sentences)


class VoiceLoop:
    """Voice REPL: record → transcribe → chat (reusing LLMClient + tools) → speak."""

    def __init__(
        self,
        client,
        stt,
        record: Callable[[], object],
        make_player: Callable[[], object],
        console,
        assistant_name: str,
        *,
        sample_rate: int = RECORD_SAMPLE_RATE,
        input_fn: Callable[[str], str] | None = None,
        once: bool = False,
        history=None,
        vad=None,
        barge_in: bool = False,
        source=None,
    ):
        self.client = client
        self.stt = stt
        self.record = record
        self.make_player = make_player
        self.console = console
        self.assistant_name = assistant_name
        self.sample_rate = sample_rate
        self.input_fn = input_fn or console.input
        self.once = once
        self.history = history
        self.messages: list[dict] = history.recent() if history else []
        self.vad = vad
        self.barge_in = barge_in
        # The session's shared microphone, so the barge-in monitor reads the
        # same open stream the recorder does instead of grabbing its own.
        self.source = source
        self._pending_prefix = None
        self._barge_in_broken = False
        self._hinted = False

    def run(self) -> None:
        self.console.print(
            f"[bold]{self.assistant_name}[/bold] is listening. "
            "Say [dim]stop[/dim] or press Ctrl+C to leave.\n"
        )
        while True:
            if self._pending_prefix is None:
                try:
                    self.input_fn("🎙️  Press Enter to start speaking... ")
                except (KeyboardInterrupt, EOFError) as exc:
                    if isinstance(exc, KeyboardInterrupt):
                        _debug_dump_interrupt("interrupted at the prompt")
                    self._goodbye()
                    return
            try:
                prefix, self._pending_prefix = self._pending_prefix, None
                audio = self.record(prefix=prefix)
            except MicPermissionError as exc:
                self.console.print(
                    f"\n[red]Microphone access was denied by the OS.[/red] {exc}\n"
                    "Grant Nero Agent microphone permission in your system settings, then retry."
                )
                return
            except MicUnavailableError as exc:
                self.console.print(
                    f"\n[red]No usable microphone was found.[/red] {exc}\n"
                    "Check that an input device is connected and accessible."
                )
                return

            timer = TurnTimer()
            transcript = asyncio.run(self.stt.transcribe(audio, self.sample_rate)).strip()
            timer.mark("stt")
            if not transcript:
                if prefix is not None and audio.size == 0:
                    # The barge-in handoff came back empty: nothing followed
                    # the trigger, so it was almost certainly Nero hearing
                    # its own voice, not the user. Not the "I didn't catch
                    # that" case -- the user never said anything to catch.
                    self.console.print(
                        "[dim]That was probably me hearing myself — ignoring it.[/dim]"
                    )
                else:
                    self.console.print("[yellow]I didn't catch that — let's try again.[/yellow]")
                continue
            self.console.print(f"[dim]🗣  heard:[/dim] {transcript}")

            if transcript.lower().strip(" .!?") in EXIT_WORDS:
                self._goodbye()
                return

            if not self._handle_turn(transcript, timer):
                return
            if self.once:
                return

    def _handle_turn(self, transcript: str, timer: TurnTimer | None = None) -> bool:
        """Run one chat turn; speak the reply. Returns False if the loop should end."""
        timer = timer or TurnTimer()
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": transcript})
        buffer = SentenceBuffer()
        player = self.make_player()
        player.start()

        barge_event = threading.Event()
        stop_monitor = threading.Event()
        # Initialised before the try so the finally block can always reference
        # it safely, even if the monitor setup below raises.
        monitor = None
        prefix_holder: list = []

        spoken_count = [0]

        def tap(chunk: str) -> None:
            if barge_event.is_set():
                raise BargeIn
            spinner.stop()
            timer.mark("ttft")
            print(chunk, end="", flush=True, file=self.console.file)
            for sentence in buffer.feed(chunk):
                timer.mark("speech")
                spoken_count[0] += 1
                player.enqueue(sentence)

        self.console.print(f"[bold magenta]{self.assistant_name}>[/bold magenta] ", end="")
        # A slow provider is worse here than in text chat: nothing prints AND
        # nothing speaks, so the turn is indistinguishable from a hung one.
        spinner = Spinner(self.console.file)
        spinner.start()
        try:

            def on_barge_in(prefix):
                prefix_holder.append(prefix)
                barge_event.set()
                player.stop_now()

            def on_monitor_error(_exc):
                # Announced immediately, on its own line: a notice that arrives
                # after the reply is useless, because the interrupt window it
                # warns about has already closed.
                if not self._barge_in_broken:
                    self._barge_in_broken = True
                    self.console.print(
                        "\n[dim]Barge-in stopped working this session "
                        "(microphone unavailable). Press Ctrl+C to interrupt.[/dim]"
                    )

            if self.barge_in and self.vad is not None and not self._barge_in_broken:
                monitor = listen_for_barge_in(
                    self.vad,
                    on_barge_in,
                    stop_monitor,
                    on_error=on_monitor_error,
                    source=self.source,
                )

            self.client.send(self.messages, on_text=tap)
            tail = buffer.flush()
            if tail:
                timer.mark("speech")
                spoken_count[0] += 1
                player.enqueue(tail)
            player.close()
            player.join()
            timer.log(spoken_count[0])
            if barge_event.is_set():
                raise BargeIn
            self.console.print()
            logger.debug("history after voice turn: %r", self.messages)
            # MAX_TOOL_ROUNDS exhaustion leaves a tool message last instead of
            # final assistant text; the role check excludes that case.
            if self.history is not None and self.messages[-1].get("role") == "assistant":
                self.history.append_turn(
                    self.messages[turn_start]["content"],
                    self.messages[-1]["content"],
                )
            return True
        except BargeIn:
            spoken = player.spoken_text()
            if prefix_holder:
                self._pending_prefix = prefix_holder[-1]
            # Feedback runs on both branches: the empty case is the most likely
            # barge-in of all (Nero hearing its own first sentence), and with no
            # output here the user would see the prompt cut off mid-word with no
            # sign anything happened and no hint that barge-in is a setting.
            self.console.print()
            self._hint_once()
            if not spoken:
                # Nothing reached the speaker: drop the whole turn, exactly like
                # the Ctrl+C path. Persisting an assistant message with no
                # content would leave a malformed exchange in context.
                del self.messages[turn_start:]
                # Deliberately asymmetric with the non-empty branch below: this
                # path has no truncated reply text on screen to make the
                # interruption self-evident, and `_hint_once()` only speaks
                # once per session -- on the second and later self-triggers it
                # says nothing at all. "Interrupted." is what has to carry
                # that signal every time, so it's printed here verbatim,
                # matching the KeyboardInterrupt branch (spec D3). Do not add
                # it to the non-empty branch below -- the visible truncated
                # text already makes that case obvious, and it would be noise.
                self.console.print("\n[dim]Interrupted.[/dim]")
                return True
            reply = f"{spoken} [interrupted]"
            del self.messages[turn_start + 1 :]
            self.messages.append({"role": "assistant", "content": reply})
            if self.history is not None:
                self.history.append_turn(self.messages[turn_start]["content"], reply)
            return True
        except KeyboardInterrupt:
            _debug_dump_interrupt("interrupted during turn")
            del self.messages[turn_start:]
            self.console.print("\n[dim]Interrupted.[/dim]")
            return True
        except litellm.exceptions.AuthenticationError:
            del self.messages[turn_start:]
            self.console.print(
                "\n[red]Your API key was rejected.[/red] Update it with [bold]nero config[/bold]."
            )
            return True
        except OllamaModelError as exc:
            # Ordered before the connection branch on purpose: the server
            # answered, so "is Ollama running?" is the wrong question.
            del self.messages[turn_start:]
            self.console.print(f"\n[red]{escape(str(exc))}[/red]")
            return True
        except litellm.exceptions.RateLimitError:
            # A free-tier quota (Gemini allows 5 requests/minute) arrives
            # mid-stream, where LiteLLM wraps it in a ServiceUnavailableError
            # subclass -- so before nero.llm.client unwrapped it this printed
            # "could not reach the model provider", which sent users to debug a
            # working network. The provider answered; it just said no.
            del self.messages[turn_start:]
            self.console.print(
                "\n[yellow]The provider is rate-limiting you.[/yellow] Wait a "
                "moment and try again, or switch models with [bold]nero config[/bold]."
            )
            return True
        except (
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.Timeout,
            httpx.HTTPError,
        ):
            logger.debug("provider unreachable", exc_info=True)
            del self.messages[turn_start:]
            hint = (
                " If you're using Ollama, make sure it's running ([bold]ollama serve[/bold])."
                if getattr(self.client, "provider", None) == "ollama"
                else " Check your connection and try again."
            )
            self.console.print(f"\n[red]Could not reach the model provider.[/red]{hint}")
            return True
        except (TTSLoadError, PlaybackError) as exc:
            del self.messages[turn_start:]
            self.console.print(
                f"\n[red]Voice output failed:[/red] {exc}\n"
                "Re-run setup to (re)download the voice model, then try [bold]nero talk[/bold] again."
            )
            return False
        except Exception as exc:  # noqa: BLE001 — the loop must never crash on a turn
            logger.debug("voice turn failed", exc_info=True)
            del self.messages[turn_start:]
            self.console.print(f"\n[red]Something went wrong with that turn:[/red] {exc}")
            return True
        finally:
            spinner.stop()
            stop_monitor.set()
            if monitor is not None:
                monitor.join(timeout=2)
                if monitor.is_alive():
                    logger.debug("barge-in monitor did not stop within 2.0s")
            # Safety net: on any abnormal exit (Ctrl+C, auth/network error) the
            # success path's close()/join() never ran, which would leave the
            # playback thread parked on the queue and hang interpreter shutdown.
            player.shutdown()

    def _hint_once(self) -> None:
        """Explain barge-in the first time it fires, then never again.

        On laptop speakers Nero can hear itself; without this the behavior reads
        as a bug rather than a toggle. Repeating it every turn would be noise.
        """
        if self._hinted:
            return
        self._hinted = True
        self.console.print(
            "[dim]Interrupted by voice. On speakers? Disable with "
            "nero config set voice.barge_in false[/dim]"
        )

    def _goodbye(self) -> None:
        self.console.print(f"\n[dim]{self.assistant_name} signing off. Bye![/dim]")
