from __future__ import annotations

import asyncio
import logging
import sys  # DEBUG(hang)
import threading  # DEBUG(hang)
import time  # DEBUG(hang)
import traceback  # DEBUG(hang)
from collections.abc import Callable

import httpx
import litellm
from rich.markup import escape

from nero.llm.ollama_adapter import OllamaModelError
from nero.voice.audio_io import RECORD_SAMPLE_RATE
from nero.voice.errors import (
    MicPermissionError,
    MicUnavailableError,
    PlaybackError,
    TTSLoadError,
)
from nero.voice.sentence_buffer import SentenceBuffer

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

    def run(self) -> None:
        self.console.print(
            f"[bold]{self.assistant_name}[/bold] is listening. "
            "Say [dim]stop[/dim] or press Ctrl+C to leave.\n"
        )
        while True:
            try:
                self.input_fn("🎙️  Press Enter to start speaking... ")
            except (KeyboardInterrupt, EOFError) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    _debug_dump_interrupt("interrupted at the prompt")
                self._goodbye()
                return
            try:
                audio = self.record()
            except MicPermissionError as exc:
                self.console.print(
                    f"\n[red]Microphone access was denied by the OS.[/red] {exc}\n"
                    "Grant Nero microphone permission in your system settings, then retry."
                )
                return
            except MicUnavailableError as exc:
                self.console.print(
                    f"\n[red]No usable microphone was found.[/red] {exc}\n"
                    "Check that an input device is connected and accessible."
                )
                return

            logger.debug("STAGE 1a: recording stopped, starting transcription")  # DEBUG(hang)
            _t_stt = time.monotonic()  # DEBUG(hang)
            transcript = asyncio.run(self.stt.transcribe(audio, self.sample_rate)).strip()
            # DEBUG(hang) STAGE 1: transcription complete
            logger.debug(
                "STAGE 1: transcription done in %.2fs -> %r",
                time.monotonic() - _t_stt,
                transcript,
            )
            if not transcript:
                self.console.print("[yellow]I didn't catch that — let's try again.[/yellow]")
                continue
            self.console.print(f"[dim]🗣  heard:[/dim] {transcript}")

            if transcript.lower().strip(" .!?") in EXIT_WORDS:
                self._goodbye()
                return

            if not self._handle_turn(transcript):
                return
            if self.once:
                return

    def _handle_turn(self, transcript: str) -> bool:
        """Run one chat turn; speak the reply. Returns False if the loop should end."""
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": transcript})
        buffer = SentenceBuffer()
        # DEBUG(hang) STAGE 1b: this used to build the TTS engine (and download
        # ~300 MB) on the first turn — the gap between "heard:" and STAGE 2.
        logger.debug("STAGE 1b: building player (TTS engine should already be loaded)")
        player = self.make_player()
        logger.debug("STAGE 1c: player built, starting playback thread")  # DEBUG(hang)
        player.start()

        _t_turn = time.monotonic()  # DEBUG(hang)
        _seen = {"first_chunk": False, "sentences": 0}  # DEBUG(hang)

        def tap(chunk: str) -> None:
            if not _seen["first_chunk"]:  # DEBUG(hang) STAGE 3: first LLM chunk
                _seen["first_chunk"] = True
                logger.debug(
                    "STAGE 3: first LLM chunk after %.2fs %r",
                    time.monotonic() - _t_turn,
                    chunk,
                )
            print(chunk, end="", flush=True, file=self.console.file)
            for sentence in buffer.feed(chunk):
                # DEBUG(hang) STAGE 4: complete sentence flushed to TTS
                _seen["sentences"] += 1
                logger.debug(
                    "STAGE 4: sentence #%d -> TTS queue at %.2fs %r",
                    _seen["sentences"],
                    time.monotonic() - _t_turn,
                    sentence,
                )
                player.enqueue(sentence)

        self.console.print(f"[bold magenta]{self.assistant_name}>[/bold magenta] ", end="")
        try:
            # DEBUG(hang) STAGE 2: about to hand the transcript to the LLM
            logger.debug(
                "STAGE 2: sending to LLM (provider=%s, %d messages)",
                getattr(self.client, "provider", "?"),
                len(self.messages),
            )
            self.client.send(self.messages, on_text=tap)
            logger.debug(  # DEBUG(hang)
                "STAGE 4b: LLM stream finished at %.2fs (%d sentences queued)",
                time.monotonic() - _t_turn,
                _seen["sentences"],
            )
            tail = buffer.flush()
            if tail:
                logger.debug("STAGE 4c: tail -> TTS queue %r", tail)  # DEBUG(hang)
                player.enqueue(tail)
            player.close()
            logger.debug("STAGE 7a: waiting for playback thread to drain")  # DEBUG(hang)
            player.join()
            logger.debug(  # DEBUG(hang)
                "STAGE 7b: turn complete, playback drained at %.2fs",
                time.monotonic() - _t_turn,
            )
            self.console.print()
            logger.debug("history after voice turn: %r", self.messages)
            if self.history is not None:
                self.history.append_turn(
                    self.messages[turn_start]["content"],
                    self.messages[-1]["content"],
                )
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
        except (
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.ServiceUnavailableError,
            litellm.exceptions.Timeout,
            httpx.HTTPError,
        ):
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
            del self.messages[turn_start:]
            self.console.print(f"\n[red]Something went wrong with that turn:[/red] {exc}")
            return True
        finally:
            # Safety net: on any abnormal exit (Ctrl+C, auth/network error) the
            # success path's close()/join() never ran, which would leave the
            # playback thread parked on the queue and hang interpreter shutdown.
            player.shutdown()

    def _goodbye(self) -> None:
        self.console.print(f"\n[dim]{self.assistant_name} signing off. Bye![/dim]")
