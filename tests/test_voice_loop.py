import numpy as np
from rich.console import Console

import nero.voice.voice_loop as voice_loop_module
from nero.voice.errors import MicPermissionError, TTSLoadError
from nero.voice.voice_loop import VoiceLoop


class FakeSTT:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts)

    async def transcribe(self, audio, sample_rate):
        return self._transcripts.pop(0)


class FakePlayer:
    instances = []

    def __init__(self):
        self.sentences = []
        self.started = self.closed = self.joined = False
        self.shutdown_called = False
        self.stop_now_calls = 0
        self._spoken = None
        FakePlayer.instances.append(self)

    def start(self):
        self.started = True

    def enqueue(self, sentence):
        self.sentences.append(sentence)

    def close(self):
        self.closed = True

    def join(self, timeout=None):
        self.joined = True

    def shutdown(self, timeout=5.0):
        # Always called from _handle_turn's finally, so an aborted turn can't
        # leave the playback thread parked.
        self.shutdown_called = True

    def stop_now(self):
        self.stop_now_calls += 1

    def spoken_text(self):
        # Barge-in tests set _spoken explicitly; otherwise derive it from
        # whatever was actually enqueued, matching the real Player's semantics.
        if self._spoken is not None:
            return self._spoken
        return " ".join(self.sentences)


class FakeClient:
    """Streams scripted chunks to on_text; records the messages it saw."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.provider = "claude"
        self.seen_messages = None

    def send(self, messages, on_text):
        self.seen_messages = list(messages)
        for chunk in self._chunks:
            on_text(chunk)


def make_loop(transcripts, chunks, once=False, record=None):
    FakePlayer.instances = []
    inputs = iter([""] * 20)
    return VoiceLoop(
        client=FakeClient(chunks),
        stt=FakeSTT(transcripts),
        record=record or (lambda prefix=None: object()),
        make_player=FakePlayer,
        console=Console(),
        assistant_name="Nero",
        input_fn=lambda *_a: next(inputs),
        once=once,
    )


def test_single_turn_transcribes_sends_and_speaks():
    loop = make_loop(["Hello there."], ["Hi ", "friend. ", "Bye now."], once=True)
    loop.run()
    assert loop.messages[0] == {"role": "user", "content": "Hello there."}
    player = FakePlayer.instances[0]
    assert player.started and player.closed and player.joined
    assert player.sentences == ["Hi friend.", "Bye now."]


def test_empty_transcript_reprompts_without_sending():
    loop = make_loop(["", "Open calculator."], ["Opening."], once=True)
    loop.run()
    assert loop.messages[0]["content"] == "Open calculator."


def test_ordinary_empty_transcript_prints_i_didnt_catch_that(capsys):
    """No prefix, no speech: the ordinary case keeps its existing message."""
    loop = make_loop(["", "Open calculator."], ["Opening."], once=True)
    loop.run()
    out = capsys.readouterr().out
    assert "I didn't catch that" in out
    assert "hearing myself" not in out.lower()


def test_prefix_empty_audio_prints_self_voice_message_not_i_didnt_catch(capsys):
    """A barge-in handoff (prefix supplied) that comes back empty means Nero
    heard itself, not the user -- a distinct message must explain that
    instead of the generic (and misleading) "I didn't catch that"."""

    def record(prefix=None):
        return np.zeros(0, dtype=np.float32)

    prompts = []
    loop = make_loop(["", "stop"], ["should not stream"], record=record)
    real_input_fn = loop.input_fn
    loop.input_fn = lambda *a: (prompts.append(a), real_input_fn(*a))[-1]
    loop._pending_prefix = np.full(4, 0.5, dtype=np.float32)

    loop.run()

    out = capsys.readouterr().out
    assert "I didn't catch that" not in out
    assert "hearing myself" in out.lower() or "self" in out.lower()
    # `_pending_prefix` must be cleared before the *next* prompt: the second
    # turn ("stop") should see the normal "Press Enter" prompt again, proving
    # the prefix consumed on turn one didn't leak into turn two.
    assert len(prompts) == 1
    assert "Press Enter" in prompts[0][0]


def test_stop_word_exits_without_sending():
    loop = make_loop(["stop"], ["should not stream"])
    loop.run()
    assert loop.messages == []


def test_exit_word_with_punctuation_exits():
    loop = make_loop(["Exit."], ["nope"])
    loop.run()
    assert loop.messages == []


def test_mic_permission_error_is_friendly(capsys):
    def boom(prefix=None):
        raise MicPermissionError("denied")

    loop = make_loop(["unused"], ["unused"], record=boom)
    loop.run()  # must not raise
    out = capsys.readouterr().out.lower()
    assert "permission" in out or "microphone" in out


def test_tts_load_error_ends_loop_gracefully(capsys):
    class BadPlayer(FakePlayer):
        def join(self):
            raise TTSLoadError("model missing")

    loop = make_loop(["Say hi."], ["Hi."], once=True)
    loop.make_player = BadPlayer
    loop.run()  # must not raise
    assert "setup" in capsys.readouterr().out.lower()


# --- Ctrl+C exit stays clean unless --debug is on ---
import logging


def _ctrl_c_at_prompt(loop):
    loop.input_fn = lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt)
    return loop


def test_ctrl_c_exit_is_silent_by_default(capsys):
    """A normal Ctrl+C prints only the sign-off — no traceback, no thread dump."""
    logger = logging.getLogger("nero.voice")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        _ctrl_c_at_prompt(make_loop(["unused"], ["unused"])).run()
        out = capsys.readouterr()
        combined = out.out + out.err
        assert "signing off" in out.out
        assert "Traceback" not in combined
        assert "thread stacks" not in combined
        assert "debug:" not in combined
    finally:
        logger.setLevel(previous)


def test_ctrl_c_dumps_diagnostics_under_debug(capsys):
    """With --debug (nero logger at DEBUG) the interrupt diagnostics come back."""
    logger = logging.getLogger("nero.voice")
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        loop = make_loop(["unused"], ["unused"])
        loop.input_fn = lambda *_a: (_ for _ in ()).throw(KeyboardInterrupt)
        loop.run()
        combined = capsys.readouterr()
        assert "thread stacks" in combined.err
        assert "interrupted at the prompt" in combined.err
    finally:
        logger.setLevel(previous)


def test_eof_exit_never_dumps_even_under_debug(capsys):
    """EOF is a normal exit, not a stall — no diagnostics regardless of --debug."""
    logger = logging.getLogger("nero.voice")
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        loop = make_loop(["unused"], ["unused"])
        loop.input_fn = lambda *_a: (_ for _ in ()).throw(EOFError)
        loop.run()
        combined = capsys.readouterr()
        assert "thread stacks" not in combined.err
        assert "signing off" in combined.out
    finally:
        logger.setLevel(previous)


def test_aborted_turn_still_shuts_player_down():
    """Ctrl+C mid-turn must not leave the playback thread parked on the queue."""

    class InterruptingClient:
        provider = "claude"

        def send(self, messages, on_text):
            on_text("Partial. ")
            raise KeyboardInterrupt

    FakePlayer.instances = []
    inputs = iter([""] * 5)
    loop = VoiceLoop(
        client=InterruptingClient(), stt=FakeSTT(["hello"]),
        record=lambda prefix=None: object(), make_player=FakePlayer,
        console=Console(), assistant_name="Nero",
        input_fn=lambda *_a: next(inputs), once=True,
    )
    loop.run()
    player = FakePlayer.instances[0]
    assert player.shutdown_called, "aborted turn leaked the playback thread"


# --- A reachable Ollama refusing a model is not a dead server ---
import io

import httpx

from nero.llm.ollama_adapter import OllamaModelError


def _run_failing_voice_turn(error):
    """One voice turn whose client raises `error`; returns everything printed."""

    class FailingClient:
        provider = "ollama"

        def send(self, messages, on_text):
            raise error

    buffer = io.StringIO()
    # Wide console so rich doesn't wrap mid-command and break substring checks.
    console = Console(file=buffer, force_terminal=False, width=200)
    inputs = iter([""] * 5)
    VoiceLoop(
        client=FailingClient(), stt=FakeSTT(["hello"]),
        record=lambda prefix=None: object(), make_player=FakePlayer,
        console=console, assistant_name="Nero",
        input_fn=lambda *_a: next(inputs), once=True,
    ).run()
    return buffer.getvalue()


def test_voice_model_error_reports_the_model_not_the_server():
    out = _run_failing_voice_turn(
        OllamaModelError(
            "Model 'gemma3n' isn't available locally. Pull it with `ollama pull gemma3n`."
        )
    )
    assert "gemma3n" in out
    assert "ollama pull gemma3n" in out
    assert "Could not reach the model provider" not in out
    assert "make sure it's running" not in out
    assert "Something went wrong" not in out


def test_voice_connection_failure_still_suggests_starting_ollama():
    out = _run_failing_voice_turn(httpx.ConnectError("connection refused"))
    assert "Could not reach the model provider" in out
    assert "ollama serve" in out


# --- Session memory: seed from history, persist only successful turns ---
class AppendingClient:
    """Like the real client: streams chunks AND appends the final assistant text."""
    provider = "claude"

    def __init__(self, reply):
        self._reply = reply

    def send(self, messages, on_text):
        on_text(self._reply)
        messages.append({"role": "assistant", "content": self._reply})


class FakeHistory:
    def __init__(self, seed=None):
        self._seed = seed or []
        self.appended = []

    def recent(self, limit=None):
        return list(self._seed)

    def append_turn(self, user, assistant):
        self.appended.append((user, assistant))


def _history_loop(transcripts, reply, history):
    return VoiceLoop(
        client=AppendingClient(reply), stt=FakeSTT(transcripts),
        record=lambda prefix=None: object(), make_player=FakePlayer,
        console=Console(), assistant_name="Nero",
        input_fn=lambda *_a: "", once=True, history=history,
    )


def test_voice_seeds_from_history():
    seed = [{"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"}]
    loop = _history_loop(["Say hi."], "Hi.", FakeHistory(seed))
    assert loop.messages == seed  # seeded in __init__, before run()


def test_voice_successful_turn_is_appended():
    hist = FakeHistory()
    _history_loop(["Hello."], "Hi there.", hist).run()
    assert hist.appended == [("Hello.", "Hi there.")]


def test_voice_max_rounds_fallthrough_not_appended():
    """Same MAX_TOOL_ROUNDS-exhaustion gap as ChatLoop: send() can return
    normally leaving a tool message last instead of final assistant text.
    That must never be persisted as the assistant's reply."""

    class MaxRoundsClient:
        provider = "claude"

        def send(self, messages, on_text):
            on_text("")
            messages.append(
                {"role": "tool", "tool_call_id": "x", "content": "raw tool output"}
            )

    hist = FakeHistory()
    loop = VoiceLoop(
        client=MaxRoundsClient(), stt=FakeSTT(["Hello."]),
        record=lambda prefix=None: object(), make_player=FakePlayer,
        console=Console(), assistant_name="Nero",
        input_fn=lambda *_a: "", once=True, history=hist,
    )
    loop.run()
    assert hist.appended == []


# --- Barge-in: only what was heard is ever recorded ---
def make_loop_with_barge_in(monkeypatch, spoken="", generated=None, prefix_value=None, turns=1):
    """A VoiceLoop wired for barge-in where the fake monitor interrupts every
    turn immediately. `spoken` is what FakePlayer.spoken_text() reports back
    (what actually reached the speaker); `generated` (defaults to `spoken`) is
    what the fake client streams, standing in for LLM text that ran ahead of
    playback. No real microphone/thread involved: `listen_for_barge_in` is
    replaced with a fake that fires `on_detect` synchronously.

    Uses the `monkeypatch` fixture (not a plain module assignment) so the fake
    is torn down at test teardown — otherwise it would leak into every test
    that runs afterwards in the same session, including Task 8's CLI tests.
    """
    FakePlayer.instances = []
    generated = spoken if generated is None else generated

    def fake_listen_for_barge_in(vad, on_detect, stop, on_error=None):
        on_detect(prefix_value)

        class _DummyThread:
            def join(self, timeout=None):
                pass

            def is_alive(self):
                return False

        return _DummyThread()

    monkeypatch.setattr(voice_loop_module, "listen_for_barge_in", fake_listen_for_barge_in)

    recorded_prefixes: list = []

    def record(prefix=None):
        recorded_prefixes.append(prefix)
        return object()

    def make_player():
        player = FakePlayer()
        player._spoken = spoken
        loop._last_player = player
        return player

    history = FakeHistory()
    # One real "question" per turn, then "stop" to end the session cleanly.
    transcripts = ["What's the weather?"] * turns + ["stop"]
    inputs = iter([""] * 20)
    loop = VoiceLoop(
        client=FakeClient([generated]),
        stt=FakeSTT(transcripts),
        record=record,
        make_player=make_player,
        console=Console(width=200),
        assistant_name="Nero",
        input_fn=lambda *_a: next(inputs),
        history=history,
        vad=object(),
        barge_in=True,
    )
    loop._recorded_prefixes = recorded_prefixes
    return loop, history


class TestBargeIn:
    def test_spoken_text_plus_marker_is_recorded(self, monkeypatch):
        """What the user heard is what the model is told it said."""
        loop, history = make_loop_with_barge_in(monkeypatch, spoken="It's 14C in Oslo.")
        loop.run()
        assert history.appended
        _user, assistant = history.appended[-1]
        assert assistant == "It's 14C in Oslo. [interrupted]"

    def test_unspoken_sentences_are_not_recorded(self, monkeypatch):
        loop, history = make_loop_with_barge_in(
            monkeypatch,
            spoken="It's 14C in Oslo.",
            generated="It's 14C in Oslo. Rain eases by six.",
        )
        loop.run()
        _user, assistant = history.appended[-1]
        assert "Rain eases by six" not in assistant

    def test_barge_in_before_any_sentence_rolls_back_entirely(self, monkeypatch):
        """Nothing spoken -> no assistant text -> the whole turn is dropped,
        including the user message. A user turn with no reply is malformed."""
        loop, history = make_loop_with_barge_in(monkeypatch, spoken="")
        loop.run()
        assert history.appended == []
        assert loop.messages == []

    def test_barge_in_before_any_sentence_still_prints_the_hint(self, monkeypatch, capsys):
        """Fix 1: this is the empty/rollback path -- the self-trigger case a
        speaker user hits most often. It must not fall silent: the user needs
        to see something happened and learn barge-in is a setting they can
        turn off, exactly like the non-empty branch already does. Per D3,
        "Interrupted." must also appear -- it's what carries the signal on
        the second and later self-triggers, once `_hint_once()` has already
        spoken and gone quiet for the rest of the session."""
        loop, history = make_loop_with_barge_in(monkeypatch, spoken="")
        loop.run()
        out = capsys.readouterr().out
        assert "voice.barge_in" in out
        assert "Interrupted." in out
        # Rollback behavior itself must be unchanged by the fix.
        assert history.appended == []
        assert loop.messages == []

    def test_barge_in_stops_the_player_immediately(self, monkeypatch):
        loop, _history = make_loop_with_barge_in(monkeypatch, spoken="One.")
        loop.run()
        assert loop._last_player.stop_now_calls == 1

    def test_prefix_audio_is_carried_into_the_next_recording(self, monkeypatch):
        loop, _history = make_loop_with_barge_in(monkeypatch, spoken="One.", prefix_value=0.7)
        loop.run()
        assert loop._recorded_prefixes and loop._recorded_prefixes[-1] is not None

    def test_first_barge_in_prints_the_speaker_hint_once(self, monkeypatch, capsys):
        """Spec D4: barge-in ships on, so a speaker user who sees Nero cut
        itself off needs to know it is a setting. Once per session
        only — a hint on every interruption becomes noise."""
        loop, _history = make_loop_with_barge_in(monkeypatch, spoken="One.", turns=2)
        loop.run()
        out = capsys.readouterr().out
        assert out.count("voice.barge_in") == 1

    def test_prompt_is_skipped_when_a_barge_in_prefix_is_pending(self, monkeypatch):
        """Rule 3: barge-in hands off directly into recording. The turn that
        begins with a pending prefix must not show "Press Enter to start
        speaking" again -- that would throw away the words that triggered the
        barge-in."""
        loop, _history = make_loop_with_barge_in(monkeypatch, spoken="One.", prefix_value=0.7)
        prompts = []
        real_input_fn = loop.input_fn
        loop.input_fn = lambda *a: (prompts.append(a), real_input_fn(*a))[-1]
        loop.run()
        # One barge-in turn, then the following "stop" turn: two iterations of
        # the loop, but the prompt is only ever needed once -- the second
        # iteration begins with a pending prefix and must skip it.
        assert len(prompts) == 1
