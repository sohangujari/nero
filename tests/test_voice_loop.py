from rich.console import Console

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
        record=record or (lambda: object()),
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


def test_stop_word_exits_without_sending():
    loop = make_loop(["stop"], ["should not stream"])
    loop.run()
    assert loop.messages == []


def test_exit_word_with_punctuation_exits():
    loop = make_loop(["Exit."], ["nope"])
    loop.run()
    assert loop.messages == []


def test_mic_permission_error_is_friendly(capsys):
    def boom():
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
        record=lambda: object(), make_player=FakePlayer,
        console=Console(), assistant_name="Nero",
        input_fn=lambda *_a: next(inputs), once=True,
    )
    loop.run()
    player = FakePlayer.instances[0]
    assert player.shutdown_called, "aborted turn leaked the playback thread"
