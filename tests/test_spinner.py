import io
import time

from nero.spinner import INTERVAL, Spinner


class FakeTTY(io.StringIO):
    def __init__(self, tty=True):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_no_output_on_a_non_terminal():
    """Piped output and captured test runs must stay byte-identical."""
    out = FakeTTY(tty=False)
    spinner = Spinner(out)
    spinner.start()
    time.sleep(INTERVAL * 3)
    spinner.stop()
    assert out.getvalue() == ""


def test_a_fast_turn_never_draws():
    """The reply's first token usually beats the first frame; nothing should flicker."""
    out = FakeTTY()
    spinner = Spinner(out)
    spinner.start()
    spinner.stop()
    assert out.getvalue() == ""


def test_frames_are_erased_so_the_reply_starts_in_place():
    out = FakeTTY()
    spinner = Spinner(out)
    spinner.start()
    time.sleep(INTERVAL * 3)
    spinner.stop()
    written = out.getvalue()
    assert written, "expected at least one frame after 3 intervals"
    # Every frame backs the cursor over itself and stop() blanks the last one,
    # so the net cursor movement is zero -- one backspace per character written.
    assert written.endswith(" \b")
    assert len(written) - written.count("\b") == written.count("\b")


def test_stop_is_idempotent():
    """_print_chunk calls it on every chunk, not just the first."""
    out = FakeTTY()
    spinner = Spinner(out)
    spinner.start()
    time.sleep(INTERVAL * 2)
    spinner.stop()
    after = out.getvalue()
    spinner.stop()
    spinner.stop()
    assert out.getvalue() == after
