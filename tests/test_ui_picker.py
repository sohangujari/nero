"""The non-TTY fallback is a correctness requirement, not polish: every config
menu test drives stdin through a pipe, and so does anyone scripting the menu."""

import pytest
from rich.console import Console

from nero import ui


class FakeStream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeSys:
    """Stands in for the `sys` module inside nero.ui.

    Patching `sys.stdout` directly does not survive here: pytest reinstalls its
    capture stdout between fixture setup and the test body, silently undoing a
    fixture's patch (an in-body patch would survive). Patching the module's own
    `sys` reference is untouched by capture and still exercises the real
    `_interactive()` logic.
    """

    def __init__(self, stdin_tty: bool, stdout_tty: bool):
        self.stdin = FakeStream(stdin_tty)
        self.stdout = FakeStream(stdout_tty)


@pytest.fixture
def piped(monkeypatch):
    monkeypatch.setattr(ui, "sys", FakeSys(False, False))


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr(ui, "sys", FakeSys(True, True))


CHOICES = [("claude", "Claude (Anthropic)"), ("mistral", "Mistral"), ("groq", "Groq")]


class TestNonTTYFallback:
    def test_uses_the_numbered_prompt_and_never_the_arrow_picker(self, piped, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("the arrow picker must not run without a TTY")

        monkeypatch.setattr(ui, "_pick_arrows", explode)
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "2")
        assert ui.pick("Provider", CHOICES, default="claude") == "mistral"

    def test_blank_answer_means_no_change(self, piped, monkeypatch):
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "")
        assert ui.pick("Provider", CHOICES, default="claude") is None

    def test_out_of_range_answer_means_no_change(self, piped, monkeypatch):
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "99")
        assert ui.pick("Provider", CHOICES, default="claude") is None

    def test_non_numeric_answer_means_no_change(self, piped, monkeypatch):
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "mistral")
        assert ui.pick("Provider", CHOICES, default="claude") is None

    def test_labels_and_current_marker_are_shown(self, piped, monkeypatch):
        console = Console(record=True, width=80)
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "")
        ui.pick("Provider", CHOICES, default="mistral", console=console)
        output = console.export_text()
        assert "Claude (Anthropic)" in output
        assert "(current)" in output


class TestTTYPath:
    def test_uses_the_arrow_picker_when_attached_to_a_terminal(self, terminal, monkeypatch):
        seen = {}

        def fake(title, choices, default):
            seen["title"] = title
            seen["default"] = default
            return "groq"

        monkeypatch.setattr(ui, "_pick_arrows", fake)
        assert ui.pick("Provider", CHOICES, default="claude") == "groq"
        assert seen == {"title": "Provider", "default": "claude"}

    def test_escape_returns_none(self, terminal, monkeypatch):
        monkeypatch.setattr(ui, "_pick_arrows", lambda *a, **k: None)
        assert ui.pick("Provider", CHOICES, default="claude") is None


class TestArrowPickerFailure:
    def test_terminal_failure_degrades_to_the_numbered_prompt(self, terminal, monkeypatch):
        """A prompt_toolkit failure must cost the user a nicer picker, never the task."""
        def boom(*args, **kwargs):
            raise RuntimeError("no console screen buffer")

        monkeypatch.setattr(ui, "_pick_arrows", boom)
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "2")
        assert ui.pick("Provider", CHOICES, default="claude") == "mistral"


class TestEmptyChoices:
    def test_no_choices_means_no_change(self, piped):
        assert ui.pick("Provider", [], default=None) is None


class TestInteractiveDetection:
    def test_both_streams_must_be_a_tty(self, monkeypatch):
        monkeypatch.setattr(ui, "sys", FakeSys(True, True))
        assert ui._interactive() is True
        monkeypatch.setattr(ui, "sys", FakeSys(True, False))
        assert ui._interactive() is False
        monkeypatch.setattr(ui, "sys", FakeSys(False, True))
        assert ui._interactive() is False
