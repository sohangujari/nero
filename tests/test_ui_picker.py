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


MANY = [("open_app", "open_app"), ("get_weather", "get_weather (needs network)"), ("play_music", "play_music")]


class TestPickManyNonTTYFallback:
    def test_never_reaches_the_checkbox_without_a_tty(self, piped, monkeypatch):
        """The discriminating test: delete the fallback and this must fail."""
        def explode(*args, **kwargs):
            raise AssertionError("the checkbox must not run without a TTY")

        monkeypatch.setattr(ui, "_pick_many_arrows", explode)
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "1")
        assert ui.pick_many("Skills", MANY, {"open_app"}) == set()

    def test_a_number_unchecks_a_checked_value(self, piped, monkeypatch):
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "2")
        assert ui.pick_many("Skills", MANY, {"open_app", "get_weather"}) == {"open_app"}

    def test_a_number_checks_an_unchecked_value(self, piped, monkeypatch):
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "3")
        assert ui.pick_many("Skills", MANY, {"open_app"}) == {"open_app", "play_music"}

    def test_blank_answer_means_no_change_not_nothing_selected(self, piped, monkeypatch):
        """None and an empty set are opposite answers: None leaves the config
        alone, an empty set disables every skill."""
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "")
        assert ui.pick_many("Skills", MANY, {"open_app"}) is None

    def test_out_of_range_answer_means_no_change(self, piped, monkeypatch):
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "99")
        assert ui.pick_many("Skills", MANY, {"open_app"}) is None

    def test_checked_markers_and_labels_are_shown(self, piped, monkeypatch):
        console = Console(record=True, width=80)
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "")
        ui.pick_many("Skills", MANY, {"open_app"}, console=console)
        output = console.export_text()
        assert "[x]" in output
        assert "[ ]" in output
        assert "get_weather (needs network)" in output


class TestPickManyTTYPath:
    def test_uses_the_checkbox_when_attached_to_a_terminal(self, terminal, monkeypatch):
        seen = {}

        def fake(title, choices, selected):
            seen["title"] = title
            seen["selected"] = selected
            return {"play_music"}

        monkeypatch.setattr(ui, "_pick_many_arrows", fake)
        assert ui.pick_many("Skills", MANY, {"open_app"}) == {"play_music"}
        assert seen == {"title": "Skills", "selected": {"open_app"}}

    def test_escape_returns_none(self, terminal, monkeypatch):
        monkeypatch.setattr(ui, "_pick_many_arrows", lambda *a, **k: None)
        assert ui.pick_many("Skills", MANY, {"open_app"}) is None

    def test_selecting_nothing_returns_an_empty_set_not_none(self, terminal, monkeypatch):
        monkeypatch.setattr(ui, "_pick_many_arrows", lambda *a, **k: set())
        assert ui.pick_many("Skills", MANY, {"open_app"}) == set()


class TestPickManyFailure:
    def test_terminal_failure_degrades_to_the_numbered_prompt(self, terminal, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("no console screen buffer")

        monkeypatch.setattr(ui, "_pick_many_arrows", boom)
        monkeypatch.setattr(ui.Prompt, "ask", lambda *a, **k: "3")
        assert ui.pick_many("Skills", MANY, {"open_app"}) == {"open_app", "play_music"}


class TestPickManyEmptyChoices:
    def test_no_choices_means_no_change(self, piped):
        assert ui.pick_many("Skills", [], set()) is None
