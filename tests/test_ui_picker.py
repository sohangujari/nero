"""The non-TTY fallback is a correctness requirement, not polish: every config
menu test drives stdin through a pipe, and so does anyone scripting the menu."""

import sys

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


# --- Fake `questionary`, for exercising the arrow-picker bodies themselves ---
#
# Every TTY-path test above monkeypatches `_pick_arrows`/`_pick_many_arrows`
# wholesale, so those functions' own bodies never run in the suite. `import
# questionary` inside them consults `sys.modules`, so installing a fake there
# lets the real bodies execute without a terminal. This is the smallest fake
# that satisfies both call sites: a `Choice` that records what it was built
# with, and `select`/`checkbox` calls that record their arguments and hand
# back a canned answer via `.ask()`.


class FakeChoice:
    def __init__(self, title, value, checked=False):
        self.title = title
        self.value = value
        self.checked = checked


class FakeQuestion:
    def __init__(self, answer):
        self._answer = answer

    def ask(self):
        return self._answer


class FakeQuestionary:
    Choice = FakeChoice

    def __init__(self, answer):
        self.answer = answer
        self.select_calls = []
        self.checkbox_calls = []

    def select(self, title, choices, default=None, qmark=""):
        self.select_calls.append(
            {"title": title, "choices": choices, "default": default, "qmark": qmark}
        )
        return FakeQuestion(self.answer)

    def checkbox(self, title, choices, qmark=""):
        self.checkbox_calls.append({"title": title, "choices": choices, "qmark": qmark})
        return FakeQuestion(self.answer)


@pytest.fixture
def fake_questionary(monkeypatch):
    """Installs a `FakeQuestionary` under `sys.modules["questionary"]` and
    returns it so the test can inspect what was called and set the answer
    `.ask()` will hand back."""

    def install(answer):
        fake = FakeQuestionary(answer)
        monkeypatch.setitem(sys.modules, "questionary", fake)
        return fake

    return install


class TestPickArrowsBody:
    """`_pick_arrows` is a near pass-through, so this only pins the two things
    that matter: the chosen value comes back, and Esc/Ctrl-C (`None`) stays
    `None` rather than being coerced into something else."""

    def test_the_chosen_value_comes_back(self, fake_questionary):
        fake_questionary("mistral")
        assert ui._pick_arrows("Provider", CHOICES, "claude") == "mistral"

    def test_none_answer_stays_none(self, fake_questionary):
        fake_questionary(None)
        assert ui._pick_arrows("Provider", CHOICES, "claude") is None


class TestPickManyArrowsBody:
    """Runs the real `_pick_many_arrows` body — the one path the rest of the
    suite cannot reach, since every other test monkeypatches this function
    away. This is where the phase's headline bug would actually surface: an
    implementation written as `set(answer) if answer else None` returns the
    same `None` as Esc/Ctrl-C for "I selected nothing", silently leaving the
    config unchanged instead of disabling every skill — and every other test
    in this file, which stubs this function out, would stay green."""

    def test_empty_answer_list_returns_an_empty_set_not_none(self, fake_questionary):
        fake_questionary([])
        result = ui._pick_many_arrows("Skills", MANY, {"open_app"})
        assert result == set()
        assert result is not None

    def test_none_answer_returns_none(self, fake_questionary):
        fake_questionary(None)
        assert ui._pick_many_arrows("Skills", MANY, {"open_app"}) is None

    def test_non_empty_answer_returns_exactly_that_set(self, fake_questionary):
        fake_questionary(["get_weather", "play_music"])
        result = ui._pick_many_arrows("Skills", MANY, {"open_app"})
        assert result == {"get_weather", "play_music"}

    def test_choices_are_checked_for_exactly_the_selected_values(self, fake_questionary):
        fake = fake_questionary([])
        ui._pick_many_arrows("Skills", MANY, {"open_app", "play_music"})
        (call,) = fake.checkbox_calls
        checked = {choice.value for choice in call["choices"] if choice.checked}
        unchecked = {choice.value for choice in call["choices"] if not choice.checked}
        assert checked == {"open_app", "play_music"}
        assert unchecked == {"get_weather"}
