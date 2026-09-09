import pytest


class TestMalformedToolCallResidue:
    """llama3.2:1b answers "hi" by echoing a tool schema back. The stripper
    cannot balance the blob and hands back its head —
    `{"type":"function","function":` — which was then printed as the reply."""

    def client(self, provider="ollama"):
        from nero.config.schema import LLMConfig, NeroConfig
        from nero.llm.client import LLMClient
        from nero.skills.registry import build_registry

        config = LLMConfig(provider=provider, model="llama3.2:1b")
        return LLMClient(config, "Nero", build_registry(NeroConfig()), api_key=None)

    def emit(self, client, cleaned):
        from nero.llm.client import _looks_like_json

        # The branch under test, in isolation: MALFORMED drops JSON residue.
        return None if _looks_like_json(cleaned) else cleaned

    @pytest.mark.parametrize(
        "residue",
        [
            '{"type":"function","function":',
            '{"name": "open_website"',
            '[{"type":"function"',
            '  {"unbalanced": ',
            '{\\"type\\": \\"function\\"',
        ],
    )
    def test_json_residue_is_never_shown_as_a_reply(self, residue):
        assert self.emit(self.client(), residue) is None

    def test_residue_starting_with_an_escaped_quote_is_a_known_gap(self):
        '''The guard tests for a leading bracket, so `\\"{...` slips through.
        Deliberately not widened: `_looks_like_json` is shared with the ollama
        path, where a prose reply opening with a quote is perfectly ordinary and
        must not be swallowed.'''
        assert self.emit(self.client(), '\\"{\\"type\\": \\"function\\"') is not None

    @pytest.mark.parametrize(
        "real", ["Sure, opening that now.", "I can't do that.", "42 is the answer."]
    )
    def test_genuine_text_alongside_a_bad_call_still_shows(self, real):
        """The branch exists to surface real conversational content the model
        produced next to its broken call. That must keep working."""
        assert self.emit(self.client(), real) == real

    def test_the_user_sees_an_apology_rather_than_protocol_noise(self):
        """Emitting nothing falls through to the apology, which is a worse
        answer than a good one but a far better one than raw JSON."""
        from nero.llm.client import APOLOGY

        shown = []
        messages = []
        self.client()._emit_final(shown.append, messages, None, already_shown=False)
        assert shown == [APOLOGY]
        assert messages[-1]["content"] == APOLOGY
