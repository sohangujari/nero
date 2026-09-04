from nero.voice.sentence_buffer import SentenceBuffer, speakable


def test_emits_on_boundary():
    buf = SentenceBuffer()
    assert buf.feed("Hello there.") == ["Hello there."]


def test_holds_partial_until_boundary():
    buf = SentenceBuffer()
    assert buf.feed("Hello ") == []
    assert buf.feed("there!") == ["Hello there!"]


def test_streamed_tokens_across_multiple_sentences():
    buf = SentenceBuffer()
    out = []
    for tok in ["Hi", " there", ". How", " are", " you", "? Bye"]:
        out += buf.feed(tok)
    assert out == ["Hi there.", "How are you?"]
    assert buf.flush() == "Bye"


def test_question_and_exclamation_boundaries():
    """"Really?!" stays whole. It used to split into "Really?" and a lone "!",
    which Kokoro then spoke as its own utterance."""
    buf = SentenceBuffer()
    assert buf.feed("Really?! Yes. No!") == ["Really?!", "Yes.", "No!"]


def test_runon_falls_back_to_max_len_at_word_boundary():
    buf = SentenceBuffer(max_len=10)
    out = buf.feed("one two three four")
    assert out and all(len(s) <= 10 for s in out)
    assert "".join(out).replace(" ", "") + buf.flush().replace(" ", "") == "onetwothreefour"


def test_flush_empty_when_clean():
    buf = SentenceBuffer()
    buf.feed("Done.")
    assert buf.flush() == ""


# --- First-segment fast path: time-to-first-audio is set by the first chunk ---
def test_first_segment_breaks_at_a_clause():
    """Synthesizing 'Sure,' costs ~830ms; the whole opening sentence can cost
    seconds. Only the first segment pays that wait in silence."""
    buf = SentenceBuffer()
    assert buf.feed("Sure thing, I can look that up for you.") == [
        "Sure thing,",
        "I can look that up for you.",
    ]


def test_later_segments_ignore_clause_boundaries():
    buf = SentenceBuffer()
    buf.feed("Done. ")
    assert buf.feed("It is sunny, dry, and warm today.") == [
        "It is sunny, dry, and warm today."
    ]


def test_short_lead_in_is_not_cut_off_on_its_own():
    """'So,' as a standalone utterance sounds worse than the latency it saves."""
    buf = SentenceBuffer()
    assert buf.feed("So, the weather today is fine.") == ["So, the weather today is fine."]


def test_first_segment_falls_back_to_first_max_len():
    buf = SentenceBuffer()
    out = buf.feed("word " * 30)
    assert out and len(out[0]) <= 60


def test_flush_counts_as_the_first_segment():
    """A one-word reply flushed at end-of-stream must not leave the next
    reply's first sentence still in eager clause mode."""
    buf = SentenceBuffer()
    assert buf.feed("Yes") == []
    assert buf.flush() == "Yes"
    assert buf.feed("It is sunny, dry, and warm today.") == [
        "It is sunny, dry, and warm today."
    ]


# --- Markdown is written to be seen, not heard ---
class TestSpeakable:
    """Measured against Kokoro: one **bold** pair costs +2.7 s of speech and a
    [link](url) +4.8 s, because the URL gets pronounced. That is the stall
    people hear "at a comma" -- the markup beside it, not the punctuation."""

    def test_bold_markers_are_not_spoken(self):
        buf = SentenceBuffer()
        buf.feed("Here you go. ")  # past the first segment's clause fast path
        assert buf.feed("**New chapter**: it starts here.") == [
            "New chapter: it starts here."
        ]

    def test_a_link_is_read_as_its_text_not_its_url(self):
        assert speakable("See [the docs](https://example.com/a/b) for more.") == (
            "See the docs for more."
        )

    def test_italics_headings_quotes_and_rules(self):
        assert speakable("## Title") == "Title"
        assert speakable("*really* good") == "really good"
        assert speakable("> quoted line") == "quoted line"
        assert speakable("---") == ""

    def test_code_spans_read_as_their_contents(self):
        assert speakable("Use `nero config` to change it.") == (
            "Use nero config to change it."
        )

    def test_an_unmatched_marker_split_across_chunks_is_still_dropped(self):
        """Streaming can cut a **pair** in half; a lone asterisk is never
        worth pronouncing."""
        assert speakable("**New chapter") == "New chapter"

    def test_underscores_inside_identifiers_survive(self):
        """Nero really does say things like voice.barge_in out loud."""
        assert speakable("Set voice.barge_in to false.") == "Set voice.barge_in to false."

    def test_a_list_number_is_not_spoken_as_its_own_sentence(self):
        """'1.' used to be cut off as a segment and spoken alone -- a full stop
        where a list should flow."""
        buf = SentenceBuffer()
        out = buf.feed("Two things.\n\n1. **Morning**: clear\n2. **Evening**: rain\n")
        out.append(buf.flush())
        assert "1." not in out
        # One utterance per item, cut before the number rather than after it.
        assert out == ["Two things.", "Morning: clear", "Evening: rain"]

    def test_a_period_after_a_number_mid_sentence_still_ends_it(self):
        """The guard is for list markers only; a sentence ending in a figure
        must not swallow the rest of the reply."""
        buf = SentenceBuffer()
        buf.feed("It is warm today. ")
        assert buf.feed("It costs 31. Nothing more.") == ["It costs 31.", "Nothing more."]

    def test_a_url_is_not_cut_in_half_at_its_dots(self):
        """A half-URL no longer matches the link pattern, so speakable() cannot
        strip it and Kokoro pronounces it -- measured at +4.8 s."""
        buf = SentenceBuffer()
        buf.feed("Here you go. ")
        assert buf.feed("See [the forecast](https://example.com/f) for more. ") == [
            "See the forecast for more."
        ]

    def test_a_decimal_point_does_not_end_a_sentence(self):
        buf = SentenceBuffer()
        buf.feed("Hello. ")
        assert buf.feed("It is 3.14 degrees today. ") == ["It is 3.14 degrees today."]
