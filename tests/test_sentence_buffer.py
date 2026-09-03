from nero.voice.sentence_buffer import SentenceBuffer


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
    buf = SentenceBuffer()
    assert buf.feed("Really?! Yes. No!") == ["Really?", "!", "Yes.", "No!"]


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
