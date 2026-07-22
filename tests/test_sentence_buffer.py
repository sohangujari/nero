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
