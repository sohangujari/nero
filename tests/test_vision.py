"""v1.5.9 subset: `/image <path> [question]` sends a photo/screenshot via
litellm's native multimodal message shape. See
docs/superpowers/specs/2026-08-23-vision-input-design.md."""
import io

from rich.console import Console

from nero.core import chat_loop
from nero.core.chat_loop import ChatLoop


def quiet_console(width=200) -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=width)


class FakeClient:
    """Records calls and appends a plain assistant reply, like FakeClient in
    tests/test_fallback.py."""

    def __init__(self, provider="claude", model="claude-sonnet-5", reply="ok"):
        self.provider = provider
        self.model = model
        self.litellm_model = model
        self._reply = reply
        self.send_calls: list[list[dict]] = []

    def send(self, messages, on_text):
        self.send_calls.append([dict(m) for m in messages])
        on_text(self._reply)
        messages.append({"role": "assistant", "content": self._reply})


class FakeHistory:
    def __init__(self):
        self.appended = []

    def recent(self, limit=None):
        return []

    def append_turn(self, user, assistant):
        self.appended.append((user, assistant))


def make_loop(inputs, client=None, history=None):
    client = client or FakeClient()
    queue = list(inputs)

    def next_input(prompt):
        if not queue:
            raise EOFError
        return queue.pop(0)

    console = quiet_console()
    loop = ChatLoop(
        client, console=console, assistant_name="Nero", input_fn=next_input, history=history
    )
    return loop, console, client


def make_image(tmp_path, name="p.png", data=b"fake image bytes"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


class TestImageMessageShape:
    def test_png_sends_block_list_message(self, tmp_path):
        image = make_image(tmp_path, "p.png")
        client = FakeClient()
        loop, _, _ = make_loop([f"/image {image} what is this", "exit"], client=client)
        loop.run()

        sent = client.send_calls[0][-1]
        assert sent["role"] == "user"
        text_block, image_block = sent["content"]
        assert text_block == {"type": "text", "text": "what is this"}
        assert image_block["type"] == "image_url"
        assert image_block["image_url"]["url"].startswith("data:image/png;base64,")

    def test_jpg_uses_jpeg_mime(self, tmp_path):
        image = make_image(tmp_path, "p.jpg")
        client = FakeClient()
        loop, _, _ = make_loop([f"/image {image} what is this", "exit"], client=client)
        loop.run()

        _, image_block = client.send_calls[0][-1]["content"]
        assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_no_question_uses_default(self, tmp_path):
        image = make_image(tmp_path, "p.png")
        client = FakeClient()
        loop, _, _ = make_loop([f"/image {image}", "exit"], client=client)
        loop.run()

        text_block, _ = client.send_calls[0][-1]["content"]
        assert text_block["text"] == chat_loop.DEFAULT_IMAGE_QUESTION

    def test_quoted_path_with_spaces_parses(self, tmp_path):
        image = make_image(tmp_path, "my photo.png")
        client = FakeClient()
        loop, _, _ = make_loop([f'/image "{image}" what is this', "exit"], client=client)
        loop.run()

        assert len(client.send_calls) == 1
        text_block, _ = client.send_calls[0][-1]["content"]
        assert text_block["text"] == "what is this"


class TestValidationFailuresConsumeNoTurn:
    def test_missing_file_prints_red_message_and_skips_client(self, tmp_path):
        missing = tmp_path / "nope.png"
        client = FakeClient()
        loop, console, _ = make_loop([f"/image {missing} what is this", "exit"], client=client)
        loop.run()

        assert client.send_calls == []
        assert loop.messages == []
        assert "No such file" in console.file.getvalue()

    def test_bad_extension_is_rejected(self, tmp_path):
        bad = tmp_path / "notes.txt"
        bad.write_text("hi")
        client = FakeClient()
        loop, console, _ = make_loop([f"/image {bad} what is this", "exit"], client=client)
        loop.run()

        assert client.send_calls == []
        assert loop.messages == []
        assert "Unsupported" in console.file.getvalue()

    def test_oversize_file_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(chat_loop, "MAX_IMAGE_BYTES", 4)
        image = make_image(tmp_path, "p.png", data=b"way more than four bytes")
        client = FakeClient()
        loop, console, _ = make_loop([f"/image {image} what is this", "exit"], client=client)
        loop.run()

        assert client.send_calls == []
        assert loop.messages == []
        assert "too large" in console.file.getvalue()


class TestProviderGates:
    def test_ollama_provider_prints_not_supported_and_skips_client(self, tmp_path):
        image = make_image(tmp_path, "p.png")
        client = FakeClient(provider="ollama", model="llama3.2:3b")
        loop, console, _ = make_loop([f"/image {image} what is this", "exit"], client=client)
        loop.run()

        assert client.send_calls == []
        assert loop.messages == []
        assert "isn't supported on the local Ollama path" in console.file.getvalue()

    def test_supports_vision_false_warns_but_still_sends(self, tmp_path, monkeypatch):
        monkeypatch.setattr(chat_loop.litellm, "supports_vision", lambda model: False)
        image = make_image(tmp_path, "p.png")
        client = FakeClient()
        loop, console, _ = make_loop([f"/image {image} what is this", "exit"], client=client)
        loop.run()

        assert len(client.send_calls) == 1
        assert "may not support images" in console.file.getvalue()


class TestHistoryTextForm:
    def test_history_records_bracketed_basename_not_base64(self, tmp_path):
        image = make_image(tmp_path, "p.png")
        client = FakeClient(reply="a description")
        hist = FakeHistory()
        loop, _, _ = make_loop(
            [f"/image {image} what is this", "exit"], client=client, history=hist
        )
        loop.run()

        assert hist.appended == [("[image: p.png] what is this", "a description")]
