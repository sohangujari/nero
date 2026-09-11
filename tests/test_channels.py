"""The parts of a chat bridge that decide whether a stranger can run skills.

Discord and Slack share their whole security model with Telegram, so these
tests are about the shared core (nero/channels.py) and the two new transports.
The transports themselves are faked: what matters is which messages reach
`ask`, not that websockets can open a socket.
"""

import json

import pytest

from nero.channels import PairingRequest, PeerStore, split
from nero import discord as discord_bridge
from nero import slack as slack_bridge


@pytest.fixture
def peers(tmp_path):
    """Never the real ~/Library state file."""
    return PeerStore("discord", tmp_path / "peers.db")


class TestPeerStore:
    def test_a_peer_keeps_one_code_across_messages(self, peers):
        """Re-issuing would let a stranger invalidate the code you are halfway
        through typing, and would mint codes without limit."""
        first = peers.request("123")
        assert peers.request("123") == first

    def test_a_code_is_six_digits(self, peers):
        code = peers.request("123")
        assert len(code) == 6 and code.isdigit()

    def test_approving_returns_the_peer_and_consumes_the_request(self, peers):
        code = peers.request("999")
        assert peers.approve(code) == "999"
        assert peers.approve(code) is None
        assert peers.pending() == []

    def test_a_wrong_code_approves_nothing(self, peers):
        peers.request("123")
        assert peers.approve("000000") is None
        assert len(peers.pending()) == 1

    def test_slack_style_ids_survive_the_round_trip(self, tmp_path):
        """Slack ids are not numeric, and Discord snowflakes are 19 digits. One
        text column has to hold both without mangling either."""
        store = PeerStore("slack", tmp_path / "slack.db")
        for peer in ("D01ABCDEF", "1319283746152637485"):
            assert store.approve(store.request(peer)) == peer

    def test_the_waiting_room_is_capped(self, tmp_path):
        """A stranger spamming a bot must not be able to flood the queue."""
        from nero.channels import MAX_PENDING_PAIRINGS

        store = PeerStore("discord", tmp_path / "many.db")
        for index in range(MAX_PENDING_PAIRINGS + 10):
            store.request(f"peer-{index}")
        assert len(store.pending()) <= MAX_PENDING_PAIRINGS

    def test_stale_requests_age_out(self, tmp_path, monkeypatch):
        from datetime import UTC, datetime, timedelta

        import nero.channels as channels

        store = PeerStore("discord", tmp_path / "stale.db")
        store.request("123")
        later = datetime.now(UTC) + timedelta(seconds=channels.PAIRING_TTL_SECONDS + 60)

        class _Later(datetime):
            @classmethod
            def now(cls, tz=None):
                return later

        monkeypatch.setattr(channels, "datetime", _Later)
        assert store.pending() == []

    def test_clearing_empties_the_waiting_room(self, peers):
        peers.request("1")
        peers.request("2")
        assert peers.clear() == 2
        assert peers.pending() == []

    def test_channels_do_not_share_a_waiting_room(self, tmp_path):
        """Clearing Discord's pending list must never touch Slack's."""
        discord = PeerStore("discord", tmp_path / "d.db")
        slack = PeerStore("slack", tmp_path / "s.db")
        discord.request("111")
        slack.request("222")
        discord.clear()
        assert [r.peer_id for r in slack.pending()] == ["222"]


class TestPairingRequestAge:
    def test_seconds_then_minutes(self):
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        assert PairingRequest("1", now - timedelta(seconds=30)).age() == "30s ago"
        assert PairingRequest("1", now - timedelta(minutes=5)).age().endswith("m ago")


class TestSplit:
    def test_a_long_reply_arrives_in_pieces_rather_than_truncated(self):
        parts = split("x" * 250, 100)
        assert all(len(part) <= 100 for part in parts)
        assert "".join(parts) == "x" * 250

    def test_it_breaks_on_a_newline_when_it_can(self):
        # Longer than the limit, or there is nothing to split.
        assert split("a" * 90 + "\n" + "b" * 50, 100)[0] == "a" * 90

    def test_an_empty_reply_is_never_sent_as_empty(self):
        assert split("   ", 100) == ["(no reply)"]


# --- Discord ----------------------------------------------------------------


class TestDiscordMarkdown:
    def test_bold_and_italics_pass_straight_through(self):
        """Discord reads the same markdown the model writes, so a converter
        would only be something new to get wrong."""
        assert discord_bridge.to_discord("**bold** and *italic*") == "**bold** and *italic*"

    def test_a_heading_becomes_bold(self):
        """Discord renders `# x` larger than the surrounding chat, which is
        wrong for a reply."""
        assert discord_bridge.to_discord("## Heading") == "**Heading**"

    def test_a_star_bullet_becomes_a_dash(self):
        assert discord_bridge.to_discord("* one\n* two") == "- one\n- two"

    def test_a_code_block_is_left_alone(self):
        text = "```python\nprint('hi')\n```"
        assert discord_bridge.to_discord(text) == text


class TestDiscordIncoming:
    def message(self, **over):
        return {"author": {"id": "222"}, "channel_id": "42", "content": "hello"} | over

    def test_a_plain_message_is_a_turn(self):
        assert discord_bridge.incoming(self.message(), "999") == ("42", "hello")

    def test_the_bots_own_message_is_never_a_turn(self):
        """Without this Nero answers itself forever: its reply arrives as
        another MESSAGE_CREATE in the same channel."""
        assert discord_bridge.incoming(self.message(author={"id": "999"}), "999") is None

    def test_another_bot_is_ignored(self):
        assert discord_bridge.incoming(
            self.message(author={"id": "555", "bot": True}), "999"
        ) is None

    def test_an_empty_message_is_not_a_turn(self):
        assert discord_bridge.incoming(self.message(content="   "), "999") is None

    def test_an_attachment_with_no_text_is_not_a_turn(self):
        assert discord_bridge.incoming(self.message(content=""), "999") is None


class _FakeSocket:
    """A Gateway that hands out canned frames, then blocks like a real idle
    socket does."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []
        self.closed = False

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def recv(self, timeout=None):
        if not self.frames:
            raise TimeoutError
        return json.dumps(self.frames.pop(0))

    def close(self):
        self.closed = True


class _FakeDiscordBot:
    def __init__(self):
        self.sent = []
        self.token = "t"

    def identity(self):
        return "999"

    def gateway_url(self):
        return "wss://example.invalid"

    def send(self, channel_id, text):
        self.sent.append((channel_id, text))

    def typing(self, channel_id):
        pass

    def close(self):
        pass


def _dispatch(channel_id="42", content="hello", author="222"):
    return {
        "op": discord_bridge.OP_DISPATCH,
        "t": "MESSAGE_CREATE",
        "s": 1,
        "d": {"author": {"id": author}, "channel_id": channel_id, "content": content},
    }


class TestDiscordServe:
    def run(self, frames, allowed, peers, **over):
        bot = _FakeDiscordBot()
        asked = []
        discord_bridge.serve(
            bot,
            allowed,
            lambda text: asked.append(text) or "answered",
            peers=peers,
            once=True,
            connect_to=lambda *a, **k: _FakeSocket(frames),
            **over,
        )
        return bot, asked

    def test_a_paired_channel_gets_an_answer(self, tmp_path):
        peers = PeerStore("discord", tmp_path / "p.db")
        bot, asked = self.run([_dispatch()], {"42"}, peers)
        assert asked == ["hello"]
        assert bot.sent == [("42", "answered")]

    def test_an_unpaired_channel_is_never_run(self, tmp_path):
        """The whole security model: a stranger's message reaches a pairing
        code and nothing else."""
        peers = PeerStore("discord", tmp_path / "p.db")
        bot, asked = self.run([_dispatch(channel_id="66")], {"42"}, peers)
        assert asked == []
        assert "pairing code" in bot.sent[0][1]
        assert [r.peer_id for r in peers.pending()] == ["66"]

    def test_with_pairing_off_a_stranger_gets_silence(self, tmp_path):
        peers = PeerStore("discord", tmp_path / "p.db")
        bot, asked = self.run(
            [_dispatch(channel_id="66")], {"42"}, peers, allow_pairing=False
        )
        assert asked == [] and bot.sent == []

    def test_a_channel_approved_mid_run_is_picked_up(self, tmp_path):
        """Without refresh you approve, the phone keeps getting pairing codes,
        and nothing says why."""
        peers = PeerStore("discord", tmp_path / "p.db")
        bot, asked = self.run(
            [_dispatch(channel_id="66")], set(), peers, refresh=lambda: {"66"}
        )
        assert asked == ["hello"]

    def test_hello_triggers_an_identify(self, tmp_path):
        peers = PeerStore("discord", tmp_path / "p.db")
        socket = _FakeSocket(
            [{"op": discord_bridge.OP_HELLO, "d": {"heartbeat_interval": 41250}}, _dispatch()]
        )
        discord_bridge.serve(
            _FakeDiscordBot(), {"42"}, lambda _t: "ok",
            peers=peers, once=True, connect_to=lambda *a, **k: socket,
        )
        assert socket.sent[0]["op"] == discord_bridge.OP_IDENTIFY
        assert socket.sent[0]["d"]["intents"] == discord_bridge.INTENTS

    def test_the_requested_intents_are_not_privileged(self):
        """Asking for Message Content without having enabled it in the portal
        makes Discord refuse the connection outright (close code 4014), so the
        default must not need a manual toggle."""
        MESSAGE_CONTENT = 1 << 15
        GUILD_MESSAGES = 1 << 9
        assert not discord_bridge.INTENTS & MESSAGE_CONTENT
        assert not discord_bridge.INTENTS & GUILD_MESSAGES

    def test_a_failing_turn_does_not_kill_the_bridge(self, tmp_path):
        peers = PeerStore("discord", tmp_path / "p.db")
        bot = _FakeDiscordBot()

        def boom(_text):
            raise RuntimeError("provider exploded")

        discord_bridge.serve(
            bot, {"42"}, boom, peers=peers, once=True,
            connect_to=lambda *a, **k: _FakeSocket([_dispatch()]),
        )
        assert "provider exploded" in bot.sent[0][1]

    def test_a_rejected_token_stops_rather_than_retrying_forever(self, tmp_path):
        from websockets.exceptions import ConnectionClosed
        from websockets.frames import Close

        peers = PeerStore("discord", tmp_path / "p.db")

        class _Rejecting(_FakeSocket):
            def recv(self, timeout=None):
                raise ConnectionClosed(Close(4004, "Authentication failed"), None)

        with pytest.raises(discord_bridge.DiscordError, match="rejected the bot token"):
            discord_bridge.serve(
                _FakeDiscordBot(), {"42"}, lambda _t: "ok", peers=peers,
                connect_to=lambda *a, **k: _Rejecting([]),
            )


# --- Slack ------------------------------------------------------------------


class TestSlackMrkdwn:
    def test_double_asterisk_bold_becomes_single(self):
        """Slack's mrkdwn is not markdown. Sent as-is, every **bold** the model
        writes arrives as literal asterisks."""
        assert slack_bridge.to_mrkdwn("**bold**") == "*bold*"

    def test_single_asterisk_italic_becomes_an_underscore(self):
        assert slack_bridge.to_mrkdwn("*italic*") == "_italic_"

    def test_a_heading_becomes_bold(self):
        assert slack_bridge.to_mrkdwn("# Title") == "*Title*"

    def test_strikethrough_loses_a_tilde(self):
        assert slack_bridge.to_mrkdwn("~~gone~~") == "~gone~"

    def test_a_bullet_becomes_a_real_bullet(self):
        assert slack_bridge.to_mrkdwn("- one") == "• one"

    def test_a_link_becomes_slacks_own_form(self):
        assert slack_bridge.to_mrkdwn("[docs](https://x.test)") == "<https://x.test|docs>"

    def test_markup_inside_a_code_span_is_shown_not_interpreted(self):
        assert slack_bridge.to_mrkdwn("`**not bold**`") == "`**not bold**`"

    def test_the_three_markup_characters_are_escaped(self):
        assert slack_bridge.to_mrkdwn("a < b & c > d") == "a &lt; b &amp; c &gt; d"


class TestSlackIncoming:
    def event(self, **over):
        return {"event": {"type": "message", "channel": "D1", "user": "U1", "text": "hi"} | over}

    def test_a_direct_message_is_a_turn(self):
        assert slack_bridge.incoming(self.event(), "UBOT") == ("D1", "hi")

    def test_the_bots_own_message_is_never_a_turn(self):
        assert slack_bridge.incoming(self.event(user="UBOT"), "UBOT") is None

    def test_a_message_from_any_bot_is_ignored(self):
        assert slack_bridge.incoming(self.event(bot_id="B1"), "UBOT") is None

    def test_an_edit_or_join_is_not_a_turn(self):
        """Slack puts edits, joins, file shares and thread broadcasts on this
        same stream, all carrying a subtype."""
        assert slack_bridge.incoming(self.event(subtype="message_changed"), "UBOT") is None

    def test_a_non_message_event_is_not_a_turn(self):
        assert slack_bridge.incoming(self.event(type="reaction_added"), "UBOT") is None

    def test_an_empty_message_is_not_a_turn(self):
        assert slack_bridge.incoming(self.event(text="  "), "UBOT") is None


class _FakeSlackBot:
    def __init__(self):
        self.sent = []

    def identity(self):
        return "UBOT"

    def socket_url(self):
        return "wss://example.invalid"

    def send(self, channel_id, text):
        self.sent.append((channel_id, text))

    def close(self):
        pass


def _envelope(channel="D1", text="hi", user="U1", envelope_id="e1"):
    return {
        "type": "events_api",
        "envelope_id": envelope_id,
        "payload": {"event": {"type": "message", "channel": channel, "user": user, "text": text}},
    }


class TestSlackServe:
    def run(self, frames, allowed, peers, **over):
        bot = _FakeSlackBot()
        socket = _FakeSocket(frames)
        asked = []
        slack_bridge.serve(
            bot,
            allowed,
            lambda text: asked.append(text) or "answered",
            peers=peers,
            once=True,
            connect_to=lambda *a, **k: socket,
            **over,
        )
        return bot, asked, socket

    def test_a_paired_channel_gets_an_answer(self, tmp_path):
        peers = PeerStore("slack", tmp_path / "p.db")
        bot, asked, _ = self.run([_envelope()], {"D1"}, peers)
        assert asked == ["hi"]
        assert bot.sent == [("D1", "answered")]

    def test_every_event_is_acknowledged(self, tmp_path):
        """Socket Mode redelivers anything unacknowledged, so a slow turn
        would otherwise be asked and answered twice."""
        peers = PeerStore("slack", tmp_path / "p.db")
        _bot, _asked, socket = self.run([_envelope()], {"D1"}, peers)
        assert socket.sent == [{"envelope_id": "e1"}]

    def test_it_acknowledges_before_running_the_turn(self, tmp_path):
        """A turn can take 30 s; Slack redelivers well inside that."""
        peers = PeerStore("slack", tmp_path / "p.db")
        socket = _FakeSocket([_envelope()])
        order = []

        def ask(_text):
            order.append(("asked", list(socket.sent)))
            return "answered"

        slack_bridge.serve(
            _FakeSlackBot(), {"D1"}, ask, peers=peers, once=True,
            connect_to=lambda *a, **k: socket,
        )
        assert order[0][1] == [{"envelope_id": "e1"}]

    def test_an_unpaired_channel_is_never_run(self, tmp_path):
        peers = PeerStore("slack", tmp_path / "p.db")
        bot, asked, _ = self.run([_envelope(channel="D9")], {"D1"}, peers)
        assert asked == []
        assert "pairing code" in bot.sent[0][1]

    def test_the_pairing_message_names_the_right_command(self, tmp_path):
        peers = PeerStore("slack", tmp_path / "p.db")
        bot, _asked, _ = self.run([_envelope(channel="D9")], set(), peers)
        assert "nero slack approve" in bot.sent[0][1]

    def test_a_channel_approved_mid_run_is_picked_up(self, tmp_path):
        peers = PeerStore("slack", tmp_path / "p.db")
        _bot, asked, _ = self.run(
            [_envelope(channel="D9")], set(), peers, refresh=lambda: {"D9"}
        )
        assert asked == ["hi"]

    def test_hello_is_not_treated_as_a_message(self, tmp_path):
        peers = PeerStore("slack", tmp_path / "p.db")
        bot, asked, _ = self.run([{"type": "hello"}, _envelope()], {"D1"}, peers)
        assert asked == ["hi"]

    def test_a_failing_turn_does_not_kill_the_bridge(self, tmp_path):
        peers = PeerStore("slack", tmp_path / "p.db")
        bot = _FakeSlackBot()

        def boom(_text):
            raise RuntimeError("provider exploded")

        slack_bridge.serve(
            bot, {"D1"}, boom, peers=peers, once=True,
            connect_to=lambda *a, **k: _FakeSocket([_envelope()]),
        )
        assert "provider exploded" in bot.sent[0][1]


class TestSlackErrors:
    def test_a_known_error_is_explained(self):
        assert "Re-run `nero slack setup`" in slack_bridge._why("invalid_auth")

    def test_an_unknown_error_is_passed_through_rather_than_swallowed(self):
        assert "ratelimited" in slack_bridge._why("ratelimited")


# --- Google Chat ------------------------------------------------------------

from nero import googlechat as gchat_bridge  # noqa: E402


class TestGoogleChatFormatting:
    def test_it_reads_the_same_dialect_as_slack(self):
        assert gchat_bridge.to_google_chat("**bold** and *italic*") == "*bold* and _italic_"

    def test_a_heading_becomes_bold(self):
        assert gchat_bridge.to_google_chat("# Title") == "*Title*"

    def test_angle_brackets_are_left_alone(self):
        """Slack reads &, < and > as markup in ordinary text; Google Chat does
        not, and escaping them would show entities to the reader."""
        assert gchat_bridge.to_google_chat("a < b & c") == "a < b & c"

    def test_a_link_becomes_the_angle_bracket_form(self):
        assert gchat_bridge.to_google_chat("[docs](https://x.test)") == "<https://x.test|docs>"


class TestGoogleChatIncoming:
    def event(self, **over):
        message = {
            "sender": {"type": "HUMAN", "name": "users/1"},
            "space": {"name": "spaces/AAA"},
            "text": "hello",
        } | over.pop("message", {})
        return {"type": "MESSAGE", "message": message} | over

    def test_a_message_is_a_turn(self):
        assert gchat_bridge.incoming(self.event()) == ("spaces/AAA", "hello")

    def test_the_apps_own_message_is_never_a_turn(self):
        """Nero's reply arrives on the same topic; without this it answers
        itself forever."""
        event = self.event(message={"sender": {"type": "BOT"}})
        assert gchat_bridge.incoming(event) is None

    def test_being_added_to_a_space_is_not_a_turn(self):
        """Chat publishes ADDED_TO_SPACE and card clicks on the same topic."""
        assert gchat_bridge.incoming(self.event(type="ADDED_TO_SPACE")) is None

    def test_an_empty_message_is_not_a_turn(self):
        assert gchat_bridge.incoming(self.event(message={"text": "   "})) is None


class _FakeGoogleChatBot:
    """Stands in for the REST calls. `pull` hands out one batch, then nothing,
    the way an idle subscription behaves."""

    def __init__(self, batches):
        self.batches = list(batches)
        self.sent = []
        self.acked = []
        self.checked = False

    def check(self):
        self.checked = True

    def pull(self, max_messages=10):
        return self.batches.pop(0) if self.batches else []

    def acknowledge(self, ack_ids):
        self.acked.extend(ack_ids)

    def send(self, space, text):
        self.sent.append((space, text))

    def close(self):
        pass


def _gc_event(space="spaces/AAA", text="hello", sender="HUMAN"):
    return {
        "type": "MESSAGE",
        "message": {"sender": {"type": sender}, "space": {"name": space}, "text": text},
    }


class TestGoogleChatServe:
    def run(self, batches, allowed, peers, **over):
        bot = _FakeGoogleChatBot(batches)
        asked = []
        gchat_bridge.serve(
            bot, allowed, lambda text: asked.append(text) or "answered",
            peers=peers, once=True, **over,
        )
        return bot, asked

    def test_a_paired_space_gets_an_answer(self, tmp_path):
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot, asked = self.run([[("ack1", _gc_event())]], {"spaces/AAA"}, peers)
        assert asked == ["hello"]
        assert bot.sent == [("spaces/AAA", "answered")]

    def test_every_event_is_acknowledged(self, tmp_path):
        """Pub/Sub redelivers anything unacknowledged, so a slow turn would
        otherwise be asked and answered twice."""
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot, _asked = self.run([[("ack1", _gc_event())]], {"spaces/AAA"}, peers)
        assert bot.acked == ["ack1"]

    def test_it_acknowledges_before_running_the_turn(self, tmp_path):
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot = _FakeGoogleChatBot([[("ack1", _gc_event())]])
        order = []
        gchat_bridge.serve(
            bot, {"spaces/AAA"},
            lambda _t: order.append(list(bot.acked)) or "answered",
            peers=peers, once=True,
        )
        assert order[0] == ["ack1"]

    def test_an_event_that_is_not_a_turn_is_still_acknowledged(self, tmp_path):
        """Otherwise one join notice comes back forever and blocks the queue."""
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot, asked = self.run(
            [[("ack1", {"type": "ADDED_TO_SPACE"})]], {"spaces/AAA"}, peers
        )
        assert asked == [] and bot.acked == ["ack1"]

    def test_an_unpaired_space_is_never_run(self, tmp_path):
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot, asked = self.run(
            [[("ack1", _gc_event(space="spaces/ZZZ"))]], {"spaces/AAA"}, peers
        )
        assert asked == []
        assert "pairing code" in bot.sent[0][1]
        assert "nero googlechat approve" in bot.sent[0][1]

    def test_a_space_approved_mid_run_is_picked_up(self, tmp_path):
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot, asked = self.run(
            [[("ack1", _gc_event(space="spaces/ZZZ"))]], set(), peers,
            refresh=lambda: {"spaces/ZZZ"},
        )
        assert asked == ["hello"]

    def test_the_subscription_is_checked_before_polling(self, tmp_path):
        """A subscription that was never created is not a transient failure;
        retrying it forever is how a bridge looks alive and answers nothing."""
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot, _asked = self.run([], {"spaces/AAA"}, peers)
        assert bot.checked

    def test_a_failing_turn_does_not_kill_the_bridge(self, tmp_path):
        peers = PeerStore("googlechat", tmp_path / "p.db")
        bot = _FakeGoogleChatBot([[("ack1", _gc_event())]])

        def boom(_text):
            raise RuntimeError("provider exploded")

        gchat_bridge.serve(bot, {"spaces/AAA"}, boom, peers=peers, once=True)
        assert "provider exploded" in bot.sent[0][1]


class TestGoogleChatCredentials:
    def test_a_key_that_is_not_json_says_so(self):
        with pytest.raises(gchat_bridge.GoogleChatError, match="not valid JSON"):
            gchat_bridge.GoogleChatBot("not json at all", "p", "s")

    def test_a_key_missing_fields_says_so(self):
        with pytest.raises(gchat_bridge.GoogleChatError, match="missing something"):
            gchat_bridge.GoogleChatBot('{"type": "service_account"}', "p", "s")
