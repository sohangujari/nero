"""Talk to Nero from Slack, over a Socket Mode app.

Slack's normal delivery route is the Events API, which posts to a public HTTPS
endpoint you have to host. Socket Mode is the alternative built for exactly
this case: Slack hands out a short-lived WebSocket URL and pushes events down
it, so Nero connects out and nothing listens for inbound connections.

One turn here is exactly one turn in the terminal: `ChatLoop.ask` runs it, so
the fallback chain, key rotation, memory recall, skills and every error message
behave identically. This module only moves text.

## Two tokens, on purpose

Slack splits them, and so does this:

    app token   xapp-…   opens the socket, and can do nothing else
    bot token   xoxb-…   posts messages, and cannot open a socket

Both are needed. Keeping them apart is Slack's design, not an inconvenience to
work around: the socket token is the one that would be exposed by a stolen
connection, and it cannot post as you.

## Every event must be acknowledged

Socket Mode redelivers anything unacknowledged, so an event is acked the
moment it is understood — before the turn runs, which can take 30 seconds.
Acking after the reply would mean Slack redelivering the question while Nero
is still answering the first copy, and the user getting the answer twice.

## The trust boundary

The same one every bridge has, and it lives in `nero/channels.py`: an
allowlist, a pairing code that travels by Slack and is approved at the
terminal, and destructive skills refused throughout.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

import httpx
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect

from nero.channels import PAIRING_REPLY, ChannelError, PeerStore, split, starred_markdown

logger = logging.getLogger("nero.slack")

API_ROOT = "https://slack.com/api"
KEYRING_APP_TOKEN = "slack_app_token"
KEYRING_BOT_TOKEN = "slack_bot_token"

# chat.postMessage accepts far more, but Slack collapses anything past roughly
# this behind a "show more" fold, which is worse to read than two messages.
SLACK_LIMIT = 3000

HTTP_TIMEOUT = 15.0
RETRY_SECONDS = 5
TICK_SECONDS = 1.0


class SlackError(ChannelError):
    """Slack could not be reached, or a token was rejected."""


class SlackBot:
    """The three Web API calls this needs, and nothing else.

    Receiving happens over Socket Mode (see `serve`); this covers identity,
    opening the socket, and talking back.
    """

    def __init__(
        self,
        app_token: str,
        bot_token: str,
        client: httpx.Client | None = None,
    ):
        self._app_token = app_token
        self._bot_token = bot_token
        self._client = client or httpx.Client(timeout=HTTP_TIMEOUT)

    def _call(self, method: str, token: str, **params):
        try:
            response = self._client.post(
                f"{API_ROOT}/{method}",
                headers={"Authorization": f"Bearer {token}"},
                json=params or {},
            )
        except httpx.HTTPError as exc:
            raise SlackError(f"Could not reach Slack: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise SlackError("Slack sent a response Nero could not read.") from exc
        if not payload.get("ok"):
            raise SlackError(_why(payload.get("error")))
        return payload

    def username(self) -> str:
        return self._call("auth.test", self._bot_token).get("user", "?")

    def identity(self) -> str:
        """The bot's own user id, so its own messages can be ignored.

        Without this the bot answers itself: its reply arrives as another
        message event in the same channel.
        """
        return str(self._call("auth.test", self._bot_token).get("user_id", ""))

    def socket_url(self) -> str:
        return self._call("apps.connections.open", self._app_token)["url"]

    def send(self, channel_id: str, text: str) -> None:
        for part in split(to_mrkdwn(text), SLACK_LIMIT):
            self._call("chat.postMessage", self._bot_token, channel=channel_id, text=part)

    def close(self) -> None:
        self._client.close()


# Slack's own error strings are terse tokens. The ones a person will actually
# hit get an explanation; everything else is passed through rather than
# swallowed, because an unmapped error is still better than "something failed".
_ERRORS = {
    "invalid_auth": "Slack rejected the token. Re-run `nero slack setup`.",
    "not_authed": "Slack rejected the token. Re-run `nero slack setup`.",
    "account_inactive": "That Slack token belongs to a deactivated app or user.",
    "channel_not_found": (
        "Slack cannot see that channel. Invite the bot to it, or message the "
        "app directly instead."
    ),
    "not_in_channel": "The bot is not in that channel. Invite it with /invite.",
    "missing_scope": (
        "The Slack app is missing a scope. It needs chat:write, im:history and "
        "connections:write — add them under OAuth & Permissions, then reinstall."
    ),
}


def _why(error: str | None) -> str:
    if not error:
        return "Slack refused the request."
    return _ERRORS.get(error, f"Slack refused the request ({error}).")


def _escape(text: str) -> str:
    """The three characters Slack reads as markup in ordinary text.

    Only these three: escaping more would show backslashes to the reader, which
    is what Slack's own guidance warns against. Google Chat needs none of this,
    which is the whole reason `starred_markdown` takes an escaper.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_mrkdwn(text: str) -> str:
    """`text` as Slack mrkdwn.

    The dialect is shared with Google Chat — both read `*bold*`, `_italic_`,
    `~strike~` and `<url|text>` — so the conversion lives in nero/channels.py.
    What is Slack's is the escaping: it is the only one of the two that reads
    `&`, `<` and `>` as markup in ordinary text.
    """
    return starred_markdown(text, _escape)


def incoming(payload: dict, bot_id: str) -> tuple[str, str] | None:
    """(channel_id, text) from a Socket Mode envelope, or None if it is not a
    turn.

    Slack puts a great deal on this one stream. Filtered out here: the bot's
    own messages (without which Nero answers itself forever), every message
    subtype — edits, joins, file shares, thread broadcasts — and anything with
    no plain text.
    """
    event = (payload.get("event") or {}) if "event" in payload else payload
    if event.get("type") != "message" or event.get("subtype"):
        return None
    if event.get("bot_id") or str(event.get("user", "")) == bot_id:
        return None
    channel_id = event.get("channel")
    text = (event.get("text") or "").strip()
    if not channel_id or not text:
        return None
    return str(channel_id), text


def serve(
    bot: SlackBot,
    allowed_channel_ids: set[str],
    ask: Callable[[str], str | None],
    on_event: Callable[[str], None] = lambda _m: None,
    once: bool = False,
    peers: PeerStore | None = None,
    allow_pairing: bool = True,
    refresh: Callable[[], set[str]] | None = None,
    stop: threading.Event | None = None,
    connect_to=connect,
) -> None:
    """Hold a Socket Mode connection open and answer messages.

    `ask` is `ChatLoop.ask`. An unpaired channel is offered a pairing code (see
    `nero/channels.py`) and nothing else; `allow_pairing=False` restores plain
    silence for anyone running with the door shut.

    `refresh` re-reads the allowlist as messages arrive. Without it a channel
    approved while this is running stays locked out until the bridge is
    restarted — you approve, Slack keeps getting pairing codes, and nothing
    says why.

    Reconnects on its own: Socket Mode URLs are short-lived by design and Slack
    closes them on a schedule, so a dropped socket is routine rather than a
    failure. Only a rejected token stops the loop.
    """
    if not allowed_channel_ids and not allow_pairing:
        raise SlackError(
            "No channel is allowed to message this app yet. Run `nero slack setup` first."
        )
    if peers is None:
        peers = PeerStore("slack")
    bot_id = bot.identity()

    while stop is None or not stop.is_set():
        try:
            socket = connect_to(bot.socket_url(), max_size=None)
        except (WebSocketException, OSError) as exc:
            on_event(f"Slack is unreachable ({exc}). Retrying.")
            if once:
                return
            time.sleep(RETRY_SECONDS)
            continue

        try:
            while stop is None or not stop.is_set():
                try:
                    raw = socket.recv(timeout=TICK_SECONDS)
                except TimeoutError:
                    continue
                frame = json.loads(raw)
                envelope = frame.get("envelope_id")
                if envelope:
                    # Before the turn, not after: a turn can take 30 s, and
                    # Slack redelivers anything unacknowledged for that long.
                    socket.send(json.dumps({"envelope_id": envelope}))
                if frame.get("type") == "disconnect":
                    on_event("Slack asked Nero to reconnect.")
                    break
                if frame.get("type") != "events_api":
                    continue  # hello, and the slash-command/interactive types
                allowed_channel_ids = _answer(
                    bot, frame.get("payload") or {}, bot_id, allowed_channel_ids,
                    ask, on_event, peers, allow_pairing, refresh,
                )
                if once:
                    return
        except ConnectionClosed:
            on_event("Slack closed the connection. Reconnecting.")
        except (WebSocketException, OSError) as exc:
            on_event(f"Slack connection failed ({exc}). Reconnecting.")
        finally:
            try:
                socket.close()
            except Exception:  # noqa: BLE001 — a socket already gone is not an error
                logger.debug("closing the socket failed", exc_info=True)
        if once:
            return
        if stop is None or not stop.is_set():
            time.sleep(RETRY_SECONDS)


def _answer(
    bot: SlackBot,
    payload: dict,
    bot_id: str,
    allowed: set[str],
    ask: Callable[[str], str | None],
    on_event: Callable[[str], None],
    peers: PeerStore,
    allow_pairing: bool,
    refresh: Callable[[], set[str]] | None,
) -> set[str]:
    """Handle one events_api envelope. Returns the allowlist to use from here
    on, which `refresh` may have just widened."""
    received = incoming(payload, bot_id)
    if received is None:
        return allowed
    channel_id, text = received
    if refresh is not None:
        allowed = refresh()
    if channel_id not in allowed:
        # The message itself is never run. All an unpaired channel can get is a
        # code, which is worth nothing without terminal access.
        logger.warning("pairing offered to channel %s (not allowed)", channel_id)
        if not allow_pairing:
            on_event(f"Ignored a message from channel {channel_id} (not paired).")
            return allowed
        code = peers.request(channel_id)
        bot.send(channel_id, PAIRING_REPLY.format(code=code, channel="slack"))
        # The code is deliberately absent here: it has to travel by Slack, or
        # approving it proves nothing about who is holding the account.
        on_event(
            f"Channel {channel_id} asked to pair. Approve with the code shown "
            "in Slack: nero slack approve <code>"
        )
        return allowed
    on_event(f"{channel_id}: {text}")
    try:
        reply = ask(text)
    except Exception as exc:  # noqa: BLE001 — the bridge must outlive a bad turn
        logger.debug("turn failed", exc_info=True)
        reply = f"Something went wrong with that turn: {exc}"
    bot.send(channel_id, reply or "(no reply)")
    return allowed
