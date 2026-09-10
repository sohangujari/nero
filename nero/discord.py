"""Talk to Nero from Discord, over a bot in your direct messages.

Unlike Telegram there is no long-poll endpoint: Discord delivers messages over
a Gateway WebSocket, and the only alternative — interaction webhooks — would
mean hosting a public HTTPS endpoint, which this app exists to avoid. So this
opens one outbound WebSocket and keeps it alive. Your laptop stays where it is;
nothing listens for inbound connections.

One turn here is exactly one turn in the terminal: `ChatLoop.ask` runs it, so
the fallback chain, key rotation, memory recall, skills and every error message
behave identically. This module only moves text.

## Direct messages, not servers

The bot answers in DMs and nowhere else, which mirrors Telegram's private chat
and, more usefully, keeps setup to one switch. Reading messages in a *server*
channel needs the Message Content intent, which Discord gates behind a manual
toggle in the developer portal and refuses the connection over (close code
4014) if you ask for it without having enabled it. Direct messages carry their
content to the bot unconditionally, so `INTENTS` asks for nothing privileged
and connecting works the moment the token exists.

## The trust boundary

The same one every bridge has, and it lives in `nero/channels.py`: an
allowlist, a pairing code that travels by Discord and is approved at the
terminal, and destructive skills refused throughout.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import threading
import time
from collections.abc import Callable

import httpx
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.sync.client import connect

from nero.channels import PAIRING_REPLY, ChannelError, PeerStore, split

logger = logging.getLogger("nero.discord")

API_ROOT = "https://discord.com/api/v10"
KEYRING_ENTRY = "discord_bot_token"

# Direct messages only — see the module docstring on why nothing privileged is
# requested here.
DIRECT_MESSAGES = 1 << 12
INTENTS = DIRECT_MESSAGES

# Discord's hard ceiling on one message's content.
DISCORD_LIMIT = 2000

HTTP_TIMEOUT = 15.0
# Backoff after a dropped connection, so a flapping network doesn't spin.
RETRY_SECONDS = 5
# How long a recv waits before the loop checks whether a heartbeat is due or
# the bridge has been asked to stop. Short enough to shut down promptly,
# long enough not to busy-wait.
TICK_SECONDS = 1.0

# Gateway opcodes. Only these five matter to a bot that reads and replies.
OP_DISPATCH = 0
OP_HEARTBEAT = 1
OP_IDENTIFY = 2
OP_RECONNECT = 7
OP_INVALID_SESSION = 9
OP_HELLO = 10
OP_HEARTBEAT_ACK = 11

# Close codes Discord will not let a retry fix. Reconnecting on these is an
# infinite loop against a wrong token or an un-toggled intent, so they are
# raised instead.
FATAL_CLOSE_CODES = {
    4004: "Discord rejected the bot token. Re-run `nero discord setup`.",
    4010: "That bot token is for a sharded connection Nero does not use.",
    4011: "This bot is in too many servers for a single connection.",
    4013: "Nero asked for a gateway intent Discord does not recognise.",
    4014: (
        "Discord refused the intents Nero asked for. Enable them for this "
        "application at https://discord.com/developers/applications, under "
        "Bot -> Privileged Gateway Intents."
    ),
}


class DiscordError(ChannelError):
    """The bot could not be reached or the token was rejected."""


class DiscordBot:
    """The four REST calls this needs, and nothing else.

    Receiving happens on the Gateway (see `serve`); this covers identity,
    finding the socket, and talking back.
    """

    def __init__(self, token: str, client: httpx.Client | None = None):
        self._token = token
        self._client = client or httpx.Client(timeout=HTTP_TIMEOUT)

    @property
    def token(self) -> str:
        return self._token

    def _call(self, method: str, path: str, **json_body):
        url = f"{API_ROOT}{path}"
        headers = {"Authorization": f"Bot {self._token}"}
        try:
            response = self._client.request(
                method, url, headers=headers, json=json_body or None
            )
        except httpx.HTTPError as exc:
            raise DiscordError(f"Could not reach Discord: {exc}") from exc
        if response.status_code == 401:
            raise DiscordError("Discord rejected the bot token. Re-run `nero discord setup`.")
        if response.status_code >= 400:
            raise DiscordError(_why(response))
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise DiscordError("Discord sent a response Nero could not read.") from exc

    def username(self) -> str:
        return (self._call("GET", "/users/@me") or {}).get("username", "?")

    def identity(self) -> str:
        """The bot's own user id, so its own messages can be ignored.

        Without this the bot answers itself: its reply is a MESSAGE_CREATE in
        the same channel, which would be treated as the next question.
        """
        return str((self._call("GET", "/users/@me") or {}).get("id", ""))

    def gateway_url(self) -> str:
        url = (self._call("GET", "/gateway/bot") or {}).get("url")
        if not url:
            raise DiscordError("Discord did not say where its gateway is.")
        return f"{url}?v=10&encoding=json"

    def send(self, channel_id: str, text: str) -> None:
        for part in split(to_discord(text), DISCORD_LIMIT):
            self._call("POST", f"/channels/{channel_id}/messages", content=part)

    def typing(self, channel_id: str) -> None:
        """Show "typing…" while a turn runs — a reply can take 30 s, and a
        silent chat is indistinguishable from a broken one."""
        try:
            self._call("POST", f"/channels/{channel_id}/typing")
        except DiscordError:
            logger.debug("could not send typing indicator", exc_info=True)

    def close(self) -> None:
        self._client.close()


def _why(response: httpx.Response) -> str:
    """Discord's own words for a failure, or the status code if it has none."""
    try:
        payload = response.json()
    except ValueError:
        return f"Discord refused the request ({response.status_code})."
    return payload.get("message") or f"Discord refused the request ({response.status_code})."


# Discord renders markdown natively, so almost nothing has to change. The two
# exceptions are headings, which it only supports at the start of a line and
# renders larger than the surrounding chat (wrong for a chat reply), and the
# `*` bullet, which it does not render as a list at all.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.M)
_BULLET = re.compile(r"^(\s*)\*[ \t]+", re.M)


def to_discord(text: str) -> str:
    """`text` as Discord-flavoured markdown.

    A passthrough with two fixes, rather than a converter: the model already
    writes the dialect Discord reads.
    """
    text = _HEADING.sub(r"**\1**", text)
    return _BULLET.sub("\\1- ", text)


def incoming(event: dict, bot_id: str) -> tuple[str, str] | None:
    """(channel_id, text) from a MESSAGE_CREATE, or None if it is not a turn.

    Three things are filtered out here, and the first is the important one:
    the bot's own messages arrive on this same event, so without that check
    Nero answers itself forever.
    """
    author = event.get("author") or {}
    if str(author.get("id", "")) == bot_id or author.get("bot"):
        return None
    channel_id = event.get("channel_id")
    text = (event.get("content") or "").strip()
    if not channel_id or not text:
        return None
    return str(channel_id), text


class _Gateway:
    """One Gateway connection: identify, heartbeat, and hand back dispatches.

    Discord's heartbeat is the connection's dead-man switch — miss it and the
    socket is closed under you — so it is interleaved with reads rather than
    run on a second thread. `recv` waits a tick at a time, which is also what
    makes a stop request take effect promptly.
    """

    def __init__(self, socket, token: str):
        self._socket = socket
        self._token = token
        self._interval = 41.25  # replaced by Hello; a sane value until then
        self._next_beat = time.monotonic() + self._interval
        self._sequence: int | None = None

    def _send(self, payload: dict) -> None:
        self._socket.send(json.dumps(payload))

    def identify(self) -> None:
        self._send(
            {
                "op": OP_IDENTIFY,
                "d": {
                    "token": self._token,
                    "intents": INTENTS,
                    "properties": {
                        "os": platform.system().lower(),
                        "browser": "nero",
                        "device": "nero",
                    },
                },
            }
        )

    def beat(self) -> None:
        self._send({"op": OP_HEARTBEAT, "d": self._sequence})
        self._next_beat = time.monotonic() + self._interval

    def beat_if_due(self) -> None:
        if time.monotonic() >= self._next_beat:
            self.beat()

    def read(self) -> dict | None:
        """The next Gateway frame, or None if the tick elapsed with nothing on
        the wire. Hello, heartbeat requests and acks are handled here; anything
        else is returned for the caller to route."""
        try:
            raw = self._socket.recv(timeout=TICK_SECONDS)
        except TimeoutError:
            return None
        frame = json.loads(raw)
        if frame.get("s") is not None:
            self._sequence = frame["s"]
        op = frame.get("op")
        if op == OP_HELLO:
            self._interval = frame["d"]["heartbeat_interval"] / 1000
            self._next_beat = time.monotonic() + self._interval
            self.identify()
            return None
        if op == OP_HEARTBEAT:
            self.beat()  # the server can ask for one out of band
            return None
        if op == OP_HEARTBEAT_ACK:
            return None
        return frame


def serve(
    bot: DiscordBot,
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
    """Hold a Gateway connection open and answer direct messages.

    `ask` is `ChatLoop.ask`. An unpaired channel is offered a pairing code (see
    `nero/channels.py`) and nothing else; `allow_pairing=False` restores plain
    silence for anyone running with the door shut.

    `refresh` re-reads the allowlist as messages arrive. Without it a channel
    approved while this is running stays locked out until the bridge is
    restarted — you approve, Discord keeps getting pairing codes, and nothing
    says why.

    Reconnects on its own after a dropped socket, because a laptop that sleeps
    drops one every time. Only the close codes Discord will not let a retry fix
    (a bad token, a refused intent) stop the loop.
    """
    if not allowed_channel_ids and not allow_pairing:
        raise DiscordError(
            "No channel is allowed to message this bot yet. Run `nero discord setup` first."
        )
    if peers is None:
        peers = PeerStore("discord")
    bot_id = bot.identity()

    while stop is None or not stop.is_set():
        try:
            socket = connect_to(bot.gateway_url(), max_size=None)
        except (WebSocketException, OSError, DiscordError) as exc:
            on_event(f"Discord is unreachable ({exc}). Retrying.")
            if once:
                return
            time.sleep(RETRY_SECONDS)
            continue

        gateway = _Gateway(socket, bot.token)
        try:
            while stop is None or not stop.is_set():
                gateway.beat_if_due()
                frame = gateway.read()
                if frame is None:
                    continue
                if frame.get("op") in (OP_RECONNECT, OP_INVALID_SESSION):
                    on_event("Discord asked Nero to reconnect.")
                    break
                if frame.get("op") != OP_DISPATCH or frame.get("t") != "MESSAGE_CREATE":
                    continue
                allowed_channel_ids = _answer(
                    bot, frame.get("d") or {}, bot_id, allowed_channel_ids,
                    ask, on_event, peers, allow_pairing, refresh,
                )
                if once:
                    return
        except ConnectionClosed as exc:
            code = exc.rcvd.code if exc.rcvd else None
            if code in FATAL_CLOSE_CODES:
                raise DiscordError(FATAL_CLOSE_CODES[code]) from exc
            on_event(f"Discord closed the connection ({code}). Reconnecting.")
        except (WebSocketException, OSError) as exc:
            on_event(f"Discord connection failed ({exc}). Reconnecting.")
        finally:
            try:
                socket.close()
            except Exception:  # noqa: BLE001 — a socket already gone is not an error
                logger.debug("closing the gateway socket failed", exc_info=True)
        if once:
            return
        if stop is None or not stop.is_set():
            time.sleep(RETRY_SECONDS)


def _answer(
    bot: DiscordBot,
    event: dict,
    bot_id: str,
    allowed: set[str],
    ask: Callable[[str], str | None],
    on_event: Callable[[str], None],
    peers: PeerStore,
    allow_pairing: bool,
    refresh: Callable[[], set[str]] | None,
) -> set[str]:
    """Handle one MESSAGE_CREATE. Returns the allowlist to use from here on,
    which `refresh` may have just widened."""
    received = incoming(event, bot_id)
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
        bot.send(channel_id, PAIRING_REPLY.format(code=code, channel="discord"))
        # The code is deliberately absent here: it has to travel by Discord, or
        # approving it proves nothing about who is holding the account.
        on_event(
            f"Channel {channel_id} asked to pair. Approve with the code shown "
            "in Discord: nero discord approve <code>"
        )
        return allowed
    on_event(f"{channel_id}: {text}")
    bot.typing(channel_id)
    try:
        reply = ask(text)
    except Exception as exc:  # noqa: BLE001 — the bridge must outlive a bad turn
        logger.debug("turn failed", exc_info=True)
        reply = f"Something went wrong with that turn: {exc}"
    bot.send(channel_id, reply or "(no reply)")
    return allowed
