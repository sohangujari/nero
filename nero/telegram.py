"""Talk to Nero from your phone, over a Telegram bot.

The Bot API is plain HTTPS and JSON, so this is httpx (already a dependency)
and long polling — no framework, no webhook, no inbound port, nothing for you
to host. Nero connects out; your laptop stays where it is.

One turn here is exactly one turn in the terminal: `ChatLoop.ask` runs it, so
the fallback chain, key rotation, memory recall, skills and every error message
behave identically. This module only moves text.

## The trust boundary

A bot token is a URL anyone can message. Nero can open apps and read files on
your machine, so `allowed_chat_ids` is not optional — an empty allowlist
answers nobody rather than defaulting to "anyone who finds the bot".

An unpaired chat gets one thing back: a six-digit pairing code, which grants
nothing on its own. The code is shown *only in Telegram*; approving it means
typing it at the terminal (`nero telegram approve <code>`). That is the point
of the two channels — approval proves whoever is at the terminal is also
holding the phone, so pairing depends on possession rather than on being the
first to message the bot. Codes expire, and one chat can never hold more than
one, so a stranger spamming the bot cannot flood the queue or wait one out.

Destructive skills stay refused. The registry fails closed without a confirm
callback (the same rule voice mode follows), and there is no safe way to
approve `rm -rf` from a phone keyboard.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from platformdirs import user_state_dir

logger = logging.getLogger("nero.telegram")

API_ROOT = "https://api.telegram.org"
KEYRING_ENTRY = "telegram_bot_token"

# Telegram holds the request open until something arrives, so the loop costs
# one connection rather than a poll per second. The client waits longer than
# the server does, or every idle poll would surface as a read timeout.
POLL_SECONDS = 25
HTTP_TIMEOUT = POLL_SECONDS + 10

# Telegram rejects anything longer; replies are split rather than truncated.
MAX_MESSAGE_CHARS = 4000

# Backoff after a network failure, so a flapping connection doesn't spin.
RETRY_SECONDS = 5

# Long enough to walk to the laptop, short enough that an unattended code does
# not stay live. A stranger's request ages out on its own.
PAIRING_TTL_SECONDS = 600
# One row per chat, so the cap only binds when many *different* chats are
# probing — at which point the oldest requests are the least interesting.
MAX_PENDING_PAIRINGS = 20
_PAIRING_SCHEMA = """
CREATE TABLE IF NOT EXISTS pairings (
    chat_id INTEGER PRIMARY KEY,
    code TEXT NOT NULL,
    requested_at TEXT NOT NULL
)
"""


class TelegramError(Exception):
    """The bot could not be reached or the token was rejected."""


def default_pairing_path() -> Path:
    return Path(user_state_dir("nero")) / "telegram.db"


@dataclass(frozen=True)
class PairingRequest:
    chat_id: int
    requested_at: datetime

    def age(self) -> str:
        seconds = int((datetime.now(UTC) - self.requested_at).total_seconds())
        return f"{seconds}s ago" if seconds < 120 else f"{seconds // 60}m ago"


class PairingStore:
    """Chats that have asked to be paired, and have not been approved yet.

    Approved chats live in config (`telegram.allowed_chat_ids`) — that stays
    the single source of truth for who may talk to Nero. This only holds the
    waiting room, in its own file so clearing it never touches anything else.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or default_pairing_path())

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(_PAIRING_SCHEMA)
        return connection

    def request(self, chat_id: int) -> str:
        """A fresh code for `chat_id`, replacing any it already had.

        Replacing rather than appending is what stops a chat from farming
        codes: one live code each, and asking again invalidates the last.
        """
        code = f"{secrets.randbelow(1_000_000):06d}"
        connection = self._connect()
        try:
            with connection:
                self._purge(connection)
                connection.execute(
                    "INSERT INTO pairings (chat_id, code, requested_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(chat_id) DO UPDATE SET code=excluded.code, "
                    "requested_at=excluded.requested_at",
                    (chat_id, code, datetime.now(UTC).isoformat()),
                )
        finally:
            connection.close()
        return code

    def pending(self) -> list[PairingRequest]:
        connection = self._connect()
        try:
            with connection:
                self._purge(connection)
            rows = connection.execute(
                "SELECT chat_id, requested_at FROM pairings ORDER BY requested_at"
            ).fetchall()
        finally:
            connection.close()
        return [PairingRequest(int(c), datetime.fromisoformat(t)) for c, t in rows]

    def approve(self, code: str) -> int | None:
        """The chat id `code` belongs to, consuming it. None if no live request
        matches.

        Compared with compare_digest against every live row: a plain `==` on a
        six-digit secret leaks its prefix through timing, and the row count
        here is tiny.
        """
        code = code.strip()
        connection = self._connect()
        try:
            with connection:
                self._purge(connection)
            rows = connection.execute("SELECT chat_id, code FROM pairings").fetchall()
            matched = None
            for chat_id, stored in rows:
                if secrets.compare_digest(stored, code):
                    matched = int(chat_id)
            if matched is None:
                return None
            with connection:
                connection.execute("DELETE FROM pairings WHERE chat_id = ?", (matched,))
            return matched
        finally:
            connection.close()

    def clear(self) -> int:
        connection = self._connect()
        try:
            with connection:
                return connection.execute("DELETE FROM pairings").rowcount
        finally:
            connection.close()

    @staticmethod
    def _purge(connection: sqlite3.Connection) -> None:
        cutoff = (datetime.now(UTC) - timedelta(seconds=PAIRING_TTL_SECONDS)).isoformat()
        connection.execute("DELETE FROM pairings WHERE requested_at < ?", (cutoff,))
        connection.execute(
            "DELETE FROM pairings WHERE chat_id NOT IN ("
            "SELECT chat_id FROM pairings ORDER BY requested_at DESC LIMIT ?)",
            (MAX_PENDING_PAIRINGS,),
        )


class TelegramBot:
    """The three Bot API calls this needs, and nothing else."""

    def __init__(self, token: str, client: httpx.Client | None = None):
        self._token = token
        self._client = client or httpx.Client(timeout=HTTP_TIMEOUT)

    def _call(self, method: str, **params):
        url = f"{API_ROOT}/bot{self._token}/{method}"
        try:
            response = self._client.post(url, json=params)
        except httpx.HTTPError as exc:
            raise TelegramError(f"Could not reach Telegram: {exc}") from exc
        if response.status_code == 401:
            raise TelegramError(
                "Telegram rejected the bot token. Re-run `nero telegram setup`."
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TelegramError("Telegram sent a response Nero could not read.") from exc
        if not payload.get("ok"):
            raise TelegramError(payload.get("description") or "Telegram refused the request.")
        return payload["result"]

    def username(self) -> str:
        return self._call("getMe").get("username", "?")

    def updates(self, offset: int | None) -> list[dict]:
        """Messages since `offset`. Blocks up to POLL_SECONDS waiting for one."""
        params = {"timeout": POLL_SECONDS, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", **params)

    def send(self, chat_id: int, text: str) -> None:
        for part in _split(text):
            self._call("sendMessage", chat_id=chat_id, text=part)

    def typing(self, chat_id: int) -> None:
        """Show "typing…" while a turn runs — a reply can take 30 s, and a
        silent chat is indistinguishable from a broken one."""
        try:
            self._call("sendChatAction", chat_id=chat_id, action="typing")
        except TelegramError:
            logger.debug("could not send typing indicator", exc_info=True)

    def close(self) -> None:
        self._client.close()


def _split(text: str) -> list[str]:
    """`text` in Telegram-sized pieces, broken at newlines where possible."""
    text = text.strip() or "(no reply)"
    parts = []
    while len(text) > MAX_MESSAGE_CHARS:
        window = text[:MAX_MESSAGE_CHARS]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = MAX_MESSAGE_CHARS
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    parts.append(text)
    return parts


def incoming(update: dict) -> tuple[int, str] | None:
    """(chat_id, text) from an update, or None if it carries no text.

    Photos, stickers, joins and edits all arrive on this endpoint; anything
    without a plain-text body is not a turn.
    """
    message = update.get("message") or {}
    chat_id = (message.get("chat") or {}).get("id")
    text = (message.get("text") or "").strip()
    if chat_id is None or not text:
        return None
    return int(chat_id), text


PAIRING_REPLY = (
    "Your pairing code is {code}\n\n"
    "To finish, run this where Nero is running:\n"
    "    nero telegram approve {code}\n\n"
    "Until someone approves it there, I can't answer you."
)


def serve(
    bot: TelegramBot,
    allowed_chat_ids: set[int],
    ask: Callable[[str], str | None],
    on_event: Callable[[str], None] = lambda _m: None,
    once: bool = False,
    pairings: "PairingStore | None" = None,
    allow_pairing: bool = True,
    refresh: Callable[[], set[int]] | None = None,
    stop: "threading.Event | None" = None,
) -> None:
    """Poll for messages and answer them until interrupted.

    `ask` is `ChatLoop.ask`. An unpaired chat is offered a pairing code (see
    the module docstring) and nothing else; `allow_pairing=False` restores
    plain silence for anyone running with the door shut.

    `refresh` re-reads the allowlist between polls. Without it a chat approved
    while this is running stays locked out until the bridge is restarted — you
    approve, the phone keeps getting pairing codes, and nothing says why.
    """
    if not allowed_chat_ids and not allow_pairing:
        raise TelegramError(
            "No chat is allowed to message this bot yet. Run `nero telegram setup` first."
        )
    if pairings is None:
        pairings = PairingStore()
    offset: int | None = None
    while stop is None or not stop.is_set():
        try:
            updates = bot.updates(offset)
        except TelegramError as exc:
            on_event(f"Telegram is unreachable ({exc}). Retrying.")
            time.sleep(RETRY_SECONDS)
            continue
        if refresh is not None and updates:
            # After the poll, not before it: the poll blocks for up to
            # POLL_SECONDS, so an allowlist read beforehand is already that
            # stale by the time a message is checked against it.
            allowed_chat_ids = refresh()
        for update in updates:
            # Advance past every update, answered or not: a message Nero
            # refuses must not be re-delivered on the next poll forever.
            offset = update["update_id"] + 1
            received = incoming(update)
            if received is None:
                continue
            chat_id, text = received
            if chat_id not in allowed_chat_ids:
                # The message itself is never run. All an unpaired chat can get
                # is a code, which is worth nothing without terminal access.
                logger.warning("pairing offered to chat %s (not allowed)", chat_id)
                if not allow_pairing:
                    on_event(f"Ignored a message from chat {chat_id} (not paired).")
                    continue
                code = pairings.request(chat_id)
                bot.send(chat_id, PAIRING_REPLY.format(code=code))
                # The code is deliberately absent here: it has to travel by
                # phone, or approving it proves nothing about who is holding it.
                on_event(
                    f"Chat {chat_id} asked to pair. Approve with the code shown "
                    "in Telegram: nero telegram approve <code>"
                )
                continue
            on_event(f"{chat_id}: {text}")
            bot.typing(chat_id)
            try:
                reply = ask(text)
            except Exception as exc:  # noqa: BLE001 — the bridge must outlive a bad turn
                logger.debug("turn failed", exc_info=True)
                reply = f"Something went wrong with that turn: {exc}"
            bot.send(chat_id, reply or "(no reply)")
        if once:
            return
