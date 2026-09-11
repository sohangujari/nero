"""What every chat bridge has in common: who is allowed to talk, and how a
reply is cut to fit.

Nero answers on Telegram, Discord and Slack. The transports have nothing in
common — Telegram long-polls HTTPS, Discord runs a Gateway WebSocket, Slack
runs Socket Mode — but the parts that decide whether a stranger can run
`delete_path` on your laptop are identical, and those are the parts worth
having exactly one copy of.

## The trust boundary, once

A bot token is an address anyone can message, and Nero can open apps and read
files. So an allowlist is not optional on any of the three: empty means "answer
nobody", never "answer whoever finds the bot".

An unpaired peer gets one thing back: a short pairing code, which grants
nothing on its own. The code is shown *only in the chat app*; approving it
means typing it at the terminal. That is the point of the two channels —
approval proves whoever is at the terminal is also holding the phone, so
pairing depends on possession rather than on being first to message the bot.
Codes expire, and one peer can never hold more than one, so a stranger
spamming a bot cannot flood the queue or wait one out.

Destructive skills stay refused on all three. The registry fails closed without
a confirm callback (the same rule voice mode follows), and there is no safe way
to approve `rm -rf` from a phone keyboard.

## Peer ids are strings

Telegram numbers its chats; Discord uses snowflakes that only just fit in a
signed 64-bit integer; Slack uses `U01ABCDEF`. One text column holds all three
without a per-platform schema, and nothing here does arithmetic on an id.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from platformdirs import user_state_dir

# Long enough to walk to the laptop, short enough that an unattended code does
# not stay live. A stranger's request ages out on its own.
PAIRING_TTL_SECONDS = 600
# One row per peer, so the cap only binds when many *different* peers are
# probing — at which point the oldest requests are the least interesting.
MAX_PENDING_PAIRINGS = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS peers (
    peer_id TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    requested_at TEXT NOT NULL
)
"""

PAIRING_REPLY = (
    "Your pairing code is {code}\n\n"
    "To finish, run this where Nero is running:\n"
    "    nero {channel} approve {code}\n\n"
    "Until someone approves it there, I can't answer you."
)


class ChannelError(Exception):
    """A bridge could not be reached, or its credentials were rejected."""


def default_pairing_path(channel: str) -> Path:
    """One file per channel, so clearing Discord's waiting room never touches
    Slack's."""
    return Path(user_state_dir("nero")) / f"{channel}.db"


@dataclass(frozen=True)
class PairingRequest:
    peer_id: str
    requested_at: datetime

    def age(self) -> str:
        seconds = int((datetime.now(UTC) - self.requested_at).total_seconds())
        return f"{seconds}s ago" if seconds < 120 else f"{seconds // 60}m ago"


class PeerStore:
    """Peers that have asked to be paired, and have not been approved yet.

    Approved peers live in config (`<channel>.allowed_peer_ids`) — that stays
    the single source of truth for who may talk to Nero. This only holds the
    waiting room, in its own file so clearing it never touches anything else.

    Connection-per-call, like the rest of Nero's small stores: cheap at this
    call volume, and it sidesteps sqlite's thread affinity, which matters
    because a bridge runs on its own thread.
    """

    def __init__(self, channel: str, path: Path | str | None = None):
        self.channel = channel
        self.path = Path(path or default_pairing_path(channel))

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute(_SCHEMA)
        return connection

    def request(self, peer_id: str) -> str:
        """The code for `peer_id`, minting one if it has none.

        Deliberately *not* a fresh code per message: re-issuing would let
        someone invalidate a code you are halfway through typing, and would let
        one peer mint codes without limit.
        """
        now = datetime.now(UTC)
        connection = self._connect()
        try:
            with connection:
                self._prune(connection, now)
                row = connection.execute(
                    "SELECT code FROM peers WHERE peer_id = ?", (str(peer_id),)
                ).fetchone()
                if row:
                    return row[0]
                code = f"{secrets.randbelow(1_000_000):06d}"
                connection.execute(
                    "INSERT INTO peers (peer_id, code, requested_at) VALUES (?, ?, ?)",
                    (str(peer_id), code, now.isoformat()),
                )
                return code
        finally:
            connection.close()

    def pending(self) -> list[PairingRequest]:
        now = datetime.now(UTC)
        connection = self._connect()
        try:
            with connection:
                self._prune(connection, now)
            rows = connection.execute(
                "SELECT peer_id, requested_at FROM peers ORDER BY requested_at"
            ).fetchall()
        finally:
            connection.close()
        return [PairingRequest(peer_id=r[0], requested_at=datetime.fromisoformat(r[1]))
                for r in rows]

    def approve(self, code: str) -> str | None:
        """The peer id this code belongs to, consuming it. None if no match.

        Compared in Python rather than in SQL so the lookup is a constant-time
        digest comparison — a six-digit code is small enough that a timing
        oracle on `WHERE code = ?` is worth not handing out.
        """
        now = datetime.now(UTC)
        connection = self._connect()
        try:
            with connection:
                self._prune(connection, now)
                rows = connection.execute("SELECT peer_id, code FROM peers").fetchall()
                matched = next(
                    (peer for peer, stored in rows
                     if secrets.compare_digest(stored, str(code).strip())),
                    None,
                )
                if matched is None:
                    return None
                connection.execute("DELETE FROM peers WHERE peer_id = ?", (matched,))
                return matched
        finally:
            connection.close()

    def clear(self) -> int:
        connection = self._connect()
        try:
            with connection:
                return connection.execute("DELETE FROM peers").rowcount
        finally:
            connection.close()

    @staticmethod
    def _prune(connection: sqlite3.Connection, now: datetime) -> None:
        cutoff = (now - timedelta(seconds=PAIRING_TTL_SECONDS)).isoformat()
        connection.execute("DELETE FROM peers WHERE requested_at < ?", (cutoff,))
        connection.execute(
            "DELETE FROM peers WHERE peer_id NOT IN ("
            "SELECT peer_id FROM peers ORDER BY requested_at DESC LIMIT ?)",
            (MAX_PENDING_PAIRINGS,),
        )


def split(text: str, limit: int) -> list[str]:
    """`text` in pieces of at most `limit` characters, broken at a newline where
    possible and a space otherwise.

    Every chat platform has a per-message ceiling and none of them will accept a
    reply that exceeds it, so a long answer has to arrive as several messages
    rather than not at all.
    """
    text = text.strip() or "(no reply)"
    parts = []
    while len(text) > limit:
        window = text[:limit]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    parts.append(text)
    return parts


# --- Formatting -------------------------------------------------------------
#
# Slack and Google Chat read the same dialect, and it is not markdown: bold is
# *one* asterisk, italic is an underscore, strikethrough is one tilde, and a
# link is <url|text>. Converting is not optional — sent as-is, every `**bold**`
# the model writes arrives as literal asterisks.
#
# Telegram is not here: it renders a subset of HTML instead, which is a
# different enough target that sharing this would mean a function with two
# unrelated halves.

_FENCE = re.compile(r"```[\w+-]*\n?([\s\S]*?)```")
_SPAN = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"!?\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")
_HELD = re.compile(r"\x00(\d+)\x00")

# Bold and headings both render as *one asterisk*, which is exactly the syntax
# the italic rule below consumes. Left in the text, `**bold**` became `*bold*`
# became `_bold_`. So their output is parked as a placeholder the italic pass
# cannot see, the same trick code spans and links already use.
_BOLD = [
    re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.M),   # heading
    re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"),      # bold
]
# Order still matters here: the bullet rule would otherwise eat the leading
# asterisk of an italic run at the start of a line.
_INLINE = [
    (re.compile(r"^(\s*)[-*+][ \t]+", re.M), "\\1\u2022 "),            # bullet
    (re.compile(r"~~(?=\S)(.+?)(?<=\S)~~"), r"~\1~"),                  # strike
    (re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])"), r"_\1_"),  # italic
]


def starred_markdown(text: str, escape: "Callable[[str], str]") -> str:
    """`text` in the *bold* / _italic_ dialect Slack and Google Chat share.

    `escape` is what differs between them: Slack reads `&`, `<` and `>` as
    markup in ordinary text and Google Chat does not, so each passes its own
    (or `str` for none).

    Code and links are lifted out and rendered before the escape pass, so
    markup inside a code block is shown rather than interpreted.
    """
    held: list[str] = []

    def hold(rendered: str) -> str:
        held.append(rendered)
        return f"\x00{len(held) - 1}\x00"

    text = _FENCE.sub(lambda m: hold(f"```\n{escape(m.group(1).strip())}\n```"), text)
    text = _SPAN.sub(lambda m: hold(f"`{escape(m.group(1))}`"), text)
    text = _LINK.sub(
        lambda m: hold(f"<{escape(m.group(2))}|{escape(m.group(1)) or 'link'}>"), text
    )
    text = escape(text)
    for pattern in _BOLD:
        # The last group is the content either way: the heading pattern has one
        # group, the bold pattern's second group is the text inside the markers.
        text = pattern.sub(lambda m: hold(f"*{m.group(m.re.groups)}*"), text)
    for pattern, replacement in _INLINE:
        text = pattern.sub(replacement, text)
    return _HELD.sub(lambda m: held[int(m.group(1))], text)
