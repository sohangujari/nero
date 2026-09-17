"""Reminders: something to say, a time to say it, and a way to reach you.

The point of this module is the last part. Nero already had routines — cron
prompts run by launchd — and they cannot remind you of anything, because an
installed routine writes its reply to `~/Library/Logs/nero/<name>.out.log` and
nothing reads a log file. A reminder that does not arrive is not a reminder.

So this is deliberately *not* built on routines:

- A routine runs a full model turn when it fires. A reminder must deliver the
  words you actually said, so no model is in the path at fire time. It cannot
  drift, cost a round trip, or apologise.
- A routine is a cron expression. "At 5pm today" has no cron form that does not
  also mean every day at 5pm.
- One launchd agent ticks for all reminders, rather than one plist each.
  Writing and unloading a file every time someone says "remind me" is how you
  end up with orphaned agents nobody can find.

## Late is better than never

A laptop asleep at 17:00 has no reminder at 17:00. Rather than dropping it,
anything still undelivered is delivered at the next tick with the time it was
meant for attached, so "take your medicine" arriving at 19:40 still says it was
for 17:00. Silently dropping a medicine reminder is the worst behaviour
available, and it is the one a naive implementation picks.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from platformdirs import user_state_dir

logger = logging.getLogger("nero.reminders")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    due_at TEXT NOT NULL,
    repeat TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL DEFAULT '',
    peer TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    delivered_at TEXT
);
CREATE INDEX IF NOT EXISTS reminders_due ON reminders (delivered_at, due_at);
"""

# How late a missed reminder may be and still be worth delivering. Past this it
# is noise rather than a reminder — a laptop shut for a week should not wake up
# and recite Monday.
MAX_LATE = timedelta(hours=12)

# How often a running session checks. Matches the launchd agent's interval;
# a reminder is never more than a minute late from either.
REMINDER_TICK_SECONDS = 60
# Overridable for tests only; nothing in production changes it.
import os as _os
TICK_SECONDS = float(_os.environ.get('NERO_REMINDER_TICK') or REMINDER_TICK_SECONDS)

REPEATS = ("", "daily", "weekdays", "weekly")


def default_reminder_path() -> Path:
    return Path(user_state_dir("nero")) / "reminders.db"


@dataclass(frozen=True)
class Reminder:
    id: int
    text: str
    due_at: datetime
    repeat: str = ""
    channel: str = ""
    peer: str = ""
    created_at: str = ""
    delivered_at: str | None = None

    def when(self) -> str:
        """The due time as a person would say it."""
        now = datetime.now().astimezone()
        due = self.due_at.astimezone()
        if due.date() == now.date():
            return f"today at {due:%H:%M}"
        if due.date() == (now + timedelta(days=1)).date():
            return f"tomorrow at {due:%H:%M}"
        return f"{due:%a %d %b} at {due:%H:%M}"

    def message(self, now: datetime | None = None) -> str:
        """What actually gets sent. Verbatim, with no model in the path."""
        now = now or datetime.now().astimezone()
        late = now - self.due_at.astimezone()
        if late > timedelta(minutes=2):
            return f"⏰ {self.text}\n\n(this was for {self.due_at.astimezone():%H:%M})"
        return f"⏰ {self.text}"


# --- Understanding a time ----------------------------------------------------
#
# The model is asked for an ISO timestamp and usually gives one, since the
# current time is already in its system prompt. But small local models do
# arithmetic badly, so the everyday forms are parsed here too rather than
# trusted to it. Anything not understood is refused, never guessed — a reminder
# set for the wrong time is worse than one that was not set, because you stop
# watching for it.

_RELATIVE = re.compile(r"\bin\s+(\d+)\s*(min(?:ute)?s?|hours?|hrs?|days?)\b", re.I)
# A dot is as common a minute separator as a colon — "5.25 pm" is how most
# people write it. Without it here the regex matched only the "5", silently
# dropped both the minutes and the "pm", and set the reminder for 05:00 the
# next morning. Wrong beats missing only in the sense that it is worse.
_CLOCK = re.compile(r"\b(\d{1,2})\s*(?:[:.]\s*(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", re.I)
# "half past five" and friends. Parsed rather than refused because people say
# them, and matched *before* the bare clock so the hour is not read alone.
_SPOKEN = re.compile(r"\b(half|quarter)\s+(past|to)\s+(\d{1,2})\b", re.I)
# Phrases that name a time this does not understand. They have to be refused
# explicitly: each contains a number the clock pattern would happily read on
# its own and turn into the wrong hour.
_UNPARSED = re.compile(r"\b(\d+\s*(?:past|to)\b|past\s+the|-ish\b|around\s+\d)", re.I)
# When the user names the day themselves, the hour is not ours to move.
_NAMES_A_DAY = re.compile(r"\b(today|tonight|tomorrow|this (?:morning|afternoon|evening))\b", re.I)
_EVENING = re.compile(r"\b(tonight|this (?:afternoon|evening)|in the (?:afternoon|evening))\b", re.I)


def parse_when(text: str, now: datetime | None = None) -> datetime | None:
    """`text` as a local datetime in the future, or None if it is not a time.

    Returns None rather than a guess. The caller turns that into a question.
    """
    now = (now or datetime.now().astimezone()).replace(microsecond=0)
    text = (text or "").strip()
    if not text:
        return None

    # An ISO timestamp, which is what the model is asked for.
    try:
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=now.tzinfo)
    except ValueError:
        pass

    lowered = text.lower()
    relative = _RELATIVE.search(lowered)
    if relative:
        count, unit = int(relative.group(1)), relative.group(2)
        if unit.startswith(("min", "mins")):
            return now + timedelta(minutes=count)
        if unit.startswith(("hour", "hr")):
            return now + timedelta(hours=count)
        return now + timedelta(days=count)

    day = now
    if "tomorrow" in lowered:
        day = now + timedelta(days=1)
    # "Tonight at 9" and "this evening at 7" name the half of the day as surely
    # as writing "pm" does.
    if _EVENING.search(lowered) and "pm" not in lowered:
        lowered += " pm"

    if "midnight" in lowered:
        return _at(day + timedelta(days=1 if "tomorrow" not in lowered else 0), 0, 0, now, lowered)
    if "noon" in lowered or "midday" in lowered:
        return _at(day, 12, 0, now, lowered)
    if _UNPARSED.search(lowered):
        return None

    spoken = _SPOKEN.search(lowered)
    if spoken:
        size, direction, raw_hour = spoken.group(1).lower(), spoken.group(2).lower(), int(spoken.group(3))
        minute = 30 if size == "half" else 15
        hour = raw_hour
        if direction == "to":
            minute = 60 - minute
            hour = (raw_hour - 1) % 24
        if "pm" in lowered and hour < 12:
            hour += 12
        return _at(day, hour, minute, now, lowered, ambiguous="pm" not in lowered)

    clock = _CLOCK.search(lowered)
    if clock is None:
        return None
    hour, minute = int(clock.group(1)), int(clock.group(2) or 0)
    meridiem = (clock.group(3) or "").replace(".", "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return _at(day, hour, minute, now, lowered, ambiguous=not meridiem)


def _at(
    day: datetime, hour: int, minute: int, now: datetime, lowered: str,
    ambiguous: bool = False,
) -> datetime | None:
    """The named time, resolved to an actual future moment.

    `ambiguous` means no am/pm was given and the hour could be either. "Half
    past 5" said at half past two means 17:30, not 05:30 tomorrow — so both
    readings are considered and the sooner future one wins. With an explicit
    am/pm, or a 24-hour clock, there is only one reading to try.
    """
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    candidates = [hour]
    if ambiguous and 1 <= hour <= 12:
        candidates.append((hour + 12) % 24)

    resolved = []
    for candidate in candidates:
        due = day.replace(hour=candidate, minute=minute, second=0, microsecond=0)
        # A time that has already gone means tomorrow — but only when the user
        # named no day at all. If they said "today", moving it to tomorrow
        # silently contradicts them; the caller is better placed to say "that
        # has already gone, which day did you mean?". "Tomorrow at 9" reading
        # as past is the user's own arithmetic, not ours to correct.
        if due <= now and not _NAMES_A_DAY.search(lowered):
            due += timedelta(days=1)
        resolved.append(due)
    # The soonest reading that has not already gone. "Tonight at 9" said at
    # 18:10 is 21:00, not this morning — and picking merely the soonest of the
    # two gave exactly that.
    future = [due for due in resolved if due > now]
    return min(future) if future else min(resolved)


def next_occurrence(due: datetime, repeat: str) -> datetime | None:
    """When a repeating reminder is next due, or None if it does not repeat."""
    if repeat == "daily":
        return due + timedelta(days=1)
    if repeat == "weekly":
        return due + timedelta(days=7)
    if repeat == "weekdays":
        following = due + timedelta(days=1)
        while following.weekday() >= 5:  # Saturday, Sunday
            following += timedelta(days=1)
        return following
    return None


# A reply that commits to having set a reminder. Matched on the *claim*, not on
# the request, so "what reminders do I have" never trips it.
_CLAIMED_SET = re.compile(
    r"\b(i(?:'ve| have| will|'ll)?\s+(?:set|scheduled|added|created)\b[^.]{0,40}\breminder"
    r"|reminder\s+(?:is|has been)\s+(?:set|scheduled|added)"
    r"|^reminder\s+set\s+for"
    r"|i(?:'ll| will)\s+remind\s+you)",
    re.I,
)


def claims_a_reminder(reply: str) -> bool:
    """Whether `reply` tells the user a reminder now exists.

    Checked against what actually ran. A small model with two such replies
    already in its transcript reproduces the wording instead of calling the
    tool — measured at 0 calls in 4 on qwen3.5:2b, every one of them claiming
    success. The user finds out at 17:40 when nothing arrives.
    """
    return bool(_CLAIMED_SET.search(reply or ""))


def notify_locally(text: str) -> bool:
    """Put a reminder on the screen of the machine Nero is running on.

    The last resort, and the only one that works for someone who has paired no
    chat app at all — which is most people on day one. Without it a CLI-only
    user sets a reminder, everything works, and nothing ever appears.

    Best effort: a notification that cannot be posted is not worth an error, and
    the caller prints the reminder to the terminal regardless.
    """
    if sys.platform != "darwin":
        return False
    body = text.replace("\\", "").replace('"', "'").replace("\n", " ")
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{body}" with title "Nero" sound name "Glass"'],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("could not post a local notification", exc_info=True)
        return False
    return True


def ensure_delivery() -> bool:
    """Make sure something will actually deliver reminders. Returns whether it
    is now in place.

    Called when a reminder is set, not left to the user to run separately. A
    reminder feature whose delivery is opt-in is a feature that silently does
    nothing the first time you rely on it — which is exactly what happened: six
    reminders sat correctly stored and correctly due while no agent existed to
    send them.

    Idempotent and never raises. If launchd will not have it, the reminder is
    still stored and still delivered by any running `nero` session.
    """
    from nero import routines

    agents = routines.default_agents_dir()
    if routines.reminder_plist_path(agents).exists():
        return True
    try:
        routines.install_reminders(routines.resolve_executable(), agents)
    except Exception:  # noqa: BLE001 — setting a reminder must not fail on this
        logger.warning("could not install reminder delivery", exc_info=True)
        return False
    return routines.reminder_plist_path(agents).exists()


class ReminderStore:
    """Connection-per-call, like Nero's other small stores: the tick process
    and a chat bridge both read this from their own threads."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or default_reminder_path())

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.executescript(_SCHEMA)
        return connection

    def add(
        self, text: str, due_at: datetime, repeat: str = "",
        channel: str = "", peer: str = "",
    ) -> Reminder:
        """Store one reminder. Asking twice for the same thing is not two
        reminders.

        A model that retries a tool call, a user who repeats themselves, and a
        turn that runs more than one round all end up here with identical
        arguments — and three copies of "drink water" at 17:25 is three
        notifications, which reads as a broken app rather than a keen one.
        """
        existing = self._select(
            "SELECT * FROM reminders WHERE delivered_at IS NULL "
            "AND text = ? AND due_at = ? LIMIT 1",
            (text.strip(), due_at.isoformat()),
        )
        if existing:
            return existing[0]
        now = datetime.now(UTC).isoformat()
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO reminders (text, due_at, repeat, channel, peer, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (text.strip(), due_at.isoformat(), repeat, channel, peer, now),
                )
        finally:
            connection.close()
        return Reminder(
            id=cursor.lastrowid, text=text.strip(), due_at=due_at,
            repeat=repeat, channel=channel, peer=peer, created_at=now,
        )

    def pending(self) -> list[Reminder]:
        """Everything still waiting, soonest first."""
        return self._select(
            "SELECT * FROM reminders WHERE delivered_at IS NULL ORDER BY due_at"
        )

    def due(self, now: datetime | None = None) -> list[Reminder]:
        """What should be delivered on this tick, including anything missed
        while the machine was asleep — but not so long ago that it is noise."""
        now = now or datetime.now().astimezone()
        floor = (now - MAX_LATE).isoformat()
        return [
            reminder
            for reminder in self._select(
                "SELECT * FROM reminders WHERE delivered_at IS NULL "
                "AND due_at <= ? AND due_at >= ? ORDER BY due_at",
                (now.isoformat(), floor),
            )
        ]

    def claim(self, reminder: Reminder) -> bool:
        """Take ownership of a reminder before sending it.

        Two things deliver reminders — the launchd agent and any running Nero
        session — so both can see the same row as due. The UPDATE only matches
        while `delivered_at` is still null, so exactly one of them wins and the
        user gets one notification rather than two.
        """
        connection = self._connect()
        try:
            with connection:
                return connection.execute(
                    "UPDATE reminders SET delivered_at = ? "
                    "WHERE id = ? AND delivered_at IS NULL",
                    (datetime.now(UTC).isoformat(), reminder.id),
                ).rowcount > 0
        finally:
            connection.close()

    def mark_delivered(self, reminder: Reminder) -> Reminder | None:
        """Close one out. A repeating reminder is re-armed for its next time,
        and the new row is returned."""
        self.claim(reminder)
        following = next_occurrence(reminder.due_at, reminder.repeat)
        if following is None:
            return None
        return self.add(
            reminder.text, following, reminder.repeat, reminder.channel, reminder.peer
        )

    def cancel(self, reminder_id: int) -> bool:
        connection = self._connect()
        try:
            with connection:
                return connection.execute(
                    "DELETE FROM reminders WHERE id = ? AND delivered_at IS NULL",
                    (reminder_id,),
                ).rowcount > 0
        finally:
            connection.close()

    def clear(self) -> int:
        connection = self._connect()
        try:
            with connection:
                return connection.execute(
                    "DELETE FROM reminders WHERE delivered_at IS NULL"
                ).rowcount
        finally:
            connection.close()

    def _select(self, sql: str, params: tuple = ()) -> list[Reminder]:
        connection = self._connect()
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            logger.warning("could not read reminders: %s", exc)
            return []
        finally:
            connection.close()
        return [
            Reminder(
                id=row["id"], text=row["text"],
                due_at=datetime.fromisoformat(row["due_at"]),
                repeat=row["repeat"], channel=row["channel"], peer=row["peer"],
                created_at=row["created_at"], delivered_at=row["delivered_at"],
            )
            for row in rows
        ]
