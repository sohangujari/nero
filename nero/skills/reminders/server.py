"""Reminder skills: remind_me, list_reminders, cancel_reminder.

Before these existed the model had nowhere to put a reminder, so it reached for
`remember_fact` and wrote "reminder = Hello at 4:46 AM" into memory — a place
nothing reads at 4:46 — and then told the user it was set. A skill that cannot
be called is worse than one that does not exist, because the model invents a
substitute and reports success.
"""

from collections.abc import Callable
from datetime import datetime

from nero.reminders import REPEATS, ReminderStore, ensure_delivery, parse_when
from nero.skills.base import Skill, SkillMeta

MAX_LISTED = 20


class RemindMeSkill(Skill):
    meta = SkillMeta(
        name="remind_me",
        description=(
            "Set a reminder that will be delivered to the user at a given time, "
            "even if they are not at their computer. Use this whenever the user "
            "asks to be reminded, told, nudged or alerted about something later. "
            "Put what they want to hear in `text`, in their own words, and the "
            "time in `when` copied from what they said — '5.40 pm', 'in 20 "
            "minutes'. Do NOT convert it to a date or a 24-hour clock; that is "
            "done for you, correctly. Leave `repeat` out unless the user "
            "actually asked for something recurring."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "What to say to them, e.g. 'take your medicine'.",
                },
                "when": {
                    "type": "string",
                    "description": "ISO timestamp, or '5pm', 'in 20 minutes', 'tomorrow at 9'.",
                },
                "repeat": {
                    "type": "string",
                    "enum": list(REPEATS),
                    "description": (
                        "Leave this out. Set it ONLY if the user literally said "
                        "the reminder recurs — 'every day', 'each weekday', "
                        "'every week'. A one-off time like '10 pm' is not a repeat."
                    ),
                },
            },
            "required": ["text", "when"],
        },
        requires_network=False,
        # State-changing, not destructive: it writes one row that the user can
        # see and cancel. Nothing is lost if it is wrong.
        permission_tier="state_changing",
        category="Memory",
    )

    def __init__(
        self,
        store: ReminderStore | None = None,
        now: Callable[[], datetime] | None = None,
        ensure: Callable[[], bool] | None = None,
    ):
        self._store = store or ReminderStore()
        self._now = now or (lambda: datetime.now().astimezone())
        # Injectable so a test never writes to the real ~/Library/LaunchAgents.
        self._ensure_delivery = ensure or ensure_delivery

    async def execute(self, **kwargs) -> str:
        text = str(kwargs.get("text") or "").strip()
        raw_when = str(kwargs.get("when") or "").strip()
        repeat = str(kwargs.get("repeat") or "").strip().lower()
        if not text:
            return "Error: no reminder text given."
        if repeat not in REPEATS:
            return f"Error: repeat must be one of {', '.join(r or 'none' for r in REPEATS)}."
        due = parse_when(raw_when, now=self._now())
        if due is None:
            # Refused rather than guessed: a reminder set for the wrong time is
            # worse than none, because the user stops watching for it.
            return (
                f"I couldn't work out when {raw_when!r} is. Could you give me "
                "the time again, like '5pm' or 'in 20 minutes'?"
            )
        if due <= self._now():
            return (
                f"{due:%H:%M} has already gone today. Which day did you mean?"
            )
        reminder = self._store.add(text, due, repeat=repeat)
        ongoing = f", repeating {repeat}" if repeat else ""
        confirmation = f"Reminder set for {reminder.when()}{ongoing}: {text}"
        # Delivery is set up here rather than left to a separate command the
        # user has to remember. Without it the reminder is stored correctly,
        # comes due correctly, and nobody is ever told.
        if not self._ensure_delivery():
            confirmation += (
                " (Note: background delivery could not be set up, so this only "
                "arrives while Nero is running. Tell the user to run "
                "`nero remind install`.)"
            )
        return confirmation


class ListRemindersSkill(Skill):
    meta = SkillMeta(
        name="list_reminders",
        description=(
            "List the reminders the user has set that have not gone off yet. "
            "Use this when they ask what reminders they have, or what is coming up."
        ),
        input_schema={"type": "object", "properties": {}, "required": []},
        requires_network=False,
        permission_tier="read_only",
        category="Memory",
    )

    def __init__(self, store: ReminderStore | None = None):
        self._store = store or ReminderStore()

    async def execute(self, **kwargs) -> str:
        pending = self._store.pending()[:MAX_LISTED]
        if not pending:
            return "No reminders set."
        return "\n".join(
            f"{r.id}. {r.when()} — {r.text}" + (f" (repeats {r.repeat})" if r.repeat else "")
            for r in pending
        )


class CancelReminderSkill(Skill):
    meta = SkillMeta(
        name="cancel_reminder",
        description=(
            "Cancel one reminder by its number. Call list_reminders first to "
            "find the number — never guess it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "reminder_id": {"type": "integer", "description": "The number from list_reminders."},
            },
            "required": ["reminder_id"],
        },
        requires_network=False,
        permission_tier="state_changing",
        category="Memory",
    )

    def __init__(self, store: ReminderStore | None = None):
        self._store = store or ReminderStore()

    async def execute(self, **kwargs) -> str:
        try:
            reminder_id = int(kwargs.get("reminder_id"))
        except (TypeError, ValueError):
            return "Error: reminder_id must be the number shown by list_reminders."
        if self._store.cancel(reminder_id):
            return f"Cancelled reminder {reminder_id}."
        return f"There is no pending reminder {reminder_id}."


REMIND_ME = RemindMeSkill()
LIST_REMINDERS = ListRemindersSkill()
CANCEL_REMINDER = CancelReminderSkill()
