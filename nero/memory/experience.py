"""Reading Nero's own audit log back: what it does often, and what goes wrong.

The audit log already records every skill call and its result, so identifying
recurring work costs no extra model calls and no per-turn bookkeeping — it is
a query over something that was being written anyway. That is the whole reason
learning here is cheap enough to leave switched on.

Nothing in this module talks to a model. It reduces a log to a handful of
observations; `nero/learn.py` is what decides whether any of them are worth
writing down as a procedure.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta

from nero.core.audit_log import AuditEntry

# A gap longer than this starts a new episode. Skill calls inside one turn land
# milliseconds apart, and a follow-up question lands seconds later; the next
# thing you sit down to do is minutes away at least.
EPISODE_GAP = timedelta(minutes=5)

# How many times something has to happen before it counts as a habit rather
# than a coincidence. Two is a repeat; three is a pattern worth a procedure.
RECURRENCE_THRESHOLD = 3

# Argument values longer than this identify one particular job rather than the
# shape of the job, so they are dropped from a signature: reading nine
# different files is one habit, not nine.
MAX_SIGNATURE_VALUE = 40

# A skill result is free text — there is no structured ok flag anywhere in the
# registry — so failure is read from what refusals and errors actually say.
# Every one of these strings is produced by nero/skills/registry.py or by a
# skill's own error path.
#
# ponytail: substring match on result text. Ceiling — a skill whose *successful*
# output happens to contain "error:" is miscounted as a failure, and a new
# failure phrasing is missed until it is added here. The upgrade path is an
# `ok` flag on the skill result envelope, which is a change to every skill;
# worth it only if these counts start driving something more than a suggestion.
FAILURE_MARKERS = (
    "error:",
    "the user declined",
    "is turned off right now",
    "needs an internet connection",
    "could not find",
    "not found",
    "failed",
    "refused",
)


@dataclass
class Episode:
    """One sitting: the skill calls that happened close together in time."""

    entries: list[AuditEntry] = field(default_factory=list)

    @property
    def skills(self) -> list[str]:
        return [entry.skill_name for entry in self.entries]

    @property
    def started_at(self):
        return self.entries[0].timestamp


@dataclass(frozen=True)
class Habit:
    """Something Nero has done enough times to be worth a procedure."""

    signature: str
    count: int
    failures: int
    examples: list[str]

    @property
    def shaky(self) -> bool:
        """Whether this habit fails often enough to be worth writing down what
        *not* to do, rather than only what to do."""
        return self.failures > 0


def failed(entry: AuditEntry) -> bool:
    summary = (entry.result_summary or "").lower()
    return any(marker in summary for marker in FAILURE_MARKERS)


def episodes(entries: list[AuditEntry], gap: timedelta = EPISODE_GAP) -> list[Episode]:
    """`entries` grouped into sittings, oldest first.

    The audit log has no session id — it is a flat append-only list shared by
    the terminal, voice and every chat bridge — so time is the only thing
    available to group by. It is also the right thing to group by: what makes
    two calls part of one job is that they happened together.
    """
    ordered = sorted(entries, key=lambda entry: entry.timestamp)
    grouped: list[Episode] = []
    for entry in ordered:
        if grouped and entry.timestamp - grouped[-1].entries[-1].timestamp <= gap:
            grouped[-1].entries.append(entry)
        else:
            grouped.append(Episode([entry]))
    return grouped


def signature(entry: AuditEntry) -> str:
    """A stable name for "this kind of call".

    Short argument values are kept because they *are* the habit — `set_volume`
    with `change: -20` is a different one from `change: 100`. Long ones are
    dropped to their key, because a path or a search query names one particular
    job rather than a recurring kind of job.
    """
    parts = []
    for key in sorted(entry.arguments or {}):
        value = entry.arguments[key]
        if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            parts.append(f"{key}={value}")
        elif isinstance(value, str) and len(value) <= MAX_SIGNATURE_VALUE:
            parts.append(f"{key}={value.strip().lower()}")
        else:
            parts.append(key)
    return f"{entry.skill_name}({', '.join(parts)})" if parts else entry.skill_name


def habits(
    entries: list[AuditEntry], threshold: int = RECURRENCE_THRESHOLD
) -> list[Habit]:
    """The recurring calls in `entries`, most frequent first."""
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for entry in entries:
        key = signature(entry)
        counts[key] += 1
        if failed(entry):
            failures[key] += 1
        examples.setdefault(key, [])
        if len(examples[key]) < 3 and entry.result_summary:
            examples[key].append(entry.result_summary)
    return [
        Habit(signature=key, count=count, failures=failures[key], examples=examples[key])
        for key, count in counts.most_common()
        if count >= threshold
    ]


def sequences(
    entries: list[AuditEntry], threshold: int = RECURRENCE_THRESHOLD, length: int = 2
) -> list[tuple[tuple[str, ...], int]]:
    """Runs of skills that keep happening in the same order, most common first.

    This is what separates a procedure from a habit: `run_shell` alone is a
    habit, but `read_file` then `edit_file` then `run_shell`, in that order,
    three times over, is a procedure that could be written down.
    """
    counts: Counter[tuple[str, ...]] = Counter()
    for episode in episodes(entries):
        names = episode.skills
        for size in range(length, min(len(names), 5) + 1):
            for start in range(len(names) - size + 1):
                window = tuple(names[start : start + size])
                # A skill repeated back to back is retrying, not a procedure.
                if len(set(window)) > 1:
                    counts[window] += 1
    return [(window, count) for window, count in counts.most_common() if count >= threshold]


_WORD = re.compile(r"[a-z][a-z0-9-]{2,}")


def suggest_name(signature_text: str) -> str:
    """A short, typeable name for a habit, e.g. `open-app-spotify`.

    Only a starting point: `nero learn` asks the model for a better one and
    falls back to this when it does not get a usable answer.
    """
    words = _WORD.findall(signature_text.lower())
    return "-".join(dict.fromkeys(words))[:40].strip("-") or "playbook"
