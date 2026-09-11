"""Turning what Nero has done into what Nero knows.

The review reads the audit log, finds work that keeps coming back, and writes
each recurring job down as a playbook: when it applies, the steps that worked,
what to avoid. Next time a matching request arrives, the playbook is in the
prompt.

## Why this is a review and not a per-turn pass

Extracting a lesson from every turn would mean a second model call on every
turn, which is exactly the cost `nero/memory/recall.py` was designed to remove.
Instead the expensive thinking happens occasionally, over a log that was being
written anyway. A turn pays nothing.

## What it is allowed to write

Playbooks, and nothing else. They are retrieved only when a request matches, so
a bad one costs one turn a poor suggestion, and `nero playbooks forget` ends it.

Facts are deliberately *not* written here, and this is the one place the
"silently" answer is qualified. A fact goes into every prompt for every turn
forever, which is a much larger blast radius than a playbook's, and Nero's own
fact store has already been polluted once by a small model storing
`greeting: Hello`. So an observed preference comes back as a nudge with the
command to accept it, and a person decides.

## The reviewer has no tools

`ask` is called with no skills offered (see `nero/cli.py:_review_client`), so a
review can read the log and write a procedure and do nothing else. A model
summarising a shell command it found in a log must never be one step away from
running it.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from nero.memory.experience import (
    RECURRENCE_THRESHOLD,
    Habit,
    habits,
    sequences,
    suggest_name,
)
from nero.memory.playbooks import Playbook, PlaybookStore

logger = logging.getLogger("nero.learn")

# How much of the log one review reads. Enough to see a habit across a few
# weeks of ordinary use, small enough that the mining stays instant.
AUDIT_WINDOW = 500

# How many playbooks one review may write. A first run over a long log would
# otherwise turn into a dozen model calls and a dozen procedures nobody asked
# for; three at a time means the most frequent work is covered first and the
# rest waits for the next review.
MAX_PER_REVIEW = 3

PROMPT = """You are writing a short procedure for an assistant called Nero, \
based on what it has actually done. Nero keeps an audit log of every action it \
takes; this is a summary of something it has done repeatedly.

{evidence}

Write the procedure as JSON with exactly these keys:
  "name":  a short kebab-case identifier, 2-4 words
  "task":  when this applies, one line, starting with a verb
  "steps": the steps, one per line, numbered
  "avoid": things that failed or should not be done, one per line, or ""

Rules:
- Use only what the evidence shows. Do not invent steps, flags or file paths.
- Write it for whoever does this next, not as a report of what happened.
- If the evidence is too thin to be useful, reply with exactly: SKIP
Reply with the JSON object and nothing else."""

REVISE = """You are revising a procedure an assistant called Nero already has, \
because it has done this work again since the procedure was written.

The current procedure:
{current}

What has happened since:
{evidence}

Reply with the same JSON keys ("name", "task", "steps", "avoid"), improved. \
Keep "name" exactly as it is. Keep what still holds; only change what the new \
evidence actually contradicts or adds. If nothing needs to change, reply with \
exactly: SKIP
Reply with the JSON object and nothing else."""


@dataclass
class Learned:
    """What one review changed."""

    created: list[str] = field(default_factory=list)
    revised: list[str] = field(default_factory=list)
    considered: int = 0
    nudges: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.created or self.revised)

    def summary(self) -> str:
        if not self.changed:
            return f"Nothing new to learn from {self.considered} recurring task(s)."
        parts = []
        if self.created:
            parts.append(f"wrote {len(self.created)} new playbook(s): {', '.join(self.created)}")
        if self.revised:
            parts.append(f"revised {len(self.revised)}: {', '.join(self.revised)}")
        return "Nero " + "; ".join(parts) + "."


def evidence_for(habit: Habit) -> str:
    """A habit as the few lines the model is asked to generalise from."""
    lines = [f"Action: {habit.signature}", f"Times run: {habit.count}"]
    if habit.failures:
        lines.append(f"Times it failed or was refused: {habit.failures}")
    if habit.examples:
        lines.append("Results seen:")
        lines.extend(f"  - {example}" for example in habit.examples)
    return "\n".join(lines)


def evidence_for_sequence(window: tuple[str, ...], count: int) -> str:
    ordered = " then ".join(window)
    return f"Action: {ordered}\nTimes run in that order: {count}"


def parse(reply: str | None) -> dict | None:
    """The JSON object in a model reply, or None if there isn't a usable one.

    Tolerant on purpose: this runs against whatever model the user configured,
    including small local ones that wrap JSON in prose or a code fence. A
    review that quietly skips a malformed answer is right; one that crashes the
    command is not.
    """
    text = (reply or "").strip()
    if not text or text.upper().startswith("SKIP"):
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    task, steps = str(parsed.get("task", "")).strip(), str(parsed.get("steps", "")).strip()
    if not task or not steps:
        return None
    return {
        "name": str(parsed.get("name", "")).strip(),
        "task": task,
        "steps": steps,
        "avoid": str(parsed.get("avoid", "")).strip(),
    }


def _covering(store: PlaybookStore, evidence: str, fallback_name: str) -> Playbook | None:
    """The playbook this evidence already belongs to, if there is one.

    Checked by name first, then by the same matcher a turn uses — otherwise a
    review would keep writing a second playbook for work the first one already
    covers, and the turn would then have two to choose between.
    """
    existing = store.get(fallback_name)
    if existing is not None:
        return existing
    matches = store.match(evidence, limit=1)
    return matches[0] if matches else None


def review(
    audit,
    store: PlaybookStore,
    ask: Callable[[str], str | None],
    limit: int = AUDIT_WINDOW,
    threshold: int = RECURRENCE_THRESHOLD,
    max_new: int = MAX_PER_REVIEW,
) -> Learned:
    """Read the audit log, write down what keeps happening.

    Never raises on a bad model reply or an unreadable playbook — a review is
    housekeeping, and failing it must not be able to break a scheduled run.
    """
    entries = audit.recent(limit=limit) if audit is not None else []
    candidates: list[tuple[str, str]] = []
    for habit in habits(entries, threshold):
        candidates.append((suggest_name(habit.signature), evidence_for(habit)))
    for window, count in sequences(entries, threshold):
        candidates.append(("-then-".join(window), evidence_for_sequence(window, count)))

    learned = Learned(considered=len(candidates))
    for name, evidence in candidates:
        if len(learned.created) + len(learned.revised) >= max_new:
            break
        existing = _covering(store, evidence, name)
        prompt = (
            REVISE.format(current=existing.render(), evidence=evidence)
            if existing
            else PROMPT.format(evidence=evidence)
        )
        try:
            written = parse(ask(prompt))
        except Exception:  # noqa: BLE001 — a review must survive a bad turn
            logger.debug("review call failed for %r", name, exc_info=True)
            continue
        if written is None:
            continue
        # The model is asked to keep an existing name and often does not; the
        # name is what `restore` and `forget` address, so the store's opinion
        # wins over the model's.
        final_name = existing.name if existing else (written["name"] or name)
        try:
            store.save(
                final_name,
                written["task"],
                written["steps"],
                written["avoid"],
                note="learned from the audit log",
            )
        except Exception:  # noqa: BLE001 — see docstring
            logger.debug("could not save playbook %r", final_name, exc_info=True)
            continue
        (learned.revised if existing else learned.created).append(final_name)

    learned.nudges = nudges(entries, store)
    return learned


def nudges(entries, store: PlaybookStore) -> list[str]:
    """Things worth telling the user, which Nero will not do on its own.

    Preferences live here rather than being written straight to the fact store:
    a fact is in every prompt of every turn forever, so it is worth one line of
    confirmation. Each nudge carries the command that accepts it.
    """
    lines = []
    for habit in habits(entries, RECURRENCE_THRESHOLD):
        match = re.fullmatch(r"open_app\(name=(.+)\)", habit.signature)
        if match:
            lines.append(
                f"You open {match.group(1)} {habit.count} times — worth telling "
                "Nero to remember, if it should be a default."
            )
        if habit.failures >= RECURRENCE_THRESHOLD:
            lines.append(
                f"{habit.signature} has failed or been refused {habit.failures} "
                f"time(s) out of {habit.count} — worth a look."
            )
    stale = [book.name for book in store.all() if book.uses == 0 and book.version > 1]
    if stale:
        lines.append(
            f"Never used since being written: {', '.join(stale)}. "
            "Remove one with: nero playbooks forget <name>"
        )
    return lines[:5]
