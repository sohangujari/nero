"""Nero learning from its own audit log.

The two things worth locking down: a playbook is text that reaches the prompt
and never anything that executes, and no version is ever lost.
"""

from datetime import UTC, datetime, timedelta

import pytest

from nero.core.audit_log import AuditEntry
from nero.learn import Learned, evidence_for, parse, review
from nero.memory import experience
from nero.memory.playbooks import PlaybookStore, playbook_block


@pytest.fixture
def store(tmp_path):
    """Never the real ~/Library state file."""
    return PlaybookStore(tmp_path / "playbooks.db")


def entry(skill="run_shell", arguments=None, result="done", minutes=0):
    return AuditEntry(
        timestamp=datetime.now(UTC) - timedelta(minutes=minutes),
        skill_name=skill,
        arguments=arguments if arguments is not None else {},
        result_summary=result,
        provider="test",
    )


class _Audit:
    def __init__(self, entries):
        self._entries = entries

    def recent(self, limit=100):
        return self._entries[:limit]


# --- The store --------------------------------------------------------------


class TestPlaybookStore:
    def test_saving_and_reading_one_back(self, store):
        store.save("Deploy Docs", "publish the docs site", "1. npm run build:docs")
        book = store.get("deploy-docs")
        assert book is not None
        assert book.task == "publish the docs site"
        assert book.version == 1

    def test_the_name_is_normalised_once(self, store):
        """Names are typed at a shell and matched against text, so
        'Deploy Docs' and 'deploy-docs' must not become two playbooks."""
        store.save("Deploy Docs", "a", "b")
        store.save("deploy-docs", "c", "d")
        assert len(store.all()) == 1
        assert store.get("DEPLOY DOCS").version == 2

    def test_saving_over_a_name_makes_a_version_rather_than_replacing(self, store):
        store.save("deploy", "publish", "1. old step")
        store.save("deploy", "publish", "1. new step")
        assert store.get("deploy").steps == "1. new step"
        history = store.history("deploy")
        assert [r.version for r in history] == [1]
        assert history[0].steps == "1. old step"

    def test_restoring_an_old_version_loses_nothing(self, store):
        """A restore is additive: the version it replaced is still there, so
        an accidental restore is itself undoable."""
        store.save("deploy", "publish", "1. first")
        store.save("deploy", "publish", "2. second")
        store.save("deploy", "publish", "3. third")
        restored = store.restore("deploy", 1)
        assert restored.steps == "1. first"
        assert restored.version == 4
        assert {r.version for r in store.history("deploy")} == {1, 2, 3}

    def test_restoring_a_version_that_never_existed_changes_nothing(self, store):
        store.save("deploy", "publish", "1. only")
        assert store.restore("deploy", 9) is None
        assert store.get("deploy").version == 1

    def test_creation_time_survives_a_revision(self, store):
        first = store.save("deploy", "publish", "1. a")
        second = store.save("deploy", "publish", "1. b")
        assert second.created_at == first.created_at

    def test_forgetting_takes_the_history_too(self, store):
        store.save("deploy", "publish", "1. a")
        store.save("deploy", "publish", "1. b")
        assert store.forget("deploy") is True
        assert store.get("deploy") is None
        assert store.history("deploy") == []

    def test_forgetting_can_keep_the_history(self, store):
        store.save("deploy", "publish", "1. a")
        store.save("deploy", "publish", "1. b")
        store.forget("deploy", keep_history=True)
        assert store.history("deploy") != []

    def test_forgetting_something_that_is_not_there_says_so(self, store):
        assert store.forget("nope") is False


class TestMatching:
    def test_a_matching_request_finds_the_playbook(self, store):
        store.save("deploy-docs", "publish the docs site to github pages", "1. build")
        assert [b.name for b in store.match("publish the docs site")] == ["deploy-docs"]

    def test_one_shared_word_is_not_a_match(self, store):
        """A single coincidental word is how an unrelated procedure ends up in
        the prompt, which is worse than no procedure at all."""
        store.save("deploy-docs", "publish the docs site to github pages", "1. build")
        assert store.match("what is the weather in Paris") == []

    def test_a_message_with_no_real_words_matches_nothing(self, store):
        store.save("deploy-docs", "publish the docs site", "1. build")
        assert store.match("hi") == []

    def test_only_one_playbook_is_ever_carried(self, store):
        """Two competing procedures is how a small model follows half of each."""
        store.save("deploy-docs", "publish the docs site", "1. a")
        store.save("deploy-site", "publish the docs site again", "1. b")
        assert len(store.match("publish the docs site")) == 1

    def test_steps_are_not_matched_on(self, store):
        """Matching on steps would retrieve a git playbook for any mention of
        a commit."""
        store.save("deploy-docs", "publish the docs site", "1. git commit -am x")
        assert store.match("git commit my work") == []


class TestPromptBlock:
    def test_a_match_is_tagged_not_headed(self, store):
        """A bare procedure above a question reads to a small model like
        something the user pasted, and gets answered instead of followed."""
        store.save("deploy-docs", "publish the docs site", "1. npm run build:docs")
        block = playbook_block(store, "publish the docs site")
        assert block.startswith("<playbook>")
        assert "npm run build:docs" in block

    def test_no_match_adds_nothing_to_the_turn(self, store):
        store.save("deploy-docs", "publish the docs site", "1. build")
        assert playbook_block(store, "what is 2+2") == ""

    def test_no_store_adds_nothing(self):
        assert playbook_block(None, "anything") == ""

    def test_a_broken_store_costs_a_hint_not_the_turn(self, store):
        class _Broken:
            def match(self, text, limit=1):
                raise RuntimeError("disk gone")

        assert playbook_block(_Broken(), "publish the docs site") == ""

    def test_retrieval_is_counted(self, store):
        store.save("deploy-docs", "publish the docs site", "1. build")
        playbook_block(store, "publish the docs site")
        assert store.get("deploy-docs").uses == 1

    def test_an_oversized_playbook_is_left_out_rather_than_truncated(self, store):
        """Half a procedure is worse than none: a cut-off step list reads as a
        complete one."""
        from nero.memory.playbooks import MAX_BLOCK_CHARS

        store.save("huge", "publish the docs site", "x" * (MAX_BLOCK_CHARS + 100))
        assert playbook_block(store, "publish the docs site") == ""


# --- Mining the audit log ---------------------------------------------------


class TestExperience:
    def test_calls_close_together_are_one_episode(self):
        entries = [entry(minutes=0), entry(minutes=1), entry(minutes=2)]
        assert len(experience.episodes(entries)) == 1

    def test_a_long_gap_starts_a_new_episode(self):
        entries = [entry(minutes=0), entry(minutes=60)]
        assert len(experience.episodes(entries)) == 2

    def test_a_short_argument_is_part_of_the_habit(self):
        """open_app(spotify) and open_app(chrome) are two habits, not one."""
        signature = experience.signature(entry("open_app", {"name": "Spotify"}))
        assert signature == "open_app(name=spotify)"

    def test_a_long_argument_is_reduced_to_its_key(self):
        """A path names one job, not a recurring kind of job."""
        long_path = "/Users/someone/very/long/path/" + "x" * 60
        signature = experience.signature(entry("read_file", {"path": long_path}))
        assert signature == "read_file(path)"

    def test_something_done_often_enough_is_a_habit(self):
        entries = [entry("open_app", {"name": "Spotify"}) for _ in range(4)]
        found = experience.habits(entries, threshold=3)
        assert [(h.signature, h.count) for h in found] == [("open_app(name=spotify)", 4)]

    def test_something_done_twice_is_not_yet_a_habit(self):
        entries = [entry("open_app", {"name": "Spotify"}) for _ in range(2)]
        assert experience.habits(entries, threshold=3) == []

    def test_failures_are_counted_separately(self):
        entries = [entry("run_shell", result="Error: no such file") for _ in range(3)]
        found = experience.habits(entries, threshold=3)
        assert found[0].failures == 3 and found[0].shaky

    def test_a_refusal_counts_as_a_failure(self):
        assert experience.failed(entry(result="The user declined the delete_path call."))

    def test_a_plain_result_is_not_a_failure(self):
        assert not experience.failed(entry(result="Opened Spotify."))

    def test_a_repeated_order_of_skills_is_a_sequence(self):
        entries = []
        for run in range(3):
            base = run * 60  # each run its own episode
            entries += [
                entry("read_file", minutes=base + 2),
                entry("edit_file", minutes=base + 1),
                entry("run_shell", minutes=base),
            ]
        found = experience.sequences(entries, threshold=3)
        assert ("read_file", "edit_file") in [window for window, _count in found]

    def test_a_skill_retried_back_to_back_is_not_a_sequence(self):
        """Retrying is not a procedure."""
        entries = [entry("run_shell", minutes=i) for i in range(6)]
        assert experience.sequences(entries, threshold=2) == []


# --- The review -------------------------------------------------------------


class TestParse:
    def test_a_plain_json_object(self):
        assert parse('{"name":"a","task":"do it","steps":"1. x"}')["task"] == "do it"

    def test_a_fenced_object(self):
        reply = '```json\n{"name":"a","task":"do it","steps":"1. x"}\n```'
        assert parse(reply)["steps"] == "1. x"

    def test_an_object_wrapped_in_prose(self):
        """Small local models narrate. A review that only accepts bare JSON
        learns nothing from them."""
        reply = 'Sure! Here you go:\n{"name":"a","task":"do it","steps":"1. x"}\nHope that helps'
        assert parse(reply)["name"] == "a"

    def test_skip_means_nothing_is_written(self):
        assert parse("SKIP") is None

    def test_an_object_with_no_steps_is_not_a_procedure(self):
        assert parse('{"name":"a","task":"do it","steps":"  "}') is None

    def test_unparseable_output_is_skipped_rather_than_raised(self):
        assert parse("I'm afraid I can't do that") is None
        assert parse(None) is None


class TestReview:
    def _audit(self, count=4):
        return _Audit([entry("open_app", {"name": "Spotify"}) for _ in range(count)])

    def test_a_recurring_habit_becomes_a_playbook(self, store):
        reply = '{"name":"open-spotify","task":"open Spotify","steps":"1. open_app Spotify"}'
        result = review(self._audit(), store, lambda _p: reply, threshold=3)
        assert result.created == ["open-spotify"]
        assert store.get("open-spotify").task == "open Spotify"

    def test_nothing_recurring_means_nothing_written(self, store):
        result = review(self._audit(count=1), store, lambda _p: "x", threshold=3)
        assert not result.changed and store.all() == []

    def test_seeing_the_same_work_again_revises_rather_than_duplicates(self, store):
        store.save("open-spotify", "open Spotify", "1. old")
        reply = '{"name":"whatever","task":"open Spotify","steps":"1. new"}'
        result = review(self._audit(), store, lambda _p: reply, threshold=3)
        assert result.revised == ["open-spotify"]
        assert result.created == []
        assert store.get("open-spotify").steps == "1. new"
        assert len(store.all()) == 1

    def test_a_revision_keeps_the_name_the_store_knows(self, store):
        """The name is what forget and restore address, so the model's opinion
        of it must not win."""
        store.save("open-spotify", "open Spotify", "1. old")
        reply = '{"name":"a-totally-different-name","task":"open Spotify","steps":"1. new"}'
        review(self._audit(), store, lambda _p: reply, threshold=3)
        assert [b.name for b in store.all()] == ["open-spotify"]

    def test_one_review_writes_only_a_few(self, store):
        """A first run over a long log must not produce a dozen procedures
        nobody asked for."""
        entries = []
        for index in range(8):
            entries += [entry("open_app", {"name": f"app{index}"}) for _ in range(3)]
        reply = '{"name":"n","task":"open the app","steps":"1. x"}'
        result = review(_Audit(entries), store, lambda _p: reply, threshold=3, max_new=3)
        assert len(result.created) + len(result.revised) == 3

    def test_a_model_that_fails_does_not_break_the_review(self, store):
        def boom(_prompt):
            raise RuntimeError("provider down")

        result = review(self._audit(), store, boom, threshold=3)
        assert not result.changed
        assert isinstance(result, Learned)

    def test_a_model_that_answers_nonsense_writes_nothing(self, store):
        result = review(self._audit(), store, lambda _p: "no idea sorry", threshold=3)
        assert not result.changed and store.all() == []

    def test_no_audit_log_at_all_is_not_an_error(self, store):
        assert review(None, store, lambda _p: "x").considered == 0

    def test_repeated_failures_are_surfaced_as_a_nudge(self, store):
        entries = [entry("run_shell", result="Error: nope") for _ in range(4)]
        result = review(_Audit(entries), store, lambda _p: "SKIP", threshold=3)
        assert any("failed" in nudge for nudge in result.nudges)

    def test_the_evidence_names_what_actually_happened(self):
        habit = experience.habits(
            [entry("open_app", {"name": "Spotify"}) for _ in range(3)], threshold=3
        )[0]
        text = evidence_for(habit)
        assert "open_app(name=spotify)" in text and "Times run: 3" in text


class TestNothingExecutes:
    def test_the_review_is_given_no_way_to_run_anything(self, store):
        """A review reads a log full of commands Nero has run. The prompt is a
        string and `ask` returns a string; there is no path from here to a
        skill call."""
        seen = []
        entries = [entry("run_shell", {"command": "rm -rf /tmp/x"}) for _ in range(3)]
        review(_Audit(entries), store, lambda prompt: seen.append(prompt) or "SKIP", threshold=3)
        assert seen and all(isinstance(prompt, str) for prompt in seen)

    def test_a_playbook_reaches_the_prompt_as_text(self, store):
        """Whatever a playbook says, it arrives as characters in a message.
        Acting on it still means a skill call, with its own gate."""
        store.save("dangerous", "clean the build directory", "1. rm -rf build")
        block = playbook_block(store, "clean the build directory")
        assert isinstance(block, str)
        assert "rm -rf build" in block


class TestRelevantFacts:
    """Every fact used to ride on every turn. Next to the question that is a
    distraction: qwen3.5:2b answered a seven-fact block instead of the command
    in front of it."""

    FACTS = [
        ("brother_name", "Somansh"),
        ("favorite_color", "blue"),
        ("favorite_day_of_week", "Thursday"),
    ]

    def _relevant(self, text):
        from nero.memory.facts import relevant

        return dict(relevant(self.FACTS, text))

    def test_a_question_finds_the_fact_it_is_about(self):
        assert self._relevant("who is my brother") == {"brother_name": "Somansh"}

    def test_a_fact_is_matched_on_its_value_too(self):
        assert "brother_name" in self._relevant("tell me about Somansh")

    def test_a_command_matches_nothing(self):
        assert self._relevant("skip this track") == {}
        assert self._relevant("pause the music") == {}

    def test_a_greeting_matches_nothing(self):
        assert self._relevant("hi") == {}

    def test_an_underscored_key_still_matches_ordinary_words(self):
        assert "favorite_day_of_week" in self._relevant("what is my favorite day")

    def test_no_facts_at_all_is_not_an_error(self):
        from nero.memory.facts import relevant

        assert relevant([], "who is my brother") == []
