from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nero.core.audit_log import AuditEntry, AuditLog, summarize


def entry(skill_name="open_app", result="Opened Safari.", provider="claude", **arguments):
    return AuditEntry(
        timestamp=datetime.now(UTC),
        skill_name=skill_name,
        arguments=arguments or {"app_name": "Safari"},
        result_summary=result,
        provider=provider,
    )


@pytest.fixture
def log(tmp_path):
    return AuditLog(tmp_path / "audit.db")


class TestRoundtrip:
    def test_record_then_read(self, log):
        assert log.record(entry()) is True
        entries = log.recent()
        assert len(entries) == 1
        assert entries[0].skill_name == "open_app"
        assert entries[0].arguments == {"app_name": "Safari"}
        assert entries[0].result_summary == "Opened Safari."
        assert entries[0].provider == "claude"

    def test_empty_log_returns_nothing(self, log):
        assert log.recent() == []

    def test_newest_first(self, log):
        log.record(entry(skill_name="first"))
        log.record(entry(skill_name="second"))
        log.record(entry(skill_name="third"))
        assert [e.skill_name for e in log.recent()] == ["third", "second", "first"]

    def test_limit_is_respected(self, log):
        for index in range(5):
            log.record(entry(skill_name=f"skill_{index}"))
        assert len(log.recent(limit=2)) == 2

    def test_creates_parent_directory(self, tmp_path):
        log = AuditLog(tmp_path / "nested" / "deeper" / "audit.db")
        assert log.record(entry()) is True
        assert len(log.recent()) == 1

    def test_arguments_stored_verbatim_not_redacted(self, log):
        log.record(entry(location="Reykjavík"))
        assert log.recent()[0].arguments == {"location": "Reykjavík"}


class TestAuditEntrySchema:
    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValidationError):
            AuditEntry(
                timestamp=datetime.now(UTC),
                skill_name="open_app",
                arguments={"app_name": "Safari"},
                result_summary="Opened Safari.",
                provider="claude",
                bogus=1,
            )


class TestSummarize:
    def test_short_text_unchanged(self):
        assert summarize("Opened Safari.") == "Opened Safari."

    def test_truncates_to_limit(self):
        assert len(summarize("x" * 500)) == 200

    def test_collapses_whitespace(self):
        assert summarize("a\n\n  b\tc") == "a b c"

    def test_handles_empty(self):
        assert summarize("") == ""


class TestFailureIsolation:
    def test_unwritable_path_returns_false_not_raises(self, tmp_path):
        # A file where the parent directory should be: mkdir will fail.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        log = AuditLog(blocker / "audit.db")
        assert log.record(entry()) is False

    def test_unreadable_log_returns_empty_not_raises(self, tmp_path):
        corrupt = tmp_path / "audit.db"
        corrupt.write_text("this is not a sqlite database at all")
        assert AuditLog(corrupt).recent() == []

    def test_unserialisable_arguments_return_false_not_raises(self, log):
        # `record` promises it never raises. json.dumps raises TypeError — not
        # sqlite3.Error or OSError — on a value it can't encode, so the promise
        # only holds if TypeError is caught too.
        assert log.record(entry(thing=object())) is False

    def test_corrupt_row_arguments_do_not_break_reading(self, log):
        import sqlite3

        log.record(entry())
        connection = sqlite3.connect(log.path)
        with connection:
            connection.execute("UPDATE invocations SET arguments = '{not json'")
        connection.close()
        # The row is still worth showing; only its arguments are unreadable.
        entries = log.recent()
        assert len(entries) == 1
        assert entries[0].arguments == {}
        assert entries[0].skill_name == "open_app"

    def test_corrupt_timestamp_does_not_break_reading(self, log):
        import sqlite3

        # A good row either side of the corrupt one, so we can also confirm
        # only the bad row is dropped rather than the whole log going empty.
        log.record(entry(skill_name="before"))
        log.record(entry(skill_name="corrupt"))
        log.record(entry(skill_name="after"))
        connection = sqlite3.connect(log.path)
        with connection:
            connection.execute(
                "UPDATE invocations SET timestamp = 'not-a-timestamp' "
                "WHERE skill_name = 'corrupt'"
            )
        connection.close()
        entries = log.recent()
        assert [e.skill_name for e in entries] == ["after", "before"]
