from datetime import UTC, datetime

from typer.testing import CliRunner

from nero import cli
from nero.core.audit_log import AuditEntry, AuditLog

runner = CliRunner()


def seed(path, count=3):
    log = AuditLog(path)
    for index in range(count):
        log.record(
            AuditEntry(
                timestamp=datetime(2026, 7, 22, 10, index, tzinfo=UTC),
                skill_name=f"skill_{index}",
                arguments={"q": f"value_{index}"},
                result_summary=f"result {index}",
                provider="claude",
            )
        )
    return log


class TestHistoryCommand:
    def test_empty_log_says_so(self, isolate_audit_log):
        result = runner.invoke(cli.app, ["history"])
        assert result.exit_code == 0
        assert "No skill invocations" in result.stdout

    def test_lists_entries(self, isolate_audit_log):
        seed(isolate_audit_log)
        result = runner.invoke(cli.app, ["history"])
        assert result.exit_code == 0
        assert "skill_0" in result.stdout
        assert "skill_2" in result.stdout
        assert "claude" in result.stdout

    def test_limit_option(self, isolate_audit_log):
        seed(isolate_audit_log, count=5)
        result = runner.invoke(cli.app, ["history", "--limit", "2"])
        assert result.exit_code == 0
        assert "skill_4" in result.stdout
        assert "skill_0" not in result.stdout
