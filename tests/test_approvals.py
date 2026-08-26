import asyncio

from typer.testing import CliRunner

from nero import cli
from nero.config.manager import ConfigManager
from nero.config.schema import NeroConfig
from nero.core.approvals import ApprovalQueue, queue_confirm
from nero.skills.base import Skill, SkillMeta
from nero.skills.registry import SkillRegistry

runner = CliRunner()


def _queue(tmp_path):
    return ApprovalQueue(tmp_path / "approvals.db")


class TestQueueRoundTrip:
    def test_record_then_pending(self, tmp_path):
        queue = _queue(tmp_path)
        entry_id = queue.record("morning", "delete_path", {"path": "/tmp/x"})
        items = queue.pending()
        assert len(items) == 1
        assert items[0].id == entry_id
        assert items[0].routine == "morning"
        assert items[0].skill == "delete_path"
        assert items[0].arguments == {"path": "/tmp/x"}

    def test_get_by_id(self, tmp_path):
        queue = _queue(tmp_path)
        entry_id = queue.record("morning", "delete_path", {"path": "/tmp/x"})
        item = queue.get(entry_id)
        assert item is not None
        assert item.id == entry_id

    def test_get_missing_id_is_none(self, tmp_path):
        queue = _queue(tmp_path)
        assert queue.get(999) is None

    def test_discard_removes_it(self, tmp_path):
        queue = _queue(tmp_path)
        entry_id = queue.record("morning", "delete_path", {"path": "/tmp/x"})
        assert queue.discard(entry_id) is True
        assert queue.pending() == []

    def test_discard_missing_id_returns_false(self, tmp_path):
        queue = _queue(tmp_path)
        assert queue.discard(999) is False

    def test_clear_empties_the_queue(self, tmp_path):
        queue = _queue(tmp_path)
        queue.record("morning", "delete_path", {"path": "/tmp/x"})
        queue.record("evening", "delete_path", {"path": "/tmp/y"})
        queue.clear()
        assert queue.pending() == []

    def test_pending_is_empty_on_a_fresh_queue(self, tmp_path):
        assert _queue(tmp_path).pending() == []


class TestQueueConfirm:
    def test_returns_false_and_records(self, tmp_path):
        queue = _queue(tmp_path)
        confirm = queue_confirm(queue, "morning")
        result = confirm("delete_path", "destructive", {"path": "/tmp/x"})
        assert result is False
        items = queue.pending()
        assert len(items) == 1
        assert items[0].routine == "morning"
        assert items[0].skill == "delete_path"


class _FakeDestructiveSkill(Skill):
    meta = SkillMeta(
        name="fake_delete",
        description="fake",
        input_schema={"type": "object", "properties": {}},
        requires_network=False,
        permission_tier="destructive",
    )

    async def execute(self, **kwargs) -> str:
        return "deleted"


class TestQueueConfirmThroughRegistry:
    def test_destructive_call_under_queue_confirm_is_refused_and_queues_exactly_once(
        self, tmp_path
    ):
        queue = _queue(tmp_path)
        registry = SkillRegistry(
            skills=[_FakeDestructiveSkill()],
            confirm=queue_confirm(queue, "morning"),
        )
        result = asyncio.run(registry.execute("fake_delete", {}))
        assert "declined" in result.lower()
        items = queue.pending()
        assert len(items) == 1
        assert items[0].skill == "fake_delete"


def _cli_queue(tmp_path):
    # Same path the autouse isolate_audit_log fixture patches nero.cli's
    # default_approvals_path to — this is how CLI-invoked commands and the
    # test see the same queue.
    return ApprovalQueue(tmp_path / "approvals.db")


class TestApprovalsListCLI:
    def test_empty_queue_says_so(self, tmp_path):
        result = runner.invoke(cli.app, ["approvals"])
        assert result.exit_code == 0
        assert "no actions waiting" in result.stdout.lower()

    def test_pending_items_show_in_a_table(self, tmp_path):
        _cli_queue(tmp_path).record("morning", "delete_path", {"path": "/tmp/x"})
        result = runner.invoke(cli.app, ["approvals"])
        assert result.exit_code == 0
        assert "morning" in result.stdout
        assert "delete_path" in result.stdout


class TestApprovalsRunCLI:
    def test_non_tty_refuses_and_leaves_row_pending(self, tmp_path, monkeypatch):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        queue = _cli_queue(tmp_path)
        entry_id = queue.record("morning", "delete_path", {"path": "/tmp/does-not-exist-x"})
        result = runner.invoke(cli.app, ["approvals", "run", str(entry_id)])
        assert "not approved" in result.stdout.lower() or "declined" in result.stdout.lower()
        # Never auto-approved: the row is still there.
        assert queue.get(entry_id) is not None

    def test_unknown_id_exits_1(self, tmp_path, monkeypatch):
        manager = ConfigManager(config_dir=tmp_path)
        manager.save(NeroConfig())
        monkeypatch.setattr(cli, "ConfigManager", lambda: manager)
        result = runner.invoke(cli.app, ["approvals", "run", "999"])
        assert result.exit_code == 1


class TestApprovalsDiscardCLI:
    def test_discards_a_pending_row(self, tmp_path):
        entry_id = _cli_queue(tmp_path).record("morning", "delete_path", {"path": "/tmp/x"})
        result = runner.invoke(cli.app, ["approvals", "discard", str(entry_id)])
        assert result.exit_code == 0
        assert "discarded" in result.stdout.lower()
        assert _cli_queue(tmp_path).get(entry_id) is None

    def test_discard_missing_id_says_so(self, tmp_path):
        result = runner.invoke(cli.app, ["approvals", "discard", "999"])
        assert result.exit_code == 0
        assert "no pending approval" in result.stdout.lower()


class TestChatStartNotice:
    def test_notice_prints_only_when_queue_non_empty(self, tmp_path, capsys):
        from nero.cli import _print_pending_approvals_notice

        _print_pending_approvals_notice()
        assert "waiting for your approval" not in capsys.readouterr().out.lower()

        _cli_queue(tmp_path).record("morning", "delete_path", {"path": "/tmp/x"})
        _print_pending_approvals_notice()
        assert "waiting for your approval" in capsys.readouterr().out.lower()
