import pytest


@pytest.fixture(autouse=True)
def isolate_audit_log(tmp_path, monkeypatch):
    """No test may write to the developer's real audit database."""
    fake = tmp_path / "audit.db"
    monkeypatch.setattr("nero.core.audit_log.default_audit_path", lambda: fake)
    monkeypatch.setattr("nero.cli.default_audit_path", lambda: fake, raising=False)
    return fake
