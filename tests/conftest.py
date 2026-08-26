import pytest


@pytest.fixture(autouse=True)
def isolate_audit_log(tmp_path, monkeypatch):
    """No test may write to the developer's real audit database."""
    fake = tmp_path / "audit.db"
    monkeypatch.setattr("nero.core.audit_log.default_audit_path", lambda: fake)
    monkeypatch.setattr("nero.cli.default_audit_path", lambda: fake, raising=False)
    monkeypatch.setattr(
        "nero.memory.history_store.default_history_path", lambda: tmp_path / "history.db"
    )
    monkeypatch.setattr(
        "nero.cli.default_history_path", lambda: tmp_path / "history.db", raising=False
    )
    monkeypatch.setattr(
        "nero.memory.facts.default_facts_path", lambda: tmp_path / "facts.db"
    )
    monkeypatch.setattr(
        "nero.cli.default_facts_path", lambda: tmp_path / "facts.db", raising=False
    )
    monkeypatch.setattr(
        "nero.memory.notes.default_notes_index_path", lambda: tmp_path / "notes_index.db"
    )
    monkeypatch.setattr(
        "nero.cli.default_notes_index_path", lambda: tmp_path / "notes_index.db", raising=False
    )
    return fake


@pytest.fixture(autouse=True)
def isolate_keyring(monkeypatch):
    """No test may write to the developer's real OS keychain.

    A menu test reached _switch_provider and stored a live `mistral_api_key`
    before this existed. Autouse because the failure is silent — the suite
    passes either way and the damage lands outside the repo.
    """
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(
        "keyring.set_password",
        lambda service, entry, value: store.__setitem__((service, entry), value),
    )
    monkeypatch.setattr("keyring.get_password", lambda service, entry: store.get((service, entry)))
    return store
