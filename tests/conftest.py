import pytest


@pytest.fixture(autouse=True)
def block_external_alerts(monkeypatch):
    """
    Autouse fixture that prevents tests from sending real outbound network calls
    to Discord webhooks or SMTP email servers during test execution.
    """
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("EMAIL_USER", raising=False)
    monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
    monkeypatch.delenv("EMAIL_RECIPIENT", raising=False)
