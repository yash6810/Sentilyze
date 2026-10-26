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


@pytest.fixture(autouse=True)
def isolate_news_cache(monkeypatch, tmp_path):
    """
    Autouse fixture that isolates the daily news cache directory to a temporary path,
    preventing any test from writing or reading files in results/.
    """
    test_cache_dir = tmp_path / "test_news_cache"
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NEWS_CACHE_DIR", str(test_cache_dir))
    monkeypatch.setenv("SENTILYZE_NEWS_CACHE_DIR", str(test_cache_dir))


@pytest.fixture(autouse=True)
def isolate_funnel_dir(monkeypatch, tmp_path):
    """
    Autouse fixture that points FUNNEL_DIR to tmp_path, preventing any test
    from writing funnel.json, funnel.md, or funnel_log.jsonl to results/.
    """
    test_funnel_dir = tmp_path / "test_funnel"
    test_funnel_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FUNNEL_DIR", str(test_funnel_dir))
