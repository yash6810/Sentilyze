import pytest
from src.alerts import (
    send_discord_wavelet_alert,
    send_discord_finjepa_alert,
    send_discord_jev_catalyst_alert,
    send_discord_crypto_arbitrage_alert,
    send_discord_sympathy_alert,
)


def test_send_discord_wavelet_alert():
    data = {
        "veto_active": True,
        "entropy": 0.89,
        "d1_energy_pct": 52.3,
        "a3_energy_pct": 14.1,
        "regime": "HIGH_ENTROPY_CHOP",
    }
    # With no webhook URL provided and none in env, returns False cleanly without crashing
    res = send_discord_wavelet_alert("NVDA", data, webhook_url="")
    assert res is False


def test_send_discord_finjepa_alert():
    data = {
        "cro_veto": True,
        "latent_energy": 0.0452,
        "dispersion_sigma": 2.85,
        "physics_state": "LIQUIDITY_DROUGHT_HAZARD",
    }
    res = send_discord_finjepa_alert("AAPL", data, webhook_url="")
    assert res is False


def test_send_discord_jev_catalyst_alert():
    verdict = {
        "conviction_tier": "TIER_1_SUPER_CATALYST",
        "event_type": "FDA_APPROVAL",
        "kelly_multiplier": 1.40,
        "latency_ms": 3.45,
    }
    res = send_discord_jev_catalyst_alert(
        "LLY", verdict, "FDA grants breakthrough designation", webhook_url=""
    )
    assert res is False


def test_send_discord_crypto_arbitrage_alert():
    data = {
        "symbol": "BTCUSDT",
        "opportunity_type": "SHORT_SQUEEZE_EXPANSION",
        "funding_rate": -0.00025,
        "annualized_rate_pct": -27.38,
        "action": "LONG_PERP_ARBITRAGE",
    }
    res = send_discord_crypto_arbitrage_alert(data, webhook_url="")
    assert res is False


def test_send_discord_sympathy_alert():
    data = {
        "impulse": 2.45,
        "expected_move_pct": 1.85,
        "delay_tau": 1,
    }
    res = send_discord_sympathy_alert("NVDA", "AMD", data, webhook_url="")
    assert res is False
