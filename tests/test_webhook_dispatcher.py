import pytest
from src.webhook_dispatcher import (
    load_webhook_config,
    save_webhook_config,
    generate_hmac_signature,
    format_broker_order_payload,
    dispatch_order_webhook,
)


def test_format_broker_order_payload():
    payload = format_broker_order_payload(
        ticker="NVDA",
        action="BUY",
        shares=20,
        price=130.50,
        tp1_target=140.00,
        tp2_target=150.00,
        sl_target=125.00,
    )
    assert payload["symbol"] == "NVDA"
    assert payload["side"] == "buy"
    assert payload["qty"] == 20
    assert payload["order_class"] == "bracket"


def test_generate_hmac_signature():
    sig = generate_hmac_signature("test_payload_string", "secret_key_123")
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA-256 hex length


def test_dispatch_order_webhook_simulation():
    payload = format_broker_order_payload(
        ticker="AAPL",
        action="BUY",
        shares=10,
        price=220.00,
        tp1_target=230.00,
        tp2_target=240.00,
        sl_target=215.00,
    )
    res = dispatch_order_webhook(payload, is_test=True)
    assert res["status"] == "SIMULATED_SUCCESS"
    assert res["status_code"] == 200
