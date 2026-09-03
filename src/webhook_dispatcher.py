"""
Automated Broker Webhooks & API Order Dispatcher for Sentilyze.
Enables secure transmission of algorithmic buy/sell/scale-out execution orders
to external brokerage APIs (Alpaca, Interactive Brokers, Custom REST Webhooks)
using cryptographic HMAC SHA-256 payload verification.
"""

from typing import Any, Dict, Optional
import os
import hmac
import hashlib
import json
import time
import requests
from datetime import datetime, timezone
from src.utils import get_logger

logger = get_logger(__name__)

WEBHOOK_CONFIG_FILE = "results/webhook_config.json"
WEBHOOK_AUDIT_LOG = "results/webhook_audit_log.json"


def load_webhook_config() -> Dict[str, Any]:
    """Loads configured external broker endpoints and secret keys."""
    if os.path.exists(WEBHOOK_CONFIG_FILE):
        try:
            with open(WEBHOOK_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "webhook_url": "https://api.your-broker.com/v2/orders",
        "broker_name": "Custom REST Webhook",
        "api_key_header": "X-API-KEY",
        "api_secret_header": "X-API-SECRET",
        "enabled": False,
        "environment": "PAPER_TRADING",
        "hmac_secret": "",
    }


def save_webhook_config(config: Dict[str, Any]) -> bool:
    """Persists external broker endpoint configuration."""
    try:
        os.makedirs(os.path.dirname(WEBHOOK_CONFIG_FILE), exist_ok=True)
        with open(WEBHOOK_CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save webhook config: {e}")
        return False


def generate_hmac_signature(payload_str: str, secret: str) -> str:
    """Generates cryptographic HMAC-SHA256 signature for payload verification."""
    return hmac.new(
        secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def format_broker_order_payload(
    ticker: str,
    action: str,
    shares: int,
    price: float,
    tp1_target: float,
    tp2_target: float,
    sl_target: float,
    order_type: str = "limit",
) -> Dict[str, Any]:
    """
    Formats institutional bracket order payload ready for Alpaca / IBKR webhook transmission.
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    return {
        "client_order_id": f"SENTILYZE_{ticker}_{int(time.time())}",
        "symbol": ticker,
        "qty": shares,
        "side": "buy" if action in ["BUY", "EXECUTE_BUY", "SCALE_IN"] else "sell",
        "type": order_type,
        "time_in_force": "day",
        "limit_price": round(price, 2),
        "order_class": "bracket",
        "take_profit": {
            "limit_price": round(tp1_target, 2),
        },
        "stop_loss": {
            "stop_price": round(sl_target, 2),
        },
        "meta": {
            "source": "Sentilyze Multi-Agent Engine",
            "tp2_runner_target": round(tp2_target, 2),
            "timestamp": now_utc,
        },
    }


def dispatch_order_webhook(
    order_payload: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    is_test: bool = False,
) -> Dict[str, Any]:
    """
    Dispatches order payload to the configured external webhook endpoint.
    """
    cfg = config or load_webhook_config()
    url = cfg.get("webhook_url", "")
    secret = cfg.get("hmac_secret", "sentilyze_key")

    payload_str = json.dumps(order_payload, sort_keys=True)
    signature = generate_hmac_signature(payload_str, secret)

    headers = {
        "Content-Type": "application/json",
        "X-Sentilyze-Signature": signature,
        "X-Sentilyze-Timestamp": str(int(time.time())),
    }

    if is_test:
        return {
            "status": "SIMULATED_SUCCESS",
            "status_code": 200,
            "url": url,
            "headers": headers,
            "payload": order_payload,
            "signature": signature,
            "message": "Test ping simulated successfully with HMAC-SHA256 signature.",
        }

    try:
        response = requests.post(url, json=order_payload, headers=headers, timeout=5)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": order_payload.get("symbol"),
            "action": order_payload.get("side"),
            "status_code": response.status_code,
            "response": response.text[:200],
        }
        _append_audit_log(log_entry)
        return {
            "status": (
                "SUCCESS" if response.status_code in [200, 201, 202] else "FAILED"
            ),
            "status_code": response.status_code,
            "response": response.text,
        }
    except Exception as e:
        logger.warning(f"Webhook dispatch error: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
        }


def _append_audit_log(entry: Dict[str, Any]):
    """Appends dispatched webhook record to audit trail."""
    try:
        logs = []
        if os.path.exists(WEBHOOK_AUDIT_LOG):
            with open(WEBHOOK_AUDIT_LOG, "r") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(WEBHOOK_AUDIT_LOG, "w") as f:
            json.dump(logs[-50:], f, indent=2)
    except Exception:
        pass
