import os
import requests
import json
from typing import Dict, List, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)

ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_BASE_URL = "https://api.alpaca.markets"


class AlpacaBrokerBridge:
    """
    Institutional Alpaca Brokerage Execution Bridge for Paper & Live Trading.
    Supports automated Bracket Orders (Buy Market + Take-Profit Limit + Stop-Loss Stop).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        is_paper: bool = True,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        raw_url = base_url or os.getenv("ALPACA_BASE_URL", (ALPACA_PAPER_BASE_URL if is_paper else ALPACA_LIVE_BASE_URL))
        # Normalize URL by removing trailing slash and /v2 if present
        self.base_url = raw_url.rstrip("/").removesuffix("/v2")
        self.is_paper = is_paper
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def is_connected(self) -> bool:
        """Verifies active connection to Alpaca Brokerage API."""
        if not self.api_key or not self.secret_key:
            return False
        try:
            res = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=6)
            return res.status_code == 200
        except Exception:
            return False

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches live Alpaca account equity, buying power, and cash."""
        if not self.is_connected():
            return {
                "status": "DISCONNECTED",
                "equity": 100000.0,
                "cash": 100000.0,
                "buying_power": 200000.0,
                "currency": "USD",
                "mode": "PAPER (Simulated)",
            }
        try:
            res = requests.get(f"{self.base_url}/v2/account", headers=self.headers, timeout=6)
            if res.status_code == 200:
                data = res.json()
                return {
                    "status": "CONNECTED",
                    "equity": float(data.get("equity", 0)),
                    "cash": float(data.get("cash", 0)),
                    "buying_power": float(data.get("buying_power", 0)),
                    "currency": data.get("currency", "USD"),
                    "mode": "ALPACA PAPER" if self.is_paper else "ALPACA LIVE",
                    "account_number": data.get("account_number", "N/A"),
                }
        except Exception as e:
            logger.error(f"Alpaca account fetch error: {e}")

        return {"status": "ERROR"}

    def submit_bracket_order(
        self,
        ticker: str,
        qty: int,
        take_profit_price: float,
        stop_loss_price: float,
        side: str = "buy",
    ) -> Dict[str, Any]:
        """
        Submits an institutional Bracket Order:
        - Entry: Market order
        - Exit 1: Limit order @ Take-Profit
        - Exit 2: Stop order @ Stop-Loss
        """
        if not self.is_connected():
            logger.info(f"[ALPACA SIMULATED] Bracket order for {qty} {ticker} @ TP ${take_profit_price:.2f} / SL ${stop_loss_price:.2f}")
            return {
                "status": "SIMULATED_SUCCESS",
                "ticker": ticker,
                "qty": qty,
                "tp": take_profit_price,
                "sl": stop_loss_price,
            }

        payload = {
            "symbol": ticker,
            "qty": qty,
            "side": side,
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "take_profit": {"limit_price": round(take_profit_price, 2)},
            "stop_loss": {"stop_price": round(stop_loss_price, 2)},
        }

        try:
            res = requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=payload, timeout=8)
            if res.status_code in [200, 201]:
                order_data = res.json()
                logger.info(f"✅ [ALPACA LIVE] Bracket order submitted: {ticker} (Order ID: {order_data.get('id')})")
                return {"status": "SUBMITTED", "order": order_data}
            else:
                logger.warning(f"Alpaca order rejected: {res.text}")
                return {"status": "REJECTED", "details": res.text}
        except Exception as e:
            logger.error(f"Alpaca order execution error: {e}")
            return {"status": "ERROR", "error": str(e)}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches active positions from Alpaca brokerage."""
        if not self.is_connected():
            return []
        try:
            res = requests.get(f"{self.base_url}/v2/positions", headers=self.headers, timeout=6)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.error(f"Alpaca positions error: {e}")
        return []
