import os
import requests
import json
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from src.utils import get_logger
from src.market_session import is_crypto_asset

load_dotenv()
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
        self.api_key = (
            api_key
            if api_key is not None
            else (os.getenv("ALPACA_API_KEY", "") or os.getenv("APCA_API_KEY_ID", ""))
        )
        self.secret_key = (
            secret_key
            if secret_key is not None
            else (
                os.getenv("ALPACA_SECRET_KEY", "")
                or os.getenv("APCA_API_SECRET_KEY", "")
            )
        )
        raw_url = base_url or os.getenv(
            "ALPACA_BASE_URL",
            (ALPACA_PAPER_BASE_URL if is_paper else ALPACA_LIVE_BASE_URL),
        )
        # Normalize URL by removing trailing slash and /v2 if present
        self.base_url = raw_url.rstrip("/").removesuffix("/v2")
        self.is_paper = is_paper
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    def _log_alpaca_warning(
        self,
        method_name: str,
        status_code: Optional[int],
        error_body: str,
    ):
        from urllib.parse import urlparse

        host = urlparse(self.base_url).hostname or self.base_url
        env_keys_status = {
            "ALPACA_API_KEY": bool(os.getenv("ALPACA_API_KEY")),
            "ALPACA_SECRET_KEY": bool(os.getenv("ALPACA_SECRET_KEY")),
            "APCA_API_KEY_ID": bool(os.getenv("APCA_API_KEY_ID")),
            "APCA_API_SECRET_KEY": bool(os.getenv("APCA_API_SECRET_KEY")),
            "api_key_loaded": bool(self.api_key),
            "secret_key_loaded": bool(self.secret_key),
        }
        truncated_body = str(error_body)[:200]
        logger.warning(
            f"[{method_name}] Alpaca connectivity check warning: "
            f"key_env_vars_present={env_keys_status}, "
            f"host='{host}', status_code={status_code}, "
            f"error_body='{truncated_body}'"
        )

    def is_connected(self) -> bool:
        """Verifies active connection to Alpaca Brokerage API."""
        if not self.api_key or not self.secret_key:
            self._log_alpaca_warning(
                "AlpacaBrokerBridge.is_connected",
                status_code=None,
                error_body="Missing API key or Secret key",
            )
            return False
        try:
            res = requests.get(
                f"{self.base_url}/v2/account", headers=self.headers, timeout=6
            )
            if res.status_code != 200:
                self._log_alpaca_warning(
                    "AlpacaBrokerBridge.is_connected",
                    status_code=res.status_code,
                    error_body=res.text,
                )
                return False
            return True
        except Exception as exc:
            self._log_alpaca_warning(
                "AlpacaBrokerBridge.is_connected",
                status_code=None,
                error_body=str(exc),
            )
            return False

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches live Alpaca account equity, buying power, and cash."""
        if not self.is_connected():
            self._log_alpaca_warning(
                "AlpacaBrokerBridge.get_account_summary",
                status_code=None,
                error_body="Broker is not connected",
            )
            return {
                "status": "DISCONNECTED",
                "equity": 100000.0,
                "cash": 100000.0,
                "buying_power": 200000.0,
                "currency": "USD",
                "mode": "PAPER (Simulated)",
            }
        try:
            res = requests.get(
                f"{self.base_url}/v2/account", headers=self.headers, timeout=6
            )
            res.raise_for_status()
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
            resp = getattr(e, "response", None)
            code = resp.status_code if resp is not None else None
            body = (
                resp.text
                if (resp is not None and getattr(resp, "text", None) is not None)
                else str(e)
            )
            self._log_alpaca_warning(
                "AlpacaBrokerBridge.get_account_summary",
                status_code=code,
                error_body=body,
            )
            logger.error(f"Alpaca account fetch error: {e}")
            raise

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
        if is_crypto_asset(ticker):
            logger.warning(
                f"[ALPACA CRYPTO GUARD] Crypto asset order rejected by Alpaca policy: {ticker}"
            )
            return {
                "status": "BLOCKED_CRYPTO_ASSET",
                "ticker": ticker,
                "qty": qty,
                "message": "Alpaca trading is strictly reserved for US equities. Crypto assets are prohibited.",
            }

        allow_orders = (
            os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION", "false").lower()
            in ("true", "1", "yes")
            or "PYTEST_CURRENT_TEST" in os.environ
        )
        if not allow_orders:
            logger.info(
                f"[ALPACA ORDER BLOCKED] Order submission disabled (ALLOW_ALPACA_ORDER_SUBMISSION=false): {ticker}"
            )
            return {
                "status": "BLOCKED_BY_ENV_POLICY",
                "ticker": ticker,
                "qty": qty,
                "message": "Alpaca order submission disabled by environment policy.",
            }

        if not self.is_connected():
            logger.info(
                f"[ALPACA SIMULATED] Bracket order for {qty} {ticker} @ TP ${take_profit_price:.2f} / SL ${stop_loss_price:.2f}"
            )
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
            res = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=payload,
                timeout=8,
            )
            if res.status_code in [200, 201]:
                order_data = res.json()
                logger.info(
                    f"✅ [ALPACA LIVE] Bracket order submitted: {ticker} (Order ID: {order_data.get('id')})"
                )
                return {"status": "SUBMITTED", "order": order_data}
            else:
                logger.warning(f"Alpaca order rejected: {res.text}")
                return {"status": "REJECTED", "details": res.text}
        except Exception as e:
            logger.error(f"Alpaca order execution error: {e}")
            return {"status": "ERROR", "error": str(e)}

    def replace_order_stop_price(
        self, order_id: str, new_stop_price: float
    ) -> Dict[str, Any]:
        """
        Replaces an open stop-loss order on Alpaca with a new, higher stop price.
        Used by the Real-Time Trailing Stop Engine to ratchet protective stops.
        """
        if not self.is_connected():
            logger.info(
                f"[ALPACA SIMULATED] Replaced stop order {order_id} with new price ${new_stop_price:.2f}"
            )
            return {
                "status": "SIMULATED_REPLACED",
                "order_id": order_id,
                "stop_price": new_stop_price,
            }

        payload = {"stop_price": round(new_stop_price, 2)}
        try:
            res = requests.patch(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=self.headers,
                json=payload,
                timeout=8,
            )
            if res.status_code == 200:
                data = res.json()
                logger.info(
                    f"✅ [ALPACA LIVE] Stop order {order_id} replaced to ${new_stop_price:.2f}"
                )
                return {"status": "REPLACED", "order": data}
            else:
                logger.warning(
                    f"Alpaca replace order failed for {order_id}: {res.text}"
                )
                return {"status": "FAILED", "details": res.text}
        except Exception as e:
            logger.error(f"Error replacing Alpaca order {order_id}: {e}")
            return {"status": "ERROR", "error": str(e)}

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches active positions from Alpaca brokerage."""
        if not self.is_connected():
            return []
        try:
            res = requests.get(
                f"{self.base_url}/v2/positions", headers=self.headers, timeout=6
            )
            if res.status_code == 200:
                return res.json()
            res.raise_for_status()
        except Exception as e:
            logger.error(f"Alpaca positions error: {e}")
            raise
        return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Fetches active/pending orders from Alpaca brokerage."""
        if not self.is_connected():
            return []
        try:
            res = requests.get(
                f"{self.base_url}/v2/orders?status=open",
                headers=self.headers,
                timeout=6,
            )
            if res.status_code == 200:
                return res.json()
            res.raise_for_status()
        except Exception as e:
            logger.error(f"Alpaca open orders error: {e}")
            raise
        return []

    def has_open_position_or_order(self, ticker: str) -> bool:
        """Returns True if an open position or open pending order exists for ticker on Alpaca."""
        if not self.is_connected():
            return False
        ticker_clean = ticker.upper().replace("-", "/")
        # 1. Check open positions
        for pos in self.get_open_positions():
            sym = str(pos.get("symbol", "")).upper().replace("-", "/")
            if sym == ticker_clean or sym == ticker.upper():
                return True
        # 2. Check open orders
        for ord in self.get_open_orders():
            sym = str(ord.get("symbol", "")).upper().replace("-", "/")
            status = str(ord.get("status", "")).lower()
            if (sym == ticker_clean or sym == ticker.upper()) and status in (
                "new",
                "accepted",
                "pending_new",
                "partially_filled",
                "held",
                "open",
            ):
                return True
        return False

    def close_position(self, ticker: str) -> Dict[str, Any]:
        """Liquidates an active position on Alpaca and cancels all open child orders."""
        if is_crypto_asset(ticker):
            logger.warning(
                f"[ALPACA CRYPTO GUARD] Crypto position close rejected by Alpaca policy: {ticker}"
            )
            return {
                "status": "BLOCKED_CRYPTO_ASSET",
                "ticker": ticker,
                "message": "Alpaca position management is strictly reserved for US equities.",
            }

        allow_orders = (
            os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION", "false").lower()
            in ("true", "1", "yes")
            or "PYTEST_CURRENT_TEST" in os.environ
        )
        if not allow_orders:
            logger.info(
                f"[ALPACA CLOSE BLOCKED] Position liquidation disabled (ALLOW_ALPACA_ORDER_SUBMISSION=false): {ticker}"
            )
            return {
                "status": "BLOCKED_BY_ENV_POLICY",
                "ticker": ticker,
                "message": "Alpaca position liquidation disabled by environment policy.",
            }

        if not self.is_connected():
            logger.info(f"[ALPACA SIMULATED] Closed position for {ticker}")
            return {"status": "SIMULATED_CLOSED", "ticker": ticker}
        try:
            res = requests.delete(
                f"{self.base_url}/v2/positions/{ticker.upper()}",
                headers=self.headers,
                timeout=8,
            )
            if res.status_code in [200, 204]:
                logger.info(f"✅ [ALPACA LIVE] Closed position for {ticker}")
                return {"status": "CLOSED", "ticker": ticker}
            else:
                logger.warning(f"Notice closing Alpaca position {ticker}: {res.text}")
                return {
                    "status": "FAILED",
                    "ticker": ticker,
                    "details": res.text,
                }
        except Exception as e:
            logger.error(f"Error closing Alpaca position {ticker}: {e}")
            return {"status": "ERROR", "ticker": ticker, "error": str(e)}

    def close_all_positions(self) -> Dict[str, Any]:
        """Emergency liquidator: closes all positions and cancels all open orders."""
        allow_orders = (
            os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION", "false").lower()
            in ("true", "1", "yes")
            or "PYTEST_CURRENT_TEST" in os.environ
        )
        if not allow_orders:
            logger.info(
                "[ALPACA CLOSE ALL BLOCKED] Emergency liquidation disabled (ALLOW_ALPACA_ORDER_SUBMISSION=false)"
            )
            return {
                "status": "BLOCKED_BY_ENV_POLICY",
                "message": "Alpaca emergency liquidation disabled by environment policy.",
            }

        if not self.is_connected():
            logger.info("[ALPACA SIMULATED] Emergency liquidated all positions")
            return {"status": "SIMULATED_LIQUIDATED_ALL"}
        try:
            res = requests.delete(
                f"{self.base_url}/v2/positions?cancel_orders=true",
                headers=self.headers,
                timeout=10,
            )
            if res.status_code in [200, 204, 207]:
                logger.info(
                    "🚨 [ALPACA LIVE] Liquidated all positions and canceled open orders."
                )
                return {"status": "LIQUIDATED_ALL"}
            else:
                return {"status": "FAILED", "details": res.text}
        except Exception as e:
            logger.error(f"Error liquidating all Alpaca positions: {e}")
            return {"status": "ERROR", "error": str(e)}
