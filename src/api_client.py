"""
Sentilyze API Client Bridge.
Provides a unified interface for Streamlit or client scripts to fetch portfolio,
market sessions, and trade telemetry from the FastAPI backend or local fallback files.
"""

import os
import json
import csv
from typing import Dict, Any, Optional, List
import requests
from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_API_URL = os.getenv("SENTILYZE_API_URL", "").rstrip("/")
PORTFOLIO_FILE = os.path.join("results", "paper_portfolio.json")
TRADES_FILE = os.path.join("results", "executed_trades.csv")


class SentilyzeClient:
    """
    Client for interacting with the Sentilyze Quant Engine API.
    Falls back gracefully to local files if API is unreachable or unset.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = (base_url or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key or os.getenv("SENTILYZE_API_KEY")
        self.headers = {"X-API-KEY": self.api_key} if self.api_key else {}

    @property
    def is_api_configured(self) -> bool:
        return bool(self.base_url and self.base_url.startswith("http"))

    def get_portfolio(self) -> Dict[str, Any]:
        """Fetches live portfolio state."""
        if self.is_api_configured:
            try:
                resp = requests.get(
                    f"{self.base_url}/api/v1/portfolio",
                    headers=self.headers,
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                logger.debug(
                    f"API call to /api/v1/portfolio failed ({e}), falling back to local file."
                )

        # Local fallback
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {
                        "status": "SUCCESS",
                        "total_equity": data.get("total_equity", 100000.0),
                        "cash": data.get("cash", 100000.0),
                        "unrealized_pnl": data.get("unrealized_pnl", 0.0),
                        "realized_pnl": data.get("realized_pnl", 0.0),
                        "win_rate": data.get("win_rate", 0.0),
                        "total_trades": data.get("total_trades", 0),
                        "open_positions": data.get("open_positions", {}),
                        "last_updated": data.get("last_updated"),
                    }
            except Exception as e:
                logger.error(f"Error reading local portfolio: {e}")

        return {
            "status": "FALLBACK",
            "total_equity": 100000.0,
            "cash": 100000.0,
            "open_positions": {},
        }

    def get_sessions(self) -> Dict[str, Any]:
        """Fetches global market sessions."""
        if self.is_api_configured:
            try:
                resp = requests.get(
                    f"{self.base_url}/api/v1/sessions",
                    headers=self.headers,
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    return resp.json().get("markets", {})
            except Exception as e:
                logger.debug(
                    f"API call to /api/v1/sessions failed ({e}), falling back to internal module."
                )

        from src.market_session import get_all_global_market_statuses

        return get_all_global_market_statuses()

    def get_trades(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetches historical executed trades."""
        if self.is_api_configured:
            try:
                resp = requests.get(
                    f"{self.base_url}/api/v1/trades?limit={limit}&offset={offset}",
                    headers=self.headers,
                    timeout=3.0,
                )
                if resp.status_code == 200:
                    return resp.json().get("trades", [])
            except Exception as e:
                logger.debug(
                    f"API call to /api/v1/trades failed ({e}), falling back to local CSV."
                )

        trades: List[Dict[str, Any]] = []
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trades.append(row)
                return trades[offset : offset + limit]
            except Exception:
                pass
        return trades
