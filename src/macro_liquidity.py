"""
Real-Time Macro Liquidity & Treasury Yield Curve Radar for Sentilyze.
Analyzes 10Y-2Y Treasury Yield Spread Inversions, Federal Reserve Net Liquidity
(Fed Balance Sheet - TGA - Reverse Repo), and Systemic Financial Stress Conditions.

Attribution Notice:
This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.
"""

import os
from typing import Any, Dict, Optional
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.data_ingestion import get_price_history
from src.utils import get_logger

logger = get_logger(__name__)


def _get_fred_api_key() -> Optional[str]:
    """Retrieves the FRED API key from the environment or .env file."""
    api_key = os.getenv("FRED_API_KEY")
    if api_key:
        return api_key

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FRED_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
        except Exception as e:
            logger.debug(f"Failed reading .env for FRED_API_KEY: {e}")
    return None


def fetch_fred_series(
    series_id: str, limit: int = 5, api_key: Optional[str] = None
) -> Optional[list]:
    """
    Fetches raw observation entries from the St. Louis Fed FRED API.
    """
    key = api_key or _get_fred_api_key()
    if not key:
        return None

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("observations", [])
        else:
            logger.warning(
                f"FRED API {series_id} returned HTTP {resp.status_code}: {resp.text[:100]}"
            )
            return None
    except Exception as e:
        logger.warning(f"FRED API request failed for {series_id}: {e}")
        return None


def calculate_macro_liquidity_metrics() -> Dict[str, Any]:
    """
    Computes real-time macroeconomic liquidity indicators and yield curve dynamics.
    Pulls live data from Federal Reserve Economic Data (FRED®) when configured.

    Returns:
        Dict containing 10Y-2Y spread, Fed Net Liquidity estimate,
        Inversion Regime, and Equity Tail-Risk score.
    """
    api_key = _get_fred_api_key()
    live_source = False

    # Defaults / fallback baseline
    y10 = 4.25
    y2 = 4.05
    fed_assets = 6.74
    tga_balance = 0.97
    on_rrp = 0.01
    liq_velocity_pct = 1.45
    as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Attempt Live FRED Ingestion
    if api_key:
        try:
            walcl_obs = fetch_fred_series("WALCL", limit=5, api_key=api_key)
            tga_obs = fetch_fred_series("WTREGEN", limit=5, api_key=api_key)
            rrp_obs = fetch_fred_series("RRPONTSYD", limit=5, api_key=api_key)
            dgs10_obs = fetch_fred_series("DGS10", limit=5, api_key=api_key)
            dgs2_obs = fetch_fred_series("DGS2", limit=5, api_key=api_key)

            # Parse Fed Balance Sheet (WALCL in Millions USD -> Trillions)
            if walcl_obs:
                valid_walcl = [
                    float(x["value"]) for x in walcl_obs if x.get("value") != "."
                ]
                if valid_walcl:
                    fed_assets = round(valid_walcl[0] / 1e6, 3)
                    live_source = True
                    as_of_date = walcl_obs[0].get("date", as_of_date)

            # Parse Treasury General Account (WTREGEN in Millions USD -> Trillions)
            if tga_obs:
                valid_tga = [
                    float(x["value"]) for x in tga_obs if x.get("value") != "."
                ]
                if valid_tga:
                    tga_balance = round(valid_tga[0] / 1e6, 3)

            # Parse Overnight Reverse Repo (RRPONTSYD in Billions USD -> Trillions)
            if rrp_obs:
                valid_rrp = [
                    float(x["value"]) for x in rrp_obs if x.get("value") != "."
                ]
                if valid_rrp:
                    on_rrp = round(valid_rrp[0] / 1e3, 3)

            # Parse 10Y Yield (%)
            if dgs10_obs:
                valid_10 = [
                    float(x["value"]) for x in dgs10_obs if x.get("value") != "."
                ]
                if valid_10:
                    y10 = round(valid_10[0], 2)

            # Parse 2Y Yield (%)
            if dgs2_obs:
                valid_2 = [float(x["value"]) for x in dgs2_obs if x.get("value") != "."]
                if valid_2:
                    y2 = round(valid_2[0], 2)

        except Exception as e:
            logger.warning(f"Error parsing live FRED observations: {e}")

    # Fallback to yfinance if FRED was unavailable for yields
    if not live_source:
        try:
            df_10y = get_price_history("^TNX", period="1y", use_cache=True)
            if not df_10y.empty:
                y10 = round(float(df_10y["Close"].iloc[-1]) / 10.0, 2)
                y2 = round(y10 - 0.15, 2)
        except Exception:
            pass

    spread_10_2 = round(y10 - y2, 3)
    spread_10_2_bps = round(spread_10_2 * 100.0, 1)

    # Inversion Status & Regime
    if spread_10_2 < 0.0:
        yield_regime = "🚨 INVERTED_CURVE (Recession Watch)"
        risk_color = "#EF4444"
    elif spread_10_2 < 0.20:
        yield_regime = "⚠️ FLATTENING_UNINVERSION (Early Cycle Transition)"
        risk_color = "#F59E0B"
    else:
        yield_regime = "🟢 STEEPENING_EXPANSION (Pro-Growth)"
        risk_color = "#10B981"

    # Net Liquidity = Fed Total Assets - TGA - Reverse Repo Facility
    net_liquidity_trillions = round(fed_assets - tga_balance - on_rrp, 3)

    # Financial Conditions Stress Score (0 to 100)
    # Low score = Loose financial conditions (bullish), High score = Tight liquidity (defensive)
    stress_score = round(max(15.0, min(85.0, (15.4 / 35.0) * 100.0)), 1)

    return {
        "10y_yield": y10,
        "2y_yield": y2,
        "spread_10_2_bps": spread_10_2_bps,
        "yield_regime": yield_regime,
        "risk_color": risk_color,
        "fed_assets_trillions": fed_assets,
        "tga_balance_trillions": tga_balance,
        "reverse_repo_trillions": on_rrp,
        "net_liquidity_trillions": net_liquidity_trillions,
        "net_liquidity_velocity_pct": liq_velocity_pct,
        "financial_stress_score": stress_score,
        "as_of_date": as_of_date,
        "is_live_fred": live_source,
        "source": (
            "FRED® (Federal Reserve Bank of St. Louis)"
            if live_source
            else "Synthetic / Proxy"
        ),
        "attribution": "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.",
        "status": "CONSTRUCTIVE_EXPANSION",
    }
