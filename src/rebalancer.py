import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_share_allocation(
    capital: float,
    selected_signals: List[Dict[str, Any]],
    method: str = "risk_parity",
    volatility_map: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Computes exact whole-share buy allocations for a given capital budget across chosen stock signals.

    Args:
        capital (float): Total dollar budget (e.g. $10,000 or $50,000).
        selected_signals (List[Dict[str, Any]]): List of signal dictionaries from daily scanner.
        method (str): Allocation model - 'risk_parity', 'equal_weight', or 'confidence'.
        volatility_map (Dict[str, float], optional): Map of ticker to 21-day annualized volatility.

    Returns:
        Dict[str, Any]: Detailed allocation breakdown with whole share counts, costs, and remaining cash.
    """
    if capital <= 0 or not selected_signals:
        return {
            "total_capital": capital,
            "total_invested": 0.0,
            "cash_reserve": capital,
            "allocation_table": pd.DataFrame(),
            "positions_count": 0,
        }

    valid_signals = [s for s in selected_signals if float(s.get("current_price", 0)) > 0]
    if not valid_signals:
        return {
            "total_capital": capital,
            "total_invested": 0.0,
            "cash_reserve": capital,
            "allocation_table": pd.DataFrame(),
            "positions_count": 0,
        }

    n_assets = len(valid_signals)
    raw_weights = {}

    if method == "equal_weight":
        for s in valid_signals:
            raw_weights[s["ticker"]] = 1.0 / n_assets

    elif method == "confidence":
        # Weight proportional to confidence above 50% baseline
        conf_diffs = {
            s["ticker"]: max(0.01, float(s.get("confidence", 0.5)) - 0.40)
            for s in valid_signals
        }
        total_conf = sum(conf_diffs.values())
        raw_weights = {t: c / total_conf for t, c in conf_diffs.items()}

    else:
        # Default: Risk Parity (Inverse Volatility)
        vols = {}
        for s in valid_signals:
            t = s["ticker"]
            if volatility_map and t in volatility_map and volatility_map[t] > 0:
                vols[t] = volatility_map[t]
            else:
                # Default baseline daily volatility estimation from ATR / Price or 25% default
                price = float(s.get("current_price", 100))
                tp = float(s.get("take_profit", price * 1.06))
                vols[t] = max(0.10, (tp - price) / price)

        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())
        raw_weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}

    # Compute whole share allocation and costs
    rows = []
    total_cost = 0.0

    for s in valid_signals:
        ticker = s["ticker"]
        price = float(s["current_price"])
        weight = raw_weights.get(ticker, 1.0 / n_assets)
        target_budget = capital * weight

        shares = int(target_budget // price)
        cost = float(shares * price)
        total_cost += cost

        tp = float(s.get("take_profit", price * 1.06))
        sl = float(s.get("stop_loss", price * 0.95))
        potential_profit = float(shares * (tp - price))
        max_risk = float(shares * (price - sl))
        reward_risk_ratio = round(potential_profit / max(1.0, max_risk), 2)

        rows.append(
            {
                "Ticker": ticker,
                "Conviction": f"{float(s.get('confidence', 0.5)) * 100:.1f}%",
                "Share Price": f"${price:,.2f}",
                "Target Weight": f"{weight * 100:.1f}%",
                "Shares to Buy": shares,
                "Total Cost ($)": f"${cost:,.2f}",
                "Actual Weight": f"{(cost / capital * 100):.1f}%" if capital > 0 else "0%",
                "Take-Profit Target": f"${tp:,.2f}",
                "Stop-Loss Target": f"${sl:,.2f}",
                "Potential Profit ($)": f"${potential_profit:+,.2f}",
                "Max Risk ($)": f"${max_risk:,.2f}",
                "Reward / Risk": f"{reward_risk_ratio}:1",
            }
        )

    df_alloc = pd.DataFrame(rows)
    cash_remaining = float(capital - total_cost)

    return {
        "total_capital": round(capital, 2),
        "total_invested": round(total_cost, 2),
        "cash_reserve": round(cash_remaining, 2),
        "invested_pct": round((total_cost / capital) * 100.0, 1) if capital > 0 else 0.0,
        "allocation_table": df_alloc,
        "positions_count": len([r for r in rows if r["Shares to Buy"] > 0]),
    }


def calculate_custom_rebalance(
    total_capital: float = 25000.0,
    method: str = "risk_parity",
    signals_file: str = "results/daily_signals_latest.json",
) -> Dict[str, Any]:
    """Helper to calculate share allocation from latest daily signals file or universe prices."""
    import os
    import json

    signals = []
    if os.path.exists(signals_file):
        try:
            with open(signals_file, "r") as f:
                data = json.load(f)
                signals = data.get("signals", [])
        except Exception:
            pass

    if not signals:
        from src.realtime_tracker import fetch_universe_live_quotes
        quotes = fetch_universe_live_quotes()
        for t, q in quotes.items():
            if q.get("price", 0) > 0:
                signals.append({
                    "ticker": t,
                    "signal": "BUY",
                    "confidence": 0.65,
                    "current_price": q["price"]
                })

    return calculate_share_allocation(capital=total_capital, selected_signals=signals, method=method)

