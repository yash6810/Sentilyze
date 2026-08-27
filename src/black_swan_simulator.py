"""
Historical Black Swan Crisis Simulator & Kelly Position Sizing for Sentilyze.
Pillar 5 Risk Engine:
- Replays historical market crashes (2008 Lehman, 2020 Covid, 2022 Fed Rate Hikes, 2000 Dot-Com) against current portfolio holdings.
- Computes Fractional Kelly Criterion (Full, Half, Quarter) optimal leverage and position sizing.
- Calculates Almgren-Chriss market impact and execution slippage under liquidity stress.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

HISTORICAL_CRISES = [
    {
        "name": "2008 Lehman Brothers & GFC",
        "date_range": "2007 - 2009",
        "market_drawdown_pct": -56.8,
        "vix_peak": 80.86,
        "tech_drawdown_pct": -50.2,
        "semis_drawdown_pct": -62.4,
        "financials_drawdown_pct": -82.1,
        "catalyst": "Subprime mortgage collapse, interbank liquidity freeze, Lehman bankruptcy.",
    },
    {
        "name": "2020 Covid-19 Liquidity Shock",
        "date_range": "Feb - Mar 2020",
        "market_drawdown_pct": -33.9,
        "vix_peak": 82.69,
        "tech_drawdown_pct": -28.5,
        "semis_drawdown_pct": -32.1,
        "financials_drawdown_pct": -43.5,
        "catalyst": "Global economic lockdown, supply chain paralysis, flash margin calls.",
    },
    {
        "name": "2022 Fed Rate Hike Tech Bear",
        "date_range": "Jan - Dec 2022",
        "market_drawdown_pct": -25.4,
        "vix_peak": 38.94,
        "tech_drawdown_pct": -35.5,
        "semis_drawdown_pct": -45.0,
        "financials_drawdown_pct": -21.0,
        "catalyst": "40-year high inflation, aggressive 75bps Fed rate hikes, multiple compression.",
    },
    {
        "name": "2000 Dot-Com Bubble Collapse",
        "date_range": "2000 - 2002",
        "market_drawdown_pct": -49.1,
        "vix_peak": 45.08,
        "tech_drawdown_pct": -78.2,
        "semis_drawdown_pct": -72.0,
        "financials_drawdown_pct": -24.0,
        "catalyst": "Unprofitable internet stock valuation implosion, telecom capex freeze.",
    },
    {
        "name": "2024 AI Rotation & Flash Drawdown",
        "date_range": "July - Aug 2024",
        "market_drawdown_pct": -9.8,
        "vix_peak": 65.73,
        "tech_drawdown_pct": -14.2,
        "semis_drawdown_pct": -22.5,
        "financials_drawdown_pct": -5.2,
        "catalyst": "Yen carry trade unwind, semiconductor capex digestion, macro growth scare.",
    },
]


def simulate_portfolio_crises(
    positions_dict: Dict[str, float], total_equity: float = 100000.0
) -> List[Dict[str, Any]]:
    """
    Stress-tests the current portfolio against major historical market crashes.

    Args:
        positions_dict: Dict of {ticker: position_dollar_value}
        total_equity: Total current portfolio equity

    Returns:
        List of simulated crisis outcomes with projected dollar losses and drawdowns.
    """
    results = []
    invested_capital = sum(positions_dict.values())
    cash = max(0.0, total_equity - invested_capital)

    # Asset sector mapping
    semis = {"NVDA", "AMD", "TSM", "AVGO"}
    big_tech = {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NFLX", "PLTR"}
    financials = {"JPM"}

    for crisis in HISTORICAL_CRISES:
        total_sim_loss = 0.0
        position_breakdown = {}

        for ticker, pos_val in positions_dict.items():
            if ticker in semis:
                shock_pct = crisis["semis_drawdown_pct"]
            elif ticker in big_tech:
                shock_pct = crisis["tech_drawdown_pct"]
            elif ticker in financials:
                shock_pct = crisis["financials_drawdown_pct"]
            else:
                shock_pct = crisis["market_drawdown_pct"]

            pos_loss = pos_val * (abs(shock_pct) / 100.0)
            total_sim_loss += pos_loss
            position_breakdown[ticker] = {
                "initial_val": round(pos_val, 2),
                "shock_pct": shock_pct,
                "projected_loss": round(pos_loss, 2),
            }

        sim_equity_after = max(0.0, total_equity - total_sim_loss)
        portfolio_dd_pct = (total_sim_loss / (total_equity + 1e-9)) * 100.0

        results.append(
            {
                "crisis_name": crisis["name"],
                "date_range": crisis["date_range"],
                "vix_peak": crisis["vix_peak"],
                "catalyst": crisis["catalyst"],
                "portfolio_drawdown_pct": round(portfolio_dd_pct, 1),
                "projected_dollar_loss": round(total_sim_loss, 2),
                "simulated_equity_after": round(sim_equity_after, 2),
                "cash_buffer_retained": round(cash, 2),
                "position_breakdown": position_breakdown,
            }
        )

    return results


def calculate_kelly_sizing(
    win_rate: float, win_loss_ratio: float, max_leverage: float = 2.0
) -> Dict[str, Any]:
    """
    Calculates optimal position sizing using the Kelly Criterion:
    Kelly % = W - (1 - W) / R

    Args:
        win_rate: Historical win rate (0.0 to 1.0, e.g. 0.56)
        win_loss_ratio: Average Win / Average Loss (e.g. 1.4)
        max_leverage: Maximum allowed portfolio leverage cap

    Returns:
        Dict with Full Kelly, Half Kelly (Recommended), Quarter Kelly, and recommended leverage.
    """
    w = float(np.clip(win_rate, 0.01, 0.99))
    r = float(max(0.1, win_loss_ratio))

    full_kelly = w - ((1.0 - w) / r)
    full_kelly = float(np.clip(full_kelly, 0.0, 1.0))

    half_kelly = full_kelly * 0.5
    quarter_kelly = full_kelly * 0.25

    # Suggested dynamic leverage
    rec_leverage = float(np.clip(1.0 + half_kelly, 1.0, max_leverage))

    return {
        "win_rate_pct": round(w * 100, 1),
        "win_loss_ratio": round(r, 2),
        "full_kelly_pct": round(full_kelly * 100, 1),
        "half_kelly_pct": round(half_kelly * 100, 1),
        "quarter_kelly_pct": round(quarter_kelly * 100, 1),
        "recommended_leverage": round(rec_leverage, 2),
        "sizing_recommendation": f"Allocate {half_kelly * 100:.1f}% capital per position (Half-Kelly).",
    }


def estimate_market_impact_slippage(
    order_size_dollars: float,
    daily_volume_dollars: float = 500_000_000.0,
    daily_volatility_pct: float = 0.025,
) -> Dict[str, Any]:
    """
    Estimates market execution slippage using the Almgren-Chriss square-root impact model:
    Impact = sigma * sqrt(OrderSize / DailyVolume) * 0.5

    Args:
        order_size_dollars: Dollar size of the trade order
        daily_volume_dollars: Average daily dollar volume of the stock
        daily_volatility_pct: Daily volatility percentage (e.g. 0.025 for 2.5%)

    Returns:
        Dict with estimated slippage bps, dollar cost, and liquidity score.
    """
    participation_rate = order_size_dollars / (daily_volume_dollars + 1e-9)
    slippage_pct = daily_volatility_pct * np.sqrt(max(0.0, participation_rate)) * 0.5
    slippage_bps = slippage_pct * 10000.0
    slippage_dollars = order_size_dollars * slippage_pct

    if slippage_bps < 3.0:
        liquidity_status = "🟢 ULTRA-HIGH LIQUIDITY (Near-Zero Slippage < 3 bps)"
    elif slippage_bps < 10.0:
        liquidity_status = "🟡 MODERATE LIQUIDITY (Acceptable 3–10 bps)"
    else:
        liquidity_status = "🔴 LOW LIQUIDITY / HIGH MARKET IMPACT (> 10 bps)"

    return {
        "order_size_dollars": round(order_size_dollars, 2),
        "estimated_slippage_bps": round(slippage_bps, 2),
        "estimated_slippage_dollars": round(slippage_dollars, 2),
        "participation_rate_pct": round(participation_rate * 100, 4),
        "liquidity_status": liquidity_status,
    }
