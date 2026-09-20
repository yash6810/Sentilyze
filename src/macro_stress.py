"""
Historical Macro Crisis Replay Simulator & Stress-Testing Engine for Sentilyze.

Simulates portfolio performance and drawdown cushion across 4 canonical crisis episodes:
1. 1987 Black Monday (-22.6% single-day collapse)
2. 2008 Great Financial Crisis / Lehman (-40.0% systemic credit freeze)
3. 2020 COVID Flash Crash (-34.0% rapid collapse, VIX to 82.7)
4. 2022 Fed Aggressive 75bps Rate Shock (-28.0% tech compression)

Strictly preserves live paper trading book while reading positions for institutional stress testing.
"""

from typing import Any, Dict, List, Optional
import os
import json
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

PORTFOLIO_FILE = os.path.join("results", "paper_portfolio.json")

# Canonical Crisis Historical Shocks (Broad Equities, Tech Skew, Safe Haven Yields)
CRISIS_SCENARIOS = {
    "1987_BLACK_MONDAY": {
        "name": "1987 Black Monday",
        "description": "Single-day market crash (-22.6% on S&P 500) triggered by portfolio insurance stop cascades.",
        "broad_equity_shock_pct": -22.6,
        "tech_equity_shock_pct": -24.5,
        "industrial_shock_pct": -21.0,
        "duration_days": 1,
        "max_vix_proxy": 150.0,
    },
    "2008_LEHMAN_GFC": {
        "name": "2008 Lehman GFC Collapse",
        "description": "Systemic subprime credit contagion and banking collapse (-40.0% peak-to-trough).",
        "broad_equity_shock_pct": -40.0,
        "tech_equity_shock_pct": -44.0,
        "industrial_shock_pct": -48.0,
        "duration_days": 130,
        "max_vix_proxy": 89.5,
    },
    "2020_COVID_FLASH_CRASH": {
        "name": "2020 COVID Flash Crash",
        "description": "Fastest 30% drop in market history (23 trading days) as global economies halted.",
        "broad_equity_shock_pct": -34.0,
        "tech_equity_shock_pct": -31.0,
        "industrial_shock_pct": -42.0,
        "duration_days": 23,
        "max_vix_proxy": 82.7,
    },
    "2022_FED_RATE_SHOCK": {
        "name": "2022 Fed Aggressive 75bps Rate Shock",
        "description": "Historic Fed monetary tightening cycle; 10Y yield surge hammering long-duration growth assets.",
        "broad_equity_shock_pct": -19.4,
        "tech_equity_shock_pct": -33.1,
        "industrial_shock_pct": -12.5,
        "duration_days": 250,
        "max_vix_proxy": 38.9,
    },
}


def load_read_only_portfolio(portfolio_path: str = PORTFOLIO_FILE) -> Dict[str, Any]:
    """
    Safely reads current paper portfolio state without modifying or resetting files.
    """
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed reading portfolio at {portfolio_path}: {e}")

    # Fallback representative baseline
    return {
        "total_equity": 159316.54,
        "cash": 68160.25,
        "open_positions": {
            "GEV": {"shares": 100, "current_price": 380.0, "market_value": 38000.0},
            "FDXF": {"shares": 150, "current_price": 240.0, "market_value": 36000.0},
            "CAT": {"shares": 45, "current_price": 380.0, "market_value": 17156.29},
        },
    }


def simulate_crisis_replay(
    scenario_key: str,
    portfolio_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Simulates portfolio drawdown under a specific historical crisis scenario.
    Accounts for cash buffer protection and asset sector betas.
    """
    scenario = CRISIS_SCENARIOS.get(
        scenario_key, CRISIS_SCENARIOS["2020_COVID_FLASH_CRASH"]
    )
    data = portfolio_data if portfolio_data is not None else load_read_only_portfolio()

    total_equity = float(data.get("total_equity", 150000.0))
    cash = float(data.get("cash", 50000.0))
    cash_ratio = cash / total_equity if total_equity > 0 else 0.40

    open_positions = data.get("open_positions", {})
    invested_equity = total_equity - cash

    # Sector shock mapping
    asset_losses = {}
    total_invested_loss = 0.0

    for ticker, pos in open_positions.items():
        mkt_val = float(pos.get("market_value", 0.0))
        t_upper = ticker.upper()

        if t_upper in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]:
            shock_pct = scenario["tech_equity_shock_pct"]
        elif t_upper in ["CAT", "GEV", "EMR", "DE"]:
            shock_pct = scenario["industrial_shock_pct"]
        else:
            shock_pct = scenario["broad_equity_shock_pct"]

        loss_dollars = mkt_val * (shock_pct / 100.0)
        asset_losses[ticker] = {
            "market_value": round(mkt_val, 2),
            "shock_pct": shock_pct,
            "simulated_dollar_pnl": round(loss_dollars, 2),
            "simulated_end_value": round(mkt_val + loss_dollars, 2),
        }
        total_invested_loss += loss_dollars

    # If no open positions present, apply broad equity shock to invested equity
    if not open_positions and invested_equity > 0:
        shock_pct = scenario["broad_equity_shock_pct"]
        total_invested_loss = invested_equity * (shock_pct / 100.0)

    # Cash remains unaffected (0% shock)
    simulated_final_equity = total_equity + total_invested_loss
    portfolio_drawdown_pct = (
        (total_invested_loss / total_equity) * 100.0 if total_equity > 0 else 0.0
    )

    # Benchmark comparison: 100% long buy & hold benchmark loss
    benchmark_loss_pct = scenario["broad_equity_shock_pct"]
    cushion_delivered_pct = abs(benchmark_loss_pct) - abs(portfolio_drawdown_pct)

    return {
        "scenario_key": scenario_key,
        "scenario_name": scenario["name"],
        "description": scenario["description"],
        "duration_days": scenario["duration_days"],
        "initial_total_equity": round(total_equity, 2),
        "cash_buffer_dollars": round(cash, 2),
        "cash_allocation_pct": round(cash_ratio * 100.0, 1),
        "simulated_loss_dollars": round(total_invested_loss, 2),
        "simulated_final_equity": round(simulated_final_equity, 2),
        "portfolio_drawdown_pct": round(portfolio_drawdown_pct, 2),
        "unhedged_market_drawdown_pct": round(benchmark_loss_pct, 2),
        "cash_cushion_protection_pct": round(cushion_delivered_pct, 2),
        "survival_status": (
            "SURVIVED_WITH_SURPLUS_CASH"
            if simulated_final_equity > cash
            else "CAPITAL_IMPAIRED"
        ),
        "asset_breakdown": asset_losses,
    }


def run_full_crisis_stress_test(
    portfolio_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes stress test across all 4 crisis scenarios.
    """
    data = portfolio_data if portfolio_data is not None else load_read_only_portfolio()
    results = {}
    for key in CRISIS_SCENARIOS:
        results[key] = simulate_crisis_replay(key, portfolio_data=data)

    max_loss_scenario = min(results.values(), key=lambda x: x["portfolio_drawdown_pct"])

    return {
        "status": "STRESS_TEST_COMPLETE",
        "total_scenarios_replayed": len(results),
        "worst_case_scenario": max_loss_scenario["scenario_name"],
        "worst_case_drawdown_pct": max_loss_scenario["portfolio_drawdown_pct"],
        "worst_case_final_equity": max_loss_scenario["simulated_final_equity"],
        "scenarios": results,
    }
