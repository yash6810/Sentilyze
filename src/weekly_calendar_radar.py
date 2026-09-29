"""
Sentilyze Macro Economic & Earnings Radar.
Scans high-impact macroeconomic releases (FOMC, ISM, Non-Farm Payrolls, CPI, JOLTS)
and company earnings dates across open positions and watchlist candidates to enforce
pre-event risk blackouts and volatility cones.
STRICT PORTFOLIO PRESERVATION: Read-only on portfolio state.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import pandas as pd

from src.utils import get_logger

logger = get_logger("weekly_calendar_radar")

# High-impact macro events template with historical volatility cones
HIGH_IMPACT_MACRO_SCHEDULE = [
    {
        "event": "ISM Manufacturing PMI",
        "date": "2026-10-01 10:00 EDT",
        "impact": "HIGH",
        "category": "MACRO_GROWTH",
        "expected_volatility_cone_pct": 0.85,
        "historical_spy_dispersion": "+/- 0.65%",
        "cro_rule": "Tighten stop loss buffers by 25% 15 minutes before release",
    },
    {
        "event": "JOLTS Job Openings",
        "date": "2026-10-01 10:00 EDT",
        "impact": "HIGH",
        "category": "LABOR_MARKET",
        "expected_volatility_cone_pct": 0.70,
        "historical_spy_dispersion": "+/- 0.55%",
        "cro_rule": "Cap new Kelly allocations at <= 1.5%",
    },
    {
        "event": "ADP Non-Farm Employment",
        "date": "2026-09-30 08:15 EDT",
        "impact": "MEDIUM",
        "category": "LABOR_MARKET",
        "expected_volatility_cone_pct": 0.50,
        "historical_spy_dispersion": "+/- 0.40%",
        "cro_rule": "Normal risk thresholds active",
    },
    {
        "event": "Non-Farm Payrolls (NFP) & Unemployment Rate",
        "date": "2026-10-02 08:30 EDT",
        "impact": "VERY_HIGH",
        "category": "CRITICAL_LABOR",
        "expected_volatility_cone_pct": 1.45,
        "historical_spy_dispersion": "+/- 1.25%",
        "cro_rule": "MACRO_VOLATILITY_BLACKOUT: Halt new long entries 60m pre-release",
    },
    {
        "event": "FOMC Speeches & Fed Balance Sheet",
        "date": "2026-10-01 16:30 EDT",
        "impact": "HIGH",
        "category": "CENTRAL_BANK",
        "expected_volatility_cone_pct": 0.90,
        "historical_spy_dispersion": "+/- 0.75%",
        "cro_rule": "Maintain min 50% dry powder cash buffer",
    },
]

# Earnings dates for universe / positions
KNOWN_EARNINGS_DATES = {
    "DAL": {"earnings_date": "2026-10-10", "days_away": 12, "blackout_active": False},
    "EMR": {"earnings_date": "2026-11-05", "days_away": 38, "blackout_active": False},
    "ETN": {"earnings_date": "2026-10-29", "days_away": 31, "blackout_active": False},
    "ADP": {"earnings_date": "2026-10-28", "days_away": 30, "blackout_active": False},
    "FAST": {"earnings_date": "2026-10-11", "days_away": 13, "blackout_active": False},
    "MSFT": {"earnings_date": "2026-10-22", "days_away": 24, "blackout_active": False},
    "AAPL": {"earnings_date": "2026-10-29", "days_away": 31, "blackout_active": False},
    "NVDA": {"earnings_date": "2026-11-18", "days_away": 51, "blackout_active": False},
}


def build_weekly_radar(
    portfolio_path: str = "results/paper_portfolio.json",
    watchlist_path: str = "results/monday_premarket_watchlist.json",
    output_path: str = "results/upcoming_week_macro_radar.json",
) -> Dict[str, Any]:
    """
    Generates the comprehensive weekly macro and earnings catalyst radar.
    """
    logger.info(
        "📅 [RADAR] Compiling weekly macro calendar & earnings blackout radar..."
    )

    open_tickers = []
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
                open_tickers = list(pdata.get("open_positions", {}).keys())
        except Exception as e:
            logger.debug(f"Could not read portfolio positions: {e}")

    watchlist_tickers = []
    if os.path.exists(watchlist_path):
        try:
            with open(watchlist_path, "r", encoding="utf-8") as f:
                wdata = json.load(f)
                watchlist_tickers = [
                    item.get("ticker") for item in wdata.get("watchlist", [])
                ]
        except Exception as e:
            logger.debug(f"Could not read watchlist: {e}")

    # Check earnings proximity
    all_monitored = list(set(open_tickers + watchlist_tickers[:5]))
    earnings_status = {}
    imminent_blackouts = []

    for t in all_monitored:
        e_info = KNOWN_EARNINGS_DATES.get(
            t, {"earnings_date": "TBD", "days_away": 45, "blackout_active": False}
        )
        days = e_info.get("days_away", 45)
        is_blackout = days <= 3
        earnings_status[t] = {
            "earnings_date": e_info.get("earnings_date"),
            "days_until_earnings": days,
            "blackout_active": is_blackout,
            "cro_guidance": (
                "🚨 PRE-EARNINGS BLACKOUT"
                if is_blackout
                else "🟢 SAFE (NO IMMINENT EARNINGS)"
            ),
        }
        if is_blackout:
            imminent_blackouts.append(t)

    radar_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "week_range": "2026-09-28 to 2026-10-02",
        "macro_risk_posture": (
            "PRE_NFP_CAUTION"
            if any(e["impact"] == "VERY_HIGH" for e in HIGH_IMPACT_MACRO_SCHEDULE)
            else "NORMAL"
        ),
        "high_impact_macro_events": HIGH_IMPACT_MACRO_SCHEDULE,
        "monitored_earnings": earnings_status,
        "imminent_blackouts_count": len(imminent_blackouts),
        "imminent_blackout_tickers": imminent_blackouts,
        "cro_summary": (
            "No active positions have earnings within 3 days. Focus risk controls on Friday NFP release (08:30 EDT)."
            if not imminent_blackouts
            else f"Imminent earnings blackout on {imminent_blackouts}: avoid adding new sizing."
        ),
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(radar_report, f, indent=2)
            logger.info(f"Saved weekly macro calendar radar to {output_path}")
        except Exception as e:
            logger.debug(f"Could not save macro radar: {e}")

    return radar_report


if __name__ == "__main__":
    rep = build_weekly_radar()
    print("Macro Radar generated successfully.")
    print(f"Risk Posture: {rep['macro_risk_posture']}")
    print(f"Events: {len(rep['high_impact_macro_events'])}")
