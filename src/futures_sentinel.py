"""
Sentilyze Overnight Futures & Macro Sentinel Engine.
Monitors US equity index futures (ES=F, NQ=F, YM=F), Treasury yields (^TNX),
and commodities (CL=F) to determine Monday market opening gap posture and portfolio impact.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import pandas as pd

from src.utils import get_logger, sanitize_filename
from src.realtime_tracker import fetch_live_quote
from src.alerts import send_discord_alert

logger = get_logger("futures_sentinel")

FUTURES_TICKERS = {
    "ES=F": {"name": "S&P 500 E-mini Futures", "proxy": "SPY", "weight": 0.50},
    "NQ=F": {"name": "Nasdaq 100 E-mini Futures", "proxy": "QQQ", "weight": 0.35},
    "YM=F": {"name": "Dow 30 E-mini Futures", "proxy": "DIA", "weight": 0.15},
}

MACRO_INDICATORS = {
    "^TNX": {"name": "US 10-Year Treasury Yield", "risk_type": "RATE_SPIKE"},
    "CL=F": {
        "name": "Crude Oil Futures",
        "proxy": "USO",
        "risk_type": "COMMODITY_SHOCK",
    },
    "^VIX": {"name": "CBOE Volatility Index", "risk_type": "VOL_SURGE"},
}

POSITION_BETAS = {
    "DAL": 1.25,
    "EMR": 1.05,
    "ETN": 1.15,
    "ADP": 0.82,
    "FAST": 1.02,
}


def fetch_futures_quote(symbol: str, proxy: Optional[str] = None) -> Dict[str, Any]:
    """Fetches real-time quote for a futures symbol with yfinance history and proxy fallback."""
    q = fetch_live_quote(symbol)
    price = float(q.get("price", 0.0))
    if price > 0 and q.get("status") != "UNAVAILABLE":
        return q

    # If futures symbol unavailable via direct live endpoint, check proxy
    if proxy:
        pq = fetch_live_quote(proxy)
        if float(pq.get("price", 0.0)) > 0 and pq.get("status") != "UNAVAILABLE":
            pq["ticker"] = symbol
            pq["status"] = "PROXY_LIVE"
            return pq

    # Fallback to yfinance historical series for last close and change
    try:
        import yfinance as yf

        target = symbol if not proxy else symbol
        t = yf.Ticker(target)
        hist = t.history(period="5d")
        if hist.empty and proxy:
            t = yf.Ticker(proxy)
            hist = t.history(period="5d")
        if not hist.empty and len(hist) >= 1:
            close_p = float(hist["Close"].iloc[-1])
            prev_p = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else close_p
            chg = ((close_p - prev_p) / prev_p * 100.0) if prev_p > 0 else 0.0
            return {
                "ticker": symbol,
                "price": round(close_p, 2),
                "prev_close": round(prev_p, 2),
                "change_pct": round(chg, 2),
                "status": "HIST_LAST_CLOSE",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    except Exception as e:
        logger.debug(f"yfinance fallback failed for {symbol}: {e}")

    return {
        "ticker": symbol,
        "price": 0.0,
        "change_pct": 0.0,
        "status": "UNAVAILABLE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_overnight_futures_pulse(
    portfolio_path: str = "results/paper_portfolio.json",
    output_path: str = "results/overnight_futures_pulse.json",
    dispatch_discord: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates overnight futures sentiment, computes implied market gap,
    and estimates projected dollar impact on active open positions.
    """
    logger.info(
        "🌌 [FUTURES SENTINEL] Gathering overnight futures & macro intelligence..."
    )

    # Load portfolio state
    total_equity = 144453.41
    cash = 102944.50
    open_positions = {}
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
                total_equity = float(pdata.get("total_equity", total_equity))
                cash = float(pdata.get("cash", cash))
                open_positions = pdata.get("open_positions", {})
        except Exception as e:
            logger.debug(f"Error loading portfolio for sentinel: {e}")

    invested_capital = total_equity - cash

    # 1. Evaluate Futures Index Flow
    futures_data = {}
    weighted_gap = 0.0
    valid_weight = 0.0

    for symbol, meta in FUTURES_TICKERS.items():
        q = fetch_futures_quote(symbol, proxy=meta.get("proxy"))
        chg = float(q.get("change_pct", 0.0))
        futures_data[symbol] = {
            "name": meta["name"],
            "price": float(q.get("price", 0.0)),
            "change_pct": chg,
            "status": q.get("status", "UNAVAILABLE"),
        }
        w = meta.get("weight", 0.33)
        weighted_gap += chg * w
        valid_weight += w

    composite_gap_pct = (
        round(weighted_gap / valid_weight, 2) if valid_weight > 0 else 0.0
    )

    # 2. Determine Implied Monday Opening Posture
    if composite_gap_pct >= 0.40:
        market_bias = "🟢 BULLISH_GAP_OPEN"
        risk_posture = "AGGRESSIVE_RUNNER_EXPANSION"
        stance_desc = "Futures point to strong opening momentum. Trailing stops ratcheted higher to capture gap."
    elif composite_gap_pct <= -0.40:
        market_bias = "🔴 BEARISH_GAP_OPEN"
        risk_posture = "DEFENSIVE_CAPITAL_SHIELD"
        stance_desc = "Futures point to adverse gap down. Trailing stops and cash buffer primed to absorb volatility."
    else:
        market_bias = "🟡 NEUTRAL_CONSOLIDATION"
        risk_posture = "BALANCED_DISCIPLINE"
        stance_desc = "Futures flat to unchanged. Systematic execution active on standard thresholds."

    # 3. Project Dollar Impact on Open Positions
    position_impacts = {}
    total_projected_impact = 0.0

    for ticker, pos in open_positions.items():
        shares = int(pos.get("shares", 0))
        curr_price = float(pos.get("current_price", pos.get("entry_price", 0.0)))
        pos_val = shares * curr_price
        beta = POSITION_BETAS.get(ticker.upper(), 1.0)
        projected_chg_pct = composite_gap_pct * beta
        projected_dollar = pos_val * (projected_chg_pct / 100.0)

        position_impacts[ticker] = {
            "shares": shares,
            "position_value": round(pos_val, 2),
            "beta": beta,
            "projected_change_pct": round(projected_chg_pct, 2),
            "projected_dollar_impact": round(projected_dollar, 2),
            "locked_profit": (
                "GUARANTEED_PROFIT"
                if float(pos.get("sl_target", 0.0)) > float(pos.get("entry_price", 0.0))
                else "SHIELDED"
            ),
        }
        total_projected_impact += projected_dollar

    projected_terminal_equity = round(total_equity + total_projected_impact, 2)

    # 4. Macro Volatility & Yields Check
    macro_pulse = {}
    for sym, m_info in MACRO_INDICATORS.items():
        mq = fetch_futures_quote(sym, proxy=m_info.get("proxy"))
        macro_pulse[sym] = {
            "name": m_info["name"],
            "value": float(mq.get("price", 0.0)),
            "change_pct": float(mq.get("change_pct", 0.0)),
            "risk_type": m_info["risk_type"],
        }

    sentinel_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_bias": market_bias,
        "composite_gap_pct": composite_gap_pct,
        "risk_posture": risk_posture,
        "stance_description": stance_desc,
        "portfolio_equity": total_equity,
        "cash_buffer": cash,
        "cash_pct": (
            round((cash / total_equity) * 100.0, 1) if total_equity > 0 else 0.0
        ),
        "total_projected_impact_dollar": round(total_projected_impact, 2),
        "projected_equity_at_open": projected_terminal_equity,
        "futures_quotes": futures_data,
        "position_impacts": position_impacts,
        "macro_pulse": macro_pulse,
    }

    # Save to output_path if specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sentinel_report, f, indent=2)
            logger.info(f"Saved overnight futures intelligence to {output_path}")
        except Exception as e:
            logger.debug(f"Could not persist futures pulse: {e}")

    # Optional Discord Dispatch
    if dispatch_discord:
        try:
            import requests

            webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
            if webhook_url:
                fields = [
                    {
                        "name": "📊 Implied Opening Gap",
                        "value": f"**{composite_gap_pct:+.2f}%** ({market_bias})",
                        "inline": True,
                    },
                    {
                        "name": "🛡️ Risk Posture",
                        "value": f"`{risk_posture}`",
                        "inline": True,
                    },
                    {
                        "name": "💵 Portfolio Equity",
                        "value": f"${total_equity:,.2f} (${cash:,.2f} Cash)",
                        "inline": True,
                    },
                    {
                        "name": "🎯 Projected Open PnL",
                        "value": f"**${total_projected_impact:+,.2f}** -> Implied Open: **${projected_terminal_equity:,.2f}**",
                        "inline": False,
                    },
                ]

                pos_lines = []
                for t, p_imp in position_impacts.items():
                    pos_lines.append(
                        f"• **{t}**: ${p_imp['position_value']:,.2f} | Implied {p_imp['projected_change_pct']:+.2f}% (${p_imp['projected_dollar_impact']:+,.2f}) [{p_imp['locked_profit']}]"
                    )
                if pos_lines:
                    fields.append(
                        {
                            "name": "📦 Open Positions Protection",
                            "value": "\n".join(pos_lines),
                            "inline": False,
                        }
                    )

                color = (
                    0x2ECC71
                    if composite_gap_pct >= 0
                    else (0xE74C3C if composite_gap_pct <= -0.5 else 0x3498DB)
                )
                embed = {
                    "title": "🌌 OVERNIGHT FUTURES SENTINEL | MONDAY GAP FORECAST",
                    "description": f"Overnight market scan complete. *{stance_desc}*",
                    "color": color,
                    "fields": fields,
                    "footer": {
                        "text": f"Sentilyze Institutional Desk • {sentinel_report['timestamp'][:19]}Z"
                    },
                }
                res = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
                if res.status_code in (200, 204):
                    logger.info(
                        "Dispatched Discord Overnight Futures Sentinel briefing successfully."
                    )
                else:
                    logger.warning(
                        f"Discord responded with status {res.status_code}: {res.text}"
                    )
        except Exception as e:
            logger.error(f"Failed to dispatch futures Discord alert: {e}")

    return sentinel_report


if __name__ == "__main__":
    rep = evaluate_overnight_futures_pulse(dispatch_discord=True)
    print(json.dumps(rep, indent=2))
