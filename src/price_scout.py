"""
Real-Time Price Action & Tape-Reading Scout Subagent (Bot) for Sentilyze.

Functions:
- Tracks live quotes, intraday high/low range, RVOL (Relative Volume), and VWAP proximity.
- Detects explosive momentum breakouts, pullback demand zones, and volume spikes.
- Reports real-time price intelligence and conviction scores directly to the Committee.
"""

from typing import Dict, Any, List, Optional
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.realtime_tracker import fetch_live_quote
from src.data_ingestion import get_price_history

logger = get_logger(__name__)

SCOUT_ALERTS_FILE = os.path.join("results", "price_scout_alerts.json")


class PriceActionScoutAgent:
    """
    Real-Time Price Action & Tape-Reading Scout Specialist.
    Monitors live intraday microstructure: spot price, day high/low position,
    RVOL (Relative Volume), VWAP proximity, and momentum velocity.
    """

    def __init__(self, name: str = "Real-Time Price & Tape Scout"):
        self.name = name

    def evaluate(
        self, ticker: str, spot_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates real-time price action and tape structure for a given ticker.
        """
        try:
            # 1. Fetch live quote
            quote = fetch_live_quote(ticker)
            price = (
                spot_price
                if spot_price and spot_price > 0
                else float(quote.get("price", 100.0))
            )
            prev_close = float(quote.get("prev_close", price))
            day_high = float(quote.get("day_high", price * 1.01))
            day_low = float(quote.get("day_low", price * 0.99))
            chg_pct = float(quote.get("change_pct", 0.0))

            # 2. Intraday Range Positioning (0.0 = Low of Day, 1.0 = High of Day)
            if day_high > day_low:
                range_pos = (price - day_low) / (day_high - day_low)
            else:
                range_pos = 0.5
            range_pos = max(0.0, min(1.0, range_pos))

            # 3. Microstructure VWAP Proxy & Multi-day Volume Analysis
            try:
                hist_df = get_price_history(ticker, period="1mo", use_cache=True)
            except Exception:
                hist_df = pd.DataFrame()

            rvol = 1.0
            momentum_velocity = 0.0
            if not hist_df.empty and len(hist_df) >= 5:
                avg_vol = hist_df["Volume"].tail(20).mean()
                latest_vol = hist_df["Volume"].iloc[-1]
                if avg_vol > 0:
                    rvol = round(float(latest_vol / avg_vol), 2)

                # 5-day momentum slope
                p_5d_ago = hist_df["Close"].iloc[-5]
                momentum_velocity = (
                    round(float((price - p_5d_ago) / p_5d_ago * 100.0), 2)
                    if p_5d_ago > 0
                    else 0.0
                )

            # 4. Synthesize Price & Tape Conviction Score (0 to 100)
            base_score = 50.0

            # High of day breakout vs low of day demand bounce
            if range_pos >= 0.75:
                # Top quartile of day's range: Breakout continuation
                base_score += 25.0
            elif range_pos <= 0.25:
                # Bottom quartile: Potential demand bottoming or downtrend
                if chg_pct > -1.5:
                    base_score += 10.0  # Mild pullback into support (bottom fishing)
                else:
                    base_score -= 20.0  # Heavy selloff
            else:
                base_score += 10.0  # Healthy mid-range consolidation

            # RVOL boost (Surging volume validates moves)
            if rvol >= 1.5:
                base_score += 15.0
            elif rvol >= 1.1:
                base_score += 8.0
            elif rvol < 0.7:
                base_score -= 10.0  # Low volume chop

            # Momentum velocity boost
            if momentum_velocity > 3.0:
                base_score += 10.0
            elif momentum_velocity < -3.0:
                base_score -= 10.0

            final_score = round(max(5.0, min(95.0, base_score)), 1)

            # Determine Vote
            if final_score >= 65.0:
                vote = "BUY"
                verdict = (
                    f"🟢 Strong Tape Momentum (RVOL {rvol}x, Range {range_pos*100:.0f}%, "
                    f"5D Vel {momentum_velocity:+.1f}%)"
                )
            elif final_score <= 35.0:
                vote = "SELL"
                verdict = (
                    f"🔴 Weak Tape / Selling Pressure (RVOL {rvol}x, Range {range_pos*100:.0f}%, "
                    f"5D Vel {momentum_velocity:+.1f}%)"
                )
            else:
                vote = "HOLD"
                verdict = (
                    f"🟡 Neutral Tape Flow (Range {range_pos*100:.0f}%, "
                    f"RVOL {rvol}x, 5D Vel {momentum_velocity:+.1f}%)"
                )

            return {
                "agent_name": self.name,
                "ticker": ticker,
                "spot_price": round(price, 2),
                "day_high": round(day_high, 2),
                "day_low": round(day_low, 2),
                "change_pct": round(chg_pct, 2),
                "range_position_pct": round(range_pos * 100.0, 1),
                "rvol": rvol,
                "momentum_velocity_pct": momentum_velocity,
                "conviction_score": final_score,
                "vote": vote,
                "verdict": verdict,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"PriceActionScout error for {ticker}: {e}")
            return {
                "agent_name": self.name,
                "ticker": ticker,
                "spot_price": spot_price or 100.0,
                "conviction_score": 50.0,
                "vote": "HOLD",
                "verdict": f"Tape data unavailable ({e})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


class PriceScoutBot:
    """
    Continuous Background Scanner Bot that scouts the 538 universe assets,
    detects real-time price anomalies and volume breakouts, and feeds them
    to the Committee and Autonomous Trading Engine.
    """

    def __init__(self):
        self.scout_agent = PriceActionScoutAgent()

    def scan_universe_breakouts(
        self, candidate_tickers: List[str], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scans given tickers for real-time volume breakout candidates.
        """
        logger.info(
            f"🔭 [PRICE SCOUT BOT] Scanning {len(candidate_tickers)} assets for price breakouts..."
        )
        reports = []
        for tk in candidate_tickers:
            res = self.scout_agent.evaluate(tk)
            if res.get("vote") == "BUY" and res.get("conviction_score", 0) >= 65.0:
                reports.append(res)

        reports.sort(key=lambda x: x.get("conviction_score", 0), reverse=True)
        top_picks = reports[:top_n]

        # Save to price scout alerts file
        try:
            os.makedirs(os.path.dirname(SCOUT_ALERTS_FILE), exist_ok=True)
            with open(SCOUT_ALERTS_FILE, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_scanned": len(candidate_tickers),
                        "breakout_candidates_count": len(top_picks),
                        "candidates": top_picks,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Failed to persist scout alerts: {e}")

        return top_picks


def get_latest_scout_alerts() -> Dict[str, Any]:
    """Retrieves the latest price scout breakout alerts."""
    if os.path.exists(SCOUT_ALERTS_FILE):
        try:
            with open(SCOUT_ALERTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"timestamp": None, "candidates": []}
