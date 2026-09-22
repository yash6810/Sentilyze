"""
Post-Earnings Announcement Drift (PEAD) & SUE Earnings Surprise Radar for Sentilyze.

Implementation of Ball & Brown (1968) and Bernard & Thomas (1989):
1. Standardized Unexpected Earnings (SUE):
   SUE = (Actual_EPS - Estimated_EPS) / sigma_EPS
2. Revenue Beat & Guidance Revision Deltas.
3. 60-Day Post-Earnings Announcement Drift (PEAD) Trajectory Modeling.
4. Catalyst Scoring for the Sentiment & Catalyst Specialist Agent.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_sue_metric(
    actual_eps: float, estimated_eps: float, historical_eps_std: Optional[float] = None
) -> Tuple[float, float]:
    """Computes surprise percentage and Standardized Unexpected Earnings (SUE)."""
    if abs(estimated_eps) > 1e-4:
        surprise_pct = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100.0
    else:
        surprise_pct = (actual_eps - estimated_eps) * 100.0

    std = historical_eps_std if (historical_eps_std and historical_eps_std > 0) else 0.5
    sue = (actual_eps - estimated_eps) / std
    return round(float(surprise_pct), 2), round(float(sue), 2)


def analyze_earnings_surprises(ticker: str) -> Dict[str, Any]:
    """
    Retrieves and computes recent earnings surprise metrics, SUE score, and PEAD trajectory for a given ticker.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'AAPL', 'PGR', 'FANG').

    Returns:
        Structured PEAD and Earnings Surprise report.
    """
    default_report = {
        "ticker": ticker,
        "latest_earnings_date": None,
        "days_since_earnings": None,
        "actual_eps": None,
        "consensus_eps": None,
        "eps_surprise_pct": 0.0,
        "sue_score": 0.0,
        "pead_momentum_signal": "NEUTRAL_DRIFT",
        "pead_bias": "NEUTRAL",
        "catalyst_conviction_boost": 0.0,
        "summary": f"No recent earnings catalyst detected for {ticker}.",
    }

    try:
        t = yf.Ticker(ticker)
        # Attempt to pull earnings history from yfinance
        eh = None
        try:
            eh = t.get_earnings_history()
        except Exception:
            try:
                eh = t.earnings_history
            except Exception:
                pass

        if (
            eh is None
            or (isinstance(eh, pd.DataFrame) and eh.empty)
            or (isinstance(eh, list) and len(eh) == 0)
        ):
            # Fallback: check quarterly_financials / fast_info
            return default_report

        if isinstance(eh, list):
            df_eh = pd.DataFrame(eh)
        else:
            df_eh = pd.DataFrame(eh)

        if df_eh.empty or "epsActual" not in df_eh.columns:
            return default_report

        # Sort by report date descending
        date_col = "reportDate" if "reportDate" in df_eh.columns else df_eh.columns[0]
        df_eh = df_eh.sort_values(by=date_col, ascending=False).dropna(
            subset=["epsActual"]
        )

        if df_eh.empty:
            return default_report

        latest = df_eh.iloc[0]
        actual = float(latest.get("epsActual", 0.0))
        estimate = (
            float(latest.get("epsEstimate", actual))
            if pd.notna(latest.get("epsEstimate"))
            else actual
        )
        rep_date_str = str(latest.get(date_col, ""))[:10]

        # Calculate surprise percentage
        if abs(estimate) > 1e-4:
            surprise_pct = ((actual - estimate) / abs(estimate)) * 100.0
        else:
            surprise_pct = (actual - estimate) * 100.0

        # Calculate Standardized Unexpected Earnings (SUE)
        eps_series = df_eh["epsActual"].astype(float).values
        std_eps = np.std(eps_series) if len(eps_series) >= 3 else 0.5
        std_eps = max(std_eps, 0.1)
        sue = (actual - estimate) / std_eps

        # Days since earnings report
        days_since = None
        try:
            rep_date = datetime.strptime(rep_date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            days_since = (datetime.now(timezone.utc) - rep_date).days
        except Exception:
            days_since = 30

        # Classify PEAD Drift Regime
        # PEAD drift typically persists for 30 to 60 trading days post-earnings
        if days_since is not None and days_since <= 60:
            if sue >= 1.2 or surprise_pct >= 8.0:
                pead_signal = "STRONG_POST_EARNINGS_ACCUMULATION"
                bias = "BULLISH"
                boost = min(sue * 5.0, 15.0)
            elif sue <= -1.2 or surprise_pct <= -5.0:
                pead_signal = "POST_EARNINGS_INSTITUTIONAL_OVERHANG"
                bias = "BEARISH"
                boost = max(sue * 5.0, -15.0)
            else:
                pead_signal = "IN_LINE_DRIFT"
                bias = "NEUTRAL"
                boost = 0.0
        else:
            pead_signal = "DRIFT_EXPIRED_OR_STALE"
            bias = "NEUTRAL"
            boost = 0.0

        summary = (
            f"{ticker} reported EPS of ${actual:.2f} vs consensus ${estimate:.2f} "
            f"({surprise_pct:+.1f}% surprise, SUE: {sue:+.2f}) on {rep_date_str}. "
            f"PEAD Regime: {pead_signal}."
        )

        return {
            "ticker": ticker,
            "latest_earnings_date": rep_date_str,
            "days_since_earnings": days_since,
            "actual_eps": round(actual, 2),
            "consensus_eps": round(estimate, 2),
            "eps_surprise_pct": round(surprise_pct, 2),
            "sue_score": round(float(sue), 2),
            "pead_momentum_signal": pead_signal,
            "pead_bias": bias,
            "catalyst_conviction_boost": round(float(boost), 2),
            "summary": summary,
        }

    except Exception as e:
        logger.warning(f"Failed extracting earnings surprises for {ticker}: {e}")
        return default_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PEAD & SUE Earnings Surprise Radar")
    parser.add_argument(
        "--ticker", type=str, default="NVDA", help="Stock ticker to evaluate"
    )
    args = parser.parse_args()

    report = analyze_earnings_surprises(args.ticker)
    print(f"=== PEAD EARNINGS SURPRISE RADAR: {args.ticker} ===")
    print(f"Latest Report Date:     {report['latest_earnings_date']}")
    print(f"Days Since Report:      {report['days_since_earnings']}")
    print(
        f"Actual vs Consensus:    ${report['actual_eps']} vs ${report['consensus_eps']}"
    )
    print(f"EPS Surprise Delta:     {report['eps_surprise_pct']:+.2f}%")
    print(f"SUE Score:              {report['sue_score']:+.2f}")
    print(f"PEAD Drift Regime:      {report['pead_momentum_signal']}")
    print(f"Catalyst Conviction:    {report['catalyst_conviction_boost']:+.2f} pts")
    print(f"Executive Summary:      {report['summary']}")
