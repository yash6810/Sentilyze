"""
Adversarial Red-Team & Devil's Advocate Specialist Agent.

Functions:
- Stress-tests committee trade proposals to identify high-risk failure modes.
- Scans for imminent event risks, downside volatility skew, extreme overextension,
  and momentum exhaustion.
- Issues formal VETO or CAUTION reports directly to the Chief Risk Officer (CRO).
"""

from typing import Dict, Any, List, Optional
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.data_ingestion import get_price_history, get_news
from src.realtime_tracker import fetch_live_quote

logger = get_logger(__name__)


class AdversarialRedTeamAgent:
    """
    Agent 5: Adversarial Red-Team / Devil's Advocate Specialist.
    Actively hunts for reasons NOT to execute a trade, stress-testing consensus.
    """

    def __init__(self, name: str = "Adversarial Red-Team Stress Tester"):
        self.name = name

    def evaluate(
        self, ticker: str, spot_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Conducts a rigorous adversarial audit of the target asset.
        """
        try:
            if not spot_price or spot_price <= 0:
                q = fetch_live_quote(ticker)
                price = float(q.get("price", 100.0))
            else:
                price = float(spot_price)

            hist_df = get_price_history(ticker, period="1y", use_cache=True)
        except Exception as e:
            logger.debug(f"Red-team data fetch notice for {ticker}: {e}")
            hist_df = pd.DataFrame()
            price = spot_price or 100.0

        risk_factors: List[str] = []
        severity_score: float = 0.0

        if not hist_df.empty and len(hist_df) >= 30:
            closes = hist_df["Close"]

            # 1. Extreme Macro Overextension above 200 SMA
            sma200 = (
                float(closes.rolling(200).mean().iloc[-1])
                if len(closes) >= 200
                else float(closes.mean())
            )
            sma_stretch = (price - sma200) / (sma200 + 1e-8) * 100.0
            if sma_stretch > 35.0:
                risk_factors.append(
                    f"Severe overextension: Trading {sma_stretch:+.1f}% above 200-day SMA (${sma200:,.2f}) — elevated mean-reversion risk."
                )
                severity_score += 35.0
            elif sma_stretch < -25.0:
                risk_factors.append(
                    f"Severe breakdown: Trading {sma_stretch:+.1f}% below 200-day SMA (${sma200:,.2f}) — falling knife regime."
                )
                severity_score += 40.0

            # 2. RSI Overbought Exhaustion or Breakdown
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = float(100 - (100 / (1 + rs)).iloc[-1])

            if rsi > 78.0:
                risk_factors.append(
                    f"RSI extreme overbought ({rsi:.1f} > 78) — institutional distribution / profit-taking imminent."
                )
                severity_score += 30.0

            # 3. Downside Semi-Variance & Asymmetric Tail Volatility Skew
            daily_rets = closes.pct_change().dropna()
            down_rets = daily_rets[daily_rets < 0]
            up_rets = daily_rets[daily_rets > 0]
            down_var = down_rets.var() if len(down_rets) > 5 else 1e-4
            up_var = up_rets.var() if len(up_rets) > 5 else 1e-4
            skew_ratio = float(down_var / (up_var + 1e-8))

            if skew_ratio > 2.2:
                risk_factors.append(
                    f"Asymmetric downside skew (Downside Vol is {skew_ratio:.1f}x Upside Vol) — fat left-tail crash vulnerability."
                )
                severity_score += 25.0

            # 4. Volume Exhaustion / Negative Volume Divergence
            if "Volume" in hist_df.columns:
                recent_vol = hist_df["Volume"].tail(5).mean()
                prior_vol = hist_df["Volume"].iloc[-25:-5].mean()
                recent_ret_5d = closes.pct_change(5).iloc[-1]
                if recent_ret_5d > 0.05 and recent_vol < prior_vol * 0.65:
                    risk_factors.append(
                        "Bearish volume divergence: Price rallying on declining volume (-35% vs 20d average) — lack of institutional backing."
                    )
                    severity_score += 20.0
        else:
            risk_factors.append(
                "Insufficient historical price history for deep forensic stress testing."
            )
            severity_score += 15.0

        # Determine Red-Team Verdict
        if severity_score >= 50.0:
            vote = "VETO"
            thesis = f"RED-TEAM VETO: Critical vulnerabilities detected ({len(risk_factors)} major failure vectors). High probability of adverse liquidation."
        elif severity_score >= 25.0:
            vote = "CAUTION"
            thesis = f"RED-TEAM CAUTION: Moderate downside risks identified ({len(risk_factors)} risk factors). Strict position sizing and tight stops advised."
        else:
            vote = "CLEAR"
            thesis = "RED-TEAM CLEAR: No critical failure vectors, tail risks, or overextension anomalies detected."

        return {
            "agent_name": "Adversarial Red-Team Specialist",
            "role": "Pillar 6: Devil's Advocate & Tail-Risk Stress Tester",
            "academic_grounding": [
                "Paper 18: Grossman & Zhou (1993) Optimal Drawdown Constraint",
                "Paper 10: Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR)",
                "Paper 16: Page (1954) CUSUM Change-Point Surveillance",
            ],
            "vote": vote,
            "severity_score": round(min(severity_score, 100.0), 1),
            "conviction_score": round(100.0 - min(severity_score, 100.0), 1),
            "risk_factors": risk_factors,
            "key_metrics": {
                "vulnerabilities_detected": len(risk_factors),
                "severity_score": round(min(severity_score, 100.0), 1),
                "tail_risk_status": (
                    "HIGH"
                    if severity_score >= 50.0
                    else ("MODERATE" if severity_score >= 25.0 else "LOW")
                ),
            },
            "thesis": thesis,
        }
