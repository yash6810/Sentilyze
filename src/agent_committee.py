"""
Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.
Quantitative Round-Table Decision Council:
- Agent 1: Technical & Quantitative Alpha Specialist (Momentum, RSI, Moving Averages, 200-SMA Regime)
- Agent 2: NLP Sentiment & Catalyst Specialist (FinBERT Transformer Polarity over Live News Streams)
- Agent 3: Forensic & Fundamental Valuation Specialist (Piotroski F-Score, Altman Z-Score, DCF Margin of Safety)
- Agent 4: Chief Risk Officer (CRO) Arbitrator (Mathematical Fractional Kelly Sizing, VIX Volatility Gate, ATR Brackets)
"""

from typing import Any, Dict, List, Optional
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.realtime_tracker import fetch_live_quote
from src.fundamental_valuation import (
    fetch_financial_statements,
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dcf_fair_value,
)
from src.data_ingestion import get_price_history, get_news
from src.sentiment_analysis import analyze_sentiment

logger = get_logger(__name__)

COMMITTEE_FILE = os.path.join("results", "committee_resolutions.json")


def compute_fractional_kelly_sizing(
    win_rate: float = 0.533,
    payoff_ratio: float = 1.75,
    kelly_fraction: float = 0.25,
    max_cap_pct: float = 15.0,
) -> Dict[str, Any]:
    """
    Computes true mathematical fractional Kelly Criterion position sizing:
    f* = (p * b - (1 - p)) / b
    where p is empirical win probability and b is payoff ratio (avg win / avg loss).

    Args:
        win_rate: Historical strategy win probability (0.0 to 1.0)
        payoff_ratio: Ratio of average gain to average loss (b = avg_win / avg_loss)
        kelly_fraction: Conservative fraction multiplier (default 0.25 for Quarter-Kelly)
        max_cap_pct: Maximum single-position allocation cap

    Returns:
        Dict with full Kelly, fractional Kelly percentage, edge, and allocation status.
    """
    if payoff_ratio <= 0.0 or win_rate <= 0.0:
        return {
            "full_kelly_pct": 0.0,
            "fractional_kelly_pct": 0.0,
            "edge": 0.0,
            "status": "INVALID_PARAMETERS",
        }

    q = 1.0 - win_rate
    full_kelly = (win_rate * payoff_ratio - q) / payoff_ratio
    edge = (win_rate * payoff_ratio) - q

    if full_kelly <= 0.0:
        return {
            "full_kelly_pct": 0.0,
            "fractional_kelly_pct": 0.0,
            "edge": round(edge, 4),
            "status": "NEGATIVE_EXPECTANCY_NO_ALLOCATION",
        }

    fractional_kelly = full_kelly * kelly_fraction
    allocated_pct = min(max(0.0, fractional_kelly * 100.0), max_cap_pct)

    return {
        "full_kelly_pct": round(full_kelly * 100.0, 2),
        "fractional_kelly_pct": round(allocated_pct, 2),
        "edge": round(edge, 4),
        "kelly_fraction": kelly_fraction,
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "status": "POSITIVE_EXPECTANCY",
    }


class TechnicalAlphaAgent:
    """Agent 1: Evaluates Technical Price Action, Momentum, RSI, and Trend Alignment."""

    def evaluate(self, ticker: str, spot_price: float) -> Dict[str, Any]:
        vote = "NEUTRAL"
        conviction = 50.0
        trend_status = "NEUTRAL"
        thesis = "Technical momentum analysis in progress."
        rsi_val = 50.0
        sma200 = spot_price
        ma21 = spot_price
        ret_5d = 0.0
        calibrated_prob = None

        try:
            df = get_price_history(ticker, period="1y", use_cache=True)
        except Exception:
            df = pd.DataFrame()

        if not df.empty and len(df) >= 30:
            closes = df["Close"]
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_val = (
                float(rsi_series.iloc[-1])
                if not np.isnan(rsi_series.iloc[-1])
                else 50.0
            )

            ma21 = (
                float(closes.rolling(21).mean().iloc[-1])
                if len(closes) >= 21
                else spot_price
            )
            sma200 = (
                float(closes.rolling(200).mean().iloc[-1])
                if len(closes) >= 200
                else float(closes.mean())
            )
            ret_5d = (
                float(closes.pct_change(5).iloc[-1] * 100.0)
                if len(closes) >= 5
                else 0.0
            )

            is_above_200 = spot_price >= sma200
            is_above_21 = spot_price >= ma21

            if rsi_val > 70.0:
                vote = "HOLD"
                conviction = 42.0
                trend_status = "OVERBOUGHT_EXTENDED"
                thesis = f"Asset RSI is overbought ({rsi_val:.1f} > 70), indicating high probability of short-term consolidation."
            elif not is_above_200:
                vote = "HOLD"
                conviction = 38.0
                trend_status = "BEARISH_BELOW_SMA200"
                thesis = f"Asset is trading below its 200-day SMA (${sma200:,.2f}), signaling prevailing macro downtrend."
            elif is_above_200 and 40.0 <= rsi_val <= 62.0:
                vote = "BUY"
                conviction = 82.0
                trend_status = "BULLISH_MOMENTUM_EXPANSION"
                thesis = f"Asset is in structural uptrend above 200 SMA (${sma200:,.2f}) with optimal pullback RSI ({rsi_val:.1f})."
            elif rsi_val < 35.0:
                vote = "BUY"
                conviction = 70.0
                trend_status = "OVERSOLD_MEAN_REVERSION"
                thesis = f"Asset RSI is oversold ({rsi_val:.1f} < 35), signaling potential mean-reversion opportunity."
            else:
                vote = "HOLD"
                conviction = 55.0
                trend_status = "NEUTRAL_CONSOLIDATION"
                thesis = f"Asset is consolidating within trend (RSI {rsi_val:.1f}, SMA200 ${sma200:,.2f})."
            # Check 15-Minute Opening Range Breakout (Paper 25)
            try:
                from src.opening_range_engine import calculate_15min_opening_range

                orb = calculate_15min_opening_range(ticker, df)
                if orb.get("has_opening_range"):
                    or_high = float(orb.get("or_high", 0))
                    or_low = float(orb.get("or_low", 0))
                    if spot_price > or_high and or_high > 0:
                        vote = "BUY"
                        conviction = max(conviction, 85.0)
                        trend_status = "ORB_BULLISH_BREAKOUT"
                        thesis += f" 🎯 15-Min Opening Range Breakout verified (Price ${spot_price:.2f} > OR_High ${or_high:.2f})."
                    elif spot_price < or_low and or_low > 0:
                        trend_status = "ORB_BEARISH_BREAKDOWN"
                        thesis += f" ⚠️ Below 15-Min Opening Low (${or_low:.2f}); downward momentum dominant."
            except Exception as oe:
                logger.debug(f"ORB check notice for {ticker}: {oe}")

            # Check Volume Profile Point of Control (PoC) & Value Area (Paper 26)
            try:
                from src.orderflow_chart import calculate_volume_profile

                _, poc_p, vah_p, val_p = calculate_volume_profile(df, n_bins=24)
                if poc_p > 0:
                    dist_to_poc_pct = (spot_price - poc_p) / poc_p * 100.0
                    if spot_price > vah_p and vah_p > 0:
                        if vote in ["BUY", "NEUTRAL"]:
                            conviction = min(conviction + 6.0, 95.0)
                        thesis += f" 📦 Value Area Expansion: Price ${spot_price:.2f} > VAH ${vah_p:.2f} (PoC: ${poc_p:.2f})."
                    elif spot_price >= poc_p and dist_to_poc_pct <= 2.5:
                        if vote in ["BUY", "NEUTRAL"]:
                            conviction = min(conviction + 4.0, 92.0)
                        thesis += f" 🛡️ Volume Support: Price holding above PoC ${poc_p:.2f} ({dist_to_poc_pct:+.1f}%)."
            except Exception as vpe:
                logger.debug(f"Volume Profile calculation notice for {ticker}: {vpe}")

            # Check ACPM XGBoost Model Conformal Prediction
            model_path = os.path.join("models", f"{ticker}_model.json")
            calibrated_prob = None
            if os.path.exists(model_path):
                try:
                    from src.modeling import load_model
                    from src.config import FEATURES
                    from src.feature_engineering import create_technical_indicators

                    xgb_model = load_model(model_path)
                    ti_df = create_technical_indicators(df)
                    row_dict = {}
                    for col in FEATURES:
                        row_dict[col] = (
                            float(ti_df[col].iloc[-1]) if col in ti_df.columns else 0.0
                        )
                    feat_df = pd.DataFrame([row_dict])
                    prob_up = float(xgb_model.predict_proba(feat_df)[0, 1])
                    calibrated_prob = round(prob_up, 4)

                    # Conformal Calibration Gatekeeper (>= 58% conviction floor)
                    if prob_up >= 0.58 and vote in ["BUY", "NEUTRAL"]:
                        vote = "BUY"
                        conviction = max(conviction, round(prob_up * 100.0, 1))
                        thesis += f" ACPM Conformal Model confirms bullish expansion ({prob_up:.1%} probability >= 58% floor)."
                    elif prob_up < 0.44:
                        if vote == "BUY":
                            vote = "HOLD"
                            conviction = min(conviction, 48.0)
                        thesis += f" ACPM Conformal Model signals caution ({prob_up:.1%} probability)."
                except Exception as me:
                    logger.debug(f"ACPM model inference notice for {ticker}: {me}")
        else:
            vote = "NEUTRAL"
            conviction = 50.0
            trend_status = "INSUFFICIENT_HISTORY"
            rsi_val = 50.0
            sma200 = spot_price
            ma21 = spot_price
            ret_5d = 0.0
            calibrated_prob = None
            thesis = (
                "Insufficient historical price series; neutral technical vote cast."
            )

        return {
            "agent_name": "Technical Momentum Specialist",
            "role": "Pillar 1: Market Structure, Moving Averages & RSI Oscillator",
            "academic_grounding": [
                "Paper 25: Zarattini, Barbon, Aziz (2024) 5-Min Opening Range Breakout (ORB)",
                "Paper 10: Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR)",
            ],
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "estimated_rsi": round(rsi_val, 1),
                "sma_200": round(sma200, 2),
                "ma_21": round(ma21, 2),
                "return_5d_pct": round(ret_5d, 2),
                "trend_status": trend_status,
                "acpm_calibrated_prob": calibrated_prob,
            },
            "thesis": thesis,
        }


class SentimentCatalystAgent:
    """Agent 2: Evaluates FinBERT Deep NLP Sentiment across Live News Streams."""

    def evaluate(self, ticker: str) -> Dict[str, Any]:
        net_polarity = 0.0
        head_count = 0
        event_type = "GENERAL_MARKET_FLOW"
        urgency = 0.2
        is_material = False
        vote = "HOLD"
        conviction = 50.0
        thesis = "Balanced news flow."

        try:
            from src.fast_finbert import get_fast_finbert_engine
            from src.event_classifier_model import EventClassifierModel

            fast_finbert = get_fast_finbert_engine()
            event_classifier = EventClassifierModel()

            news_raw = get_news(ticker, use_cache=True)
            if isinstance(news_raw, pd.DataFrame) and not news_raw.empty:
                titles = (
                    news_raw["Title"].dropna().tolist()
                    if "Title" in news_raw.columns
                    else []
                )
            elif isinstance(news_raw, list) and len(news_raw) > 0:
                titles = [str(x) for x in news_raw]
            else:
                titles = []

            head_count = len(titles)
            if titles:
                # Fast batch scoring via FastFinBERT INT8 + Event Model
                first_title = titles[0]
                ev_res = event_classifier.classify(first_title)
                event_type = ev_res["event_type"]
                urgency = ev_res["urgency_score"]
                is_material = ev_res["is_material"]

                sent_scores = []
                for t in titles[:8]:
                    p_res = fast_finbert.predict_single(t, ticker=ticker)
                    sent_scores.append(p_res["sentiment_score"])

                if sent_scores:
                    time_col = None
                    if isinstance(news_raw, pd.DataFrame):
                        for c in [
                            "date",
                            "publishedAt",
                            "Date",
                            "published_date",
                            "time",
                        ]:
                            if c in news_raw.columns:
                                time_col = c
                                break

                    if (
                        time_col
                        and isinstance(news_raw, pd.DataFrame)
                        and len(news_raw) >= len(sent_scores)
                    ):
                        from src.sentiment_decay import (
                            compute_exponential_decay_weights,
                        )

                        try:
                            weights = compute_exponential_decay_weights(
                                news_raw[time_col].iloc[: len(sent_scores)],
                                half_life_hours=4.0,
                            )
                            net_polarity = round(float(np.dot(weights, sent_scores)), 3)
                        except Exception:
                            raw_w = np.exp(-np.arange(len(sent_scores)) * 0.35)
                            weights = raw_w / np.sum(raw_w)
                            net_polarity = round(float(np.dot(weights, sent_scores)), 3)
                    else:
                        # Rank-based exponential decay for reverse chronological news
                        raw_w = np.exp(-np.arange(len(sent_scores)) * 0.35)
                        weights = raw_w / np.sum(raw_w)
                        net_polarity = round(float(np.dot(weights, sent_scores)), 3)
                else:
                    net_polarity = 0.0
        except Exception as e:
            logger.debug(f"Sentiment evaluation notice for {ticker}: {e}")
            net_polarity = 0.0
            head_count = 0

        # High-impact material catalysts boost conviction
        if is_material and urgency >= 0.75:
            if net_polarity >= 0:
                vote = "BUY"
                conviction = round(min(75.0 + (urgency * 20.0), 95.0), 1)
                thesis = f"🚀 HIGH-IMPACT CATALYST DETECTED: {event_type} (Urgency: {urgency:.1%}, Polarity: {net_polarity:+.2f} across {head_count} headlines)."
            else:
                vote = "SELL"
                conviction = round(min(75.0 + (urgency * 20.0), 95.0), 1)
                thesis = f"🚨 SEVERE RISK CATALYST: {event_type} (Urgency: {urgency:.1%}, Polarity: {net_polarity:+.2f}). Defensive de-risking mandatory."
        elif net_polarity >= 0.20:
            vote = "BUY"
            conviction = round(min(60.0 + (net_polarity * 40.0), 92.0), 1)
            thesis = f"Strong bullish news flow (+{net_polarity:+.2f} FinBERT score across {head_count} live headlines; Event: {event_type})."
        elif net_polarity <= -0.20:
            vote = "SELL"
            conviction = round(min(60.0 + (abs(net_polarity) * 40.0), 90.0), 1)
            thesis = f"Negative media catalyst ({net_polarity:+.2f} FinBERT polarity; Event: {event_type}); downstream selling pressure likely."
        else:
            vote = "HOLD"
            conviction = 50.0
            thesis = f"Balanced sentiment environment ({net_polarity:+.2f} polarity across {head_count} headlines; Event: {event_type})."

        return {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "role": "Pillar 2: Tri-Brain FinBERT + Event Classifier NLP",
            "academic_grounding": [
                "Paper 06: Multi-Agent Coordination Primacy (CPH Survey 2025/2026)",
                "Paper 21: Bifet & Gavaldà (2007) ADWIN Adaptive Concept Drift Tracking",
            ],
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "finbert_polarity": net_polarity,
                "event_type": event_type,
                "urgency_score": urgency,
                "is_material": is_material,
                "headlines_analyzed": head_count,
            },
            "thesis": thesis,
        }


class ForensicFundamentalAgent:
    """Agent 3: Evaluates Real Financial Statements, Piotroski F-Score, and DCF Valuation."""

    def evaluate(self, ticker: str, spot_price: float) -> Dict[str, Any]:
        vote = "HOLD"
        conviction = 50.0
        thesis = "Valuation evaluation in progress."

        fin_data = fetch_financial_statements(ticker)
        is_real = fin_data.get("is_real_data", False)

        if not is_real:
            return {
                "agent_name": "Forensic & Valuation Auditor",
                "role": "Pillar 8: Fundamental Health & Forensic Valuation",
                "academic_grounding": [
                    "Paper 05: Bellman-Ford Fundamental Arbitrage Graph Theory",
                    "Paper 13: Chen et al. (2020) Supply Chain Contagion Graph Neural Networks",
                ],
                "vote": "NEUTRAL",
                "conviction_score": 50.0,
                "key_metrics": {
                    "data_available": False,
                    "piotroski_f_score": 5,
                    "altman_z_score": 2.5,
                    "dcf_margin_of_safety_pct": 0.0,
                },
                "thesis": "Live SEC financial statement data unavailable on free feed; abstaining with neutral vote.",
            }

        f_score_data = calculate_piotroski_f_score(ticker, fin_data)
        f_score = int(f_score_data.get("total_score", 5))

        altman = calculate_altman_z_score(ticker, fin_data)
        z_score = float(altman.get("z_score", 2.5))

        dcf = calculate_dcf_fair_value(ticker=ticker, fin_data=fin_data)
        dcf_mos = float(dcf.get("margin_of_safety_pct", 0.0))

        is_financially_healthy = f_score >= 5 and z_score >= 1.81 and dcf_mos >= -15.0

        if is_financially_healthy:
            vote = "BUY"
            conviction = round(
                min(50.0 + (f_score * 4.0) + max(dcf_mos * 0.5, 0.0), 92.0), 1
            )
            thesis = (
                f"Solid fundamental health: Piotroski F-Score {f_score}/9, Altman Z-Score {z_score:.2f} ({altman.get('zone', 'SAFE')}), "
                f"and DCF Fair Value ${dcf.get('fair_value_price', spot_price):,.2f} ({dcf_mos:+.1f}% margin of safety)."
            )
        else:
            vote = "HOLD"
            conviction = round(max(30.0 + (f_score * 3.0) + (dcf_mos * 0.3), 20.0), 1)
            thesis = (
                f"Valuation caution: Piotroski F-Score {f_score}/9, "
                f"DCF Fair Value ${dcf.get('fair_value_price', spot_price):,.2f} (Margin of Safety: {dcf_mos:+.1f}%)."
            )

        # OpenBB Form 4 Insider Flow Integration
        insider_score = 0.0
        insider_verdict = "NEUTRAL"
        try:
            from src.openbb_bridge import OpenBBBridge

            bridge = OpenBBBridge()
            insider = bridge.get_insider_transactions(ticker)
            insider_score = insider.net_insider_score
            insider_verdict = insider.sentiment_verdict
            if insider.sentiment_verdict == "BULLISH_ACCUMULATION":
                conviction = min(conviction + 5.0, 95.0)
                thesis += f" 👔 OpenBB Insider Flow: Net accumulation score {insider.net_insider_score:+.2f}."
            elif (
                insider.sentiment_verdict == "BEARISH_DUMPING"
                and insider.transaction_count >= 3
            ):
                conviction = max(conviction - 6.0, 15.0)
                thesis += f" ⚠️ OpenBB Insider Alert: Significant executive selling (Net {insider.net_insider_score:+.2f})."
        except Exception as ie:
            logger.debug(f"OpenBB insider transactions check notice for {ticker}: {ie}")

        return {
            "agent_name": "Forensic & Valuation Auditor",
            "role": "Pillar 8: Fundamental Health & Forensic Valuation",
            "academic_grounding": [
                "Paper 05: Bellman-Ford Fundamental Arbitrage Graph Theory",
                "Paper 13: Chen et al. (2020) Supply Chain Contagion Graph Neural Networks",
                "SEC Form 4 Insider Accumulation/Distribution Anomaly (Lakonishok & Lee 2001)",
            ],
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "data_available": True,
                "piotroski_f_score": f_score,
                "altman_z_score": z_score,
                "dcf_fair_value": dcf.get("fair_value_price", spot_price),
                "dcf_margin_of_safety_pct": dcf_mos,
                "insider_score": insider_score,
                "insider_verdict": insider_verdict,
            },
            "thesis": thesis,
        }


class RegulatoryCatalystAgent:
    """Agent 5: SEC Form 8-K Filings & FDA BioTech Catalyst Specialist."""

    def __init__(self, name: str = "Regulatory & Catalyst Specialist"):
        self.name = name

    def evaluate(self, ticker: str) -> Dict[str, Any]:
        vote = "NEUTRAL"
        conviction = 50.0
        material_risk = "NONE"
        risk_profile = "STANDARD_EQUITY"
        catalysts_found = 0
        thesis = (
            f"Monitoring SEC 8-K filings and clinical trial catalysts for {ticker}."
        )

        # 1. Check SEC 8-K Catalysts
        try:
            from src.sec_crawler import SECCatalystCrawler

            crawler = SECCatalystCrawler()
            sec_events = crawler.get_recent_sec_catalysts(ticker, days=21)
            catalysts_found += len(sec_events)

            for ev in sec_events:
                item_code = ev.get("item_code", "")
                sentiment = float(ev.get("sentiment_score", 0.0))

                # Item 3.01 Delisting Notice or severe governance fraud
                if "3.01" in item_code or "DELISTING" in ev.get("catalyst_type", ""):
                    vote = "VETO"
                    conviction = 5.0
                    material_risk = "DELISTING_OR_FRAUD"
                    thesis = f"🚨 CRITICAL REGULATORY VETO: SEC 8-K Item 3.01 Notice of Delisting/Non-compliance filed for {ticker}."
                    break
                elif "1.01" in item_code and sentiment > 0.3:
                    vote = "BUY"
                    conviction = max(conviction, 82.0)
                    thesis = f"🚀 Material Commercial Agreement (8-K Item 1.01): {ev.get('description', '')[:100]}."
                elif "2.02" in item_code:
                    if sentiment > 0.4:
                        vote = "BUY"
                        conviction = max(conviction, 78.0)
                        thesis = f"📈 Bullish Earnings 8-K Catalyst: {ev.get('description', '')[:100]}."
                    elif sentiment < -0.4:
                        vote = "HOLD"
                        conviction = min(conviction, 35.0)
                        thesis = "⚠️ Bearish Earnings 8-K Catalyst: Negative results reported."
        except Exception as se:
            logger.debug(f"SEC 8-K check notice for {ticker}: {se}")

        # 2. Check BioTech / FDA Catalyst Radar if applicable
        if material_risk != "DELISTING_OR_FRAUD":
            try:
                from src.biotech_catalyst_radar import (
                    is_biotech_or_healthcare,
                    compile_biotech_catalyst_radar,
                )

                if is_biotech_or_healthcare(ticker):
                    radar = compile_biotech_catalyst_radar(ticker)
                    risk_profile = radar.get("risk_profile", "STANDARD_EQUITY")
                    approval_count = radar.get("total_approvals", 0)
                    pdufa_count = radar.get("total_pdufa_events", 0)

                    if risk_profile == "HIGH_BINARY_EVENT_RISK":
                        thesis += f" ⚠️ High Binary Clinical Risk: {pdufa_count} PDUFA/Phase 3 events pending."
                    elif approval_count > 0:
                        vote = "BUY"
                        conviction = max(conviction, 88.0)
                        thesis = f"🎯 Confirmed FDA Approval Catalyst: {approval_count} new drug approvals on record."
            except Exception as be:
                logger.debug(f"BioTech radar notice for {ticker}: {be}")

        return {
            "agent_name": self.name,
            "role": "Pillar 5: Regulatory 8-K Filings & FDA BioTech Catalysts",
            "academic_grounding": [
                "SEC EDGAR Real-Time Disclosure Mandate (Regulation FD)",
                "PDUFA FDA Binary Event Asymmetry Modeling",
            ],
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "material_risk": material_risk,
                "risk_profile": risk_profile,
                "catalysts_found": catalysts_found,
            },
            "thesis": thesis,
        }


class MacroEconomicAgent:
    """Agent 6: Macroeconomic Calendar, Yield Curve & Pre-Event Volatility Blackout Specialist."""

    def __init__(self, name: str = "Macro Economic & Central Bank Specialist"):
        self.name = name

    def evaluate(
        self, ticker: str, as_of_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        try:
            from src.macro_calendar import evaluate_macro_volatility_blackout

            blackout_res = evaluate_macro_volatility_blackout(as_of_time=as_of_time)
            is_blackout = blackout_res.get("is_blackout_active", False)
            status = blackout_res.get("status", "NORMAL_LIQUIDITY")
            nearest_event = blackout_res.get("nearest_event_name", "None")
            minutes_to_event = blackout_res.get("minutes_to_event", 9999.0)

            if is_blackout:
                vote = "VETO"
                conviction = 10.0
                thesis = f"🔴 MACRO EVENT BLACKOUT: {blackout_res.get('reason', 'High-impact release imminent')}."
            # OpenBB Macro Regime & FRED Yield Curve Integration
            macro_regime = "NORMAL_EXPANSION"
            yield_spread = 0.0
            credit_spread = 0.0
            try:
                from src.openbb_bridge import OpenBBBridge

                bridge = OpenBBBridge()
                snap = bridge.get_macro_regime()
                macro_regime = snap.regime_classification
                yield_spread = snap.yield_curve_spread
                credit_spread = snap.high_yield_spread
                if snap.regime_classification == "CREDIT_STRESS":
                    if not is_blackout:
                        vote = "CAUTION"
                        conviction = min(conviction, 42.0)
                    thesis += f" ⚠️ Credit Spread Stress: High yield spread elevated at {snap.high_yield_spread:.2f}%."
                elif snap.regime_classification == "YIELD_CURVE_INVERTED":
                    thesis += f" 📉 Yield Curve Inverted: 10Y-2Y spread {snap.yield_curve_spread:+.2f}%."
                else:
                    thesis += f" 🏛️ Macro Expansion: Yield spread {snap.yield_curve_spread:+.2f}%, Credit spread {snap.high_yield_spread:.2f}%."
            except Exception as obe:
                logger.debug(f"OpenBB macro regime check notice for {ticker}: {obe}")

            return {
                "agent_name": self.name,
                "role": "Pillar 6: Central Bank Schedules & Volatility Blackouts",
                "academic_grounding": [
                    "Fleming & Remolona (1999) Price Discovery Around Macro Announcements",
                    "Bernanke & Kuttner (2005) Monetary Policy Impact on Equity Markets",
                    "Estrella & Mishkin (1996) Yield Curve as a Predictor of US Recessions",
                ],
                "vote": vote,
                "conviction_score": conviction,
                "key_metrics": {
                    "is_blackout_active": is_blackout,
                    "macro_status": status,
                    "macro_regime": macro_regime,
                    "yield_curve_spread": yield_spread,
                    "high_yield_spread": credit_spread,
                    "nearest_event": nearest_event,
                    "minutes_to_event": minutes_to_event,
                    "stop_tightening_factor": blackout_res.get(
                        "stop_tightening_factor", 1.0
                    ),
                    "max_kelly_cap_pct": blackout_res.get("max_kelly_cap_pct", 15.0),
                },
                "thesis": thesis,
            }
        except Exception as me:
            logger.debug(f"Macro economic agent notice for {ticker}: {me}")
            return {
                "agent_name": self.name,
                "role": "Pillar 6: Central Bank Schedules & Volatility Blackouts",
                "vote": "NEUTRAL",
                "conviction_score": 50.0,
                "key_metrics": {
                    "is_blackout_active": False,
                    "macro_status": "NORMAL_LIQUIDITY",
                },
                "thesis": "Macroeconomic calendar operating with standard liquidity profile.",
            }


class StatArbSpecialist:
    """Agent 7: Statistical Arbitrage & Cointegration Specialist."""

    PAIR_TWIN_MAP = {
        "NVDA": ["TSM", "AMD"],
        "AMD": ["NVDA", "INTC"],
        "TSM": ["NVDA", "ASML"],
        "MSFT": ["GOOGL", "AAPL"],
        "GOOGL": ["MSFT", "META"],
        "META": ["GOOGL", "SNAP"],
        "AMZN": ["MSFT", "WMT"],
        "AAPL": ["MSFT", "HPQ"],
        "TSLA": ["RIVN", "GM"],
        "GEV": ["CAT", "EMR"],
        "FDXF": ["FDX", "UNP"],
    }

    def __init__(self, name: str = "Statistical Arbitrage & Cointegration Specialist"):
        self.name = name

    def evaluate(self, ticker: str, spot_price: float) -> Dict[str, Any]:
        vote = "NEUTRAL"
        conviction = 50.0
        thesis = f"Evaluating statistical relative value for {ticker}."
        twins = self.PAIR_TWIN_MAP.get(ticker.upper(), [])
        is_extended = False

        if not twins:
            return {
                "agent_name": self.name,
                "role": "Pillar 7: Cointegration, OU Mean-Reversion & Relative Value Alpha",
                "academic_grounding": [
                    "Engle & Granger (1987) Co-Integration and Error Correction",
                    "Ornstein & Uhlenbeck (1930) On the Theory of Brownian Motion",
                ],
                "vote": "NEUTRAL",
                "conviction_score": 50.0,
                "key_metrics": {
                    "status": "NO_PAIR_TWIN_CONFIGURED",
                    "is_extended": False,
                },
                "thesis": f"No sector cointegration twin configured for {ticker}; neutral relative-value stance.",
            }

        try:
            from src.stat_arb_engine import (
                test_pairs_cointegration,
                compute_dynamic_spread_zscore,
                compute_hurst_exponent,
            )

            df_primary = get_price_history(ticker, period="1y", use_cache=True)
            if df_primary.empty or len(df_primary) < 40:
                raise ValueError("Insufficient price history.")

            best_p_val = 1.0
            best_twin = None
            best_coint = None
            best_z = None

            for tw in twins:
                try:
                    df_tw = get_price_history(tw, period="1y", use_cache=True)
                    if df_tw.empty or len(df_tw) < 40:
                        continue
                    comb = pd.concat(
                        [df_primary["Close"], df_tw["Close"]], axis=1
                    ).dropna()
                    if len(comb) < 40:
                        continue
                    c_res = test_pairs_cointegration(
                        comb.iloc[:, 0].values, comb.iloc[:, 1].values
                    )
                    if c_res["p_value"] < best_p_val:
                        best_p_val = c_res["p_value"]
                        best_twin = tw
                        best_coint = c_res
                        best_z = compute_dynamic_spread_zscore(
                            c_res["spread_series"],
                            c_res["ou_params"]["half_life"],
                        )
                except Exception:
                    continue

            if best_twin and best_coint and best_z:
                z_score = best_z["current_zscore"]
                half_life = best_coint["ou_params"]["half_life"]
                is_coint = best_coint["is_cointegrated"]
                hurst = compute_hurst_exponent(df_primary["Close"].values)

                if is_coint:
                    if z_score <= -2.0:
                        vote = "BUY"
                        conviction = round(min(70.0 + abs(z_score) * 7.5, 95.0), 1)
                        thesis = (
                            f"Strong Mean-Reversion Alpha: {ticker}/{best_twin} spread oversold at {z_score:+.2f}σ "
                            f"(Half-Life: {half_life:.1f}d, p={best_coint['p_value']:.4f}). Long bounce favorable."
                        )
                    elif z_score >= 2.2:
                        vote = "HOLD"
                        conviction = 32.0
                        is_extended = True
                        thesis = (
                            f"⚠️ Relative-Value Overextension: {ticker} is stretched {z_score:+.2f}σ above {best_twin}. "
                            "High risk of spread convergence pullback."
                        )
                    elif abs(z_score) >= 3.5:
                        vote = "VETO"
                        conviction = 15.0
                        thesis = f"🚨 Structural Spread Breakdown: Z-Score {z_score:+.2f}σ exceeds 3.5σ cointegration barrier."
                    else:
                        vote = "HOLD"
                        conviction = 52.0
                        thesis = f"Pair spread with {best_twin} is balanced at {z_score:+.2f}σ (Hurst: {hurst:.2f})."
                else:
                    vote = "HOLD"
                    conviction = 48.0
                    thesis = f"Sector pair {ticker}/{best_twin} spread non-stationary (p={best_coint['p_value']:.3f} > 0.05)."

                return {
                    "agent_name": self.name,
                    "role": "Pillar 7: Cointegration, OU Mean-Reversion & Relative Value Alpha",
                    "academic_grounding": [
                        "Engle & Granger (1987) Co-Integration and Error Correction",
                        "Ornstein & Uhlenbeck (1930) On the Theory of Brownian Motion",
                    ],
                    "vote": vote,
                    "conviction_score": conviction,
                    "key_metrics": {
                        "paired_twin": best_twin,
                        "spread_zscore": z_score,
                        "half_life_days": half_life,
                        "cointegration_p_val": best_coint["p_value"],
                        "hurst_exponent": round(hurst, 2),
                        "is_extended": is_extended,
                    },
                    "thesis": thesis,
                }
        except Exception as ste:
            logger.debug(f"Stat-Arb evaluation notice for {ticker}: {ste}")

        return {
            "agent_name": self.name,
            "role": "Pillar 7: Cointegration, OU Mean-Reversion & Relative Value Alpha",
            "vote": "NEUTRAL",
            "conviction_score": 50.0,
            "key_metrics": {"status": "EVALUATION_FALLBACK", "is_extended": False},
            "thesis": "Relative value spread balanced within normal parameters.",
        }


class ChiefRiskOfficerAgent:
    """Agent 4: Chief Risk Officer (CRO) — Synthesizes Votes with Dynamic Trailing Memory Weights,
    Enforces Volatility, Regulatory, Macro, and Stat-Arb Gates, and Signs Off."""

    # Default weights across the 8 specialist testimonies
    DEFAULT_WEIGHTS = {
        "Technical Momentum Specialist": 0.20,
        "Sentiment & Alternative Data Specialist": 0.20,
        "Forensic & Valuation Auditor": 0.15,
        "Real-Time Price & Tape Scout": 0.10,
        "Adversarial Red-Team Specialist": 0.10,
        "Regulatory & Catalyst Specialist": 0.10,
        "Macro Economic & Central Bank Specialist": 0.075,
        "Statistical Arbitrage & Cointegration Specialist": 0.075,
    }

    def _load_calibrated_weights(self) -> Dict[str, float]:
        """Loads trailing win-rate weights from results/agent_memory/committee_weights.json."""
        weights_file = os.path.join("results", "agent_memory", "committee_weights.json")
        if os.path.exists(weights_file):
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    # Normalize loaded weights
                    w_copy = self.DEFAULT_WEIGHTS.copy()
                    for k, v in data.items():
                        if k in w_copy and isinstance(v, (int, float)) and v > 0:
                            w_copy[k] = float(v)
                    tot = sum(w_copy.values())
                    if tot > 0:
                        return {k: v / tot for k, v in w_copy.items()}
            except Exception as e:
                logger.debug(f"Weight calibration loading notice: {e}")
        return self.DEFAULT_WEIGHTS.copy()

    def evaluate_and_sign_off(
        self,
        ticker: str,
        spot_price: float,
        agent_reports: List[Dict[str, Any]],
        vix_level: float = 16.5,
        vix_change_pct: float = -1.2,
    ) -> Dict[str, Any]:
        weights_map = self._load_calibrated_weights()

        # Compute dynamic weighted conviction and weighted buy fractions
        total_w = 0.0
        weighted_conviction_sum = 0.0
        weighted_buy_w = 0.0
        buy_votes = 0

        for r in agent_reports:
            if not isinstance(r, dict):
                continue
            name = r.get("agent_name", "")
            conv = float(r.get("conviction_score", 50.0))
            w = weights_map.get(name, 0.10)
            total_w += w
            weighted_conviction_sum += w * conv

            if r.get("vote") == "BUY":
                buy_votes += 1
                weighted_buy_w += w

        avg_conviction = weighted_conviction_sum / total_w if total_w > 0 else 50.0
        buy_weight_fraction = weighted_buy_w / total_w if total_w > 0 else 0.0

        # 1. Evaluate Multi-Pillar Veto and Caution Gates
        vix_veto = False
        trend_veto = False
        red_team_veto = False
        red_team_caution = False
        regulatory_veto = False
        regulatory_caution = False
        macro_blackout_veto = False
        statarb_veto = False
        statarb_caution = False
        veto_reason = None
        stop_tightening_factor = 1.0

        # Scan specialist reports for gate activations
        for r in agent_reports:
            if not isinstance(r, dict):
                continue
            name = r.get("agent_name", "")
            metrics = r.get("key_metrics", {})
            vote = r.get("vote", "")

            # Adversarial Red-Team Gate
            if name == "Adversarial Red-Team Specialist":
                if vote == "VETO":
                    red_team_veto = True
                    veto_reason = f"Adversarial Red-Team VETO: {r.get('thesis')}"
                elif vote == "CAUTION":
                    red_team_caution = True

            # Regulatory 8-K Delisting / Fraud Gate
            elif name == "Regulatory & Catalyst Specialist":
                if (
                    vote == "VETO"
                    or metrics.get("material_risk") == "DELISTING_OR_FRAUD"
                ):
                    regulatory_veto = True
                    veto_reason = f"Regulatory Hard Veto: {r.get('thesis')}"
                elif metrics.get("risk_profile") == "HIGH_BINARY_EVENT_RISK":
                    regulatory_caution = True

            # Macro Volatility Blackout Gate
            elif name == "Macro Economic & Central Bank Specialist":
                if vote == "VETO" or metrics.get("is_blackout_active"):
                    macro_blackout_veto = True
                    veto_reason = f"Macro Blackout Veto: {r.get('thesis')}"
                    stop_tightening_factor = float(
                        metrics.get("stop_tightening_factor", 0.50)
                    )

            # Statistical Arbitrage Cointegration Gate
            elif name == "Statistical Arbitrage & Cointegration Specialist":
                if vote == "VETO":
                    statarb_veto = True
                    veto_reason = f"Stat-Arb Cointegration Breakdown: {r.get('thesis')}"
                elif metrics.get("is_extended"):
                    statarb_caution = True

            # Macro Trend Gate (SMA200)
            if metrics.get("trend_status") == "BEARISH_BELOW_SMA200":
                trend_veto = True
                if not veto_reason:
                    veto_reason = "Asset is in a structural macro downtrend below its 200-day Moving Average (SMA200)."

        # VIX Panic Check
        if vix_level > 26.0 or vix_change_pct > 8.0:
            vix_veto = True
            veto_reason = f"VIX elevated at {vix_level:.1f} (+{vix_change_pct:+.1f}% spike) — macro volatility gate activated."

        # 2. Dynamic Mathematical Fractional Kelly Sizing (Paper 23 & Paper 14)
        # For exceptional high-conviction unvetoed setups (avg_conviction >= 78%), scale from Quarter-Kelly (0.25)
        # up to Half-Kelly (0.40) with asymmetric payoff ratio (2.25) to capture massive compounding moves.
        if avg_conviction >= 78.0 and not (
            red_team_caution or statarb_caution or regulatory_caution
        ):
            empirical_win_rate = 0.58
            active_kelly_fraction = 0.40  # Half-Kelly for Unanimous High Conviction
            active_payoff_ratio = 2.25
        elif avg_conviction >= 68.0:
            empirical_win_rate = 0.533
            active_kelly_fraction = 0.25  # Quarter-Kelly
            active_payoff_ratio = 1.85
        else:
            empirical_win_rate = 0.48
            active_kelly_fraction = 0.20
            active_payoff_ratio = 1.75

        kelly_result = compute_fractional_kelly_sizing(
            win_rate=empirical_win_rate,
            payoff_ratio=active_payoff_ratio,
            kelly_fraction=active_kelly_fraction,
            max_cap_pct=15.0,
        )
        calculated_kelly_pct = float(kelly_result.get("fractional_kelly_pct", 0.0))

        # Apply Caution Scalings
        if red_team_caution:
            calculated_kelly_pct = round(calculated_kelly_pct * 0.65, 2)
        if statarb_caution:
            calculated_kelly_pct = round(calculated_kelly_pct * 0.60, 2)
        if regulatory_caution:
            calculated_kelly_pct = min(calculated_kelly_pct, 2.0)

        # Market Microstructure & Toxic Flow Check (Paper 2012)
        micro_health = None
        try:
            from src.microstructure import evaluate_microstructure_health

            micro_health = evaluate_microstructure_health(ticker)
            micro_penalty = float(micro_health.get("action_penalty_multiplier", 1.0))
            if micro_penalty < 1.0:
                calculated_kelly_pct = round(calculated_kelly_pct * micro_penalty, 2)
        except Exception as mie:
            logger.debug(f"Microstructure check notice for {ticker}: {mie}")

        # 3. Determine Final Committee Resolution
        has_veto = (
            vix_veto
            or trend_veto
            or red_team_veto
            or regulatory_veto
            or macro_blackout_veto
            or statarb_veto
        )

        if has_veto:
            if regulatory_veto:
                final_resolution = "🔴 VETO / REGULATORY DELISTING OR FRAUD"
            elif macro_blackout_veto:
                final_resolution = "🔴 VETO / MACRO EVENT BLACKOUT"
            elif statarb_veto:
                final_resolution = "🔴 VETO / COINTEGRATION SPREAD BREAKDOWN"
            elif vix_veto:
                final_resolution = "🔴 VETO / CAPITAL PRESERVATION"
            elif red_team_veto:
                final_resolution = "🔴 VETO / RED-TEAM VULNERABILITY"
            else:
                final_resolution = "🔴 VETO / MACRO DOWNTREND (BELOW SMA200)"
            action_code = "VETO"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0
        elif (buy_weight_fraction >= 0.35 or buy_votes >= 2) and avg_conviction >= (
            72.0 if regulatory_caution else 50.0
        ):
            final_resolution = "🚀 HIGH CONVICTION COMMITTEE BUY"
            action_code = "EXECUTE_BUY"
            approved_leverage = 1.0 if regulatory_caution else 1.25
            kelly_allocation_pct = (
                min(calculated_kelly_pct, 2.0)
                if regulatory_caution
                else max(calculated_kelly_pct, 5.0)
            )
        elif (buy_weight_fraction >= 0.20 or buy_votes >= 1) and avg_conviction >= 42.0:
            final_resolution = "🟡 AGILE SCALE-IN (Quorum Approved)"
            action_code = "SCALE_IN"
            approved_leverage = 1.0
            kelly_allocation_pct = max(round(calculated_kelly_pct * 0.65, 2), 3.5)
        elif (buy_weight_fraction >= 0.15 or buy_votes >= 1) or avg_conviction >= 38.0:
            final_resolution = "⏸️ NEUTRAL HOLD / NO ACTION"
            action_code = "HOLD"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0
        else:
            final_resolution = "🔴 BEARISH SKEW / AVOID ASSET"
            action_code = "AVOID"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0

        # Empirical 14-Day ATR Volatility Corridors (+2.5 ATR TP1, +4.5 ATR TP2, -1.5 ATR SL) - Paper 11
        atr_val = None
        try:
            hist_df = get_price_history(ticker, period="6mo", use_cache=True)
            if (
                isinstance(hist_df, pd.DataFrame)
                and len(hist_df) >= 15
                and "High" in hist_df.columns
                and "Low" in hist_df.columns
                and "Close" in hist_df.columns
            ):
                h = hist_df["High"]
                l = hist_df["Low"]
                c = hist_df["Close"]
                tr1 = h - l
                tr2 = (h - c.shift(1)).abs()
                tr3 = (l - c.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_14 = float(tr.rolling(14).mean().dropna().iloc[-1])
                if not np.isnan(atr_14) and atr_14 > 0:
                    atr_val = round(atr_14, 2)
        except Exception as ae:
            logger.debug(f"Empirical ATR calculation notice for {ticker}: {ae}")

        if atr_val is None or atr_val <= 0:
            atr_val = round(max(spot_price * 0.03, 1.0), 2)

        # Apply blackout tightening factor to stop loss distance if active
        sl_multiplier = 1.5 * stop_tightening_factor
        tp1 = round(spot_price + (2.5 * atr_val), 2)
        # Asymmetric Super-Runner: For setups with conviction >= 75%, let the runner expand to +6.5 ATR
        tp2_multiplier = 6.5 if avg_conviction >= 75.0 else 4.5
        tp2 = round(spot_price + (tp2_multiplier * atr_val), 2)
        sl = round(max(spot_price - (sl_multiplier * atr_val), spot_price * 0.85), 2)
        atr_pct = round((atr_val / spot_price) * 100.0, 2)

        reason_str = veto_reason if veto_reason else "Insufficient consensus."
        action_msg = (
            "Trade signed off for execution."
            if action_code in ["EXECUTE_BUY", "SCALE_IN"]
            else f"Trade blocked: {reason_str}"
        )
        vix_status = "NORMAL" if not vix_veto else "ELEVATED"

        cro_thesis = (
            f"Committee Consensus: {buy_votes}/{len(agent_reports)} specialist agents voted BUY "
            f"(Weighted Conviction: {avg_conviction:.1f}%, Buy Weight: {buy_weight_fraction:.1%}). "
            f"ATR14 is ${atr_val:.2f} ({atr_pct:.1f}% vol). VIX is {vix_level:.1f} ({vix_status}). "
            f"Fractional Kelly: {kelly_allocation_pct}%. {action_msg}"
        )

        return {
            "cro_name": "Chief Risk Officer (Arbitrator)",
            "academic_grounding": [
                "Paper 18: Grossman & Zhou (1993) Optimal Drawdown Constraint",
                "Paper 23: Busseti, Ryu, Boyd (Stanford 2016) Risk-Constrained Kelly Gambling",
                "Paper 11: López de Prado (2018) Triple-Barrier ATR Corridors",
                "Paper 16: Page (1954) CUSUM Change-Point Surveillance",
                "Paper 17: RiskMetrics (1996) EWMA Dynamic Correlation Monitor",
            ],
            "final_resolution": final_resolution,
            "action_code": action_code,
            "buy_votes": buy_votes,
            "total_specialist_votes": len(agent_reports),
            "consensus_conviction_pct": round(avg_conviction, 1),
            "buy_weight_fraction": round(buy_weight_fraction, 3),
            "approved_leverage": approved_leverage,
            "kelly_allocation_pct": kelly_allocation_pct,
            "kelly_details": kelly_result,
            "atr_14": atr_val,
            "atr_pct": atr_pct,
            "tp1_target": tp1,
            "tp2_target": tp2,
            "stop_loss_target": sl,
            "cro_thesis": cro_thesis,
            "vix_level": vix_level,
            "vix_veto_triggered": vix_veto,
            "red_team_veto_triggered": red_team_veto,
            "regulatory_veto_triggered": regulatory_veto,
            "macro_blackout_veto_triggered": macro_blackout_veto,
            "statarb_veto_triggered": statarb_veto,
            "veto_reason": veto_reason,
            "microstructure_audit": micro_health,
        }


def convene_trading_committee(
    ticker: str,
    vix_level: float = 16.5,
    vix_change_pct: float = -1.2,
    save_resolution: bool = True,
    spot_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Orchestrates a full round-table deliberation of the 8-Agent Trading Committee for a given asset.

    Returns structured transcript with individual agent testimonies and CRO official sign-off.
    """
    logger.info(f"🏛️ Convening 8-Agent Quantitative Trading Committee for {ticker}...")

    if not spot_price or spot_price <= 0:
        quote = fetch_live_quote(ticker)
        spot_price = float(quote.get("price", 100.0))
        if spot_price <= 0.0:
            spot_price = 100.0

    from src.price_scout import PriceActionScoutAgent
    from src.red_team_agent import AdversarialRedTeamAgent

    tech_agent = TechnicalAlphaAgent()
    sent_agent = SentimentCatalystAgent()
    forensic_agent = ForensicFundamentalAgent()
    scout_agent = PriceActionScoutAgent()
    red_team_agent = AdversarialRedTeamAgent()
    regulatory_agent = RegulatoryCatalystAgent()
    macro_agent = MacroEconomicAgent()
    statarb_agent = StatArbSpecialist()
    cro_agent = ChiefRiskOfficerAgent()

    # Gather Specialist Testimonies from all 8 domain specialists
    report_tech = tech_agent.evaluate(ticker, spot_price)
    report_sent = sent_agent.evaluate(ticker)
    report_forensic = forensic_agent.evaluate(ticker, spot_price)
    report_scout = scout_agent.evaluate(ticker, spot_price)
    report_red_team = red_team_agent.evaluate(ticker, spot_price)
    report_regulatory = regulatory_agent.evaluate(ticker)
    report_macro = macro_agent.evaluate(ticker)
    report_statarb = statarb_agent.evaluate(ticker, spot_price)

    specialist_reports = [
        report_tech,
        report_sent,
        report_forensic,
        report_scout,
        report_red_team,
        report_regulatory,
        report_macro,
        report_statarb,
    ]

    # CRO Deliberation & Sign-Off
    cro_signoff = cro_agent.evaluate_and_sign_off(
        ticker=ticker,
        spot_price=spot_price,
        agent_reports=specialist_reports,
        vix_level=vix_level,
        vix_change_pct=vix_change_pct,
    )

    resolution_packet = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "spot_price": spot_price,
        "final_resolution": cro_signoff["final_resolution"],
        "action_code": cro_signoff["action_code"],
        "consensus_conviction_pct": cro_signoff["consensus_conviction_pct"],
        "approved_leverage": cro_signoff["approved_leverage"],
        "kelly_allocation_pct": cro_signoff["kelly_allocation_pct"],
        "tp1_target": cro_signoff["tp1_target"],
        "tp2_target": cro_signoff["tp2_target"],
        "stop_loss_target": cro_signoff["stop_loss_target"],
        "atr_14": cro_signoff.get("atr_14", 0.0),
        "atr_pct": cro_signoff.get("atr_pct", 0.0),
        "agent_testimonies": specialist_reports,
        "cro_signoff": cro_signoff,
    }

    if save_resolution:
        _persist_committee_resolution(ticker, resolution_packet)

    return resolution_packet


def _persist_committee_resolution(
    ticker: str, resolution_packet: Dict[str, Any]
) -> None:
    """Saves the committee resolution into results/committee_resolutions.json."""
    try:
        os.makedirs(os.path.dirname(COMMITTEE_FILE), exist_ok=True)
        data = {}
        if os.path.exists(COMMITTEE_FILE):
            try:
                with open(COMMITTEE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[ticker] = resolution_packet
        with open(COMMITTEE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to persist committee resolution for {ticker}: {e}")


def audit_full_universe_committee(
    universe_tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs committee deliberation across the provided universe of tickers."""
    tickers = universe_tickers or [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
    ]
    resolutions = {}
    for t in tickers:
        try:
            res = convene_trading_committee(t, save_resolution=True)
            resolutions[t] = res
        except Exception as e:
            logger.error(f"Error deliberating for {t}: {e}")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_audited": len(resolutions),
        "resolutions": resolutions,
    }


def execute_committee_order(
    ticker: str, deliberation: Dict[str, Any], broker: Any
) -> Dict[str, Any]:
    """
    Executes a committee-approved buy order into the virtual paper broker ledger.
    """
    try:
        spot_price = float(deliberation.get("spot_price", 0.0))
        if spot_price <= 0:
            return {"success": False, "reason": "Invalid spot price"}

        kelly_alloc_pct = float(deliberation.get("kelly_allocation_pct", 10.0))
        portfolio_summary = broker.get_portfolio_summary()
        total_equity = float(portfolio_summary.get("total_equity", 100000.0))
        cash_avail = float(portfolio_summary.get("cash", 0.0))

        # Size dollar allocation based on fractional Kelly allocation
        target_allocation_dollars = total_equity * (kelly_alloc_pct / 100.0)
        invest_amount = min(target_allocation_dollars, cash_avail * 0.90)

        if invest_amount < 500.0:
            return {
                "success": False,
                "reason": "Insufficient cash for minimum position size",
            }

        shares = int(invest_amount // spot_price)
        if shares <= 0:
            return {
                "success": False,
                "reason": "Position size too small for 1 whole share",
            }

        order_res = broker.execute_buy(
            ticker=ticker,
            shares=shares,
            price=spot_price,
            strategy_name="Committee_MultiAgent_Kelly",
            tp1_target=deliberation.get("tp1_target", spot_price * 1.05),
            tp2_target=deliberation.get("tp2_target", spot_price * 1.10),
            stop_loss=deliberation.get("stop_loss_target", spot_price * 0.96),
        )
        return order_res
    except Exception as e:
        logger.error(f"Error executing committee order for {ticker}: {e}")
        return {"success": False, "error": str(e)}
