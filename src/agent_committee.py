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
            thesis = (
                "Insufficient historical price series; neutral technical vote cast."
            )

        # Check Options Market Maker Gamma Exposure (GEX) & Strike Walls
        gex_metrics = {}
        try:
            from src.options_gex import calculate_options_gex_profile

            gex_profile = calculate_options_gex_profile(
                ticker=ticker, spot_price=spot_price
            )
            if gex_profile.get("is_real_data", False):
                call_wall = gex_profile.get("call_wall", spot_price * 1.05)
                put_wall = gex_profile.get("put_wall", spot_price * 0.95)
                flip_level = gex_profile.get("gamma_flip", spot_price)
                net_gex = gex_profile.get("total_net_gex_m", 0.0)
                mm_regime = gex_profile.get("mm_regime", "NEUTRAL")

                gex_metrics = {
                    "call_wall": round(call_wall, 2),
                    "put_wall": round(put_wall, 2),
                    "gamma_flip": round(flip_level, 2),
                    "total_net_gex_m": round(net_gex, 2),
                    "mm_regime": mm_regime,
                }

                # If above gamma flip in negative gamma -> momentum explosive acceleration
                if spot_price > flip_level and net_gex < -2.0:
                    conviction = min(95.0, conviction + 6.0)
                    thesis += f" ⚡ Negative GEX Regime (${net_gex:+.1f}M): MM short gamma accelerates momentum above Flip ${flip_level:.2f}."
                elif spot_price < put_wall:
                    thesis += f" 🛡️ Approaching Put Wall support (${put_wall:.2f})."
                elif spot_price > call_wall:
                    thesis += f" ⚠️ Approaching Call Wall ceiling resistance (${call_wall:.2f})."
        except Exception as ge:
            logger.debug(f"Options GEX calculation notice for {ticker}: {ge}")

        # Check Microsoft Qlib Alpha158 Top 25 Orthogonal Factors
        alpha158_metrics = {}
        try:
            from src.ultra_quant_engine import get_ultra_quant_engine

            uq = get_ultra_quant_engine()
            a158 = uq.evaluate_alpha158_factors(ticker, df)
            if a158:
                alpha158_metrics = {
                    "kmid": a158.get("KMID", 0.0),
                    "klen": a158.get("KLEN", 0.0),
                    "corr5": a158.get("CORR5", 0.0),
                    "wvma10": a158.get("WVMA10", 0.0),
                }
                kmid = a158.get("KMID", 0.0)
                corr5 = a158.get("CORR5", 0.0)
                if kmid > 0.008 and corr5 > 0.35 and vote in ["BUY", "NEUTRAL"]:
                    conviction = min(95.0, conviction + 4.0)
                    thesis += f" Qlib Alpha158 confirms real-body expansion (KMID: {kmid:+.4f}, CORR5: {corr5:.2f})."
                elif kmid < -0.012:
                    conviction = max(25.0, conviction - 5.0)
                    thesis += (
                        f" Qlib Alpha158 warns of body rejection (KMID: {kmid:+.4f})."
                    )
        except Exception as a_err:
            logger.debug(f"Alpha158 check notice for {ticker}: {a_err}")

        return {
            "agent_name": "Technical Momentum Specialist",
            "role": "Pillar 1: Market Structure, Moving Averages, RSI, Options GEX & Alpha158",
            "academic_grounding": [
                "Microsoft Qlib: Alpha158 Orthogonal Price-Volume Factors",
                "Paper 25: Zarattini, Barbon, Aziz (2024) 5-Min Opening Range Breakout (ORB)",
                "Paper 10: Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR)",
                "Institutional Market Microstructure: SqueezeMetrics Gamma Exposure (GEX)",
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
                **gex_metrics,
                **alpha158_metrics,
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

        # Integrate Post-Earnings Announcement Drift (PEAD) & SUE Earnings Surprise
        pead_metrics = {}
        try:
            from src.earnings_whisper import analyze_earnings_surprises

            pead_report = analyze_earnings_surprises(ticker)
            pead_boost = float(pead_report.get("catalyst_conviction_boost", 0.0))
            if pead_boost != 0.0:
                conviction = round(
                    float(np.clip(conviction + pead_boost, 15.0, 95.0)), 1
                )
                thesis += f" PEAD Catalyst ({pead_report.get('pead_momentum_signal')}): {pead_report.get('eps_surprise_pct'):+.1f}% EPS Surprise."
            pead_metrics = {
                "pead_signal": pead_report.get("pead_momentum_signal"),
                "sue_score": pead_report.get("sue_score"),
                "eps_surprise_pct": pead_report.get("eps_surprise_pct"),
            }
        except Exception as pe:
            logger.debug(f"PEAD earnings analysis notice for {ticker}: {pe}")

        # Integrate Alternative Data: SEC Form 8-K, Insider Form 4, Social Buzz, Biotech Radar
        alt_data_metrics = {}
        try:
            from src.ultra_quant_engine import get_ultra_quant_engine

            uq = get_ultra_quant_engine()
            # 1. SEC Form 8-K Filings
            sec_filings = uq.evaluate_sec_catalysts(ticker)
            if sec_filings:
                for sf in sec_filings[:3]:
                    tags = sf.get("tags", [])
                    item_str = ", ".join(tags)
                    if any("DELISTING" in t or "WEAKNESS" in t for t in tags):
                        vote = "SELL"
                        conviction = 95.0
                        thesis += f" 🚨 CRITICAL SEC 8-K: Delisting/material weakness warning ({item_str})."
                        break
                    elif any(
                        "MATERIAL_AGREEMENT" in t or "EARNINGS" in t for t in tags
                    ):
                        conviction = min(95.0, conviction + 6.0)
                        thesis += f" 📄 SEC Form 8-K Filing: {item_str}."

            # 2. SEC Form 4 Insider Conviction & Cluster Buys
            insider = uq.evaluate_insider_signals(ticker)
            if insider.get("cluster_buy_detected", False):
                conviction = min(95.0, conviction + 7.0)
                thesis += (
                    f" 👔 EXECUTIVE CLUSTER BUY: Multiple officers buying shares "
                    f"(Score: {insider.get('conviction_score')}/100)."
                )
            elif insider.get("conviction_score", 50.0) < 30.0:
                conviction = max(20.0, conviction - 5.0)
                thesis += " ⚠️ INSIDER DISTRIBUTION: Heavy insider selling recorded."

            # 3. Social Sentiment Velocity
            social = uq.evaluate_social_sentiment(ticker)
            soc_vel = float(social.get("mention_velocity_ratio", 1.0))
            if soc_vel >= 2.0:
                thesis += f" ⚡ Social Buzz ({soc_vel:.1f}x velocity): {social.get('regime')}."

            # 4. Biotech FDA Radar
            biotech = uq.evaluate_biotech_radar(ticker)
            if biotech.get("is_biotech", False):
                thesis += (
                    f" 🧬 BioTech Catalyst Profile: {biotech.get('risk_profile')}."
                )

            alt_data_metrics = {
                "sec_8k_count": len(sec_filings),
                "insider_score": insider.get("conviction_score", 50.0),
                "insider_cluster_buy": insider.get("cluster_buy_detected", False),
                "social_velocity": soc_vel,
                "is_biotech": biotech.get("is_biotech", False),
            }
        except Exception as alt_err:
            logger.debug(f"Alternative data integration notice for {ticker}: {alt_err}")

        return {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "role": "Pillar 2: Tri-Brain FinBERT, SEC 8-K, Insider Form 4 & Social Velocity",
            "academic_grounding": [
                "SEC EDGAR Real-Time Form 8-K & Form 4 Microstructure",
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
                **pead_metrics,
                **alt_data_metrics,
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

        return {
            "agent_name": "Forensic & Valuation Auditor",
            "role": "Pillar 8: Fundamental Health & Forensic Valuation",
            "academic_grounding": [
                "Paper 05: Bellman-Ford Fundamental Arbitrage Graph Theory",
                "Paper 13: Chen et al. (2020) Supply Chain Contagion Graph Neural Networks",
            ],
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "data_available": True,
                "piotroski_f_score": f_score,
                "altman_z_score": z_score,
                "dcf_fair_value": dcf.get("fair_value_price", spot_price),
                "dcf_margin_of_safety_pct": dcf_mos,
            },
            "thesis": thesis,
        }


class ChiefRiskOfficerAgent:
    """Agent 4: Chief Risk Officer (CRO) — Synthesizes Votes, Computes Kelly Sizing, Enforces Volatility Gates, and Signs Off."""

    def evaluate_and_sign_off(
        self,
        ticker: str,
        spot_price: float,
        agent_reports: List[Dict[str, Any]],
        vix_level: float = 16.5,
        vix_change_pct: float = -1.2,
    ) -> Dict[str, Any]:
        # Tally Votes across the specialist domain agents safely
        buy_votes = sum(
            1 for r in agent_reports if isinstance(r, dict) and r.get("vote") == "BUY"
        )
        avg_conviction = sum(
            float(r.get("conviction_score", 50.0))
            for r in agent_reports
            if isinstance(r, dict)
        ) / max(len(agent_reports), 1)

        # Load calibrated specialist weights from Bayesian Post-Mortem Learner
        weights_file = os.path.join("results", "agent_memory", "committee_weights.json")
        agent_weights = {
            "Technical Momentum Specialist": 0.25,
            "Sentiment & Alternative Data Specialist": 0.25,
            "Fundamental Valuation Specialist": 0.20,
            "Adversarial Red-Team Specialist": 0.15,
            "Chief Risk Officer": 0.15,
        }
        if os.path.exists(weights_file):
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    wdata = json.load(f)
                    if isinstance(wdata, dict):
                        agent_weights["Technical Momentum Specialist"] = float(
                            wdata.get("Technical Momentum", 0.25)
                        )
                        agent_weights["Sentiment & Alternative Data Specialist"] = (
                            float(wdata.get("FinBERT Sentiment", 0.25))
                        )
                        agent_weights["Fundamental Valuation Specialist"] = float(
                            wdata.get("Fundamental Valuation", 0.20)
                        )
                        agent_weights["Adversarial Red-Team Specialist"] = float(
                            wdata.get("Adversarial Red-Team", 0.15)
                        )
            except Exception:
                pass

        total_weight = sum(
            agent_weights.get(r.get("agent_name", ""), 0.25)
            for r in agent_reports
            if isinstance(r, dict)
        )
        if total_weight > 0:
            weighted_conviction = (
                sum(
                    float(r.get("conviction_score", 50.0))
                    * agent_weights.get(r.get("agent_name", ""), 0.25)
                    for r in agent_reports
                    if isinstance(r, dict)
                )
                / total_weight
            )
        else:
            weighted_conviction = avg_conviction

        effective_conviction = round(
            0.5 * avg_conviction + 0.5 * weighted_conviction, 1
        )

        # 1. Check Macro Volatility Gate (VIX Panic Check) & VPIN Toxicity Shield
        vix_veto = False
        trend_veto = False
        red_team_veto = False
        red_team_caution = False
        vpin_veto = False
        veto_reason = None

        # Check VPIN Order Flow Toxicity (Easley, Lopez de Prado & O'Hara)
        vpin_metrics = {}
        try:
            from src.vpin_toxicity import evaluate_ticker_toxicity

            vpin_report = evaluate_ticker_toxicity(ticker)
            vpin_val = float(vpin_report.get("vpin", 0.35))
            vpin_reg = vpin_report.get("toxicity_regime", "NORMAL_LIQUIDITY")
            vpin_metrics = {"vpin": vpin_val, "toxicity_regime": vpin_reg}
            if vpin_report.get("cro_veto_recommended", False):
                vpin_veto = True
                veto_reason = f"VPIN Order Flow Toxicity Critical ({vpin_val:.2f}) — Informed institutional dumping detected."
        except Exception as ve:
            logger.debug(f"VPIN toxicity check notice for {ticker}: {ve}")

        # Check Red-Team Adversarial Veto
        for r in agent_reports:
            if (
                isinstance(r, dict)
                and r.get("agent_name") == "Adversarial Red-Team Specialist"
            ):
                if r.get("vote") == "VETO":
                    red_team_veto = True
                    veto_reason = f"Adversarial Red-Team VETO: {r.get('thesis')}"
                    break
                elif r.get("vote") == "CAUTION":
                    red_team_caution = True

        # Check Ultra-Low-Latency Quantum Core (HMM Regime, CUSUM Structural Break, PoC Magnet)
        cusum_veto = False
        hmm_veto = False
        quantum_telemetry = {}
        try:
            from src.ultra_quant_engine import get_ultra_quant_engine

            ultra_engine = get_ultra_quant_engine()
            quantum_telemetry = ultra_engine.evaluate_asset_full_quantum(
                ticker=ticker, spot_price=spot_price
            )
            if (
                quantum_telemetry.get("cusum_alarm")
                and quantum_telemetry.get("cusum_direction") == "DOWN"
            ):
                cusum_veto = True
                veto_reason = "CUSUM Structural Break VETO (Page 1954): Acute downward changepoint detected in price series."
            elif quantum_telemetry.get("hmm_regime") == "Crisis":
                hmm_veto = True
                veto_reason = "HMM Regime VETO (Hamilton 1989): High-volatility crisis regime detected with high probability."
        except Exception as qe:
            logger.debug(f"Ultra-quant check notice for {ticker}: {qe}")

        # Check Macro Blackout Gate (FOMC / CPI / NFP releases)
        macro_blackout_veto = False
        m_blackout = quantum_telemetry.get("macro_blackout", {})
        if m_blackout.get("is_blackout_active", False):
            macro_blackout_veto = True
            veto_reason = f"Macro Blackout VETO: {m_blackout.get('reason')}"

        if macro_blackout_veto or cusum_veto or hmm_veto:
            pass
        elif vpin_veto:
            pass  # VPIN veto already set
        elif vix_level > 26.0 or vix_change_pct > 8.0:
            vix_veto = True
            veto_reason = f"VIX elevated at {vix_level:.1f} (+{vix_change_pct:+.1f}% spike) — macro volatility gate activated."
        elif not red_team_veto:
            # Check Trend Regime Gate: Long positions blocked if asset is below 200 SMA
            for r in agent_reports:
                if (
                    isinstance(r, dict)
                    and r.get("key_metrics", {}).get("trend_status")
                    == "BEARISH_BELOW_SMA200"
                ):
                    trend_veto = True
                    veto_reason = "Asset is in a structural macro downtrend below its 200-day Moving Average (SMA200)."
                    break

        # 2. Dynamic Mathematical Fractional Kelly Sizing (Paper 23 & Paper 14)
        empirical_win_rate = 0.533 if effective_conviction >= 70.0 else 0.48
        kelly_result = compute_fractional_kelly_sizing(
            win_rate=empirical_win_rate,
            payoff_ratio=1.75,
            kelly_fraction=0.25,  # Quarter-Kelly
            max_cap_pct=15.0,
        )
        calculated_kelly_pct = float(kelly_result.get("fractional_kelly_pct", 0.0))
        if red_team_caution:
            calculated_kelly_pct = round(calculated_kelly_pct * 0.65, 2)

        # Scale by Quantum Core Risk Multiplier (Credit spread radar / Value area adjustment)
        q_scalar = float(quantum_telemetry.get("risk_multiplier", 1.0))
        if q_scalar != 1.0:
            calculated_kelly_pct = round(calculated_kelly_pct * q_scalar, 2)

        # 3. Check Options Market Maker Gamma Exposure (GEX) Regime
        gex_haircut_applied = False
        try:
            from src.options_gex import compute_gamma_exposure_profile

            gex_data = compute_gamma_exposure_profile(
                ticker, spot_price=spot_price, max_expiries=2
            )
            if gex_data.get("gamma_regime") == "NEGATIVE_GAMMA_VOLATILITY_EXPANSION":
                gex_haircut_applied = True
                calculated_kelly_pct = round(calculated_kelly_pct * 0.50, 2)
        except Exception as ge:
            logger.debug(f"GEX check notice for {ticker}: {ge}")

        # 4. Integrate Episodic Memory Risk Brain & Ticker Reputation
        mem_adjustment = {}
        require_supermajority = False
        try:
            from src.agent_memory import AgentMemoryStore

            mem_store = AgentMemoryStore()
            mem_adjustment = mem_store.get_memory_risk_adjustment(ticker)
            mem_conviction_delta = float(mem_adjustment.get("conviction_delta", 0.0))
            if mem_conviction_delta != 0.0:
                effective_conviction = float(
                    np.clip(effective_conviction + mem_conviction_delta, 10.0, 95.0)
                )
            mem_kelly_scale = float(mem_adjustment.get("kelly_scale", 1.0))
            calculated_kelly_pct = round(calculated_kelly_pct * mem_kelly_scale, 2)
            require_supermajority = bool(
                mem_adjustment.get("require_supermajority", False)
            )
        except Exception as me:
            logger.debug(f"Episodic memory risk check notice for {ticker}: {me}")

        # 5. Check Portfolio Diversity & Effective Bets (src/portfolio_diversity.py)
        diversity_grade = "A"
        try:
            from src.portfolio_diversity import calculate_portfolio_diversity_grade

            portfolio_file = os.path.join("results", "paper_portfolio.json")
            if os.path.exists(portfolio_file):
                with open(portfolio_file, "r", encoding="utf-8") as pf:
                    p_state = json.load(pf)
                open_tickers = list(p_state.get("open_positions", {}).keys())
                if len(open_tickers) >= 2:
                    div_report = calculate_portfolio_diversity_grade(
                        open_tickers + [ticker]
                    )
                    diversity_grade = div_report.get("grade", "A")
                    if diversity_grade in ["C", "D"]:
                        calculated_kelly_pct = round(calculated_kelly_pct * 0.75, 2)
                        logger.info(
                            f"🛡️ [PORTFOLIO DIVERSITY THROTTLE] Diversity Grade '{diversity_grade}'. Sizing throttled by 25% to protect effective bets."
                        )
        except Exception as de:
            logger.debug(f"Portfolio diversity check notice for {ticker}: {de}")

        # Check Grossman-Zhou Stochastic Drawdown Floor (Paper 18)
        try:
            from src.ultra_quant_engine import get_ultra_quant_engine

            uq_engine = get_ultra_quant_engine()
            portfolio_file = os.path.join("results", "paper_portfolio.json")
            if os.path.exists(portfolio_file):
                with open(portfolio_file, "r", encoding="utf-8") as pf:
                    p_state = json.load(pf)
                curr_eq = float(p_state.get("total_equity", 100000.0))
                gz_alloc = uq_engine.evaluate_grossman_zhou_allocation(
                    current_wealth=curr_eq, peak_wealth=159316.0
                )
                if gz_alloc.get("at_floor", False):
                    calculated_kelly_pct = 0.0
                    logger.warning(
                        "🛡️ [GROSSMAN-ZHOU FLOOR] At capital preservation floor. New risk halted."
                    )
        except Exception as gz_err:
            logger.debug(f"Grossman-Zhou check notice: {gz_err}")

        # Determine Final Committee Resolution with Tightened Consensus Thresholds
        if (
            macro_blackout_veto
            or cusum_veto
            or hmm_veto
            or vpin_veto
            or vix_veto
            or trend_veto
            or red_team_veto
        ):
            if macro_blackout_veto:
                final_resolution = "🔴 VETO / MACRO EVENT BLACKOUT (FOMC/CPI/NFP)"
            elif cusum_veto:
                final_resolution = "🔴 VETO / CUSUM STRUCTURAL BREAKDOWN"
            elif hmm_veto:
                final_resolution = "🔴 VETO / HMM CRISIS REGIME"
            elif vpin_veto:
                final_resolution = "🔴 VETO / TOXIC LIQUIDITY DUMPING (VPIN SPIKE)"
            elif vix_veto:
                final_resolution = "🔴 VETO / CAPITAL PRESERVATION"
            elif red_team_veto:
                final_resolution = "🔴 VETO / RED-TEAM VULNERABILITY"
            else:
                final_resolution = "🔴 VETO / MACRO DOWNTREND (BELOW SMA200)"
            action_code = "VETO"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0
        elif (
            buy_votes >= 3
            and effective_conviction >= (65.0 if require_supermajority else 58.0)
        ) or (
            not require_supermajority
            and buy_votes >= 2
            and effective_conviction >= 72.0
        ):
            final_resolution = "🚀 HIGH CONVICTION COMMITTEE BUY"
            action_code = "EXECUTE_BUY"
            approved_leverage = 1.25
            kelly_allocation_pct = max(calculated_kelly_pct, 5.0)
        elif (
            not require_supermajority
            and buy_votes >= 2
            and effective_conviction >= 55.0
        ):
            final_resolution = "🟡 AGILE SCALE-IN (Quorum Approved)"
            action_code = "SCALE_IN"
            approved_leverage = 1.0
            kelly_allocation_pct = max(round(calculated_kelly_pct * 0.65, 2), 3.5)
        elif buy_votes >= 1 or effective_conviction >= 38.0:
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

        tp1 = round(spot_price + (2.5 * atr_val), 2)
        tp2 = round(spot_price + (4.5 * atr_val), 2)
        sl = round(max(spot_price - (1.5 * atr_val), spot_price * 0.85), 2)
        atr_pct = round((atr_val / spot_price) * 100.0, 2)

        reason_str = veto_reason if veto_reason else "Insufficient consensus."
        action_msg = (
            "Trade signed off for execution."
            if action_code in ["EXECUTE_BUY", "SCALE_IN"]
            else f"Trade blocked: {reason_str}"
        )
        vix_status = "NORMAL" if not vix_veto else "ELEVATED"

        cro_thesis = (
            f"Committee Consensus: {buy_votes}/{len(agent_reports)} specialist agents voted BUY (Average Conviction: {avg_conviction:.1f}%). "
            f"ATR14 is ${atr_val:.2f} ({atr_pct:.1f}% vol). VIX is {vix_level:.1f} ({vix_status}). Fractional Kelly: {kelly_allocation_pct}%. {action_msg}"
        )

        # Conformal Prediction 95% Statistical Boundary
        conf_low = quantum_telemetry.get("conformal_lower_bound")
        if conf_low and conf_low > 0:
            sl = round(max(sl, conf_low), 2)

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
            "consensus_conviction_pct": round(effective_conviction, 1),
            "unweighted_conviction_pct": round(avg_conviction, 1),
            "weighted_conviction_pct": round(weighted_conviction, 1),
            "approved_leverage": approved_leverage,
            "kelly_allocation_pct": kelly_allocation_pct,
            "kelly_details": kelly_result,
            "atr_14": atr_val,
            "atr_pct": atr_pct,
            "tp1_target": tp1,
            "tp2_target": tp2,
            "stop_loss_target": sl,
            "portfolio_diversity_grade": diversity_grade,
            "conformal_lower_bound": quantum_telemetry.get("conformal_lower_bound"),
            "conformal_upper_bound": quantum_telemetry.get("conformal_upper_bound"),
            "cro_thesis": cro_thesis,
            "vix_level": vix_level,
            "vix_veto_triggered": vix_veto,
            "red_team_veto_triggered": red_team_veto,
            "vpin_veto_triggered": vpin_veto,
            "cusum_veto_triggered": cusum_veto,
            "hmm_veto_triggered": hmm_veto,
            "quantum_telemetry": quantum_telemetry,
            "memory_risk_adjustment": mem_adjustment,
            "require_supermajority": require_supermajority,
            **vpin_metrics,
        }


def convene_trading_committee(
    ticker: str,
    vix_level: float = 16.5,
    vix_change_pct: float = -1.2,
    save_resolution: bool = True,
    spot_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Orchestrates a full round-table deliberation of the 5-Agent Trading Committee for a given asset.

    Returns structured transcript with individual agent testimonies and CRO official sign-off.
    """
    logger.info(f"🏛️ Convening 5-Agent Quantitative Trading Committee for {ticker}...")

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
    cro_agent = ChiefRiskOfficerAgent()

    # Gather Specialist Testimonies from domain agents + real-time price scout + red-team stress tester
    report_tech = tech_agent.evaluate(ticker, spot_price)
    report_sent = sent_agent.evaluate(ticker)
    report_forensic = forensic_agent.evaluate(ticker, spot_price)
    report_scout = scout_agent.evaluate(ticker, spot_price)
    report_red_team = red_team_agent.evaluate(ticker, spot_price)

    specialist_reports = [
        report_tech,
        report_sent,
        report_forensic,
        report_scout,
        report_red_team,
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
        "quantum_telemetry": cro_signoff.get("quantum_telemetry", {}),
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

        # 🛡️ CPPI Dynamic Cushion Scaling (Black & Jones 1987, Chekhlov 2005 CDaR)
        cppi_factor = 1.0
        try:
            from src.cppi_insurance import get_cppi_cushion_multiplier

            equity_hist = [
                h.get("total_equity", 100000.0)
                for h in broker.state.get("equity_history", [])
            ]
            peak_eq = max(equity_hist + [total_equity])
            cppi_eval = get_cppi_cushion_multiplier(
                portfolio_value=total_equity,
                peak_equity=peak_eq,
                floor_pct=0.95,
                multiplier=2.85,
            )
            cppi_factor = float(cppi_eval.get("allocation_factor", 1.0))
        except Exception:
            cppi_factor = 1.0

        if cppi_factor <= 0.0:
            return {
                "success": False,
                "reason": "CPPI Capital Preservation Floor Active (No new equity risk permitted)",
            }

        # Size dollar allocation based on fractional Kelly allocation scaled by CPPI cushion
        target_allocation_dollars = (
            total_equity * (kelly_alloc_pct / 100.0) * cppi_factor
        )

        # 🐢 Turtle 0.5N Pyramiding: Initial entry is 1/3 Seed Unit to truncate left-tail breakout risk
        seed_allocation = target_allocation_dollars / 3.0
        invest_amount = min(seed_allocation, cash_avail * 0.90)

        # Fallback for small portfolios: ensure at least 1 whole share if target allocation is valid
        if invest_amount < spot_price and target_allocation_dollars >= spot_price:
            invest_amount = min(target_allocation_dollars, cash_avail * 0.90)

        if invest_amount < 200.0:
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

        atr_val = deliberation.get("atr_14")
        order_res = broker.execute_buy(
            ticker=ticker,
            shares=shares,
            price=spot_price,
            strategy_name="Committee_Turtle_Pyramid_Kelly",
            tp1_target=deliberation.get("tp1_target", spot_price * 1.05),
            tp2_target=deliberation.get("tp2_target", spot_price * 1.10),
            stop_loss=deliberation.get("stop_loss_target", spot_price * 0.96),
            atr=atr_val,
        )
        if order_res.get("success") and ticker in broker.state.get(
            "open_positions", {}
        ):
            pos_dict = broker.state["open_positions"][ticker]
            pos_dict["pyramid_units_filled"] = 1
            pos_dict["target_units"] = 3
            pos_dict["unit_shares"] = shares
            pos_dict["atr_14"] = atr_val
            pos_dict["highest_price_seen"] = spot_price
            broker._save()

        return order_res
    except Exception as e:
        logger.error(f"Error executing committee order for {ticker}: {e}")
        return {"success": False, "error": str(e)}
