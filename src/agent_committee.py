"""
Autonomous Multi-Agent Trading Committee & Deliberation Engine for Sentilyze.
Institutional Round-Table Debate Desk:
- Agent 1: Technical & Quantitative Alpha Specialist (Momentum, RSI, Moving Averages, TFT Attention)
- Agent 2: NLP Sentiment & Alternative Data Specialist (FinBERT Headline Score, Social Velocity, Insider Flow)
- Agent 3: Forensic & Fundamental Health Specialist (Piotroski F-Score, Altman Z-Score, Beneish M-Score, DCF Margin of Safety)
- Agent 4: Chief Risk Officer (CRO) Arbitrator (VIX Volatility Gate, Kelly Allocation, Veto Authority & Sign-Off)
"""

from typing import Any, Dict, List, Optional
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.realtime_tracker import fetch_live_quote
from src.forensic_accounting import calculate_beneish_m_score
from src.fundamental_valuation import (
    fetch_financial_statements,
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dcf_fair_value,
)
from src.earnings_sentiment import analyze_earnings_call_transcript
from src.social_sentiment import calculate_social_buzz_metrics
from src.insider_tracker import compute_smart_money_insider_score
from src.patent_contract_radar import compute_government_and_patent_index

logger = get_logger(__name__)

COMMITTEE_FILE = os.path.join("results", "committee_resolutions.json")


from src.data_ingestion import get_price_history


class TechnicalAlphaAgent:
    """Agent 1: Evaluates Technical Price Action, Momentum, RSI, and Multi-Horizon Forecasts."""

    def evaluate(self, ticker: str, spot_price: float) -> Dict[str, Any]:
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
                thesis = f"Asset RSI is severely overbought ({rsi_val:.1f} > 70), indicating high probability of mean-reversion pullback."
            elif not is_above_200:
                vote = "HOLD"
                conviction = 38.0
                trend_status = "BEARISH_BELOW_SMA200"
                thesis = f"Asset is trading below its 200-day SMA (${sma200:,.2f}), trapped in a macro downtrend regime."
            elif is_above_200 and 40.0 <= rsi_val <= 62.0:
                vote = "BUY"
                conviction = 82.0
                trend_status = "BULLISH_MOMENTUM_EXPANSION"
                thesis = f"Asset is in strong structural uptrend above 200 SMA (${sma200:,.2f}) with optimal pullback RSI ({rsi_val:.1f})."
            elif rsi_val < 35.0:
                vote = "BUY"
                conviction = 70.0
                trend_status = "OVERSOLD_MEAN_REVERSION"
                thesis = f"Asset is deeply oversold (RSI: {rsi_val:.1f} < 35), presenting high-probability rebound setup."
            else:
                vote = "NEUTRAL"
                conviction = 50.0
                trend_status = "CONSOLIDATION_RANGE"
                thesis = f"Asset is range-bound around 21 MA (${ma21:,.2f}) with neutral RSI ({rsi_val:.1f})."
        else:
            rsi_val = 52.0
            ret_5d = 1.5
            trend_status = "NEUTRAL_BASE"
            vote = "NEUTRAL"
            conviction = 50.0
            thesis = "Insufficient historical bar depth; standing neutral."

        return {
            "agent_name": "Technical Momentum Specialist",
            "role": "Pillar 1: Technical & Quant Alpha",
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "estimated_rsi": round(rsi_val, 1),
                "trend": trend_status,
                "tft_5d_forecast_pct": round(ret_5d, 1),
            },
            "thesis": thesis,
        }


class SentimentCatalystAgent:
    """Agent 2: Evaluates FinBERT NLP News Sentiment, Earnings Call Tone, and Social Velocity."""

    def evaluate(self, ticker: str) -> Dict[str, Any]:
        earn_res = analyze_earnings_call_transcript(ticker)
        soc_res = calculate_social_buzz_metrics(ticker)
        insider_res = compute_smart_money_insider_score(ticker)
        gov_res = compute_government_and_patent_index(ticker)

        # Check FinBERT news sentiment from processed CSV if present
        sent_path = os.path.join("data", "processed", f"{ticker}_sentiment.csv")
        finbert_score = 0.0
        if os.path.exists(sent_path):
            try:
                sdf = pd.read_csv(sent_path)
                if "sentiment_score" in sdf.columns and not sdf.empty:
                    finbert_score = float(sdf["sentiment_score"].tail(10).mean())
            except Exception as e:
                logger.debug(f"FinBERT sentiment cache notice for {ticker}: {e}")

        compound_score = (
            earn_res.get("executive_optimism_score", 60.0) * 0.35
            + insider_res.get("smart_money_score", 50.0) * 0.25
            + gov_res.get("composite_innovation_score", 50.0) * 0.20
            + (max(min(finbert_score * 50.0 + 50.0, 100.0), 0.0)) * 0.20
        )

        if compound_score >= 68.0:
            vote = "BUY"
        elif compound_score >= 50.0:
            vote = "HOLD"
        else:
            vote = "SELL"
        conviction = round(compound_score, 1)

        thesis = (
            f"Earnings tone is {earn_res.get('verdict', 'NEUTRAL')}. "
            f"Social velocity is {soc_res.get('mention_velocity_ratio', 1.0):.1f}x. "
            f"Insider smart money score: {insider_res.get('smart_money_score', 50.0):.0f}/100. "
            f"FinBERT NLP Score: {finbert_score:+.2f}."
        )

        return {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "role": "Pillar 2: NLP & Catalyst Intelligence",
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "earnings_tone": earn_res.get("verdict", "N/A"),
                "social_velocity": soc_res.get("mention_velocity_ratio", 1.0),
                "insider_sentiment": insider_res.get("sentiment_verdict", "N/A"),
                "gov_procurement": gov_res.get("badge", "N/A"),
            },
            "thesis": thesis,
        }


class ForensicFundamentalAgent:
    """Agent 3: Audits Piotroski 9-Point Score, Altman Z-Score, Beneish M-Score, and DCF Valuation."""

    def evaluate(self, ticker: str, spot_price: float) -> Dict[str, Any]:
        fin_data = fetch_financial_statements(ticker)
        piotroski = calculate_piotroski_f_score(ticker, fin_data)
        altman = calculate_altman_z_score(ticker, fin_data)
        beneish = calculate_beneish_m_score(ticker)
        dcf = calculate_dcf_fair_value(ticker, fin_data)

        f_score = piotroski.get("f_score", 0)
        z_score = altman.get("z_score", 0.0)
        m_score = beneish.get("beneish_m_score", -2.5)
        dcf_mos = dcf.get("margin_of_safety_pct", 0.0)

        # High valuation or weak fundamentals lower the score
        is_financially_healthy = (
            f_score >= 5 and z_score >= 1.81 and m_score < -1.78 and dcf_mos >= -15.0
        )

        if is_financially_healthy:
            vote = "BUY"
            conviction = round(
                min(50.0 + (f_score * 4.0) + max(dcf_mos * 0.5, 0.0), 92.0), 1
            )
            thesis = (
                f"Robust fundamentals: Piotroski F-Score {f_score}/9, Altman Z-Score {z_score:.2f} ({altman.get('zone', 'SAFE')}), "
                f"and DCF Fair Value ${dcf.get('fair_value_price', spot_price):,.2f} ({dcf_mos:+.1f}% margin of safety)."
            )
        else:
            vote = "HOLD"
            conviction = round(max(30.0 + (f_score * 3.0) + (dcf_mos * 0.3), 20.0), 1)
            thesis = (
                f"Valuation/Quality caution: Piotroski F-Score {f_score}/9, "
                f"DCF Fair Value ${dcf.get('fair_value_price', spot_price):,.2f} (Margin of Safety: {dcf_mos:+.1f}%)."
            )

        return {
            "agent_name": "Forensic & Valuation Auditor",
            "role": "Pillar 8: Fundamental Health & Forensic Accounting",
            "vote": vote,
            "conviction_score": conviction,
            "key_metrics": {
                "piotroski_f_score": f_score,
                "altman_z_score": z_score,
                "beneish_m_score": m_score,
                "dcf_fair_value": dcf.get("fair_value_price", spot_price),
                "dcf_margin_of_safety_pct": dcf_mos,
            },
            "thesis": thesis,
        }


class ChiefRiskOfficerAgent:
    """Agent 4: Chief Risk Officer (CRO) — Synthesizes Votes, Enforces Macro Volatility Gates, and Signs Off."""

    def evaluate_and_sign_off(
        self,
        ticker: str,
        spot_price: float,
        agent_reports: List[Dict[str, Any]],
        vix_level: float = 16.5,
        vix_change_pct: float = -1.2,
    ) -> Dict[str, Any]:
        # Tally Votes
        buy_votes = sum(1 for r in agent_reports if r["vote"] == "BUY")
        avg_conviction = sum(r["conviction_score"] for r in agent_reports) / max(
            len(agent_reports), 1
        )

        # 1. Check Macro Volatility Gate (VIX Panic Check)
        vix_veto = False
        veto_reason = None
        if vix_level > 26.0 or vix_change_pct > 8.0:
            vix_veto = True
            veto_reason = f"VIX elevated at {vix_level:.1f} (+{vix_change_pct:+.1f}% spike) — macro volatility gate activated."

        # 2. Check Forensic Red Flags
        forensic_report = next(
            (r for r in agent_reports if "Forensic" in r["agent_name"]), None
        )
        forensic_veto = False
        if forensic_report:
            m_score = forensic_report["key_metrics"].get("beneish_m_score", -2.5)
            if m_score >= -1.78:
                forensic_veto = True
                veto_reason = f"Beneish M-Score flagged possible earnings distortion ({m_score:.2f} >= -1.78)."

        # Determine Final Committee Resolution
        if vix_veto or forensic_veto:
            final_resolution = "🔴 VETO / CAPITAL PRESERVATION"
            action_code = "VETO"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0
        elif buy_votes == 3 and avg_conviction >= 72.0:
            final_resolution = "🚀 CONVICTION INSTITUTIONAL BUY"
            action_code = "EXECUTE_BUY"
            approved_leverage = 1.5
            kelly_allocation_pct = 12.5
        elif buy_votes >= 2 and avg_conviction >= 58.0:
            final_resolution = "🟡 CAUTIOUS SCALE-IN (Moderate Conviction)"
            action_code = "SCALE_IN"
            approved_leverage = 1.0
            kelly_allocation_pct = 6.0
        elif buy_votes == 1 or avg_conviction >= 45.0:
            final_resolution = "⏸️ NEUTRAL HOLD / NO ACTION"
            action_code = "HOLD"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0
        else:
            final_resolution = "🔴 BEARISH SKEW / AVOID ASSET"
            action_code = "AVOID"
            approved_leverage = 0.0
            kelly_allocation_pct = 0.0

        # ATR Targets
        atr_est = spot_price * 0.03
        tp1 = round(spot_price + (2.5 * atr_est), 2)
        tp2 = round(spot_price + (4.5 * atr_est), 2)
        sl = round(spot_price - (1.5 * atr_est), 2)

        reason_str = veto_reason if veto_reason else "Insufficient consensus."
        action_msg = (
            "Trade signed off for execution."
            if action_code in ["EXECUTE_BUY", "SCALE_IN"]
            else f"Trade blocked: {reason_str}"
        )
        vix_status = "NORMAL" if not vix_veto else "ELEVATED"

        cro_thesis = (
            f"Committee Consensus: {buy_votes}/3 specialist agents voted BUY (Average Conviction: {avg_conviction:.1f}%). "
            f"VIX is {vix_level:.1f} ({vix_status}). {action_msg}"
        )

        return {
            "cro_name": "Chief Risk Officer (Arbitrator)",
            "final_resolution": final_resolution,
            "action_code": action_code,
            "buy_votes": buy_votes,
            "total_specialist_votes": len(agent_reports),
            "consensus_conviction_pct": round(avg_conviction, 1),
            "approved_leverage": approved_leverage,
            "kelly_allocation_pct": kelly_allocation_pct,
            "tp1_target": tp1,
            "tp2_target": tp2,
            "stop_loss_target": sl,
            "cro_thesis": cro_thesis,
            "vix_level": vix_level,
            "vix_veto_triggered": vix_veto,
        }


def convene_trading_committee(
    ticker: str,
    vix_level: float = 16.5,
    vix_change_pct: float = -1.2,
    save_resolution: bool = True,
    spot_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Orchestrates a full round-table deliberation of the 4-Agent Trading Committee for a given asset.

    Returns structured transcript with individual agent testimonies and CRO official sign-off.
    """
    logger.info(f"🏛️ Convening Multi-Agent Trading Committee for {ticker}...")

    if not spot_price or spot_price <= 0:
        quote = fetch_live_quote(ticker)
        spot_price = float(quote.get("price", 100.0))
        if spot_price <= 0.0:
            spot_price = 100.0

    tech_agent = TechnicalAlphaAgent()
    sent_agent = SentimentCatalystAgent()
    forensic_agent = ForensicFundamentalAgent()
    cro_agent = ChiefRiskOfficerAgent()

    # Gather Specialist Testimonies
    report_tech = tech_agent.evaluate(ticker, spot_price)
    report_sent = sent_agent.evaluate(ticker)
    report_forensic = forensic_agent.evaluate(ticker, spot_price)

    specialist_reports = [report_tech, report_sent, report_forensic]

    # CRO Deliberation & Sign-Off
    cro_signoff = cro_agent.evaluate_and_sign_off(
        ticker=ticker,
        spot_price=spot_price,
        agent_reports=specialist_reports,
        vix_level=vix_level,
        vix_change_pct=vix_change_pct,
    )

    deliberation = {
        "ticker": ticker,
        "spot_price": spot_price,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_resolution": cro_signoff["final_resolution"],
        "action_code": cro_signoff["action_code"],
        "consensus_conviction_pct": cro_signoff["consensus_conviction_pct"],
        "approved_leverage": cro_signoff["approved_leverage"],
        "kelly_allocation_pct": cro_signoff["kelly_allocation_pct"],
        "tp1_target": cro_signoff["tp1_target"],
        "tp2_target": cro_signoff["tp2_target"],
        "stop_loss_target": cro_signoff["stop_loss_target"],
        "cro_signoff": cro_signoff,
        "agent_testimonies": specialist_reports,
    }

    if save_resolution:
        os.makedirs("results", exist_ok=True)
        resolutions_db = {}
        if os.path.exists(COMMITTEE_FILE):
            try:
                with open(COMMITTEE_FILE, "r") as f:
                    resolutions_db = json.load(f)
            except Exception as e:
                logger.debug(f"Could not load existing committee file: {e}")

        resolutions_db[ticker] = deliberation
        resolutions_db["last_deliberated_at"] = datetime.now(timezone.utc).isoformat()
        with open(COMMITTEE_FILE, "w") as f:
            json.dump(resolutions_db, f, indent=2)

    logger.info(
        f"🏛️ Committee Resolution for {ticker}: {deliberation['final_resolution']} ({deliberation['consensus_conviction_pct']}%)"
    )
    return deliberation


def audit_full_universe_committee(
    universe_tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Audits the entire universe of assets through the Autonomous Trading Committee.
    """
    tickers = universe_tickers or [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "TSLA",
        "AMZN",
        "AVGO",
        "AMD",
        "PLTR",
        "LLY",
        "QQQ",
        "SPY",
        "JPM",
        "COST",
        "NFLX",
        "TSM",
    ]
    results = {}
    for t in tickers:
        try:
            results[t] = convene_trading_committee(t, save_resolution=False)
        except Exception as e:
            logger.error(f"Committee error for {t}: {e}")

    os.makedirs("results", exist_ok=True)
    summary_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_audited": len(results),
        "resolutions": results,
    }
    with open(COMMITTEE_FILE, "w") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload


def execute_committee_order(
    ticker: str,
    deliberation: Optional[Dict[str, Any]] = None,
    broker: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the autonomous Buying or Selling action sanctioned by the Multi-Agent Trading Committee.
    - If action_code is EXECUTE_BUY / SCALE_IN: calculates Kelly shares and buys the asset.
    - If action_code is VETO / AVOID and asset is currently held: immediately closes the position.
    """
    from src.paper_broker import PaperBroker

    broker = broker or PaperBroker()
    if deliberation is None:
        deliberation = convene_trading_committee(ticker, save_resolution=True)

    action_code = deliberation.get("action_code", "HOLD")
    spot_price = float(deliberation.get("spot_price", 100.0))
    cro = deliberation.get("cro_signoff", {})
    kelly_pct = float(cro.get("kelly_allocation_pct", 8.0))
    leverage = float(cro.get("approved_leverage", 1.0))
    conviction = float(deliberation.get("consensus_conviction_pct", 70.0)) / 100.0

    summary = broker.get_portfolio_summary()
    equity = float(summary.get("total_equity", 100000.0))
    cash = float(summary.get("cash", equity))

    if action_code in ["EXECUTE_BUY", "SCALE_IN"]:
        target_budget = equity * (kelly_pct / 100.0) * leverage
        usable_capital = min(target_budget, cash * 0.95)
        shares_to_buy = int(usable_capital / spot_price)

        if shares_to_buy > 0:
            atr_est = spot_price * 0.03
            buy_res = broker.execute_manual_buy(
                ticker=ticker,
                shares=shares_to_buy,
                price=spot_price,
                atr=atr_est,
                confidence=conviction,
            )
            return {
                "success": buy_res.get("success", False),
                "action": "BUY_EXECUTED",
                "ticker": ticker,
                "shares": shares_to_buy,
                "price": spot_price,
                "tp1_target": deliberation.get("tp1_target"),
                "tp2_target": deliberation.get("tp2_target"),
                "stop_loss_target": deliberation.get("stop_loss_target"),
                "resolution": deliberation.get("final_resolution"),
            }
        else:
            return {
                "success": False,
                "action": "INSUFFICIENT_CASH",
                "ticker": ticker,
                "message": f"Insufficient available cash (${cash:,.2f}) for Kelly sizing (${target_budget:,.2f}).",
            }

    elif action_code in ["VETO", "AVOID"]:
        if ticker in broker.state.get("open_positions", {}):
            sell_res = broker.execute_manual_sell(ticker=ticker, price=spot_price)
            return {
                "success": sell_res.get("success", False),
                "action": "SELL_VETO_EXECUTED",
                "ticker": ticker,
                "price": spot_price,
                "trade": sell_res.get("trade"),
                "resolution": deliberation.get("final_resolution"),
            }
        else:
            return {
                "success": True,
                "action": "NO_POSITION_TO_VETO",
                "ticker": ticker,
                "message": "Asset not currently held in portfolio.",
            }

    return {
        "success": True,
        "action": "HOLD_NO_ACTION",
        "ticker": ticker,
        "message": "Committee verdict is HOLD; standing in cash reserve.",
    }
