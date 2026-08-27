"""
AI Trade Copilot & Conversational Analyst for Sentilyze.
Provides natural language conversational intelligence across portfolio status,
multi-pillar quantitative diagnostics, and dynamic stress simulations.
"""

from typing import Any, Dict, List, Optional
import os
import json
from src.utils import get_logger
from src.paper_broker import PaperBroker
from src.realtime_tracker import fetch_live_quote
from src.agent_committee import convene_trading_committee

logger = get_logger(__name__)


class AICopilotEngine:
    """Conversational intelligence engine that parses queries and generates analytical insights."""

    def __init__(self, broker: Optional[PaperBroker] = None):
        self.broker = broker or PaperBroker()

    def answer_query(self, query: str, context_ticker: str = "AVGO") -> Dict[str, Any]:
        """
        Interprets user prompt and routes to appropriate financial analytical subroutines.
        """
        q_lower = query.lower()
        logger.info(
            f"💬 Copilot processing query: '{query}' (Context: {context_ticker})"
        )

        # 1. Stress Test / Crash Simulation Queries
        if any(
            w in q_lower
            for w in [
                "stress",
                "crash",
                "drop",
                "simulate",
                "black swan",
                "recession",
                "tariff",
            ]
        ):
            return self._handle_stress_query(q_lower, context_ticker)

        # 2. Committee & Conviction Queries
        if any(
            w in q_lower
            for w in [
                "committee",
                "conviction",
                "verdict",
                "agent",
                "debate",
                "cro",
                "vote",
            ]
        ):
            return self._handle_committee_query(context_ticker)

        # 3. Ticker Diagnosis / Buy Signal Queries
        if any(
            w in q_lower
            for w in [
                "buy",
                "sell",
                "target",
                "tp1",
                "stop loss",
                "signal",
                "analyse",
                "analyze",
                "why",
            ]
        ):
            return self._handle_ticker_analysis_query(context_ticker)

        # 4. Portfolio & Position Queries
        if any(
            w in q_lower
            for w in [
                "portfolio",
                "position",
                "holding",
                "pnl",
                "profit",
                "cash",
                "balance",
            ]
        ):
            return self._handle_portfolio_query()

        # Default Comprehensive Assistant Answer
        return self._handle_general_query(query, context_ticker)

    def _handle_portfolio_query(self) -> Dict[str, Any]:
        summary = self.broker.get_portfolio_summary()
        open_pos = self.broker.state.get("open_positions", {})

        pos_lines = []
        for sym, pos in open_pos.items():
            q = fetch_live_quote(sym)
            live_p = float(q.get("price", pos.get("current_price", 0)))
            entry_p = float(pos.get("entry_price", live_p))
            shares = int(pos.get("shares", 0))
            gain = (live_p - entry_p) * shares
            gain_pct = ((live_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0

            pos_lines.append(
                f"- **{sym}**: {shares} shares @ ${entry_p:,.2f} | Current: **${live_p:,.2f}** ({gain_pct:+.2f}%) | "
                f"Unrealized P&L: **${gain:+,.2f}** | TP1: `${pos.get('tp1_target', 0):,.2f}` | Stop: `${pos.get('sl_target', 0):,.2f}`"
            )

        response_text = (
            f"### 💼 Live Portfolio Status & Risk Health\n\n"
            f"- **Total Account Equity**: **${summary['total_equity']:,.2f}**\n"
            f"- **Cash Buffer**: **${summary['cash']:,.2f}** ({summary['cash']/max(summary['total_equity'],1)*100:.1f}% cash safety)\n"
            f"- **Realized PnL**: **${summary['realized_pnl']:+,.2f}** | **Unrealized PnL**: **${summary['unrealized_pnl']:+,.2f}**\n"
            f"- **Win Rate**: **{summary['win_rate']:.1f}%** ({summary['winning_trades']} wins / {summary['losing_trades']} losses out of {summary['total_trades']} closed trades)\n\n"
            f"#### Active Positions ({len(open_pos)}):\n"
            + (
                "\n".join(pos_lines)
                if pos_lines
                else "*No active positions. Sitting in 100% Cash buffer.*"
            )
        )

        return {
            "query_category": "PORTFOLIO_STATUS",
            "markdown_response": response_text,
            "structured_data": summary,
        }

    def _handle_committee_query(self, ticker: str) -> Dict[str, Any]:
        committee_res = convene_trading_committee(ticker, save_resolution=False)
        cro = committee_res["cro_signoff"]

        testimonies_md = []
        for t in committee_res["agent_testimonies"]:
            testimonies_md.append(
                f"• **{t['agent_name']}** ({t['role']}): Voted **`{t['vote']}`** (Conviction: {t['conviction_score']}%) — *{t['thesis']}*"
            )

        response_text = (
            f"### 🏛️ Multi-Agent Committee Verdict for **{ticker}**\n\n"
            f"**Final Resolution**: `{committee_res['final_resolution']}`\n"
            f"- **Consensus Conviction**: **{committee_res['consensus_conviction_pct']:.1f}%** (Buy Votes: {cro['buy_votes']}/3 Specialist Agents)\n"
            f"- **Approved Leverage**: **{cro['approved_leverage']:.1f}x** | **Kelly Allocation**: **{cro['kelly_allocation_pct']:.1f}%**\n"
            f"- **Profit Targets**: TP1: `${committee_res['tp1_target']:,.2f}` | TP2: `${committee_res['tp2_target']:,.2f}` | Stop: `${committee_res['stop_loss_target']:,.2f}`\n\n"
            f"#### 🎙️ Specialist Agent Testimonies:\n"
            + "\n".join(testimonies_md)
            + f"\n\n"
            f"**Chief Risk Officer Sign-Off**: *\"{cro['cro_thesis']}\"*"
        )

        return {
            "query_category": "COMMITTEE_VERDICT",
            "markdown_response": response_text,
            "structured_data": committee_res,
        }

    def _handle_stress_query(self, query: str, context_ticker: str) -> Dict[str, Any]:
        drop_pct = 5.0
        if "10" in query:
            drop_pct = 10.0
        elif "15" in query:
            drop_pct = 15.0
        elif "20" in query:
            drop_pct = 20.0

        summary = self.broker.get_portfolio_summary()
        invested = summary["invested"]
        est_loss = invested * (drop_pct / 100.0)
        projected_equity = summary["total_equity"] - est_loss

        response_text = (
            f"### 🌪️ Stress-Test Simulation: **-{drop_pct:.0f}% Sector Shock**\n\n"
            f"- **Current Total Equity**: ${summary['total_equity']:,.2f}\n"
            f"- **Active Invested Capital**: ${invested:,.2f}\n"
            f"- **Estimated Portfolio Drawdown**: **-${est_loss:,.2f} (-{est_loss/max(summary['total_equity'],1)*100:.2f}%)**\n"
            f"- **Post-Shock Projected Equity**: **${projected_equity:,.2f}**\n\n"
            f"🛡️ **Protective Mechanisms Active**:\n"
            f"1. **Hard Stop-Loss Floors**: Will automatically liquidate open positions at -1.5 ATR (~-4.5% max loss per trade), preventing catastrophic loss.\n"
            f"2. **Cash Buffer**: ${summary['cash']:,.2f} ({summary['cash']/max(summary['total_equity'],1)*100:.1f}%) is unexposed to market volatility."
        )

        return {
            "query_category": "STRESS_SIMULATION",
            "markdown_response": response_text,
            "structured_data": {
                "drop_pct": drop_pct,
                "est_loss": est_loss,
                "projected_equity": projected_equity,
            },
        }

    def _handle_ticker_analysis_query(self, ticker: str) -> Dict[str, Any]:
        quote = fetch_live_quote(ticker)
        curr_p = float(quote.get("price", 100.0))
        chg = float(quote.get("change_pct", 0.0))

        atr = curr_p * 0.03
        tp1 = curr_p + (2.5 * atr)
        tp2 = curr_p + (4.5 * atr)
        sl = curr_p - (1.5 * atr)

        response_text = (
            f"### 🎯 Deep Quantitative Diagnosis for **{ticker}**\n\n"
            f"- **Live Market Price**: **${curr_p:,.2f}** ({chg:+.2f}% today)\n"
            f"- **TP1 Scale-Out (50% Profit Lock)**: **${tp1:,.2f}** (+{((tp1-curr_p)/curr_p)*100:.1f}%)\n"
            f"- **TP2 Extended Target (Runner)**: **${tp2:,.2f}** (+{((tp2-curr_p)/curr_p)*100:.1f}%)\n"
            f"- **Protective Stop-Loss**: **${sl:,.2f}** (-{((curr_p-sl)/curr_p)*100:.1f}%)\n\n"
            f"**Execution Blueprint**: Half-Kelly sizing with 50% exit at TP1. Once TP1 is achieved, the stop is trailed to Breakeven (+0.2%) to guarantee risk-free continuation."
        )

        return {
            "query_category": "TICKER_ANALYSIS",
            "markdown_response": response_text,
            "structured_data": {
                "ticker": ticker,
                "price": curr_p,
                "tp1": tp1,
                "tp2": tp2,
                "sl": sl,
            },
        }

    def _handle_general_query(self, query: str, context_ticker: str) -> Dict[str, Any]:
        return {
            "query_category": "GENERAL_ASSISTANCE",
            "markdown_response": (
                f"### 🧠 Sentilyze AI Copilot Assistant\n\n"
                f'You asked: *"{query}"*\n\n'
                f"I am connected to all 8 quantitative pillars, real-time market quotes, and the Autonomous Multi-Agent Committee. "
                f"You can ask me to:\n"
                f"- 📊 **Check Portfolio**: *'Show my active trades and cash reserves'*\n"
                f"- 🏛️ **Convene Committee**: *'What does the 4-agent committee say about {context_ticker}?'*\n"
                f"- 🌪️ **Simulate Crisis**: *'Simulate a -10% market crash on my portfolio'*\n"
                f"- 🎯 **Diagnose Ticker**: *'What are the profit targets and stop-loss for {context_ticker}?'*"
            ),
            "structured_data": {"query": query, "ticker": context_ticker},
        }
