"""
Counterfactual Trade Replay Simulator (Sprint 1, Module 1.4 / Idea 43)

Mines executed trades ledger (results/executed_trades.csv) and simulates
counterfactual 'what-if' trajectories across varied trailing stop distances (1.5x - 3.5x ATR),
profit target thresholds (3% - 10%), and post-exit forward drift.

Calculates:
- Baseline realized PnL vs Counterfactual PnL across parameter grids
- 'Alpha Left on the Table' (post-exit runner potential) vs 'Disaster Averted' (loss protection)
- Optimal risk-reward parameters maximizing Expectancy and Profit Factor
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from src.data_ingestion import get_price_history

logger = logging.getLogger("Sentilyze.CounterfactualSimulator")
logging.basicConfig(level=logging.INFO)

DEFAULT_TRADES_PATH = os.path.join("results", "executed_trades.csv")
DEFAULT_REPORT_PATH = os.path.join("results", "counterfactual_report.json")


class CounterfactualTradeSimulator:
    """
    Simulates counterfactual execution paths on historical closed trades.
    Evaluates alternative risk-management rules to optimize future trade sizing & exits.
    """

    def __init__(self, trades_path: str = DEFAULT_TRADES_PATH):
        self.trades_path = trades_path

    def load_trades(self) -> pd.DataFrame:
        """Loads executed trades ledger safely without modifying original state."""
        if not os.path.exists(self.trades_path):
            logger.warning(f"Executed trades file not found at {self.trades_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.trades_path)
            # Ensure necessary columns exist
            required = [
                "ticker",
                "shares",
                "entry_price",
                "exit_price",
                "pnl",
                "return_pct",
            ]
            if not all(col in df.columns for col in required):
                logger.error(f"Missing required columns in {self.trades_path}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error reading executed trades: {e}")
            return pd.DataFrame()

    def simulate_single_trade(
        self,
        trade: pd.Series,
        stop_multiplier: float = 2.0,
        tp_pct: float = 0.05,
        use_trailing: bool = True,
        hist_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Simulates counterfactual exit for a single trade.
        If intraday/daily history is available, replays daily High/Low/Close.
        Otherwise, approximates using baseline return, high/low bounds, and volatility proxy.
        """
        ticker = str(trade.get("ticker", "UNKNOWN"))
        entry_price = float(trade.get("entry_price", 100.0))
        baseline_exit = float(trade.get("exit_price", entry_price))
        shares = float(trade.get("shares", 1.0))
        baseline_pnl = float(trade.get("pnl", 0.0))
        baseline_ret = float(trade.get("return_pct", 0.0))

        # Default ATR proxy: 2.5% of entry price if not explicitly in data
        atr_proxy = entry_price * 0.025

        cf_stop_loss = entry_price - (stop_multiplier * atr_proxy)
        cf_take_profit = entry_price * (1.0 + tp_pct)

        # If historical bars provided for trade window, perform bar-by-bar walk
        if hist_df is not None and not hist_df.empty and len(hist_df) >= 2:
            highest_close = entry_price
            exit_price = baseline_exit
            exit_reason = "COUNTERFACTUAL_EXPIRY"

            for _, row in hist_df.iterrows():
                high = float(row.get("High", row.get("Close", entry_price)))
                low = float(row.get("Low", row.get("Close", entry_price)))
                close = float(row.get("Close", entry_price))

                if close > highest_close:
                    highest_close = close

                # Update trailing stop if enabled
                current_stop = (
                    highest_close - (stop_multiplier * atr_proxy)
                    if use_trailing
                    else cf_stop_loss
                )

                if low <= current_stop:
                    exit_price = current_stop
                    exit_reason = "COUNTERFACTUAL_STOP_LOSS"
                    break
                elif high >= cf_take_profit:
                    exit_price = cf_take_profit
                    exit_reason = "COUNTERFACTUAL_TAKE_PROFIT"
                    break

            cf_pnl = round((exit_price - entry_price) * shares, 2)
            cf_ret = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
        else:
            # Analytical synthetic replay based on baseline return and barrier distances
            # If baseline won substantially (> tp_pct), check if looser stop allowed capture
            if baseline_ret >= (tp_pct * 100.0):
                cf_exit = max(
                    cf_take_profit, baseline_exit if use_trailing else cf_take_profit
                )
                cf_pnl = round((cf_exit - entry_price) * shares, 2)
                cf_ret = round(((cf_exit - entry_price) / entry_price) * 100.0, 2)
                exit_reason = "COUNTERFACTUAL_TAKE_PROFIT"
            elif baseline_ret < -1.5:
                # Baseline took a loss: a tighter stop would have cut loss earlier; looser stop might take bigger hit
                cf_exit = max(
                    cf_stop_loss, entry_price * (1.0 - (stop_multiplier * 0.025))
                )
                cf_pnl = round((cf_exit - entry_price) * shares, 2)
                cf_ret = round(((cf_exit - entry_price) / entry_price) * 100.0, 2)
                exit_reason = "COUNTERFACTUAL_STOP_LOSS"
            else:
                cf_exit = baseline_exit
                cf_pnl = baseline_pnl
                cf_ret = baseline_ret
                exit_reason = "COUNTERFACTUAL_BASELINE_PARITY"

        return {
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry_price,
            "baseline_exit": baseline_exit,
            "baseline_pnl": baseline_pnl,
            "baseline_ret_pct": baseline_ret,
            "cf_exit_price": round(cf_exit if "cf_exit" in locals() else exit_price, 2),
            "cf_pnl": cf_pnl,
            "cf_ret_pct": cf_ret,
            "pnl_delta": round(cf_pnl - baseline_pnl, 2),
            "exit_reason": exit_reason,
            "stop_multiplier": stop_multiplier,
            "tp_pct": tp_pct,
        }

    def simulate_grid(
        self,
        stop_multipliers: Optional[List[float]] = None,
        tp_pcts: Optional[List[float]] = None,
        max_trades: int = 100,
    ) -> Dict[str, Any]:
        """
        Executes a 2D parameter grid search across stop ATR multipliers and Take Profit targets.
        Compares baseline total realized PnL against each counterfactual parameter set.
        """
        if stop_multipliers is None:
            stop_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5]
        if tp_pcts is None:
            tp_pcts = [0.03, 0.05, 0.07, 0.10]

        df = self.load_trades()
        if df.empty:
            return {
                "status": "NO_DATA",
                "message": "Executed trades ledger is empty.",
                "grid_results": [],
            }

        trades_to_eval = df.tail(max_trades)
        baseline_total_pnl = round(float(trades_to_eval["pnl"].sum()), 2)
        baseline_win_rate = round(float((trades_to_eval["pnl"] > 0).mean()) * 100.0, 1)

        grid_results = []
        best_pnl = -float("inf")
        optimal_params = {}

        for sm in stop_multipliers:
            for tp in tp_pcts:
                sim_pnls = []
                sim_wins = 0

                for _, trade in trades_to_eval.iterrows():
                    res = self.simulate_single_trade(
                        trade, stop_multiplier=sm, tp_pct=tp
                    )
                    pnl = res["cf_pnl"]
                    sim_pnls.append(pnl)
                    if pnl > 0:
                        sim_wins += 1

                total_pnl = round(float(np.sum(sim_pnls)), 2)
                win_rate = round((sim_wins / max(len(sim_pnls), 1)) * 100.0, 1)
                pnl_delta = round(total_pnl - baseline_total_pnl, 2)

                gross_profit = sum(p for p in sim_pnls if p > 0)
                gross_loss = abs(sum(p for p in sim_pnls if p < 0))
                profit_factor = (
                    round(gross_profit / gross_loss, 2)
                    if gross_loss > 0
                    else (99.0 if gross_profit > 0 else 1.0)
                )

                row_result = {
                    "stop_atr_multiplier": sm,
                    "take_profit_pct": tp,
                    "total_pnl": total_pnl,
                    "pnl_delta_vs_baseline": pnl_delta,
                    "win_rate_pct": win_rate,
                    "profit_factor": profit_factor,
                    "trades_evaluated": len(sim_pnls),
                }
                grid_results.append(row_result)

                if total_pnl > best_pnl:
                    best_pnl = total_pnl
                    optimal_params = row_result

        return {
            "status": "SUCCESS",
            "trades_count": len(trades_to_eval),
            "baseline_total_pnl": baseline_total_pnl,
            "baseline_win_rate_pct": baseline_win_rate,
            "optimal_policy": optimal_params,
            "grid_results": grid_results,
        }

    def analyze_post_exit_drift(
        self, ticker: str, exit_price: float, lookahead_days: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluates what the stock did AFTER exiting:
        - If stock continued dropping, exit saved money ('Disaster Averted')
        - If stock surged higher, exit was premature ('Alpha Left on Table')
        """
        try:
            df = get_price_history(ticker, period="1mo", use_cache=True)
            if df.empty or len(df) < 5:
                return {"drift_status": "INSUFFICIENT_DATA", "alpha_left_pct": 0.0}

            recent_closes = df["Close"].iloc[-lookahead_days:]
            max_post_close = float(recent_closes.max())
            min_post_close = float(recent_closes.min())

            max_run_pct = round(((max_post_close - exit_price) / exit_price) * 100.0, 2)
            max_drop_pct = round(
                ((min_post_close - exit_price) / exit_price) * 100.0, 2
            )

            verdict = "WELL_TIMED_EXIT"
            if max_run_pct > 4.0:
                verdict = "PREMATURE_EXIT_RUNNER_MISSED"
            elif max_drop_pct < -4.0:
                verdict = "EXCELLENT_CAPITAL_PRESERVATION"

            return {
                "drift_status": verdict,
                "max_run_post_exit_pct": max_run_pct,
                "max_drop_post_exit_pct": max_drop_pct,
                "lookahead_days": lookahead_days,
            }
        except Exception as e:
            logger.debug(f"Post-exit drift notice for {ticker}: {e}")
            return {"drift_status": "DATA_UNAVAILABLE", "alpha_left_pct": 0.0}

    def generate_report(self, output_path: str = DEFAULT_REPORT_PATH) -> Dict[str, Any]:
        """Runs full grid simulation and saves results/counterfactual_report.json."""
        grid_analysis = self.simulate_grid()
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "module": "Sprint 1.4: Counterfactual Trade Replay Simulator (Idea 43)",
            "analysis": grid_analysis,
        }
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"✅ Saved counterfactual report to {output_path}")
        except Exception as e:
            logger.error(f"Failed to persist counterfactual report: {e}")
        return report


def get_counterfactual_simulator() -> CounterfactualTradeSimulator:
    return CounterfactualTradeSimulator()
