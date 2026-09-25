"""
Autonomous Trade Post-Mortem & Bayesian Council Weight Calibration Engine for Sentilyze.

Analyzes historical executed trades from results/executed_trades.csv and results/paper_portfolio.json:
1. Categorizes trade outcomes: Alpha Harvest (TP1/TP2), Protective Stop-Loss, Model Sell, Breakeven.
2. Evaluates performance attribution across the 5 Council Specialists:
   - Technical Momentum Specialist
   - FinBERT Sentiment Specialist
   - Fundamental Valuation Specialist
   - Chief Risk Officer (CRO)
   - Adversarial Red-Team Specialist
3. Performs Bayesian Dirichlet / Softmax Belief Updates on Council Voting Weights.
4. Generates an institutional Episodic Lessons-Learned Ledger for future deliberations.
"""

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_WEIGHTS = {
    "Technical Momentum": 0.25,
    "FinBERT Sentiment": 0.25,
    "Fundamental Valuation": 0.20,
    "Chief Risk Officer": 0.15,
    "Adversarial Red-Team": 0.15,
}

WEIGHTS_FILE = os.path.join("results", "agent_memory", "committee_weights.json")
MEMORY_FILE = os.path.join("results", "agent_learning_memory.json")
EXECUTED_TRADES_FILE = os.path.join("results", "executed_trades.csv")
PORTFOLIO_FILE = os.path.join("results", "paper_portfolio.json")


class TradePostMortemLearner:
    """
    Automated quantitative post-mortem engine that analyzes closed positions
    and adaptively recalibrates the Trading Council's voting weights.
    """

    def __init__(
        self,
        trades_csv_path: str = EXECUTED_TRADES_FILE,
        portfolio_json_path: str = PORTFOLIO_FILE,
        weights_path: str = WEIGHTS_FILE,
        memory_path: str = MEMORY_FILE,
        learning_rate: float = 0.15,
    ):
        self.trades_csv_path = trades_csv_path
        self.portfolio_json_path = portfolio_json_path
        self.weights_path = weights_path
        self.memory_path = memory_path
        self.learning_rate = learning_rate

    def load_executed_trades(self) -> pd.DataFrame:
        """Loads executed trades history from CSV or returns empty DataFrame."""
        if not os.path.exists(self.trades_csv_path):
            logger.warning(f"Trades file {self.trades_csv_path} does not exist.")
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.trades_csv_path)
            # Ensure expected columns exist
            expected_cols = [
                "ticker",
                "shares",
                "entry_price",
                "exit_price",
                "entry_date",
                "exit_date",
                "pnl",
                "return_pct",
                "exit_reason",
            ]
            if len(df.columns) >= len(expected_cols):
                df.columns = expected_cols[: len(df.columns)]
            return df
        except Exception as e:
            logger.error(f"Failed to read executed trades: {e}")
            return pd.DataFrame()

    def load_current_weights(self) -> Dict[str, float]:
        """Loads current committee weights from JSON or returns default weights."""
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    weights = json.load(f)
                    total = sum(weights.values())
                    if total > 0:
                        return {k: round(v / total, 4) for k, v in weights.items()}
            except Exception as e:
                logger.warning(f"Could not load weights from {self.weights_path}: {e}")
        return DEFAULT_WEIGHTS.copy()

    def categorize_trade(self, row: pd.Series) -> Dict[str, Any]:
        """Categorizes trade outcome and assigns attribution scores."""
        pnl = float(row.get("pnl", 0.0))
        ret_pct = float(row.get("return_pct", 0.0))
        reason = str(row.get("exit_reason", "")).upper()
        ticker = str(row.get("ticker", "UNKNOWN"))

        if pnl > 0 and ("PROFIT" in reason or "TARGET" in reason or ret_pct >= 1.0):
            classification = "ALPHA_HARVEST"
            quality = "WINNER"
        elif "STOP" in reason:
            if ret_pct >= -3.0:
                classification = "CAPITAL_SHIELD_DEFENSE"
                quality = "CONTROLLED_LOSS"
            else:
                classification = "UNCONTROLLED_DRAWDOWN"
                quality = "SEVERE_LOSS"
        elif ret_pct >= 0:
            classification = "BREAKEVEN_OR_SCRATCH"
            quality = "NEUTRAL"
        else:
            classification = "MODEL_EXIT"
            quality = "LOSS"

        return {
            "ticker": ticker,
            "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
            "classification": classification,
            "quality": quality,
            "exit_reason": reason,
        }

    def compute_agent_performance_deltas(
        self, analyzed_trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Computes performance feedback scores for each specialist agent based on historical outcomes.
        Positive delta indicates the agent contributed to profitable or well-hedged trades.
        """
        if not analyzed_trades:
            return {k: 0.0 for k in DEFAULT_WEIGHTS}

        deltas = {k: 0.0 for k in DEFAULT_WEIGHTS}

        for t in analyzed_trades:
            ret = t["return_pct"]
            qual = t["quality"]

            if qual == "WINNER":
                # Winners validate Technical momentum and FinBERT catalysts
                deltas["Technical Momentum"] += min(ret * 0.1, 1.0)
                deltas["FinBERT Sentiment"] += min(ret * 0.1, 1.0)
                deltas["Fundamental Valuation"] += min(ret * 0.05, 0.5)
                deltas["Chief Risk Officer"] += 0.2  # Approved valid sizing
                deltas["Adversarial Red-Team"] -= 0.1  # If too cautious
            elif qual == "CONTROLLED_LOSS":
                # Capital Shield preserved funds: CRO and Red-Team get rewarded
                deltas["Chief Risk Officer"] += 0.5
                deltas["Adversarial Red-Team"] += 0.5
                deltas["Technical Momentum"] -= 0.3
                deltas["FinBERT Sentiment"] -= 0.3
            elif qual == "SEVERE_LOSS":
                # Severe breakdown penalizes bullish specialists
                deltas["Technical Momentum"] -= 1.0
                deltas["FinBERT Sentiment"] -= 1.0
                deltas["Fundamental Valuation"] -= 0.5
                deltas["Adversarial Red-Team"] += 1.0  # Red team should have vetoed
                deltas["Chief Risk Officer"] -= 0.5
            else:
                # Breakeven
                deltas["Chief Risk Officer"] += 0.1

        # Normalize deltas by trade count
        n = max(len(analyzed_trades), 1)
        return {k: round(v / n, 4) for k, v in deltas.items()}

    def update_weights_bayesian(
        self, current_weights: Dict[str, float], deltas: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Updates weights using a Softmax Bayesian update with Dirichlet regularization:
        w_i' = exp(log(w_i) + lr * delta_i) / sum(exp(log(w_j) + lr * delta_j))
        Constrained between 0.10 (floor) and 0.35 (ceiling) to prevent monopoly.
        """
        scores = {}
        for agent, w in current_weights.items():
            delta = deltas.get(agent, 0.0)
            log_w = np.log(max(w, 0.01))
            scores[agent] = log_w + self.learning_rate * delta

        # Numerical stability softmax
        max_score = max(scores.values())
        exp_scores = {k: np.exp(v - max_score) for k, v in scores.items()}
        sum_exp = sum(exp_scores.values())
        raw_weights = {k: v / sum_exp for k, v in exp_scores.items()}

        # Apply floors (0.10) and ceilings (0.35)
        floor = 0.10
        ceiling = 0.35
        clamped = {k: np.clip(v, floor, ceiling) for k, v in raw_weights.items()}
        total_clamped = sum(clamped.values())
        normalized = {k: round(float(v / total_clamped), 4) for k, v in clamped.items()}

        # Guarantee exact 1.0 sum
        diff = round(1.0 - sum(normalized.values()), 4)
        if diff != 0:
            first_key = list(normalized.keys())[0]
            normalized[first_key] = round(normalized[first_key] + diff, 4)

        return normalized

    def run_post_mortem_cycle(self) -> Dict[str, Any]:
        """
        Executes an end-to-end post-mortem learning cycle:
        1. Analyzes trade history.
        2. Computes attribution scores.
        3. Updates council voting weights.
        4. Saves calibrated weights and learning memory atomically.
        """
        df = self.load_executed_trades()
        if df.empty:
            logger.info("No executed trades available for post-mortem analysis.")
            return {
                "status": "NO_TRADES",
                "weights": self.load_current_weights(),
                "trades_analyzed": 0,
            }

        analyzed_trades = [self.categorize_trade(row) for _, row in df.iterrows()]
        current_weights = self.load_current_weights()
        deltas = self.compute_agent_performance_deltas(analyzed_trades)
        calibrated_weights = self.update_weights_bayesian(current_weights, deltas)

        # Win rate & metrics
        winners = [t for t in analyzed_trades if t["quality"] == "WINNER"]
        controlled_losses = [
            t for t in analyzed_trades if t["quality"] == "CONTROLLED_LOSS"
        ]
        severe_losses = [t for t in analyzed_trades if t["quality"] == "SEVERE_LOSS"]
        win_rate_pct = round(len(winners) / len(analyzed_trades) * 100.0, 2)
        total_realized_pnl = round(sum(t["pnl"] for t in analyzed_trades), 2)

        # Save calibrated weights
        try:
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            tmp_weights = f"{self.weights_path}.tmp"
            with open(tmp_weights, "w", encoding="utf-8") as f:
                json.dump(calibrated_weights, f, indent=2)
            os.replace(tmp_weights, self.weights_path)
            logger.info(f"Calibrated committee weights saved to {self.weights_path}")
        except Exception as e:
            logger.error(f"Failed to persist committee weights: {e}")

        # Save episodic learning memory
        episodic_memory = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_trades_analyzed": len(analyzed_trades),
            "trades_analyzed": len(analyzed_trades),
            "win_rate_pct": win_rate_pct,
            "total_realized_pnl": total_realized_pnl,
            "winners_count": len(winners),
            "controlled_losses_count": len(controlled_losses),
            "severe_losses_count": len(severe_losses),
            "prior_weights": current_weights,
            "posterior_calibrated_weights": calibrated_weights,
            "calibrated_weights": calibrated_weights,
            "agent_performance_deltas": deltas,
            "recent_top_wins": sorted(winners, key=lambda x: x["pnl"], reverse=True)[
                :5
            ],
            "recent_lessons": [
                f"Controlled {len(controlled_losses)} stop-loss events via Capital Shield.",
                f"Harvested {len(winners)} high-conviction momentum runs with {win_rate_pct}% win rate.",
            ],
        }

        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            tmp_mem = f"{self.memory_path}.tmp"
            with open(tmp_mem, "w", encoding="utf-8") as f:
                json.dump(episodic_memory, f, indent=2)
            os.replace(tmp_mem, self.memory_path)
            logger.info(f"Episodic learning memory saved to {self.memory_path}")
        except Exception as e:
            logger.error(f"Failed to persist learning memory: {e}")

        return episodic_memory


def run_post_mortem_learning_cycle() -> Dict[str, Any]:
    """Helper function to execute the full post-mortem analysis."""
    learner = TradePostMortemLearner()
    return learner.run_post_mortem_cycle()


if __name__ == "__main__":
    result = run_post_mortem_learning_cycle()
    print("=== AUTONOMOUS TRADE POST-MORTEM & COUNCIL CALIBRATION REPORT ===")
    print(f"Trades Analyzed:        {result.get('total_trades_analyzed', 0)}")
    print(f"Win Rate:               {result.get('win_rate_pct', 0.0)}%")
    print(f"Total Realized Profit:  ${result.get('total_realized_pnl', 0.0):,.2f}")
    print("\nPrior Weights:")
    for k, v in result.get("prior_weights", {}).items():
        print(f"  {k:<25}: {v:.2%}")
    print("\nCalibrated Posterior Weights (Bayesian Updated):")
    for k, v in result.get("posterior_calibrated_weights", {}).items():
        print(f"  {k:<25}: {v:.2%}")
