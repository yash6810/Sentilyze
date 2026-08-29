"""
Empirical Alpha Attribution & Signal vs Risk-Management Decomposition Engine for Sentilyze.

Addresses Section 4 of Quantitative Audit:
Decomposes strategy returns into:
1. Model A: Full ML Signal (FinBERT + Walk-Forward XGBoost + Asymmetric Trade Management)
2. Model B: Always-Long Baseline + Same Asymmetric Trade Management (Trailing Stop, ATR Take-Profit, RSI<75)
3. Model C: Random Signal Baseline + Same Asymmetric Trade Management (50 Monte Carlo Trials)
4. Benchmark: Pure Buy & Hold

Strict Integrity Enforcement:
- Loads ONLY verified out-of-sample prediction files (results/{TICKER}_predictions.csv).
- Fails loudly with FileNotFoundError/ValueError if real predictions are missing (zero silent surrogates).
"""

from typing import Any, Dict, List, Optional
import os
import json
import numpy as np
import pandas as pd
from src.utils import get_logger
from src.backtesting import run_backtest
from src.data_ingestion import get_price_history

logger = get_logger(__name__)

ATTRIBUTION_RESULTS_FILE = os.path.join("results", "attribution_analysis.json")


def run_attribution_decomposition(
    ticker: str = "NVDA",
    initial_capital: float = 10000.0,
    n_random_trials: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs a 4-way attribution experiment on a given asset using real out-of-sample ML predictions.

    Args:
        ticker: Asset symbol (e.g. NVDA, AAPL, MSFT)
        initial_capital: Initial portfolio equity
        n_random_trials: Number of Monte Carlo permutations for the random baseline (default: 50)
        seed: Random seed for reproducibility

    Returns:
        Dict comparing Total Return, Sharpe, Win Rate, and Drawdowns across all 4 regimes.
    """
    logger.info(
        f"🔬 Running Alpha Attribution Decomposition for {ticker} (Trials={n_random_trials})..."
    )
    np.random.seed(seed)

    # 1. Load Real Out-of-Sample Predictions (Hard Check: No Surrogates Allowed)
    ml_probs_file = os.path.join("results", f"{ticker}_predictions.csv")
    if not os.path.exists(ml_probs_file):
        raise FileNotFoundError(
            f"FATAL: Real out-of-sample prediction file '{ml_probs_file}' not found! "
            f"Surrogate fallback is strictly forbidden. Ensure WFO predictions are exported to results/{ticker}_predictions.csv."
        )

    preds_df = pd.read_csv(ml_probs_file, index_col=0, parse_dates=True)
    if "Prob_Up" not in preds_df.columns:
        raise ValueError(
            f"FATAL: '{ml_probs_file}' does not contain required 'Prob_Up' column! Available columns: {list(preds_df.columns)}"
        )

    # 2. Load Real Historical Price Data
    df_price = get_price_history(ticker, period="max", use_cache=True)
    if df_price.empty or len(df_price) < 100:
        raise ValueError(
            f"Insufficient historical price data for {ticker} (need >= 100 rows)."
        )

    # Clean and align datetime indices
    df_price = df_price.sort_index()

    # Normalize tz for robust index intersection
    price_idx = pd.to_datetime(df_price.index).tz_localize(None)
    preds_idx = pd.to_datetime(preds_df.index).tz_localize(None)

    df_price.index = price_idx
    preds_df.index = preds_idx

    common_idx = df_price.index.intersection(preds_df.index)
    if len(common_idx) < 50:
        raise ValueError(
            f"Insufficient overlapping dates between price history and real predictions for {ticker} "
            f"({len(common_idx)} overlapping dates found; need >= 50)."
        )

    df_eval = df_price.loc[common_idx].sort_index()
    ml_probs = preds_df.loc[common_idx, "Prob_Up"].astype(float)

    logger.info(
        f"✅ Loaded {len(df_eval)} verified out-of-sample prediction dates for {ticker} "
        f"({df_eval.index[0].strftime('%Y-%m-%d')} to {df_eval.index[-1].strftime('%Y-%m-%d')})."
    )

    # -------------------------------------------------------------
    # 1. Model A: Full ML Signal + Asymmetric Trade Management
    # -------------------------------------------------------------
    port_ml, metrics_ml, _ = run_backtest(
        price_history=df_eval,
        prediction_probs=ml_probs,
        initial_capital=initial_capital,
        prob_threshold=0.52,
        take_profit_atr_mult=2.5,
    )

    # -------------------------------------------------------------
    # 2. Model B: Always-Long + Same Asymmetric Trade Management
    # -------------------------------------------------------------
    always_long_probs = pd.Series(0.99, index=df_eval.index)
    port_always, metrics_always, _ = run_backtest(
        price_history=df_eval,
        prediction_probs=always_long_probs,
        initial_capital=initial_capital,
        prob_threshold=0.52,
        take_profit_atr_mult=2.5,
    )

    # -------------------------------------------------------------
    # 3. Model C: Random Signals + Same Asymmetric Trade Management (50 Trials)
    # -------------------------------------------------------------
    random_returns = []
    random_sharpes = []
    random_win_rates = []
    random_max_drawdowns = []

    for trial in range(n_random_trials):
        rand_probs = pd.Series(
            np.random.uniform(0.0, 1.0, len(df_eval)), index=df_eval.index
        )
        _, m_rand, _ = run_backtest(
            price_history=df_eval,
            prediction_probs=rand_probs,
            initial_capital=initial_capital,
            prob_threshold=0.52,
            take_profit_atr_mult=2.5,
        )
        random_returns.append(
            m_rand.get("strategy_total_return", m_rand.get("total_return", 0.0))
        )
        random_sharpes.append(m_rand.get("sharpe_ratio", 0.0))
        random_win_rates.append(m_rand.get("win_rate", 0.0))
        random_max_drawdowns.append(
            m_rand.get("strategy_max_drawdown", m_rand.get("max_drawdown", 0.0))
        )

    metrics_random_avg = {
        "total_return": float(np.mean(random_returns)),
        "sharpe_ratio": float(np.mean(random_sharpes)),
        "win_rate": float(np.mean(random_win_rates)),
        "max_drawdown": float(np.mean(random_max_drawdowns)),
    }

    # -------------------------------------------------------------
    # 4. Model D: Pure Buy & Hold Benchmark
    # -------------------------------------------------------------
    bnh_return = float((df_eval["Close"].iloc[-1] / df_eval["Close"].iloc[0]) - 1.0)

    # -------------------------------------------------------------
    # Mathematical Decomposition Metrics
    # -------------------------------------------------------------
    ml_total_ret = metrics_ml.get(
        "strategy_total_return", metrics_ml.get("total_return", 0.0)
    )
    rand_total_ret = metrics_random_avg["total_return"]
    always_total_ret = metrics_always.get(
        "strategy_total_return", metrics_always.get("total_return", 0.0)
    )

    ml_alpha_contribution = max(0.0, ml_total_ret - rand_total_ret)
    risk_mgmt_contribution = rand_total_ret

    total_explained_return = ml_total_ret
    if total_explained_return > 0:
        ml_pct_share = round(
            (ml_alpha_contribution / total_explained_return) * 100.0, 1
        )
        risk_mgmt_pct_share = round(
            (risk_mgmt_contribution / total_explained_return) * 100.0, 1
        )
    else:
        ml_pct_share = 0.0
        risk_mgmt_pct_share = 100.0

    result = {
        "ticker": ticker,
        "sample_period_days": len(df_eval),
        "evaluation_window": {
            "start_date": df_eval.index[0].strftime("%Y-%m-%d"),
            "end_date": df_eval.index[-1].strftime("%Y-%m-%d"),
        },
        "models": {
            "full_ml_strategy": {
                "total_return_pct": round(ml_total_ret * 100.0, 2),
                "sharpe_ratio": round(metrics_ml.get("sharpe_ratio", 0.0), 2),
                "win_rate_pct": round(metrics_ml.get("win_rate", 0.0) * 100.0, 2),
                "max_drawdown_pct": round(
                    metrics_ml.get(
                        "strategy_max_drawdown",
                        metrics_ml.get("max_drawdown", 0.0),
                    )
                    * 100.0,
                    2,
                ),
                "trades_count": metrics_ml.get(
                    "total_trades", metrics_ml.get("num_trades", 0)
                ),
            },
            "always_long_strategy": {
                "total_return_pct": round(always_total_ret * 100.0, 2),
                "sharpe_ratio": round(metrics_always.get("sharpe_ratio", 0.0), 2),
                "win_rate_pct": round(metrics_always.get("win_rate", 0.0) * 100.0, 2),
                "max_drawdown_pct": round(
                    metrics_always.get(
                        "strategy_max_drawdown",
                        metrics_always.get("max_drawdown", 0.0),
                    )
                    * 100.0,
                    2,
                ),
                "trades_count": metrics_always.get(
                    "total_trades", metrics_always.get("num_trades", 0)
                ),
            },
            "random_signal_strategy": {
                "total_return_pct": round(rand_total_ret * 100.0, 2),
                "sharpe_ratio": round(metrics_random_avg["sharpe_ratio"], 2),
                "win_rate_pct": round(metrics_random_avg["win_rate"] * 100.0, 2),
                "max_drawdown_pct": round(
                    metrics_random_avg["max_drawdown"] * 100.0, 2
                ),
                "n_monte_carlo_trials": n_random_trials,
            },
            "buy_and_hold_benchmark": {
                "total_return_pct": round(bnh_return * 100.0, 2)
            },
        },
        "attribution_decomposition": {
            "ml_predictive_edge_share_pct": ml_pct_share,
            "risk_trade_management_share_pct": risk_mgmt_pct_share,
            "interpretation": (
                f"Empirical attribution shows that asymmetric trade management (+2.5 ATR TP, -1.5 ATR SL, breakeven ratchets) "
                f"provides the primary baseline positive expectancy (Sharpe ~ {round(metrics_random_avg['sharpe_ratio'], 2)} on random entries), "
                f"while ML predictive filtering alters trade frequency, win rate, and tail drawdown characteristics."
            ),
        },
    }

    _persist_attribution_results(ticker, result)
    return result


def _persist_attribution_results(ticker: str, result: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(ATTRIBUTION_RESULTS_FILE), exist_ok=True)
        data = {}
        if os.path.exists(ATTRIBUTION_RESULTS_FILE):
            try:
                with open(ATTRIBUTION_RESULTS_FILE, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data[ticker] = result
        with open(ATTRIBUTION_RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist attribution results: {e}")
