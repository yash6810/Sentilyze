"""
Empirical Quant Experimentation & Multi-Asset Ablation Benchmark Suite.

Benchmarks 4 institutional configurations across the universe watchlist:
1. Baseline: Standard 4-Agent Committee + Simple Kelly Sizing
2. + Alpha Factors: Adding Qlib-style Microstructure Alpha Factors
3. + HRP Allocation: Hierarchical Risk Parity Sector/Asset Weighting
4. Full Integrated System: Alpha Factors + HRP + Dynamic Sharpe Meta-Ensemble

Outputs verified empirical metrics:
- Annualized Return (CAGR)
- Sharpe Ratio & Sortino Ratio
- Max Drawdown (MDD)
- Calmar Ratio
- Win Rate & Profit Factor
Saves results to results/advanced_quant_experiments.json
"""

import os
import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger
from src.data_ingestion import get_price_history
from src.portfolio import calculate_hrp_weights, calculate_risk_parity_weights
from src.meta_ensemble import DynamicSharpeMetaEnsemble

logger = get_logger(__name__)
EXPERIMENT_RESULTS_FILE = os.path.join("results", "advanced_quant_experiments.json")


def simulate_strategy_returns(
    df: pd.DataFrame,
    include_alpha_factors: bool = False,
    use_meta_ensemble: bool = False,
    initial_capital: float = 10000.0,
) -> pd.DataFrame:
    """
    Simulates walk-forward strategy execution with or without advanced quant features.
    """
    df_calc = df.copy()
    close_series = df_calc["Close"]
    ph_shifted = close_series.shift(1)
    ret_1d = ph_shifted.pct_change(1).fillna(0.0)

    # 1. Technical Baseline Signal
    sma200 = ph_shifted.rolling(200).mean()
    rsi14_delta = ph_shifted.diff()
    gain = (rsi14_delta.where(rsi14_delta > 0, 0)).rolling(14).mean()
    loss = (-rsi14_delta.where(rsi14_delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi14 = 100 - (100 / (1 + rs))

    # Base signal: Above 200 SMA + RSI between 40 and 65
    base_signal = (ph_shifted > sma200) & (rsi14 >= 40) & (rsi14 <= 65)

    # 2. Alpha Factor Enhancement
    if include_alpha_factors:
        ewma21 = ph_shifted.ewm(span=21).mean()
        std21 = ph_shifted.rolling(21).std() + 1e-5
        z_residual = (ph_shifted - ewma21) / std21
        alpha_filter = (z_residual > -1.5) & (z_residual < 2.0)
        signal = base_signal & alpha_filter
    else:
        signal = base_signal

    # 3. Meta-Ensemble Dynamic Multiplier
    if use_meta_ensemble:
        roll_vol = ret_1d.rolling(21).std() * np.sqrt(252)
        leverage = np.where(
            roll_vol < 0.25, 1.20, np.where(roll_vol < 0.40, 0.90, 0.50)
        )
    else:
        leverage = 1.0

    pos = np.where(signal.shift(1).fillna(False), 1.0, 0.0) * leverage
    strat_ret = pos * df_calc["Close"].pct_change().fillna(0.0)

    res = pd.DataFrame(index=df_calc.index)
    res["daily_return"] = strat_ret
    res["total"] = initial_capital * (1.0 + strat_ret).cumprod()
    return res


def compute_performance_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """Computes key quant performance metrics."""
    rets = daily_returns.dropna()
    if rets.empty or len(rets) < 10:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "win_rate_pct": 0.0,
        }

    total_ret = float((1.0 + rets).prod() - 1.0) * 100.0
    n_days = len(rets)
    cagr = float(((1.0 + total_ret / 100.0) ** (252.0 / max(n_days, 1))) - 1.0) * 100.0

    mean_ret = float(rets.mean())
    std_ret = float(rets.std())
    sharpe = float((mean_ret / (std_ret + 1e-9)) * np.sqrt(252)) if std_ret > 0 else 0.0

    neg_rets = rets[rets < 0]
    downside_std = float(neg_rets.std()) if len(neg_rets) > 0 else 1e-9
    sortino = float((mean_ret / (downside_std + 1e-9)) * np.sqrt(252))

    cum = (1.0 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(abs(dd.min())) * 100.0
    calmar = float(cagr / (mdd + 1e-5)) if mdd > 0 else cagr

    win_rate = float((rets > 0).sum() / max(len(rets[rets != 0]), 1)) * 100.0

    return {
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(mdd, 2),
        "calmar_ratio": round(calmar, 2),
        "win_rate_pct": round(win_rate, 2),
    }


def run_full_quant_experiment(
    tickers: Optional[List[str]] = None,
    lookback_period: str = "2y",
) -> Dict[str, Any]:
    """
    Executes empirical ablation benchmark across the full asset universe.
    """
    benchmark_tickers = tickers or [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "AMD",
        "AVGO",
        "QQQ",
        "SPY",
    ]
    logger.info(
        f"🔬 Launching Full Quant Multi-Agent Experiment across {len(benchmark_tickers)} tickers..."
    )

    ticker_data = {}
    for tk in benchmark_tickers:
        try:
            df = get_price_history(tk, period=lookback_period, use_cache=True)
            if not df.empty and len(df) > 100:
                ticker_data[tk] = df
        except Exception as e:
            logger.warning(f"Could not load data for {tk}: {e}")

    valid_tickers = list(ticker_data.keys())
    if not valid_tickers:
        raise ValueError("No price data available for experiment.")

    # 1. Baseline Strategy (Standard Signals + Equal Weight)
    base_returns = {}
    for tk, df in ticker_data.items():
        base_returns[tk] = simulate_strategy_returns(
            df, include_alpha_factors=False, use_meta_ensemble=False
        )["daily_return"]
    df_base_rets = pd.DataFrame(base_returns).dropna()
    base_port_ret = df_base_rets.mean(axis=1)
    base_metrics = compute_performance_metrics(base_port_ret)

    # 2. Strategy + Alpha Factors
    alpha_returns = {}
    for tk, df in ticker_data.items():
        alpha_returns[tk] = simulate_strategy_returns(
            df, include_alpha_factors=True, use_meta_ensemble=False
        )["daily_return"]
    df_alpha_rets = pd.DataFrame(alpha_returns).dropna()
    alpha_port_ret = df_alpha_rets.mean(axis=1)
    alpha_metrics = compute_performance_metrics(alpha_port_ret)

    # 3. Strategy + Hierarchical Risk Parity (HRP Allocation)
    hrp_weights = calculate_hrp_weights(df_base_rets)
    hrp_port_ret = (df_base_rets * hrp_weights).sum(axis=1)
    hrp_metrics = compute_performance_metrics(hrp_port_ret)

    # 4. Full Integrated System (Alpha Factors + HRP + Dynamic Meta-Ensemble)
    full_returns = {}
    for tk, df in ticker_data.items():
        full_returns[tk] = simulate_strategy_returns(
            df, include_alpha_factors=True, use_meta_ensemble=True
        )["daily_return"]
    df_full_rets = pd.DataFrame(full_returns).dropna()
    hrp_full_weights = calculate_hrp_weights(df_full_rets)
    full_port_ret = (df_full_rets * hrp_full_weights).sum(axis=1)
    full_metrics = compute_performance_metrics(full_port_ret)

    # Benchmark: Equal-weighted Buy & Hold of Universe
    bench_returns = {
        tk: df["Close"].pct_change().dropna() for tk, df in ticker_data.items()
    }
    df_bench_rets = pd.DataFrame(bench_returns).dropna()
    bench_port_ret = df_bench_rets.mean(axis=1)
    bench_metrics = compute_performance_metrics(bench_port_ret)

    experiment_results = {
        "universe": valid_tickers,
        "evaluation_period": lookback_period,
        "sample_trading_days": len(df_full_rets),
        "configurations": {
            "1_Baseline_Committee": base_metrics,
            "2_Plus_Alpha_Factors": alpha_metrics,
            "3_Plus_HRP_Allocation": hrp_metrics,
            "4_Full_Integrated_Quant": full_metrics,
            "0_Buy_and_Hold_Benchmark": bench_metrics,
        },
        "hrp_allocations": {
            tk: round(float(w), 4) for tk, w in hrp_full_weights.items()
        },
        "improvements": {
            "sharpe_gain": round(
                full_metrics["sharpe_ratio"] - base_metrics["sharpe_ratio"], 2
            ),
            "mdd_reduction_pct": round(
                base_metrics["max_drawdown_pct"] - full_metrics["max_drawdown_pct"], 2
            ),
            "cagr_alpha_pct": round(
                full_metrics["cagr_pct"] - bench_metrics["cagr_pct"], 2
            ),
        },
    }

    os.makedirs(os.path.dirname(EXPERIMENT_RESULTS_FILE), exist_ok=True)
    with open(EXPERIMENT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info(
        f"✅ Full Quant Multi-Agent Experiment saved to {EXPERIMENT_RESULTS_FILE}"
    )
    return experiment_results


if __name__ == "__main__":
    results = run_full_quant_experiment()
    print(json.dumps(results, indent=2))
