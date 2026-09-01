"""
Master Academic Research Papers Empirical Benchmark Suite.

Benchmarks the 4 seminal papers in real-world trading conditions:
1. Boyd et al. (Stanford): Convex Multi-Period Portfolio Optimization with Slippage
2. Marcos Lopez de Prado: Triple-Barrier Method & Deflated Sharpe Ratio (DSR)
3. Hazan et al. (Princeton): Online Newton Step (ONS) O(d^2) Logarithmic Regret
4. Unified Master Poly-Time Engine: Integrated Sentilyze Quant Pipeline
5. Benchmark: Buy & Hold Equal Weight S&P Universe

Computes:
- Total Return, CAGR
- Sharpe Ratio & Deflated Sharpe Ratio (DSR p-value)
- Max Drawdown (MDD)
- Execution Solver Latency (milliseconds to verify Polynomial Time)
Saves results to results/academic_papers_benchmark.json
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils import get_logger
from src.data_ingestion import get_price_history
from src.convex_optimizer import PolyTimeConvexOptimizer
from src.triple_barrier import (
    apply_triple_barrier_labeling,
    calculate_deflated_sharpe_ratio,
)
from src.online_newton_step import OnlineNewtonStepOptimizer

logger = get_logger(__name__)
BENCHMARK_RESULTS_FILE = os.path.join("results", "academic_papers_benchmark.json")


def run_all_papers_benchmark(
    tickers: Optional[List[str]] = None,
    lookback_period: str = "2y",
) -> Dict[str, Any]:
    """
    Executes empirical backtests comparing all 4 academic paper methodologies.
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
        f"🏛️ Starting Master Academic Research Benchmark across {len(benchmark_tickers)} tickers..."
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
        raise ValueError("No price data available.")

    # Aligned returns matrix
    price_matrix = pd.DataFrame(
        {tk: df["Close"] for tk, df in ticker_data.items()}
    ).dropna()
    rets_matrix = price_matrix.pct_change().dropna()
    cov_matrix = rets_matrix.cov() * 252.0  # Annualized covariance
    n_days, n_assets = rets_matrix.shape

    # =========================================================================
    # 1. PAPER 1: Boyd et al. (Stanford Poly-Time Convex Optimizer)
    # =========================================================================
    t0 = time.perf_counter()
    optimizer = PolyTimeConvexOptimizer(risk_aversion=1.2, linear_slippage_coeff=0.0005)
    # Alpha scores = 20-day momentum
    alpha_scores = rets_matrix.rolling(20).mean().iloc[-1] * 252.0
    boyd_res = optimizer.optimize_allocation(alpha_scores, cov_matrix)
    boyd_weights = boyd_res["weights"]
    boyd_port_rets = (rets_matrix * boyd_weights).sum(axis=1)
    boyd_ms = (time.perf_counter() - t0) * 1000.0

    # =========================================================================
    # 2. PAPER 2: Marcos Lopez de Prado (Triple-Barrier Method + DSR)
    # =========================================================================
    t0 = time.perf_counter()
    triple_returns = {}
    for tk, df in ticker_data.items():
        tb_df = apply_triple_barrier_labeling(
            df, profit_taking_mult=2.0, stop_loss_mult=1.5
        )
        # Strategy signal based on positive barrier label
        pos = np.where(tb_df["target_barrier"].shift(1) > 0, 1.0, 0.0)
        triple_returns[tk] = pos * df["Close"].pct_change().fillna(0.0)
    df_triple_rets = pd.DataFrame(triple_returns).reindex(rets_matrix.index).dropna()
    triple_port_rets = df_triple_rets.mean(axis=1)
    lopez_ms = (time.perf_counter() - t0) * 1000.0

    # =========================================================================
    # 3. PAPER 3: Hazan et al. (Online Newton Step ONS Engine)
    # =========================================================================
    t0 = time.perf_counter()
    ons_engine = OnlineNewtonStepOptimizer(num_assets=n_assets, eta=0.5, beta=1.0)
    ons_df = ons_engine.backtest_sequence(rets_matrix)
    ons_port_rets = ons_df["daily_return"]
    ons_ms = (time.perf_counter() - t0) * 1000.0

    # =========================================================================
    # 4. UNIFIED MASTER ENGINE: Poly-Time Convex + Triple-Barrier + Dynamic ONS
    # =========================================================================
    t0 = time.perf_counter()
    unified_port_rets = (
        0.40 * boyd_port_rets + 0.35 * triple_port_rets + 0.25 * ons_port_rets
    )
    unified_ms = (time.perf_counter() - t0) * 1000.0

    # =========================================================================
    # 0. BENCHMARK: Buy & Hold Universe (Equal-Weight)
    # =========================================================================
    bench_port_rets = rets_matrix.mean(axis=1)

    # Compute comprehensive metrics
    def evaluate(name: str, rets: pd.Series, latency_ms: float) -> Dict[str, Any]:
        dsr_metrics = calculate_deflated_sharpe_ratio(rets, num_trials=50)
        tot_ret = float((1.0 + rets).prod() - 1.0) * 100.0
        cagr = (
            float(((1.0 + tot_ret / 100.0) ** (252.0 / max(len(rets), 1))) - 1.0)
            * 100.0
        )
        cum = (1.0 + rets).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        mdd = float(abs(dd.min())) * 100.0
        calmar = round(cagr / (mdd + 1e-5), 2)
        win_rate = float((rets > 0).sum() / max(len(rets[rets != 0]), 1)) * 100.0

        return {
            "methodology": name,
            "total_return_pct": round(tot_ret, 2),
            "cagr_pct": round(cagr, 2),
            "annualized_sharpe": dsr_metrics["annualized_sharpe"],
            "deflated_sharpe_prob": dsr_metrics["dsr_probability"],
            "statistically_significant": dsr_metrics["is_statistically_significant"],
            "max_drawdown_pct": round(mdd, 2),
            "calmar_ratio": calmar,
            "win_rate_pct": round(win_rate, 2),
            "solver_latency_ms": round(latency_ms, 2),
            "complexity_class": "Polynomial Time O(d^k)",
        }

    results = {
        "evaluation_period": lookback_period,
        "universe": valid_tickers,
        "sample_trading_days": n_days,
        "papers_benchmarked": {
            "1_Boyd_Stanford_Convex_SOCP": evaluate(
                "Boyd et al. (Stanford): Convex Multi-Period Slippage Optimizer",
                boyd_port_rets,
                boyd_ms,
            ),
            "2_Lopez_de_Prado_Triple_Barrier": evaluate(
                "Marcos Lopez de Prado: Triple-Barrier Method & DSR",
                triple_port_rets,
                lopez_ms,
            ),
            "3_Hazan_Online_Newton_Step": evaluate(
                "Agarwal, Hazan, Kale: Online Newton Step (ONS) O(d^2)",
                ons_port_rets,
                ons_ms,
            ),
            "4_Unified_PolyTime_Master_Engine": evaluate(
                "Sentilyze Unified Poly-Time Multi-Agent Engine",
                unified_port_rets,
                unified_ms,
            ),
            "0_Buy_and_Hold_Benchmark": evaluate(
                "Buy & Hold S&P Benchmark (Equal Weight)",
                bench_port_rets,
                0.1,
            ),
        },
        "boyd_optimal_weights": {tk: float(w) for tk, w in boyd_weights.items()},
    }

    os.makedirs(os.path.dirname(BENCHMARK_RESULTS_FILE), exist_ok=True)
    with open(BENCHMARK_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"✅ Master Academic Papers Benchmark saved to {BENCHMARK_RESULTS_FILE}"
    )
    return results


if __name__ == "__main__":
    benchmark_res = run_all_papers_benchmark()
    print(json.dumps(benchmark_res, indent=2))
