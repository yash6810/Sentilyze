"""
Benchmark runner for Papers 15-24: Lightweight Safety Algorithms.

Runs each algorithm on real market data and collects empirical metrics.
Saves results to results/papers_15_24_benchmark.json.
"""

import json
import time
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cusum_detector import CUSUMDetector
from src.ewma_monitor import EWMACorrelationMonitor
from src.grossman_zhou import grossman_zhou_allocation
from src.page_hinkley import PageHinkleyDetector
from src.hmm_regime import GaussianHMMRegimeDetector
from src.cppi_insurance import run_cppi_backtest
from src.adwin_detector import ADWINDetector
from src.risk_constrained_kelly import risk_constrained_kelly_allocation
from src.cdar_optimizer import optimize_cdar_portfolio, calculate_cdar
from src.dcc_correlation import DCCCorrelation


def fetch_data():
    """Fetch multi-asset data for benchmarking. Uses cached CSVs first."""
    tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    raw_dir = os.path.join(base_dir, "data", "raw")

    # Try cached CSV files first
    frames = {}
    for tk in tickers:
        csv_path = os.path.join(raw_dir, f"{tk}_price_history.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            if "Close" in df.columns:
                frames[tk] = df["Close"]

    if len(frames) >= 3:
        prices = pd.DataFrame(frames).dropna()
        returns = prices.pct_change().dropna()
        print(f"  (Using cached CSVs: {len(frames)} tickers)")
        return prices, returns

    # Fallback to yfinance
    print("  (Falling back to yfinance...)")
    import yfinance as yf

    data = yf.download(tickers, period="2y", progress=False)
    if "Adj Close" in data.columns.get_level_values(0):
        prices = data["Adj Close"]
    elif "Close" in data.columns.get_level_values(0):
        prices = data["Close"]
    else:
        prices = data
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    return prices, returns


def benchmark_cusum(returns):
    """Benchmark Paper 16: CUSUM."""
    print("  [16] CUSUM Sequential Change Detection...")
    t0 = time.perf_counter()
    detector = CUSUMDetector(threshold_h=0.15, drift_k=0.005, target_mean=0.0)
    # Monitor the first ticker's returns
    first_col = returns.columns[0]
    results = detector.update_batch(returns[first_col].values)
    latency_ns = (time.perf_counter() - t0) * 1e9
    n_alarms = len(detector.alarm_history)
    return {
        "paper": "Paper 16 - CUSUM (Page 1954)",
        "ticker_monitored": first_col,
        "n_observations": len(results),
        "n_alarms": n_alarms,
        "alarm_rate_pct": round(n_alarms / max(len(results), 1) * 100, 2),
        "latency_per_obs_ns": round(latency_ns / max(len(results), 1), 1),
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(1) per observation",
    }


def benchmark_ewma(returns):
    """Benchmark Paper 17: EWMA Correlation Monitor."""
    print("  [17] EWMA Correlation Monitor...")
    t0 = time.perf_counter()
    monitor = EWMACorrelationMonitor(
        decay_lambda=0.94, correlation_alert_threshold=0.75
    )
    monitor.initialize(returns.iloc[:30])

    alert_days = 0
    avg_corrs = []
    for i in range(30, len(returns)):
        r = returns.iloc[i].values
        result = monitor.update(r)
        if result["correlation_breakdown_alert"]:
            alert_days += 1
        avg_corrs.append(result["avg_pairwise_correlation"])

    latency_ns = (time.perf_counter() - t0) * 1e9
    n_updates = len(returns) - 30
    return {
        "paper": "Paper 17 - EWMA (RiskMetrics 1996)",
        "n_assets": len(returns.columns),
        "n_updates": n_updates,
        "alert_days": alert_days,
        "alert_rate_pct": round(alert_days / max(n_updates, 1) * 100, 2),
        "mean_avg_correlation": round(float(np.mean(avg_corrs)), 4),
        "max_avg_correlation": round(float(np.max(avg_corrs)), 4),
        "latency_per_obs_ns": round(latency_ns / max(n_updates, 1), 1),
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(1) per observation per pair",
    }


def benchmark_grossman_zhou(returns):
    """Benchmark Paper 18: Grossman-Zhou Drawdown Constraint."""
    print("  [18] Grossman-Zhou Drawdown Constraint...")
    t0 = time.perf_counter()

    # Simulate a portfolio using the first ticker
    first_col = returns.columns[0]
    rets = returns[first_col].values

    wealth = 100000.0
    running_max = wealth
    portfolio_values = [wealth]
    risky_weights = []

    for r in rets:
        alloc = grossman_zhou_allocation(
            current_wealth=wealth,
            running_max_wealth=running_max,
            max_drawdown_tolerance=0.15,
        )
        risky_weights.append(alloc["risky_weight"])
        wealth *= 1.0 + alloc["risky_weight"] * r
        running_max = max(running_max, wealth)
        portfolio_values.append(wealth)

    latency_ns = (time.perf_counter() - t0) * 1e9

    pv = np.array(portfolio_values)
    peak = np.maximum.accumulate(pv)
    max_dd = float(((peak - pv) / peak).max()) * 100
    total_ret = (pv[-1] / pv[0] - 1.0) * 100
    ann_ret = ((pv[-1] / pv[0]) ** (252.0 / len(rets)) - 1.0) * 100
    daily_rets = np.diff(pv) / pv[:-1]
    sharpe = float(np.mean(daily_rets) / max(np.std(daily_rets), 1e-9) * np.sqrt(252))

    return {
        "paper": "Paper 18 - Grossman-Zhou (1993)",
        "ticker": first_col,
        "total_return_pct": round(total_ret, 2),
        "annualized_return_pct": round(ann_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "drawdown_tolerance_pct": 15.0,
        "constraint_respected": bool(max_dd <= 15.5),
        "sharpe_ratio": round(sharpe, 2),
        "avg_risky_weight": round(float(np.mean(risky_weights)), 4),
        "latency_per_obs_ns": round(latency_ns / max(len(rets), 1), 1),
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(1) per rebalance",
    }


def benchmark_page_hinkley(returns):
    """Benchmark Paper 22: Page-Hinkley Test."""
    print("  [22] Page-Hinkley Drift Detector...")
    t0 = time.perf_counter()
    detector = PageHinkleyDetector(threshold_lambda=0.5, min_magnitude_delta=0.001)
    first_col = returns.columns[0]
    results = detector.update_batch(returns[first_col].values)
    latency_ns = (time.perf_counter() - t0) * 1e9
    n_drifts = len(detector.alarm_history)
    return {
        "paper": "Paper 22 - Page-Hinkley (1954/1971)",
        "ticker_monitored": first_col,
        "n_observations": len(results),
        "n_drifts_detected": n_drifts,
        "drift_rate_pct": round(n_drifts / max(len(results), 1) * 100, 2),
        "latency_per_obs_ns": round(latency_ns / max(len(results), 1), 1),
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(1) per observation",
    }


def benchmark_hmm(returns):
    """Benchmark Paper 15: Gaussian HMM Regime Detector."""
    print("  [15] Gaussian HMM Regime Detection...")
    t0 = time.perf_counter()
    detector = GaussianHMMRegimeDetector(n_states=3)
    first_col = returns.columns[0]
    regime_df = detector.classify_series(returns[first_col].values)
    latency_ns = (time.perf_counter() - t0) * 1e9
    regime_counts = regime_df["regime"].value_counts().to_dict()
    crisis_days = int(regime_df["is_crisis"].sum())
    return {
        "paper": "Paper 15 - Gaussian HMM (Hamilton 1989)",
        "ticker": first_col,
        "n_observations": len(regime_df),
        "regime_distribution": regime_counts,
        "crisis_days": crisis_days,
        "crisis_pct": round(crisis_days / max(len(regime_df), 1) * 100, 2),
        "latency_per_obs_ns": round(latency_ns / max(len(regime_df), 1), 1),
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(T * K^2)",
    }


def benchmark_cppi(returns):
    """Benchmark Paper 20: CPPI Portfolio Insurance."""
    print("  [20] CPPI Portfolio Insurance...")
    t0 = time.perf_counter()
    first_col = returns.columns[0]
    result = run_cppi_backtest(
        returns=returns[first_col].values,
        initial_capital=100000.0,
        floor_pct=0.85,
        multiplier=3.0,
    )
    latency_ns = (time.perf_counter() - t0) * 1e9
    result["paper"] = "Paper 20 - CPPI (Black-Jones 1987)"
    result["ticker"] = first_col
    result["total_latency_ms"] = round(latency_ns / 1e6, 2)
    result["complexity"] = "O(1) per rebalance"
    return result


def benchmark_adwin(returns):
    """Benchmark Paper 21: ADWIN Drift Detector."""
    print("  [21] ADWIN Drift Detection...")
    t0 = time.perf_counter()
    detector = ADWINDetector(confidence_delta=0.002)
    first_col = returns.columns[0]
    results = detector.update_batch(returns[first_col].values)
    latency_ns = (time.perf_counter() - t0) * 1e9
    drift_count = sum(1 for r in results if r["drift_detected"])
    return {
        "paper": "Paper 21 - ADWIN (Bifet-Gavalda 2007)",
        "ticker_monitored": first_col,
        "n_observations": len(results),
        "n_drifts_detected": drift_count,
        "drift_rate_pct": round(drift_count / max(len(results), 1) * 100, 2),
        "final_window_size": results[-1]["window_size"] if results else 0,
        "latency_per_obs_ns": round(latency_ns / max(len(results), 1), 1),
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(log W) amortized",
    }


def benchmark_risk_kelly(returns):
    """Benchmark Paper 23: Risk-Constrained Kelly."""
    print("  [23] Risk-Constrained Kelly...")
    t0 = time.perf_counter()
    mu = returns.mean().values * 252.0
    cov = returns.cov().values * 252.0
    result = risk_constrained_kelly_allocation(
        expected_returns=mu,
        cov_matrix=cov,
        max_drawdown_prob=0.05,
        max_drawdown_level=0.15,
        max_leverage=1.0,
    )
    latency_ns = (time.perf_counter() - t0) * 1e9

    weights_dict = {
        tk: round(float(w), 4) for tk, w in zip(returns.columns, result["weights"])
    }
    return {
        "paper": "Paper 23 - Risk-Constrained Kelly (Busseti-Boyd 2016)",
        "n_assets": len(returns.columns),
        "weights": weights_dict,
        "log_growth_rate": result["log_growth_rate"],
        "expected_return": result["expected_return"],
        "portfolio_sharpe": result["portfolio_sharpe"],
        "variance_used_pct": result["variance_used_pct"],
        "solver_converged": result["solver_converged"],
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(d^3) convex solver",
    }


def benchmark_cdar(returns):
    """Benchmark Paper 19: CDaR Optimization."""
    print("  [19] CDaR Portfolio Optimization...")
    t0 = time.perf_counter()
    result = optimize_cdar_portfolio(returns, alpha=0.05, max_weight=0.30)
    latency_ns = (time.perf_counter() - t0) * 1e9

    weights_dict = {tk: round(float(w), 4) for tk, w in result["weights"].items()}
    return {
        "paper": "Paper 19 - CDaR (Chekhlov-Uryasev-Zabarankin 2003)",
        "n_assets": len(returns.columns),
        "weights": weights_dict,
        "portfolio_cdar": result["portfolio_cdar"],
        "annualized_return_pct": result["annualized_return"],
        "annualized_volatility_pct": result["annualized_volatility"],
        "asset_cdars": result["asset_cdars"],
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(T * d) LP",
    }


def benchmark_dcc(returns):
    """Benchmark Paper 24: DCC Correlation."""
    print("  [24] DCC Dynamic Conditional Correlation...")
    t0 = time.perf_counter()
    model = DCCCorrelation()
    result = model.fit(returns)
    latency_ns = (time.perf_counter() - t0) * 1e9

    # Extract avg corr timeseries stats
    avg_ts = result["avg_correlation_timeseries"]
    return {
        "paper": "Paper 24 - DCC (Engle 2002)",
        "n_assets": len(returns.columns),
        "n_observations": result["n_observations"],
        "final_avg_correlation": result["final_avg_pairwise_correlation"],
        "correlation_breakdown_alert": result["correlation_breakdown_alert"],
        "mean_avg_correlation": round(float(np.mean(avg_ts)), 4),
        "max_avg_correlation": round(float(np.max(avg_ts)), 4),
        "min_avg_correlation": round(float(np.min(avg_ts)), 4),
        "garch_volatilities": result["annualized_garch_volatilities"],
        "total_latency_ms": round(latency_ns / 1e6, 2),
        "complexity": "O(d^2 * T)",
    }


def run_all_benchmarks():
    """Run all 10 paper benchmarks."""
    print("Fetching market data...")
    prices, returns = fetch_data()
    print(f"  Data: {len(returns)} days, {len(returns.columns)} assets")
    print(f"  Tickers: {list(returns.columns)}")
    print()

    all_results = {}

    print("Running benchmarks:")
    all_results["paper_16_cusum"] = benchmark_cusum(returns)
    all_results["paper_17_ewma"] = benchmark_ewma(returns)
    all_results["paper_18_grossman_zhou"] = benchmark_grossman_zhou(returns)
    all_results["paper_22_page_hinkley"] = benchmark_page_hinkley(returns)
    all_results["paper_15_hmm"] = benchmark_hmm(returns)
    all_results["paper_20_cppi"] = benchmark_cppi(returns)
    all_results["paper_21_adwin"] = benchmark_adwin(returns)
    all_results["paper_23_risk_kelly"] = benchmark_risk_kelly(returns)
    all_results["paper_19_cdar"] = benchmark_cdar(returns)
    all_results["paper_24_dcc"] = benchmark_dcc(returns)

    # Summary
    all_results["_summary"] = {
        "total_papers": 10,
        "data_period": "2y",
        "n_trading_days": len(returns),
        "n_assets": len(returns.columns),
        "tickers": list(returns.columns),
    }

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "papers_15_24_benchmark.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results/papers_15_24_benchmark.json")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
