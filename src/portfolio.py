import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from src.utils import get_logger

logger = get_logger(__name__)


def load_all_ticker_portfolios(
    results_dir: str = "results", tickers: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load individual backtest portfolio CSVs for all available tickers.

    Args:
        results_dir (str): Directory where {ticker}_portfolio.csv files are stored.
        tickers (List[str], optional): List of tickers to load. If None, loaded from stocks.txt.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping ticker symbol to its portfolio DataFrame.
    """
    if tickers is None:
        stocks_file = "stocks.txt"
        if os.path.exists(stocks_file):
            with open(stocks_file, "r") as f:
                tickers = [line.strip() for line in f if line.strip()]
        else:
            tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN"]

    portfolios = {}
    for ticker in tickers:
        file_path = os.path.join(results_dir, f"{ticker}_portfolio.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            portfolios[ticker] = df
            logger.info(f"Loaded portfolio for {ticker} ({len(df)} days)")
        else:
            logger.warning(f"Portfolio file not found for {ticker}: {file_path}")

    return portfolios


def calculate_risk_parity_weights(
    returns_df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate Inverse-Volatility (Naive Risk Parity) weights for each asset.
    Assets with lower daily volatility receive higher capital weights.

    Args:
        returns_df (pd.DataFrame): DataFrame of daily returns for each ticker.

    Returns:
        pd.Series: Normalized portfolio weights summing to 1.0.
    """
    volatilities = returns_df.std()
    # Handle zero or NaN volatility
    volatilities = volatilities.replace(0, np.nan).fillna(volatilities.mean())
    inv_vol = 1.0 / (volatilities + 1e-10)
    weights = inv_vol / inv_vol.sum()
    return weights


def build_unified_portfolio(
    initial_capital: float = 100000.0,
    results_dir: str = "results",
    tickers: List[str] = None,
    allocation_method: str = "risk_parity",
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Combines individual stock strategies into a single managed multi-asset fund.

    Args:
        initial_capital (float): Total starting capital for the fund.
        results_dir (str): Directory containing backtest CSV results.
        tickers (List[str], optional): List of tickers.
        allocation_method (str): 'risk_parity' (inverse-volatility) or 'equal_weight'.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
            - unified_df: Daily master fund valuation, daily returns, drawdown, and benchmark.
            - metrics: Summary statistics (Sharpe, Sortino, Total Return, Max Drawdown).
            - weights_df: Allocation breakdown per ticker.
    """
    portfolios = load_all_ticker_portfolios(results_dir=results_dir, tickers=tickers)

    if not portfolios:
        raise ValueError("No portfolio data found to build unified fund.")

    # Align all portfolios on the common intersection of dates
    common_dates = None
    for ticker, df in portfolios.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    common_dates = common_dates.sort_values()

    # Extract daily strategy returns and benchmark returns per ticker
    strat_returns = pd.DataFrame(index=common_dates)
    bench_returns = pd.DataFrame(index=common_dates)

    for ticker, df in portfolios.items():
        aligned_df = df.loc[common_dates]
        strat_returns[ticker] = aligned_df["total"].pct_change().fillna(0)
        bench_returns[ticker] = aligned_df["benchmark"].pct_change().fillna(0)

    # Determine allocation weights
    if allocation_method == "risk_parity":
        weights = calculate_risk_parity_weights(strat_returns)
    else:  # equal weight
        n = len(portfolios)
        weights = pd.Series(1.0 / n, index=portfolios.keys())

    # Calculate weighted daily returns for Strategy and Benchmark
    unified_strat_return = (strat_returns * weights).sum(axis=1)
    unified_bench_return = (bench_returns * (1.0 / len(portfolios))).sum(axis=1)

    # Reconstruct portfolio equity curves starting from initial_capital
    unified_df = pd.DataFrame(index=common_dates)
    unified_df["daily_return"] = unified_strat_return
    unified_df["benchmark_daily_return"] = unified_bench_return

    unified_df["total"] = initial_capital * (1.0 + unified_strat_return).cumprod()
    unified_df["benchmark"] = (
        initial_capital * (1.0 + unified_bench_return).cumprod()
    )

    # Drawdown calculations
    running_max = unified_df["total"].cummax()
    unified_df["drawdown"] = (unified_df["total"] - running_max) / running_max

    bench_running_max = unified_df["benchmark"].cummax()
    unified_df["benchmark_drawdown"] = (
        unified_df["benchmark"] - bench_running_max
    ) / bench_running_max

    # Performance Metrics
    total_strat_return = (
        (unified_df["total"].iloc[-1] - initial_capital) / initial_capital
    )
    total_bench_return = (
        (unified_df["benchmark"].iloc[-1] - initial_capital) / initial_capital
    )
    max_drawdown = float(unified_df["drawdown"].min())
    bench_max_drawdown = float(unified_df["benchmark_drawdown"].min())

    # Annualized Sharpe Ratio (252 trading days)
    mean_ret = unified_strat_return.mean()
    std_ret = unified_strat_return.std()
    sharpe_ratio = float((mean_ret / (std_ret + 1e-10)) * np.sqrt(252))

    # Sortino Ratio (Downside deviation only)
    downside_returns = unified_strat_return[unified_strat_return < 0]
    downside_std = (
        downside_returns.std() if len(downside_returns) > 0 else std_ret
    )
    sortino_ratio = float(
        (mean_ret / (downside_std + 1e-10)) * np.sqrt(252)
    )

    metrics = {
        "initial_capital": initial_capital,
        "final_value": float(unified_df["total"].iloc[-1]),
        "strategy_total_return": float(total_strat_return),
        "benchmark_total_return": float(total_bench_return),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "sortino_ratio": round(sortino_ratio, 2),
        "max_drawdown": float(max_drawdown),
        "benchmark_max_drawdown": float(bench_max_drawdown),
        "allocation_method": allocation_method,
        "num_assets": len(portfolios),
    }

    weights_df = pd.DataFrame(
        {"ticker": weights.index, "weight": weights.values}
    ).sort_values(by="weight", ascending=False)

    return unified_df, metrics, weights_df
