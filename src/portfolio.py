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
                tickers = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
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
    returns_data: Any,
) -> pd.Series:
    """
    Calculate Inverse-Volatility (Naive Risk Parity) weights for each asset.
    Assets with lower daily volatility receive higher capital weights.

    Args:
        returns_data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
            Either a DataFrame of daily returns or a dictionary of portfolio DataFrames.

    Returns:
        pd.Series: Normalized portfolio weights summing to 1.0.
    """
    if isinstance(returns_data, dict):
        if not returns_data:
            return pd.Series(dtype=float)
        extracted = {}
        for ticker, df in returns_data.items():
            if isinstance(df, pd.DataFrame):
                if "total" in df.columns:
                    extracted[ticker] = df["total"].pct_change().dropna()
                elif "Strategy_Cumulative" in df.columns:
                    extracted[ticker] = df["Strategy_Cumulative"].pct_change().dropna()
                elif "Close" in df.columns:
                    extracted[ticker] = df["Close"].pct_change().dropna()
        if extracted:
            returns_df = pd.DataFrame(extracted).dropna()
        else:
            n = len(returns_data)
            return pd.Series(1.0 / n, index=list(returns_data.keys()))
    elif isinstance(returns_data, pd.DataFrame):
        returns_df = returns_data
    else:
        return pd.Series(dtype=float)

    if returns_df.empty:
        return pd.Series(dtype=float)

    volatilities = returns_df.std()
    # Handle zero or NaN volatility
    volatilities = volatilities.replace(0, np.nan).fillna(volatilities.mean())
    inv_vol = 1.0 / (volatilities + 1e-10)
    weights = inv_vol / inv_vol.sum()
    return weights


def get_quasi_diag(link: np.ndarray) -> List[int]:
    """Sort clustered items by hierarchical tree order."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def get_cluster_var(cov: np.ndarray, c_items: List[int]) -> float:
    """Compute risk variance of a sub-cluster under inverse-variance weighting."""
    cov_slice = cov[np.ix_(c_items, c_items)]
    diag = np.diagonal(cov_slice)
    w = 1.0 / np.maximum(diag, 1e-8)
    w /= np.sum(w)
    return float(np.dot(np.dot(w, cov_slice), w))


def get_rec_bisection(cov: np.ndarray, sort_ix: List[int]) -> pd.Series:
    """Recursively bisect clusters and compute inverse-cluster-variance weights."""
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            var0 = get_cluster_var(cov, c_items0)
            var1 = get_cluster_var(cov, c_items1)
            alpha = 1.0 - var0 / (var0 + var1 + 1e-10)
            w[c_items0] *= alpha
            w[c_items1] *= 1.0 - alpha
    return w


def calculate_hrp_weights(returns_data: Any) -> pd.Series:
    """
    Marcos Lopez de Prado's Hierarchical Risk Parity (HRP) Portfolio Allocation.
    Uses machine learning agglomerative clustering on the asset correlation matrix
    to allocate risk across mutually uncorrelated clusters.

    Args:
        returns_data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
            Returns DataFrame or dictionary of portfolio DataFrames.

    Returns:
        pd.Series: Normalized HRP portfolio weights summing to 1.0.
    """
    if isinstance(returns_data, dict):
        if not returns_data:
            return pd.Series(dtype=float)
        extracted = {}
        for ticker, df in returns_data.items():
            if isinstance(df, pd.DataFrame):
                if "total" in df.columns:
                    extracted[ticker] = df["total"].pct_change().dropna()
                elif "Strategy_Cumulative" in df.columns:
                    extracted[ticker] = df["Strategy_Cumulative"].pct_change().dropna()
                elif "Close" in df.columns:
                    extracted[ticker] = df["Close"].pct_change().dropna()
        if extracted:
            returns_df = pd.DataFrame(extracted).dropna()
        else:
            n = len(returns_data)
            return pd.Series(1.0 / n, index=list(returns_data.keys()))
    elif isinstance(returns_data, pd.DataFrame):
        returns_df = returns_data
    else:
        return pd.Series(dtype=float)

    if returns_df.empty or len(returns_df.columns) == 0:
        return pd.Series(dtype=float)
    if len(returns_df.columns) == 1:
        return pd.Series([1.0], index=returns_df.columns)

    import scipy.cluster.hierarchy as sch

    cov = np.array(returns_df.cov().values, copy=True, dtype=float)
    corr = np.array(returns_df.corr().fillna(0.0).values, copy=True, dtype=float)
    np.fill_diagonal(corr, 1.0)

    # 1. Tree clustering: correlation distance metric
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    link = sch.linkage(sch.distance.squareform(dist), method="single")

    # 2. Quasi-diagonalization
    sort_ix = get_quasi_diag(link)

    # 3. Recursive bisection
    hrp_weights = get_rec_bisection(cov, sort_ix)
    hrp_weights = hrp_weights.sort_index()
    hrp_weights.index = returns_df.columns[hrp_weights.index]

    # Normalize to 1.0
    hrp_weights = hrp_weights / hrp_weights.sum()
    return hrp_weights


def build_unified_portfolio(
    initial_capital: float = 100000.0,
    results_dir: str = "results",
    tickers: List[str] = None,
    allocation_method: str = "hrp",
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Combines individual stock strategies into a single managed multi-asset fund.

    Args:
        initial_capital (float): Total starting capital for the fund.
        results_dir (str): Directory containing backtest CSV results.
        tickers (List[str], optional): List of tickers.
        allocation_method (str): 'hrp' (Hierarchical Risk Parity), 'risk_parity', or 'equal_weight'.

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

    # Extract daily strategy returns and benchmark returns per ticker in one vectorized step
    strat_dict = {}
    bench_dict = {}
    for ticker, df in portfolios.items():
        aligned_df = df.loc[common_dates]
        strat_dict[ticker] = aligned_df["total"].pct_change().fillna(0)
        bench_dict[ticker] = aligned_df["benchmark"].pct_change().fillna(0)

    strat_returns = pd.DataFrame(strat_dict, index=common_dates)
    bench_returns = pd.DataFrame(bench_dict, index=common_dates)

    # Determine allocation weights
    if allocation_method == "hrp":
        weights = calculate_hrp_weights(strat_returns)
    elif allocation_method == "risk_parity":
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
    unified_df["benchmark"] = initial_capital * (1.0 + unified_bench_return).cumprod()

    # Drawdown calculations
    running_max = unified_df["total"].cummax()
    unified_df["drawdown"] = (unified_df["total"] - running_max) / running_max

    bench_running_max = unified_df["benchmark"].cummax()
    unified_df["benchmark_drawdown"] = (
        unified_df["benchmark"] - bench_running_max
    ) / bench_running_max

    # Performance Metrics
    total_strat_return = (
        unified_df["total"].iloc[-1] - initial_capital
    ) / initial_capital
    total_bench_return = (
        unified_df["benchmark"].iloc[-1] - initial_capital
    ) / initial_capital
    max_drawdown = float(unified_df["drawdown"].min())
    bench_max_drawdown = float(unified_df["benchmark_drawdown"].min())

    # Annualized Sharpe Ratio (252 trading days)
    mean_ret = unified_strat_return.mean()
    std_ret = unified_strat_return.std()
    sharpe_ratio = float((mean_ret / (std_ret + 1e-10)) * np.sqrt(252))

    # Sortino Ratio (Downside deviation only)
    downside_returns = unified_strat_return[unified_strat_return < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else std_ret
    sortino_ratio = float((mean_ret / (downside_std + 1e-10)) * np.sqrt(252))

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
