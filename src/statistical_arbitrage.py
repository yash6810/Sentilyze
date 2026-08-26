"""
Statistical Arbitrage & Cointegration Pairs Trading Engine for Sentilyze.
Provides Engle-Granger cointegration testing, Ornstein-Uhlenbeck half-life estimation,
rolling Z-score spread tracking, dynamic mean-reversion signal generation,
and historical pairs backtesting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_hedge_ratio_and_spread(
    series_a: pd.Series, series_b: pd.Series
) -> Tuple[float, float, pd.Series]:
    """
    Computes optimal OLS Hedge Ratio (beta) and spread:
    Spread_t = Series_A_t - (alpha + beta * Series_B_t)

    Args:
        series_a: Price series of Asset A (Dependent variable Y)
        series_b: Price series of Asset B (Independent variable X)

    Returns:
        Tuple of (hedge_ratio, intercept, spread_series)
    """
    df = pd.DataFrame({"A": series_a, "B": series_b}).dropna()
    if len(df) < 10:
        return 1.0, 0.0, series_a - series_b

    slope, intercept, _, _, _ = stats.linregress(df["B"], df["A"])
    hedge_ratio = float(slope)
    alpha = float(intercept)
    spread = df["A"] - (alpha + hedge_ratio * df["B"])
    return hedge_ratio, alpha, spread


def evaluate_cointegration_adf(spread: pd.Series) -> Dict[str, Any]:
    """
    Performs Augmented Dickey-Fuller (ADF) unit root test on spread residuals
    to evaluate statistical stationarity and cointegration.

    Args:
        spread: Residual spread time series

    Returns:
        Dict with adf_statistic, p_value, is_cointegrated, and confidence level.
    """
    clean_spread = spread.dropna().values
    n = len(clean_spread)
    if n < 15:
        return {
            "adf_statistic": 0.0,
            "p_value": 1.0,
            "is_cointegrated": False,
            "confidence": "Weak / Insufficient Data",
        }

    # First difference: Delta y_t = gamma * y_{t-1} + error
    dy = np.diff(clean_spread)
    y_lag = clean_spread[:-1]

    # OLS regression: dy ~ y_lag
    slope, _, _, _, stderr = stats.linregress(y_lag, dy)
    t_stat = float(slope / (stderr + 1e-9))

    # Empirical MacKinnon-style critical values for ADF with constant (n > 100)
    # Critical Values: 1%: -3.45, 5%: -2.87, 10%: -2.57
    if t_stat < -3.45:
        p_val = 0.01
        conf = "99% High Confidence Cointegration"
        is_coint = True
    elif t_stat < -2.87:
        p_val = 0.05
        conf = "95% Statistically Significant"
        is_coint = True
    elif t_stat < -2.57:
        p_val = 0.10
        conf = "90% Moderate Cointegration"
        is_coint = True
    else:
        p_val = min(1.0, 0.10 + max(0.0, (t_stat + 2.57) * 0.2))
        conf = "No Cointegration (Non-Stationary)"
        is_coint = False

    return {
        "adf_statistic": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "is_cointegrated": is_coint,
        "confidence": conf,
    }


def calculate_half_life(spread: pd.Series) -> float:
    """
    Estimates the Ornstein-Uhlenbeck mean-reversion half-life (in days):
    dy_t = -theta * y_{t-1} * dt + eps
    Half-life = -ln(2) / theta

    Args:
        spread: Residual spread series

    Returns:
        Estimated mean-reversion half-life in days (capped between 1 and 252).
    """
    clean = spread.dropna().values
    if len(clean) < 15:
        return 30.0

    dy = np.diff(clean)
    y_lag = clean[:-1]
    slope, _, _, _, _ = stats.linregress(y_lag, dy)
    theta = float(slope)

    if theta >= 0:
        return 252.0  # Divergent or non-mean-reverting

    half_life = -np.log(2.0) / theta
    return float(np.clip(round(half_life, 1), 1.0, 252.0))


def calculate_rolling_zscore(
    spread: pd.Series, window: int = 30
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates rolling mean, rolling standard deviation, and standardized Z-score.

    Args:
        spread: Residual spread series
        window: Rolling lookback window in days (default: 30)

    Returns:
        Tuple of (zscore_series, rolling_mean, rolling_std)
    """
    rolling_mean = spread.rolling(window=window, min_periods=max(5, window // 2)).mean()
    rolling_std = spread.rolling(window=window, min_periods=max(5, window // 2)).std()
    rolling_std = rolling_std.replace(0, np.nan).ffill().bfill().fillna(1.0)

    zscore = (spread - rolling_mean) / rolling_std
    return zscore, rolling_mean, rolling_std


def generate_pairs_trading_signals(
    series_a: pd.Series,
    series_b: pd.Series,
    ticker_a: str,
    ticker_b: str,
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> Dict[str, Any]:
    """
    Generates actionable pairs trading signals based on current rolling Z-score.

    Args:
        series_a: Price series of Ticker A
        series_b: Price series of Ticker B
        ticker_a: Symbol A
        ticker_b: Symbol B
        window: Lookback window
        entry_z: Z-score threshold to trigger entry (e.g. 2.0)
        exit_z: Z-score threshold to exit/mean-revert (e.g. 0.5)

    Returns:
        Structured dictionary with pair signal, Z-score, hedge ratio, and metrics.
    """
    hedge_ratio, alpha, spread = calculate_hedge_ratio_and_spread(series_a, series_b)
    adf_res = evaluate_cointegration_adf(spread)
    half_life = calculate_half_life(spread)
    zscore, roll_mean, roll_std = calculate_rolling_zscore(spread, window=window)

    curr_z = float(zscore.iloc[-1]) if not zscore.empty and not pd.isna(zscore.iloc[-1]) else 0.0
    curr_price_a = float(series_a.iloc[-1]) if not series_a.empty else 0.0
    curr_price_b = float(series_b.iloc[-1]) if not series_b.empty else 0.0

    if curr_z >= entry_z:
        action = f"SHORT {ticker_a} / LONG {ticker_b}"
        status = "🔴 OVERBOUGHT SPREAD (Mean Reversion Downward Expected)"
        signal_code = -1
    elif curr_z <= -entry_z:
        action = f"LONG {ticker_a} / SHORT {ticker_b}"
        status = "🟢 OVERSOLD SPREAD (Mean Reversion Upward Expected)"
        signal_code = 1
    elif abs(curr_z) <= exit_z:
        action = "EQUILIBRIUM / CLOSE PAIR"
        status = "⚪ AT FAIR VALUE SPREAD"
        signal_code = 0
    else:
        action = "HOLD / MONITOR"
        status = "🟡 WITHIN NORMAL CONVERGENCE BAND"
        signal_code = 0

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "pair_name": f"{ticker_a} / {ticker_b}",
        "action": action,
        "status": status,
        "signal_code": signal_code,
        "current_zscore": round(curr_z, 2),
        "hedge_ratio": round(hedge_ratio, 4),
        "alpha": round(alpha, 4),
        "half_life_days": half_life,
        "adf_statistic": adf_res["adf_statistic"],
        "p_value": adf_res["p_value"],
        "is_cointegrated": adf_res["is_cointegrated"],
        "confidence": adf_res["confidence"],
        "current_price_a": curr_price_a,
        "current_price_b": curr_price_b,
        "spread_series": spread,
        "zscore_series": zscore,
        "rolling_mean": roll_mean,
        "rolling_std": roll_std,
    }


def scan_pairs_universe(
    prices_dict: Dict[str, pd.Series],
    candidate_pairs: Optional[List[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Scans candidate stock pairs across the universe and ranks them by
    statistical cointegration and current Z-score trading opportunities.

    Args:
        prices_dict: Dictionary of {ticker: close_price_series}
        candidate_pairs: Optional list of explicit (Ticker_A, Ticker_B) pairs

    Returns:
        List of pair analyses sorted by ADF significance.
    """
    if candidate_pairs is None:
        # Default institutional sector pairs
        tickers = list(prices_dict.keys())
        candidate_pairs = [
            ("NVDA", "AMD"),
            ("MSFT", "GOOGL"),
            ("TSM", "AVGO"),
            ("QQQ", "SPY"),
            ("AAPL", "MSFT"),
            ("META", "GOOGL"),
            ("AMZN", "COST"),
            ("JPM", "SPY"),
        ]
        # Filter to available tickers
        candidate_pairs = [
            (a, b) for a, b in candidate_pairs if a in prices_dict and b in prices_dict
        ]

    results = []
    for a, b in candidate_pairs:
        try:
            res = generate_pairs_trading_signals(
                prices_dict[a], prices_dict[b], a, b
            )
            results.append(res)
        except Exception as e:
            logger.warning(f"Failed analyzing pair {a}/{b}: {e}")

    # Sort by cointegration p-value (lowest/best first)
    results.sort(key=lambda x: x["p_value"])
    return results


def backtest_pairs_strategy(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float = 3.5,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """
    Runs a historical backtest of the Cointegration Pairs Mean-Reversion strategy.

    Args:
        series_a: Price series A
        series_b: Price series B
        window: Lookback window
        entry_z: Entry Z-score (+2.0 / -2.0)
        exit_z: Exit Z-score (+0.5 / -0.5)
        stop_loss_z: Hard exit Z-score for pair breakdown (+3.5 / -3.5)
        initial_capital: Starting capital in dollars

    Returns:
        Backtest summary with total return, Sharpe, Max Drawdown, and equity curve.
    """
    hedge_ratio, _, spread = calculate_hedge_ratio_and_spread(series_a, series_b)
    zscore, _, _ = calculate_rolling_zscore(spread, window=window)

    df = pd.DataFrame({"A": series_a, "B": series_b, "Z": zscore}).dropna()
    if len(df) < 50:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "equity_curve": pd.Series([initial_capital]),
        }

    position = 0  # 1: Long A/Short B, -1: Short A/Long B, 0: Flat
    capital = initial_capital
    equity_history = []
    trades = []
    entry_val = 0.0

    ret_a = df["A"].pct_change().fillna(0)
    ret_b = df["B"].pct_change().fillna(0)

    for i in range(len(df)):
        z = df["Z"].iloc[i]
        ra = ret_a.iloc[i]
        rb = ret_b.iloc[i]

        # Calculate daily pair return based on current position
        if position == 1:
            daily_ret = ra - hedge_ratio * rb
            capital *= (1.0 + daily_ret * 0.5)  # 50% capital allocation
        elif position == -1:
            daily_ret = -ra + hedge_ratio * rb
            capital *= (1.0 + daily_ret * 0.5)

        equity_history.append(capital)

        # Signal evaluation
        if position == 0:
            if z >= entry_z:
                position = -1  # Short A, Long B
                entry_val = capital
            elif z <= -entry_z:
                position = 1   # Long A, Short B
                entry_val = capital
        elif position == 1:
            if z >= -exit_z or z <= -stop_loss_z:
                # Close trade
                trade_pnl = (capital - entry_val) / entry_val
                trades.append(trade_pnl)
                position = 0
        elif position == -1:
            if z <= exit_z or z >= stop_loss_z:
                # Close trade
                trade_pnl = (capital - entry_val) / entry_val
                trades.append(trade_pnl)
                position = 0

    equity_series = pd.Series(equity_history, index=df.index)
    daily_rets = equity_series.pct_change().dropna()
    total_ret = (capital - initial_capital) / initial_capital

    sharpe = float(
        np.sqrt(252) * (daily_rets.mean() / (daily_rets.std() + 1e-9))
    ) if not daily_rets.empty else 0.0

    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = float(drawdown.min())

    win_rate = float(
        np.sum(np.array(trades) > 0) / len(trades)
    ) if len(trades) > 0 else 0.0

    return {
        "total_return": round(total_ret * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "total_trades": len(trades),
        "final_equity": round(capital, 2),
        "equity_curve": equity_series,
    }
