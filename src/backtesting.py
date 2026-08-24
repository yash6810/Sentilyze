import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils import get_logger
from typing import Tuple, Dict, List, Any

logger = get_logger(__name__)


def create_monthly_returns_heatmap(portfolio: pd.DataFrame) -> plt.Figure:
    """
    Creates a heatmap of monthly returns from a portfolio with improved aesthetics.

    Args:
        portfolio (pd.DataFrame): A DataFrame containing the portfolio history with a 'total' column.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object containing the heatmap.
    """
    plt.clf()  # Clear the current figure to prevent overlap
    plt.style.use("dark_background")  # Set style for dark theme
    daily_returns = portfolio["total"].pct_change().fillna(0)
    monthly_returns = daily_returns.resample("ME").apply(lambda x: (x + 1).prod() - 1)
    monthly_returns.index = monthly_returns.index.to_period("M")

    # Create pivot table for heatmap
    returns_pivot = monthly_returns.to_frame(name="returns").pivot_table(
        values="returns",
        index=monthly_returns.index.year,
        columns=monthly_returns.index.month,
        aggfunc="sum",
    )
    returns_pivot.columns = returns_pivot.columns.map(
        lambda x: pd.to_datetime(str(x), format="%m").strftime("%b")
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        returns_pivot,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={
            "format": "%.0f%%",
            "label": "Monthly Return",
        },
    )
    ax.set_title("Monthly Returns Heatmap (Strategy Performance)", fontsize=16)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def run_backtest(
    price_history: pd.DataFrame,
    prediction_probs: pd.Series,
    initial_capital: float = 10000.0,
    transaction_cost_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    prob_threshold: float = 0.52,
    take_profit_atr_mult: float = 2.5,
    max_leverage: float = 1.5,
    annual_margin_interest_rate: float = 0.05,
    maintenance_margin_pct: float = 0.25,
) -> Tuple[pd.DataFrame, Dict, plt.Figure]:
    """
    Runs an advanced regime-aware backtest with realistic margin constraints.

    Regime Filter: Buy only if P(Up) > prob_threshold AND RSI < 75
    Dynamic Risk: Trailing Stop-Loss = Entry - (3.0 * ATR) in Uptrend, or (1.5 * ATR) in Downtrend
    Take-Profit: Target = Entry + (2.5 * ATR) locks in swing profits at cycle peaks
    Margin Safeguard: Reg T 25% maintenance margin requirement. Triggers forced liquidation on breach.
    Daily Financing Cost: 5% annual interest rate charged daily on borrowed margin.

    Args:
        price_history (pd.DataFrame): Historical price data containing features like SMA200, RSI, ATR.
        prediction_probs (pd.Series): The model's predicted probability of a positive return.
        initial_capital (float): Starting cash ($10,000).
        transaction_cost_pct (float): Broker fee percentage (0.1%).
        slippage_pct (float): Estimated slippage percentage (0.05%).
        prob_threshold (float): Minimum confidence to trigger a buy (0.52).
        take_profit_atr_mult (float): Multiplier for ATR take-profit target (2.5x).
        max_leverage (float): Maximum leverage factor (1.5x).
        annual_margin_interest_rate (float): Annual margin loan interest rate (5%).
        maintenance_margin_pct (float): Minimum equity ratio before margin call liquidation (25%).

    Returns:
        tuple: Portfolio history DataFrame, performance metrics dict, and monthly returns heatmap figure.
    """
    logger.info(
        f"Starting regime-aware backtest. Capital: ${initial_capital:,.2f}, Threshold: {prob_threshold}"
    )

    # Align dates between price history and predictions
    common_dates = price_history.index.intersection(prediction_probs.index)
    if len(common_dates) == 0:
        logger.error("No overlapping dates between price history and predictions.")
        raise ValueError("No overlapping dates between price history and predictions.")

    price_history = price_history.loc[common_dates].copy()
    prediction_probs = prediction_probs.loc[common_dates].copy()

    # Pre-allocate numpy arrays/DataFrame columns
    portfolio = pd.DataFrame(index=price_history.index)
    portfolio["Open"] = price_history["Open"]
    portfolio["Close"] = price_history["Close"]
    portfolio["sma200"] = (
        price_history["sma200"]
        if "sma200" in price_history.columns
        else price_history["Close"].rolling(200).mean()
    )
    portfolio["rsi"] = price_history["rsi"] if "rsi" in price_history.columns else 50.0
    portfolio["atr"] = (
        price_history["atr"]
        if "atr" in price_history.columns
        else price_history["Close"] * 0.02
    )
    portfolio["prob_up"] = prediction_probs

    portfolio["cash"] = initial_capital
    portfolio["holdings"] = 0.0
    portfolio["borrowed_margin"] = 0.0
    portfolio["total"] = initial_capital
    portfolio["signal"] = 0
    portfolio["stop_loss"] = 0.0

    position_open = False
    entry_price = 0.0
    entry_atr = 0.0
    current_stop_loss = 0.0
    borrowed_margin = 0.0
    days_held = 0

    # Locate column indices to avoid expensive lookups inside the loop
    cash_idx = portfolio.columns.get_loc("cash")
    hold_idx = portfolio.columns.get_loc("holdings")
    bm_idx = portfolio.columns.get_loc("borrowed_margin")
    tot_idx = portfolio.columns.get_loc("total")
    open_idx = portfolio.columns.get_loc("Open")
    close_idx = portfolio.columns.get_loc("Close")
    sma_idx = portfolio.columns.get_loc("sma200")
    rsi_idx = portfolio.columns.get_loc("rsi")
    atr_idx = portfolio.columns.get_loc("atr")
    prob_idx = portfolio.columns.get_loc("prob_up")
    sig_idx = portfolio.columns.get_loc("signal")
    sl_idx = portfolio.columns.get_loc("stop_loss")

    daily_margin_rate = annual_margin_interest_rate / 252.0

    # --- Iterative Simulation ---
    for i in range(1, len(portfolio)):
        # Carry over previous day's balances and state
        portfolio.iloc[i, cash_idx] = portfolio.iloc[i - 1, cash_idx]
        portfolio.iloc[i, hold_idx] = portfolio.iloc[i - 1, hold_idx]
        portfolio.iloc[i, bm_idx] = borrowed_margin
        portfolio.iloc[i, sl_idx] = current_stop_loss

        # Get data from previous day for decision making
        prev_prob_up = portfolio.iloc[i - 1, prob_idx]
        prev_sma200 = portfolio.iloc[i - 1, sma_idx]
        prev_rsi = portfolio.iloc[i - 1, rsi_idx]
        prev_atr = portfolio.iloc[i - 1, atr_idx]
        prev_close = portfolio.iloc[i - 1, close_idx]

        today_open_price = portfolio.iloc[i, open_idx]
        today_close_price = portfolio.iloc[i, close_idx]

        # 1. Update holdings value based on today's price action
        if position_open and prev_close > 0:
            price_change_pct = (today_close_price / prev_close) - 1
            portfolio.iloc[i, hold_idx] *= 1 + price_change_pct

            # Daily margin financing fee deducted from cash equity
            if borrowed_margin > 0:
                daily_interest = borrowed_margin * daily_margin_rate
                portfolio.iloc[i, cash_idx] -= daily_interest

        execute_trade = 0  # 0=no action, 1=buy, -1=sell

        # 2. Risk Management & Profit Targets
        if position_open:
            days_held += 1
            current_holdings = portfolio.iloc[i, hold_idx]
            current_equity = (
                current_holdings + portfolio.iloc[i, cash_idx] - borrowed_margin
            )

            # A. Maintenance Margin Call Check (Reg T 25% threshold)
            if (
                current_holdings > 0
                and (current_equity / current_holdings) < maintenance_margin_pct
            ):
                execute_trade = -1
                logger.warning(
                    f"Margin Call: Forced liquidation on Day {portfolio.index[i].date()} at ${today_open_price:.2f}"
                )
            # B. Take-Profit Target Hit (+2.5 ATR or +6%)
            elif entry_price > 0 and today_open_price >= (
                entry_price + (take_profit_atr_mult * entry_atr)
            ):
                execute_trade = -1
                logger.debug(
                    f"Take-Profit triggered on Day {portfolio.index[i].date()} at ${today_open_price:.2f}"
                )
            # C. Trailing Stop-Loss Check at today's open
            elif today_open_price < current_stop_loss:
                execute_trade = -1
                logger.debug(
                    f"Stop-loss triggered on Day {portfolio.index[i].date()} at ${today_open_price:.2f}"
                )
            else:
                # Update Trailing Stop-Loss based on previous day's close
                atr_multiplier = 3.0 if prev_close > prev_sma200 else 1.5
                new_stop = prev_close - (atr_multiplier * prev_atr)
                if new_stop > current_stop_loss:
                    current_stop_loss = new_stop
                    portfolio.iloc[i, sl_idx] = current_stop_loss

        # 3. Regime Filter (Buy Logic)
        elif not position_open:
            available_cash = portfolio.iloc[i, cash_idx]
            if (
                prev_prob_up >= prob_threshold
                and prev_rsi < 75
                and available_cash > 100.0
            ):
                execute_trade = 1

        # 4. Execute Trades at today's open price
        portfolio.iloc[i, sig_idx] = execute_trade

        if execute_trade == 1:  # BUY
            current_cash = portfolio.iloc[i, cash_idx]
            leverage = max_leverage if prev_prob_up > 0.80 else 1.0

            # Position sizing: leverage is applied to available un-leveraged equity
            total_investment = current_cash * leverage
            borrowed_margin = max(0.0, total_investment - current_cash)
            portfolio.iloc[i, bm_idx] = borrowed_margin

            cost = total_investment * transaction_cost_pct
            effective_investment = total_investment / (1 + slippage_pct)

            portfolio.iloc[i, cash_idx] -= total_investment + cost
            portfolio.iloc[i, hold_idx] += effective_investment

            position_open = True
            entry_price = today_open_price
            entry_atr = prev_atr
            atr_multiplier = 3.0 if prev_close > prev_sma200 else 1.5
            current_stop_loss = prev_close - (atr_multiplier * prev_atr)
            portfolio.iloc[i, sl_idx] = current_stop_loss

        elif execute_trade == -1:  # SELL / LIQUIDATION / TAKE-PROFIT
            proceeds = portfolio.iloc[i, hold_idx]
            if proceeds > 0:
                cost = proceeds * transaction_cost_pct
                effective_proceeds = proceeds * (1 - slippage_pct)

                # Return borrowed margin loan back to the broker
                net_proceeds = effective_proceeds - cost - borrowed_margin
                portfolio.iloc[i, cash_idx] += net_proceeds + borrowed_margin
                portfolio.iloc[i, hold_idx] = 0.0

                position_open = False
                entry_price = 0.0
                entry_atr = 0.0
                current_stop_loss = 0.0
                borrowed_margin = 0.0
                days_held = 0
                portfolio.iloc[i, sl_idx] = 0.0
                portfolio.iloc[i, bm_idx] = 0.0

        # Total Net Liquidation Value
        portfolio.iloc[i, tot_idx] = (
            portfolio.iloc[i, cash_idx]
            + portfolio.iloc[i, hold_idx]
            - portfolio.iloc[i, bm_idx]
        )

    # --- Benchmark Simulation ---
    benchmark_returns = portfolio["Close"].pct_change().fillna(0)
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    portfolio["benchmark"] = initial_capital * benchmark_cumulative_returns

    logger.info(
        f"Backtest complete. Final portfolio value: ${portfolio['total'].iloc[-1]:,.2f}"
    )

    # --- Performance Metrics & Visuals ---
    metrics = calculate_performance_metrics(portfolio)
    heatmap_fig = create_monthly_returns_heatmap(portfolio)

    return portfolio, metrics, heatmap_fig


def _calculate_trade_outcomes(portfolio: pd.DataFrame) -> List[float]:
    """
    Identifies individual trades and calculates their profit/loss.

    Args:
        portfolio (pd.DataFrame): A DataFrame containing the portfolio history with 'signal' and 'Close' columns.

    Returns:
        list: A list of PnL for each trade.
    """
    pnl_list = []
    position_open = False
    entry_price = 0.0

    trade_events = portfolio[portfolio["signal"] != 0]

    for i, row in trade_events.iterrows():
        signal = row["signal"]

        if signal == 1 and not position_open:
            entry_price = row["Open"]
            position_open = True
        elif signal == -1 and position_open:
            exit_price = row["Open"]
            pnl_list.append(exit_price - entry_price)
            position_open = False

    return pnl_list


def calculate_performance_metrics(portfolio: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates key performance metrics from the portfolio history.

    Args:
        portfolio (pd.DataFrame): A DataFrame containing the portfolio history with 'total' and 'benchmark' columns.

    Returns:
        dict: A dictionary of performance metrics.
    """
    metrics = {}
    daily_returns = portfolio["total"].pct_change().fillna(0)

    # Total Return
    total_return = (portfolio["total"].iloc[-1] / portfolio["total"].iloc[0]) - 1
    metrics["strategy_total_return"] = float(total_return)

    # Benchmark (Buy and Hold) Return
    benchmark_return = (
        portfolio["benchmark"].iloc[-1] / portfolio["benchmark"].iloc[0]
    ) - 1
    metrics["buy_and_hold_total_return"] = float(benchmark_return)

    # Max Drawdown
    rolling_max = portfolio["total"].cummax()
    daily_drawdown = portfolio["total"] / (rolling_max + 1e-10) - 1.0
    max_drawdown = float(daily_drawdown.min())
    metrics["strategy_max_drawdown"] = max_drawdown

    # Benchmark Max Drawdown
    benchmark_rolling_max = portfolio["benchmark"].cummax()
    benchmark_daily_drawdown = (
        portfolio["benchmark"] / (benchmark_rolling_max + 1e-10)
    ) - 1.0
    benchmark_max_drawdown = float(benchmark_daily_drawdown.min())
    metrics["buy_and_hold_max_drawdown"] = benchmark_max_drawdown

    # Sharpe Ratio (annualized)
    daily_std = daily_returns.std()
    sharpe_ratio = (
        float((daily_returns.mean() / daily_std) * (252**0.5)) if daily_std > 0 else 0.0
    )
    metrics["sharpe_ratio"] = sharpe_ratio

    # Sortino Ratio (annualized)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (
        float((daily_returns.mean() / downside_std) * (252**0.5))
        if downside_std > 0
        else 0.0
    )
    metrics["sortino_ratio"] = sortino_ratio

    # Win Rate & Trades
    trade_outcomes = _calculate_trade_outcomes(portfolio)
    total_trades = len(trade_outcomes)
    win_rate = (
        float(sum(1 for pnl in trade_outcomes if pnl > 0) / total_trades)
        if total_trades > 0
        else 0.0
    )
    metrics["total_trades"] = total_trades
    metrics["win_rate"] = win_rate

    logger.info(f"Performance Metrics: {metrics}")
    return metrics


def run_significance_test(
    portfolio: pd.DataFrame,
    price_history: pd.DataFrame,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs a Monte Carlo permutation significance test comparing the strategy Sharpe ratio
    against N randomized entry strategies with the same trade frequency.

    Args:
        portfolio (pd.DataFrame): Backtested portfolio with strategy returns.
        price_history (pd.DataFrame): OHLCV price DataFrame.
        n_simulations (int): Number of permutation runs (default 1000).
        random_seed (int): Seed for deterministic replication.

    Returns:
        dict: Significance metrics including p-value and 95% confidence bounds.
    """
    logger.info(
        f"Running Monte Carlo significance test with {n_simulations} simulations..."
    )
    rng = np.random.default_rng(random_seed)

    strat_daily_returns = portfolio["total"].pct_change().fillna(0)
    strat_std = strat_daily_returns.std()
    strategy_sharpe = (
        float((strat_daily_returns.mean() / strat_std) * (252**0.5))
        if strat_std > 0
        else 0.0
    )

    trade_signals = portfolio["signal"].values
    n_days = len(portfolio)
    n_trades = int(np.sum(trade_signals == 1))

    if n_trades == 0 or n_days < 20:
        return {
            "strategy_sharpe": strategy_sharpe,
            "random_sharpe_mean": 0.0,
            "p_value": 1.0,
            "confidence_interval_95": [0.0, 0.0],
            "is_statistically_significant": False,
            "n_simulations": n_simulations,
        }

    close_prices = price_history["Close"].values
    daily_price_returns = np.diff(close_prices) / (close_prices[:-1] + 1e-10)

    # Generate N random Sharpe ratios
    random_sharpes = []
    avg_hold_days = max(1, int(n_days / (n_trades * 2)))

    for _ in range(n_simulations):
        # Pick random entry points
        random_entries = rng.choice(
            len(daily_price_returns) - avg_hold_days, size=n_trades, replace=False
        )
        sim_returns = np.zeros_like(daily_price_returns)

        for entry in random_entries:
            sim_returns[entry : entry + avg_hold_days] = daily_price_returns[
                entry : entry + avg_hold_days
            ]

        sim_std = np.std(sim_returns)
        if sim_std > 0:
            sim_sharpe = (np.mean(sim_returns) / sim_std) * np.sqrt(252)
        else:
            sim_sharpe = 0.0
        random_sharpes.append(sim_sharpe)

    random_sharpes_arr = np.array(random_sharpes)
    p_value = float(np.mean(random_sharpes_arr >= strategy_sharpe))
    ci_lower = float(np.percentile(random_sharpes_arr, 2.5))
    ci_upper = float(np.percentile(random_sharpes_arr, 97.5))

    results = {
        "strategy_sharpe": float(round(strategy_sharpe, 4)),
        "random_sharpe_mean": float(round(float(np.mean(random_sharpes_arr)), 4)),
        "p_value": float(round(p_value, 4)),
        "confidence_interval_95": [round(ci_lower, 4), round(ci_upper, 4)],
        "is_statistically_significant": bool(p_value < 0.05),
        "n_simulations": n_simulations,
    }

    logger.info(f"Significance test results: {results}")
    return results
