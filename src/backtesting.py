import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils import get_logger
from typing import Tuple, Dict, List

logger = get_logger(__name__)


def create_monthly_returns_heatmap(portfolio: pd.DataFrame) -> plt.Figure:
    """
    Creates a heatmap of monthly returns from a portfolio with improved aesthetics.

    Args:
        portfolio (pd.DataFrame): A DataFrame containing the portfolio history with a 'total' column.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object containing the heatmap.
    """
    plt.clf() # Clear the current figure to prevent overlap
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
    fig, ax = plt.subplots(figsize=(14, 10)) # Increased figure size
    sns.heatmap(
        returns_pivot,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn", # Changed colormap for better contrast
        center=0,
        ax=ax,
        linewidths=.5, # Add lines between cells
        linecolor='gray', # Color of the lines
        cbar_kws={'format': '%.0f%%', 'label': 'Monthly Return'} # Colorbar formatting and label
    )
    ax.set_title("Monthly Returns Heatmap (Strategy Performance)", fontsize=16) # More descriptive title and larger font
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)
    plt.yticks(rotation=0) # Ensure years are horizontal
    plt.xticks(rotation=45, ha='right') # Rotate month labels for better readability
    plt.tight_layout()

    return fig


def run_backtest(
    price_history: pd.DataFrame, signals: pd.Series, initial_capital: float = 10000.0, transaction_cost_pct: float = 0.001, slippage_pct: float = 0.0005
) -> Tuple[pd.DataFrame, Dict, plt.Figure]:
    """
    Runs a more realistic, iterative backtest on a given set of price history and trading signals.

    Args:
        price_history (pd.DataFrame): A DataFrame containing the historical price data with a 'Close' column.
        signals (pd.Series): A Series containing the trading signals (1 for buy, -1 for sell).
        initial_capital (float): The initial capital for the backtest.
        transaction_cost_pct (float): The transaction cost as a percentage of the trade value.
        slippage_pct (float): The slippage as a percentage of the trade value.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The portfolio history.
            - dict: A dictionary of performance metrics.
            - matplotlib.figure.Figure: A matplotlib Figure object containing the monthly returns heatmap.
    """
    logger.info(
        f"Starting iterative backtest with initial capital: ${initial_capital:,.2f}, transaction cost: {transaction_cost_pct:.2%}, and slippage: {slippage_pct:.2%}"
    )

    # --- Initialization ---
    portfolio = pd.DataFrame(index=price_history.index)
    portfolio["signal"] = signals
    portfolio["price"] = price_history["Close"]
    portfolio["cash"] = 0.0
    portfolio["holdings"] = 0.0
    portfolio["total"] = 0.0

    # Set initial capital
    portfolio.iloc[0, portfolio.columns.get_loc("cash")] = initial_capital
    portfolio.iloc[0, portfolio.columns.get_loc("total")] = initial_capital

    # --- Iterative Simulation ---
    for i in range(1, len(portfolio)):
        # Carry over previous day's values
        portfolio.iloc[i, portfolio.columns.get_loc("cash")] = portfolio.iloc[
            i - 1, portfolio.columns.get_loc("cash")
        ]
        portfolio.iloc[i, portfolio.columns.get_loc("holdings")] = portfolio.iloc[
            i - 1, portfolio.columns.get_loc("holdings")
        ]

        # Get previous day's signal to decide today's trade
        signal = portfolio.iloc[i - 1, portfolio.columns.get_loc("signal")]
        prev_signal = (
            portfolio.iloc[i - 2, portfolio.columns.get_loc("signal")] if i > 1 else 0
        )

        # Update holdings value based on price change
        if portfolio.iloc[i - 1, portfolio.columns.get_loc("holdings")] > 0:
            price_change_pct = (
                portfolio.iloc[i, portfolio.columns.get_loc("price")]
                / portfolio.iloc[i - 1, portfolio.columns.get_loc("price")]
            ) - 1
            portfolio.iloc[i, portfolio.columns.get_loc("holdings")] *= (
                1 + price_change_pct
            )

        # Execute trades if signal changes
        if signal != prev_signal:
            if signal == 1:  # Buy signal
                if portfolio.iloc[i, portfolio.columns.get_loc("cash")] > 0:
                    investment = portfolio.iloc[i, portfolio.columns.get_loc("cash")]
                    cost = investment * transaction_cost_pct
                    effective_investment = investment / (1 + slippage_pct)
                    portfolio.iloc[i, portfolio.columns.get_loc("cash")] -= (investment + cost)
                    portfolio.iloc[i, portfolio.columns.get_loc("holdings")] += effective_investment
            elif signal == -1:  # Sell signal
                if portfolio.iloc[i - 1, portfolio.columns.get_loc("holdings")] > 0:
                    proceeds = portfolio.iloc[
                        i - 1, portfolio.columns.get_loc("holdings")
                    ]
                    cost = proceeds * transaction_cost_pct
                    effective_proceeds = proceeds * (1 - slippage_pct)
                    portfolio.iloc[i, portfolio.columns.get_loc("cash")] += (effective_proceeds - cost)
                    portfolio.iloc[i, portfolio.columns.get_loc("holdings")] = 0

        # Update total portfolio value for the day
        portfolio.iloc[i, portfolio.columns.get_loc("total")] = (
            portfolio.iloc[i, portfolio.columns.get_loc("cash")]
            + portfolio.iloc[i, portfolio.columns.get_loc("holdings")]
        )

    # --- Benchmark Simulation ---
    benchmark_returns = portfolio["price"].pct_change().fillna(0)
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
        portfolio (pd.DataFrame): A DataFrame containing the portfolio history with 'signal' and 'price' columns.

    Returns:
        list: A list of PnL for each trade.
    """
    trades = portfolio["signal"].diff().fillna(0)
    trade_entry_exit = trades[trades != 0]

    pnl_list = []
    position_open = False
    entry_price = 0

    if portfolio["signal"].iloc[0] == 1:
        entry_price = portfolio["price"].iloc[0]
        position_open = True

    for i, trade in trade_entry_exit.items():
        if not position_open and trade == 2:  # From -1 (sell) to 1 (buy)
            entry_price = portfolio.loc[i, "price"]
            position_open = True
        elif position_open and trade == -2:  # From 1 (buy) to -1 (sell)
            exit_price = portfolio.loc[i, "price"]
            pnl_list.append(exit_price - entry_price)
            position_open = False

    return pnl_list


def calculate_performance_metrics(portfolio: pd.DataFrame) -> Dict:
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
    metrics["strategy_total_return"] = total_return

    # Benchmark (Buy and Hold) Return
    benchmark_return = (
        portfolio["benchmark"].iloc[-1] / portfolio["benchmark"].iloc[0]
    ) - 1
    metrics["buy_and_hold_total_return"] = benchmark_return

    # Max Drawdown
    rolling_max = portfolio["total"].cummax()
    daily_drawdown = portfolio["total"] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    metrics["strategy_max_drawdown"] = max_drawdown

    # Benchmark Max Drawdown
    benchmark_rolling_max = portfolio["benchmark"].cummax()
    benchmark_daily_drawdown = (
        portfolio["benchmark"] / benchmark_rolling_max
    ) - 1.0
    benchmark_max_drawdown = benchmark_daily_drawdown.min()
    metrics["buy_and_hold_max_drawdown"] = benchmark_max_drawdown

    # Sharpe Ratio (annualized)
    sharpe_ratio = (
        (daily_returns.mean() / daily_returns.std()) * (252**0.5)
        if daily_returns.std() != 0
        else 0
    )
    metrics["sharpe_ratio"] = sharpe_ratio

    # Sortino Ratio (annualized)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (
        (daily_returns.mean() / downside_std) * (252**0.5) if downside_std != 0 else 0
    )
    metrics["sortino_ratio"] = sortino_ratio

    # Win Rate & Trades
    trade_outcomes = _calculate_trade_outcomes(portfolio)
    total_trades = len(trade_outcomes)
    win_rate = (
        (sum(1 for pnl in trade_outcomes if pnl > 0) / total_trades)
        if total_trades > 0
        else 0
    )
    metrics["total_trades"] = total_trades
    metrics["win_rate"] = win_rate

    logger.info(f"Performance Metrics: {metrics}")
    return metrics
