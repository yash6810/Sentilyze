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
    price_history: pd.DataFrame, 
    prediction_probs: pd.Series, 
    initial_capital: float = 10000.0, 
    transaction_cost_pct: float = 0.001, 
    slippage_pct: float = 0.0005,
    prob_threshold: float = 0.50,
    max_leverage: float = 1.5
) -> Tuple[pd.DataFrame, Dict, plt.Figure]:
    """
    Runs an advanced iterative backtest implementing aggressive Leveraged Alpha regime mechanics.
    
    Regime Filter: Buy only if P(Up) > prob_threshold (default 0.50) AND RSI < 70
    Dynamic Risk: Trailing Stop-Loss = Entry - (3.0 * ATR) in Uptrend, or (1.5 * ATR) in Downtrend
    Take-Profit: None (Let runners run until Stop-Loss)

    Args:
        price_history (pd.DataFrame): Historical price data containing features like SMA200, RSI, ATR.
        prediction_probs (pd.Series): The model's predicted probability of a positive return.
        initial_capital (float): Starting cash.
        transaction_cost_pct (float): Broker fee percentage.
        slippage_pct (float): Estimated slippage percentage.
        prob_threshold (float): Minimum confidence to trigger a buy.

    Returns:
        tuple: Portfolio history, performance metrics dict, and monthly returns heatmap.
    """
    logger.info(
        f"Starting regime-aware backtest. Capital: ${initial_capital:,.2f}, Threshold: {prob_threshold}"
    )

    # --- Initialization ---
    portfolio = pd.DataFrame(index=price_history.index)
    portfolio["prob_up"] = prediction_probs
    
    # Copy necessary pricing and indicator columns securely
    for col in ["Open", "Close", "sma200", "rsi", "atr"]:
        if col in price_history.columns:
            portfolio[col] = price_history[col]
        else:
            logger.warning(f"Missing essential feature '{col}' in price_history for backtesting. Filling with 0.")
            portfolio[col] = 0

    portfolio["cash"] = 0.0
    portfolio["holdings"] = 0.0
    portfolio["total"] = 0.0
    portfolio["signal"] = 0  # 1 for Buy, -1 for Sell, 0 for Hold
    portfolio["stop_loss"] = 0.0 

    # Set initial capital
    portfolio.iloc[0, portfolio.columns.get_loc("cash")] = initial_capital
    portfolio.iloc[0, portfolio.columns.get_loc("total")] = initial_capital

    # State tracking variables
    position_open = False
    current_stop_loss = 0.0
    borrowed_margin = 0.0
    days_held = 0

    # Locate column indices to avoid expensive lookups inside the loop
    cash_idx = portfolio.columns.get_loc("cash")
    hold_idx = portfolio.columns.get_loc("holdings")
    tot_idx = portfolio.columns.get_loc("total")
    open_idx = portfolio.columns.get_loc("Open")
    close_idx = portfolio.columns.get_loc("Close")
    sma_idx = portfolio.columns.get_loc("sma200")
    rsi_idx = portfolio.columns.get_loc("rsi")
    atr_idx = portfolio.columns.get_loc("atr")
    prob_idx = portfolio.columns.get_loc("prob_up")
    sig_idx = portfolio.columns.get_loc("signal")
    sl_idx = portfolio.columns.get_loc("stop_loss")

    # --- Iterative Simulation ---
    for i in range(1, len(portfolio)):
        # Carry over previous day's balances and state
        portfolio.iloc[i, cash_idx] = portfolio.iloc[i - 1, cash_idx]
        portfolio.iloc[i, hold_idx] = portfolio.iloc[i - 1, hold_idx]
        portfolio.iloc[i, sl_idx] = current_stop_loss

        # Get data from previous day for decision making
        prev_prob_up = portfolio.iloc[i - 1, prob_idx]
        prev_sma200 = portfolio.iloc[i - 1, sma_idx]
        prev_rsi = portfolio.iloc[i - 1, rsi_idx]
        prev_atr = portfolio.iloc[i - 1, atr_idx]
        prev_close = portfolio.iloc[i-1, close_idx]

        today_open_price = portfolio.iloc[i, open_idx]
        today_close_price = portfolio.iloc[i, close_idx]

        # 1. Update holdings value based on today's price action
        if position_open and prev_close > 0:
            price_change_pct = (today_close_price / prev_close) - 1
            portfolio.iloc[i, hold_idx] *= (1 + price_change_pct)

        execute_trade = 0 # 0=no action, 1=buy, -1=sell

        # 2. Dynamic Risk Management (Sell Logic)
        if position_open:
            days_held += 1
            # Check Stop-Loss at today's open
            if today_open_price < current_stop_loss:
                execute_trade = -1
                logger.debug(f"Stop-loss triggered on Day {portfolio.index[i].date()} at ${today_open_price:.2f}")
            else:
                # Trailing Stop-Loss update based on previous day's close
                atr_multiplier = 3.0 if prev_close > prev_sma200 else 1.5
                new_stop = prev_close - (atr_multiplier * prev_atr)
                if new_stop > current_stop_loss:
                    current_stop_loss = new_stop
                    portfolio.iloc[i, sl_idx] = current_stop_loss

        # 3. Regime Filter (Buy Logic)
        elif not position_open:
            if prev_prob_up > prob_threshold and prev_rsi < 70:
                execute_trade = 1

        # 4. Execute Trades at today's open price
        portfolio.iloc[i, sig_idx] = execute_trade
        trade_price = today_open_price

        if execute_trade == 1:  # BUY
            leverage = max_leverage if prev_prob_up > 0.80 else 1.0
            investment = portfolio.iloc[i, cash_idx] * leverage
            if investment > 0:
                borrowed_margin = investment - portfolio.iloc[i, cash_idx]
                
                cost = investment * transaction_cost_pct
                effective_investment = investment / (1 + slippage_pct)
                portfolio.iloc[i, cash_idx] -= (investment + cost)
                portfolio.iloc[i, hold_idx] += effective_investment
                
                position_open = True
                atr_multiplier = 3.0 if prev_close > prev_sma200 else 1.5
                current_stop_loss = prev_close - (atr_multiplier * prev_atr)
                portfolio.iloc[i, sl_idx] = current_stop_loss

        elif execute_trade == -1:  # SELL
            proceeds = portfolio.iloc[i, hold_idx]
            if proceeds > 0:
                if borrowed_margin > 0:
                    margin_interest_cost = borrowed_margin * 0.05 * (days_held / 365)
                else:
                    margin_interest_cost = 0.0

                cost = proceeds * transaction_cost_pct
                effective_proceeds = proceeds * (1 - slippage_pct)
                
                portfolio.iloc[i, cash_idx] += (effective_proceeds - cost - margin_interest_cost + borrowed_margin)
                portfolio.iloc[i, hold_idx] = 0
                
                position_open = False
                current_stop_loss = 0.0
                borrowed_margin = 0.0
                days_held = 0
                portfolio.iloc[i, sl_idx] = 0.0

        # Update total portfolio value for the day
        portfolio.iloc[i, tot_idx] = portfolio.iloc[i, cash_idx] + portfolio.iloc[i, hold_idx]

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
    entry_price = 0

    # Filter only days where a trade actually occurred
    trade_events = portfolio[portfolio["signal"] != 0]

    for i, row in trade_events.iterrows():
        signal = row["signal"]
        
        if signal == 1 and not position_open:
            entry_price = row["Open"] # Use Open for entry price
            position_open = True
        elif signal == -1 and position_open:
            exit_price = row["Open"] # Use Open for exit price
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
