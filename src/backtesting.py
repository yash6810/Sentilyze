
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

import matplotlib.pyplot as plt
import seaborn as sns

def create_monthly_returns_heatmap(portfolio):
    """
    Creates a heatmap of monthly returns.
    """
    daily_returns = portfolio['total'].pct_change().fillna(0)
    monthly_returns = daily_returns.resample('M').apply(lambda x: (x + 1).prod() - 1)
    monthly_returns.index = monthly_returns.index.to_period('M')

    # Create pivot table for heatmap
    returns_pivot = monthly_returns.to_frame(name='returns').pivot_table(
        values='returns', index=monthly_returns.index.year, columns=monthly_returns.index.month, aggfunc='sum'
    )
    returns_pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(returns_pivot, annot=True, fmt=".2%", cmap="vlag", center=0, ax=ax)
    ax.set_title('Monthly Returns Heatmap')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    plt.tight_layout()

    return fig

def run_backtest(price_history, signals, initial_capital=10000.0):
    """
    Runs a simple backtest on a given set of price history and trading signals.
    """
    logger.info(f"Starting backtest with initial capital: ${initial_capital:,.2f}")

    # --- Initialization ---
    portfolio = pd.DataFrame(index=price_history.index)
    portfolio['signal'] = signals
    portfolio['price'] = price_history['Close']

    # --- Strategy Simulation ---
    positions = portfolio['signal'].shift().fillna(0) # Shift signals to trade on next day's open
    portfolio['holdings'] = (positions * portfolio['price']).cumsum()
    portfolio['cash'] = initial_capital - (positions.diff().fillna(positions.iloc[0]) * portfolio['price']).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    # --- Benchmark Simulation ---
    benchmark_returns = portfolio['price'].pct_change().fillna(0)
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod()
    portfolio['benchmark'] = initial_capital * benchmark_cumulative_returns

    logger.info(f"Backtest complete. Final portfolio value: ${portfolio['total'].iloc[-1]:,.2f}")

    # --- Performance Metrics & Visuals ---
    metrics = calculate_performance_metrics(portfolio)
    heatmap_fig = create_monthly_returns_heatmap(portfolio)

    return portfolio, metrics, heatmap_fig

def calculate_performance_metrics(portfolio):
    """
    Calculates key performance metrics from the portfolio history.
    """
    metrics = {}
    daily_returns = portfolio['total'].pct_change().fillna(0)
    
    # Total Return
    total_return = (portfolio['total'].iloc[-1] / portfolio['total'].iloc[0]) - 1
    metrics['Strategy Total Return'] = f"{total_return:.2%}"

    # Benchmark (Buy and Hold) Return
    benchmark_return = (portfolio['benchmark'].iloc[-1] / portfolio['benchmark'].iloc[0]) - 1
    metrics['Buy & Hold Total Return'] = f"{benchmark_return:.2%}"

    # Max Drawdown
    rolling_max = portfolio['total'].cummax()
    daily_drawdown = portfolio['total'] / rolling_max - 1.0
    max_drawdown = daily_drawdown.min()
    metrics['Strategy Max Drawdown'] = f"{max_drawdown:.2%}"

    # Sharpe Ratio (annualized)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252**0.5) if daily_returns.std() != 0 else 0
    metrics['Sharpe Ratio'] = f"{sharpe_ratio:.2f}"

    # Sortino Ratio (annualized)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (daily_returns.mean() / downside_std) * (252**0.5) if downside_std != 0 else 0
    metrics['Sortino Ratio'] = f"{sortino_ratio:.2f}"

    # TODO: Add Win/Loss Rate, etc.

    logger.info(f"Performance Metrics: {metrics}")
    return metrics
