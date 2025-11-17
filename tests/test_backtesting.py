import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
import pytest
from src.backtesting import run_backtest, _calculate_trade_outcomes, calculate_performance_metrics

@pytest.fixture
def sample_backtest_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample price history and signals for backtesting."""
    price_history = pd.DataFrame({
        'Close': [100, 102, 105, 103, 106, 108, 107, 110, 112, 111]
    }, index=pd.to_datetime(pd.date_range('2025-01-01', periods=10)))

    signals = pd.Series([1, 1, -1, -1, 1, 1, 1, -1, -1, -1], index=price_history.index)
    return price_history, signals

def test_calculate_trade_outcomes(sample_backtest_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test the _calculate_trade_outcomes function."""
    price_history, signals = sample_backtest_data
    portfolio = pd.DataFrame({
        'signal': signals,
        'price': price_history['Close']
    })

    trade_outcomes = _calculate_trade_outcomes(portfolio)

    assert len(trade_outcomes) == 2
    assert trade_outcomes[0] == pytest.approx(5.0)
    assert trade_outcomes[1] == pytest.approx(4.0)

def test_calculate_performance_metrics(sample_backtest_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test the calculate_performance_metrics function."""
    price_history, signals = sample_backtest_data
    portfolio, metrics, _ = run_backtest(price_history, signals)

    assert metrics['total_trades'] == 2
    assert metrics['win_rate'] == 1.0
    assert metrics['strategy_total_return'] == pytest.approx(0.0422, abs=1e-4)
    assert metrics['buy_and_hold_total_return'] == pytest.approx(0.11, abs=1e-4)
    assert metrics['sharpe_ratio'] == pytest.approx(5.07, abs=1e-2)
    assert metrics['sortino_ratio'] == pytest.approx(19.30, abs=1e-2)
    assert metrics['strategy_max_drawdown'] == pytest.approx(-0.0122, abs=1e-4)

def test_run_backtest(sample_backtest_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test the run_backtest function."""
    price_history, signals = sample_backtest_data
    portfolio, metrics, heatmap_fig = run_backtest(price_history, signals)

    assert isinstance(portfolio, pd.DataFrame)
    assert len(portfolio) == 10
    assert 'total' in portfolio.columns
    assert 'benchmark' in portfolio.columns

    assert isinstance(metrics, dict)
    assert 'total_trades' in metrics
    assert 'win_rate' in metrics

    assert heatmap_fig is not None
