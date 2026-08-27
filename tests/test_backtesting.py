import matplotlib

matplotlib.use("Agg")
import pandas as pd
import pytest
from src.backtesting import (
    run_backtest,
    _calculate_trade_outcomes,
    run_significance_test,
)


@pytest.fixture
def sample_backtest_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample price history and prediction probabilities for backtesting."""
    price_history = pd.DataFrame(
        {
            "Open": [100, 101, 102, 104, 96, 106, 108, 108, 110, 81],
            "Close": [100, 102, 105, 95, 106, 108, 107, 110, 80, 111],
            "sma200": [
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
                90,
            ],  # Price > SMA200 (Uptrend)
            "rsi": [40, 45, 50, 48, 55, 60, 58, 65, 75, 70],  # RSI < 70 mostly
            "atr": [
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],  # Average True Range
        },
        index=pd.to_datetime(pd.date_range("2025-01-01", periods=10)).normalize(),
    )

    prediction_probs = pd.Series(
        [0.6, 0.6, 0.6, 0.4, 0.7, 0.8, 0.9, 0.3, 0.2, 0.1], index=price_history.index
    )
    return price_history, prediction_probs


def test_calculate_trade_outcomes(
    sample_backtest_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test the _calculate_trade_outcomes function."""
    price_history, probs = sample_backtest_data
    portfolio, _, _ = run_backtest(price_history, probs)

    trade_outcomes = _calculate_trade_outcomes(portfolio)

    assert len(trade_outcomes) == 2
    assert trade_outcomes[0] == pytest.approx(-5.0)
    assert isinstance(trade_outcomes[1], float)


def test_calculate_performance_metrics(
    sample_backtest_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test the calculate_performance_metrics function."""
    price_history, probs = sample_backtest_data
    portfolio, metrics, _ = run_backtest(price_history, probs)

    assert "total_trades" in metrics
    assert "win_rate" in metrics
    assert "strategy_total_return" in metrics
    assert "buy_and_hold_total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "strategy_max_drawdown" in metrics

    assert isinstance(metrics["total_trades"], int)


def test_run_backtest(sample_backtest_data: tuple[pd.DataFrame, pd.Series]) -> None:
    """Test the run_backtest function."""
    price_history, signals = sample_backtest_data
    portfolio, metrics, heatmap_fig = run_backtest(price_history, signals)

    assert isinstance(portfolio, pd.DataFrame)
    assert len(portfolio) == 10
    assert "total" in portfolio.columns
    assert "benchmark" in portfolio.columns

    assert isinstance(metrics, dict)
    assert "total_trades" in metrics
    assert "win_rate" in metrics

    assert heatmap_fig is not None


def test_run_significance_test(
    sample_backtest_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Test the run_significance_test function."""
    price_history, signals = sample_backtest_data
    portfolio, _, _ = run_backtest(price_history, signals)
    sig_res = run_significance_test(portfolio, price_history, n_simulations=50)

    assert "p_value" in sig_res
    assert "strategy_sharpe" in sig_res
    assert "confidence_interval_95" in sig_res
    assert isinstance(sig_res["is_statistically_significant"], bool)
    assert 0.0 <= sig_res["p_value"] <= 1.0
