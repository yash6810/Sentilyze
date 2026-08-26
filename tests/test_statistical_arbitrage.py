import numpy as np
import pandas as pd
from src.statistical_arbitrage import (
    calculate_hedge_ratio_and_spread,
    evaluate_cointegration_adf,
    calculate_half_life,
    calculate_rolling_zscore,
    generate_pairs_trading_signals,
    scan_pairs_universe,
    backtest_pairs_strategy,
)


def _generate_synthetic_cointegrated_series(n: int = 200, seed: int = 42):
    np.random.seed(seed)
    x = np.cumsum(np.random.normal(0, 1, n)) + 100
    noise = np.random.normal(0, 0.5, n)
    y = 1.5 * x + 5.0 + noise
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(y, index=dates), pd.Series(x, index=dates)


def test_calculate_hedge_ratio_and_spread():
    series_a, series_b = _generate_synthetic_cointegrated_series(100)
    hedge_ratio, alpha, spread = calculate_hedge_ratio_and_spread(series_a, series_b)

    assert abs(hedge_ratio - 1.5) < 0.2
    assert abs(alpha - 5.0) < 2.0
    assert len(spread) == 100
    assert isinstance(spread, pd.Series)


def test_cointegration_adf_eval():
    series_a, series_b = _generate_synthetic_cointegrated_series(150)
    _, _, spread = calculate_hedge_ratio_and_spread(series_a, series_b)
    res = evaluate_cointegration_adf(spread)

    assert "adf_statistic" in res
    assert "p_value" in res
    assert res["is_cointegrated"] is True
    assert res["p_value"] <= 0.10


def test_calculate_half_life():
    series_a, series_b = _generate_synthetic_cointegrated_series(150)
    _, _, spread = calculate_hedge_ratio_and_spread(series_a, series_b)
    half_life = calculate_half_life(spread)

    assert isinstance(half_life, float)
    assert 1.0 <= half_life <= 252.0


def test_calculate_rolling_zscore():
    series_a, series_b = _generate_synthetic_cointegrated_series(100)
    _, _, spread = calculate_hedge_ratio_and_spread(series_a, series_b)
    zscore, roll_mean, roll_std = calculate_rolling_zscore(spread, window=20)

    assert len(zscore) == 100
    assert not zscore.dropna().empty
    assert (roll_std > 0).all()


def test_generate_pairs_trading_signals():
    series_a, series_b = _generate_synthetic_cointegrated_series(120)
    res = generate_pairs_trading_signals(
        series_a, series_b, "NVDA", "AMD", window=20, entry_z=1.5
    )

    assert res["ticker_a"] == "NVDA"
    assert res["ticker_b"] == "AMD"
    assert "action" in res
    assert "status" in res
    assert "current_zscore" in res
    assert "hedge_ratio" in res
    assert "p_value" in res


def test_scan_pairs_universe():
    series_a, series_b = _generate_synthetic_cointegrated_series(100)
    prices_dict = {
        "NVDA": series_a,
        "AMD": series_b,
        "MSFT": series_b * 1.1,
    }
    results = scan_pairs_universe(prices_dict)

    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["p_value"] <= results[-1]["p_value"]


def test_backtest_pairs_strategy():
    series_a, series_b = _generate_synthetic_cointegrated_series(250)
    res = backtest_pairs_strategy(
        series_a, series_b, window=20, entry_z=1.5, exit_z=0.5
    )

    assert "total_return" in res
    assert "sharpe_ratio" in res
    assert "max_drawdown" in res
    assert "total_trades" in res
    assert "equity_curve" in res
    assert len(res["equity_curve"]) > 0
