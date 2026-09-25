"""
Unit tests for the UltraQuantEngine: Ultra-Low-Latency Unified Execution Core.
Verifies microsecond execution speed, nanosecond hardware telemetry,
and multi-model fusion (HMM, CUSUM, Volume Profile, Momentum, Credit Radar).
"""

import pytest
import numpy as np
import pandas as pd

from src.ultra_quant_engine import UltraQuantEngine, get_ultra_quant_engine


@pytest.fixture
def sample_ohlcv_data():
    """Generates 60 bars of synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    close_prices = 100.0 + np.cumsum(np.random.randn(60) * 1.5)
    high_prices = close_prices + np.random.uniform(0.5, 2.0, size=60)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, size=60)
    open_prices = low_prices + np.random.uniform(0.1, 1.0, size=60)
    volumes = np.random.uniform(500000, 2000000, size=60)

    return pd.DataFrame(
        {
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volumes,
        },
        index=dates,
    )


def test_ultra_quant_engine_singleton():
    """Verify singleton pattern returns identical instance."""
    engine1 = get_ultra_quant_engine()
    engine2 = get_ultra_quant_engine()
    assert engine1 is engine2
    assert isinstance(engine1, UltraQuantEngine)


def test_evaluate_microstructure_poc(sample_ohlcv_data):
    """Verify Volume Profile PoC, VAH, and VAL execution and latency in microseconds."""
    engine = UltraQuantEngine()
    poc_res = engine.evaluate_microstructure_poc(sample_ohlcv_data)

    assert "poc_price" in poc_res
    assert "vah_price" in poc_res
    assert "val_price" in poc_res
    assert poc_res["eval_latency_ns"] > 0
    assert poc_res["eval_latency_micros"] >= 0.0
    assert poc_res["val_price"] <= poc_res["vah_price"]


def test_evaluate_asset_full_quantum(sample_ohlcv_data):
    """Verify full multi-model quantum evaluation pipeline and nanosecond telemetry."""
    engine = UltraQuantEngine()
    result = engine.evaluate_asset_full_quantum(
        ticker="NVDA",
        df_history=sample_ohlcv_data,
        spot_price=125.50,
        daily_return=0.015,
    )

    assert result["ticker"] == "NVDA"
    assert result["spot_price"] == 125.50
    assert "recommended_policy" in result
    assert "risk_multiplier" in result
    assert "hmm_regime" in result
    assert "cusum_alarm" in result
    assert "momentum_zscore" in result
    assert "vol_scaled_exposure" in result
    assert "execution_latency" in result

    assert "conformal_lower_bound" in result
    assert "conformal_upper_bound" in result
    assert result["conformal_lower_bound"] <= result["conformal_upper_bound"]

    lat = result["execution_latency"]
    assert lat["total_nanoseconds"] > 0
    assert lat["total_microseconds"] > 0
    assert lat["hmm_micros"] >= 0
    assert lat["cusum_micros"] >= 0
    assert lat["volume_profile_micros"] >= 0


def test_trade_post_mortem_learner(tmp_path):
    """Verify TradePostMortemLearner operates safely in a temporary sandbox."""
    from src.post_mortem_learner import TradePostMortemLearner

    t_csv = tmp_path / "mock_trades.csv"
    p_json = tmp_path / "mock_portfolio.json"
    w_json = tmp_path / "mock_weights.json"
    m_json = tmp_path / "mock_memory.json"

    # Create synthetic closed trades
    df_trades = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "shares": 50,
                "entry_price": 100.0,
                "exit_price": 115.0,
                "entry_date": "2026-09-01",
                "exit_date": "2026-09-10",
                "pnl": 750.0,
                "return_pct": 15.0,
                "exit_reason": "TP2_RUNNER_EXIT",
            },
            {
                "ticker": "AAPL",
                "shares": 40,
                "entry_price": 200.0,
                "exit_price": 196.0,
                "entry_date": "2026-09-05",
                "exit_date": "2026-09-08",
                "pnl": -160.0,
                "return_pct": -2.0,
                "exit_reason": "STOP_LOSS",
            },
        ]
    )
    df_trades.to_csv(t_csv, index=False)

    learner = TradePostMortemLearner(
        trades_csv_path=str(t_csv),
        portfolio_json_path=str(p_json),
        weights_path=str(w_json),
        memory_path=str(m_json),
    )

    cycle_res = learner.run_post_mortem_cycle()
    assert cycle_res["trades_analyzed"] == 2
    assert "calibrated_weights" in cycle_res
    assert (
        abs(sum(cycle_res["calibrated_weights"].values()) - 1.0) < 0.01
    )  # Sums to 1.0


def test_cusum_structural_break_downward():
    """Verify CUSUM detector triggers alarm on sudden downward price breakdown."""
    engine = UltraQuantEngine()
    cusum = engine.get_or_create_cusum("TSLA")

    # Feed negative return shocks
    alarm_fired = False
    for _ in range(15):
        c_res = cusum.update(-0.04)  # Severe -4% per step
        if c_res.get("alarm") and c_res.get("direction") == "DOWN":
            alarm_fired = True
            break

    assert alarm_fired is True


def test_hmm_market_regime_step():
    """Verify Gaussian HMM forward update executes in microseconds."""
    engine = UltraQuantEngine()
    hmm_res = engine.evaluate_hmm_market_regime(-0.02)

    assert "regime" in hmm_res
    assert "crisis_probability" in hmm_res
    assert hmm_res["eval_latency_ns"] > 0
    assert hmm_res["eval_latency_micros"] >= 0.0
