"""
Tests for Cloud Lakehouse (Parquet/DuckDB), 24/7 Crypto Isolated Sandbox Broker,
and Fair Value Gap (FVG) Liquidity Void Execution.
"""

import os
import json
import pytest
import pandas as pd
import numpy as np

from src.cloud_lakehouse import (
    CloudDataLake,
    export_trades_to_parquet,
    export_equity_to_parquet,
    query_lakehouse,
)
from src.paper_broker import PaperBroker
from src.price_scout import calculate_fvg_corridor, PriceActionScoutAgent


def test_parquet_export_and_duckdb_query(tmp_path):
    trades = [
        {
            "ticker": "NVDA",
            "shares": 10,
            "entry_price": 120.0,
            "exit_price": 130.0,
            "entry_date": "2026-10-01",
            "exit_date": "2026-10-05",
            "pnl": 100.0,
            "return_pct": 8.33,
            "reason": "TP1_PROFIT_LOCK",
        },
        {
            "ticker": "AAPL",
            "shares": 5,
            "entry_price": 220.0,
            "exit_price": 235.0,
            "entry_date": "2026-10-02",
            "exit_date": "2026-10-06",
            "pnl": 75.0,
            "return_pct": 6.82,
            "reason": "TP2_RUNNER_EXIT",
        },
    ]

    out_file = str(tmp_path / "test_trades.parquet")
    saved_path = export_trades_to_parquet(trades, output_path=out_file)
    assert os.path.exists(saved_path)

    df = pd.read_parquet(saved_path)
    assert len(df) == 2
    assert "ticker" in df.columns
    assert df["pnl"].sum() == 175.0


def test_cloud_data_lake_sync(tmp_path):
    lake = CloudDataLake()
    schema = lake.initialize_schema()
    assert schema["status"] == "READY"
    assert "sentilyze_trades" in schema["tables"]

    trades = [
        {
            "ticker": "BTC-USD",
            "shares": 0.5,
            "entry_price": 60000.0,
            "exit_price": 63000.0,
            "pnl": 1500.0,
            "return_pct": 5.0,
            "reason": "TP1_PROFIT_LOCK",
        }
    ]
    res = lake.sync_trades_to_lakehouse(trades)
    assert res["status"] == "SUCCESS"
    assert res["synced_trades"] == 1


def test_crypto_isolated_sandbox_broker(tmp_path):
    crypto_portfolio_path = str(tmp_path / "test_crypto_portfolio.json")
    crypto_trades_path = str(tmp_path / "test_crypto_trades.csv")

    broker = PaperBroker(
        portfolio_path=crypto_portfolio_path,
        initial_cash=25000.0,
        trades_path=crypto_trades_path,
    )

    assert broker.state["cash"] == 25000.0
    assert broker.state["total_equity"] == 25000.0
    assert broker.state["open_positions"] == {}

    # Execute fractional crypto buy
    res = broker.execute_buy(
        ticker="BTC-USD",
        shares=0.25,
        price=64000.0,
        strategy_name="Crypto_Pilot_Sandbox",
        tp1_target=67000.0,
        tp2_target=70000.0,
        stop_loss=62000.0,
    )
    assert res["success"] is True
    assert "BTC-USD" in broker.state["open_positions"]
    assert broker.state["open_positions"]["BTC-USD"]["shares"] == 0.25

    # Check file persistence in tmp_path
    assert os.path.exists(crypto_portfolio_path)
    with open(crypto_portfolio_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "BTC-USD" in data["open_positions"]


def test_calculate_fvg_corridor():
    # Synthetic DataFrame with a clear 3-bar Bullish Fair Value Gap
    # Candle 0: Low 98, High 100
    # Candle 1: Big expansion green bar, Low 99, High 106
    # Candle 2: Pullback bar, Low 102, High 107 -> Low(2) = 102 > High(0) = 100
    # Bullish FVG gap is between [100.0, 102.0]
    dates = pd.date_range("2026-09-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "Open": [
                98.0,
                99.0,
                103.0,
                105.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "High": [
                100.0,
                106.0,
                107.0,
                108.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
                112.0,
            ],
            "Low": [97.0, 99.0, 102.0, 104.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "Close": [
                99.0,
                105.0,
                106.0,
                107.0,
                105.0,
                107.0,
                108.0,
                109.0,
                110.0,
                111.0,
            ],
            "Volume": [100000] * 10,
        },
        index=dates,
    )

    fvg_res = calculate_fvg_corridor("TEST", spot_price=101.0, df=df)
    assert fvg_res["has_active_fvg"] is True
    assert fvg_res["active_bullish_fvg"] is not None
    assert fvg_res["active_bullish_fvg"]["bottom"] == 100.0
    assert fvg_res["active_bullish_fvg"]["top"] == 102.0
    assert fvg_res["active_bullish_fvg"]["consequent_encroachment"] == 101.0
    assert fvg_res["is_mitigating"] is True


def test_price_scout_agent_evaluates_fvg():
    agent = PriceActionScoutAgent()
    # Evaluate a major liquid ticker with mocked or real history
    rep = agent.evaluate("NVDA", spot_price=135.0)
    assert rep["agent_name"] == "Real-Time Price & Tape Scout"
    assert "fvg_analysis" in rep
    assert "tactical_limit_entry" in rep
    assert rep["tactical_limit_entry"] > 0
    assert "conviction_score" in rep
