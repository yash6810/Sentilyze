import pytest
import os
import numpy as np
import pandas as pd
from src.duckdb_store import DuckDBStore


def test_duckdb_store_register_and_query(tmp_path):
    store = DuckDBStore(store_dir=str(tmp_path))

    df = pd.DataFrame(
        {
            "ticker": ["NVDA", "AAPL", "MSFT"],
            "price": [120.5, 230.0, 440.0],
            "volume": [1000000, 2000000, 1500000],
        }
    )

    store.register_dataframe("quotes", df)
    res = store.query(
        "SELECT ticker, price FROM quotes WHERE price > 200 ORDER BY price ASC"
    )

    assert len(res) == 2
    assert list(res["ticker"]) == ["AAPL", "MSFT"]


def test_duckdb_store_parquet_roundtrip(tmp_path):
    store = DuckDBStore(store_dir=str(tmp_path))

    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5),
            "close": [100.0, 102.0, 101.5, 105.0, 108.0],
        }
    )

    pq_path = store.save_dataframe_as_parquet(df, "sample_prices.parquet")
    assert os.path.exists(pq_path)

    query_res = store.query_parquet_file(
        pq_path, "SELECT avg(close) AS avg_close FROM {FILE}"
    )
    assert round(float(query_res.iloc[0]["avg_close"]), 2) == 103.3


def test_duckdb_store_vectorized_metrics(tmp_path):
    store = DuckDBStore(store_dir=str(tmp_path))

    np.random.seed(42)
    # Generate 100 days of slight positive returns
    rets = np.random.normal(0.001, 0.01, size=100)
    rets_df = pd.DataFrame({"return": rets})

    metrics = store.compute_vectorized_equity_curve_metrics(rets_df)
    assert metrics["status"] == "SUCCESS"
    assert metrics["total_bars"] == 100
    assert metrics["max_drawdown_pct"] <= 0.0
