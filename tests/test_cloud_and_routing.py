from src.cloud_lakehouse import CloudDataLake
from src.order_routing import (
    generate_vwap_order_schedule,
    generate_twap_order_schedule,
)


def test_cloud_data_lake():
    lake = CloudDataLake()
    schema = lake.initialize_schema()
    assert schema["status"] == "READY"
    assert "sentilyze_trades" in schema["tables"]

    sync_res = lake.sync_trades_to_lakehouse(
        [{"trade_id": "T1", "ticker": "NVDA", "pnl": 500.0}]
    )
    assert sync_res["status"] == "SUCCESS"
    assert sync_res["synced_trades"] == 1

    stream_res = lake.stream_live_portfolio_snapshot(
        total_equity=105000.0, cash=40000.0, open_positions=3
    )
    assert stream_res["delivered"] is True


def test_order_routing_vwap_twap():
    vwap = generate_vwap_order_schedule("NVDA", total_shares=5000, current_price=130.0)
    assert vwap["ticker"] == "NVDA"
    assert vwap["total_child_slices"] == 7
    assert vwap["estimated_execution_savings_dollars"] > 0

    twap = generate_twap_order_schedule(
        "AAPL", total_shares=6000, current_price=220.0, num_slices=6
    )
    assert twap["ticker"] == "AAPL"
    assert twap["total_slices"] == 6
    assert twap["shares_per_slice"] == 1000
