from src.liquidity_heatmap import (
    compute_order_book_depth_and_clusters,
    compute_volume_profile_and_poc,
)


def test_compute_order_book_depth_and_clusters():
    res = compute_order_book_depth_and_clusters("AVGO", spot_price=350.0)
    assert res["ticker"] == "AVGO"
    assert len(res["bids"]) == 15
    assert len(res["asks"]) == 15
    assert res["total_bid_volume"] > 0
    assert res["total_ask_volume"] > 0
    assert res["depth_sentiment"] in [
        "BULLISH_BUY_PRESSURE",
        "BEARISH_SUPPLY_WALL",
        "BALANCED",
    ]


def test_compute_volume_profile_and_poc():
    res = compute_volume_profile_and_poc("AVGO", spot_price=350.0)
    assert res["ticker"] == "AVGO"
    assert res["poc_price"] > 0.0
    assert res["value_area_high"] > res["poc_price"]
    assert res["value_area_low"] < res["poc_price"]
    assert len(res["price_bins"]) == 30
    assert len(res["volumes"]) == 30
