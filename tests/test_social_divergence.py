import numpy as np
import pandas as pd
import pytest
from src.streaming_news import (
    SimHashDeduper,
    NewsRingBuffer,
    calculate_materiality_score,
)
from src.social_divergence import (
    compute_sentiment_exhaustion_top,
    compute_short_squeeze_probability,
)


def test_simhash_deduplication():
    deduper = SimHashDeduper(hash_bits=64)
    h1 = deduper.compute_fingerprint(
        "Nvidia reports record quarterly earnings beating Wall Street expectations"
    )
    # Near identical wire syndicate headline
    h2 = deduper.compute_fingerprint(
        "Nvidia reports record quarterly earnings beating Wall Street estimates"
    )
    # Completely distinct headline
    h3 = deduper.compute_fingerprint(
        "Federal Reserve cuts interest rates by 25 basis points amid slowing inflation"
    )

    dist_near = deduper.hamming_distance(h1, h2)
    dist_diff = deduper.hamming_distance(h1, h3)

    assert dist_near <= 15, f"Expected near-duplicate distance <= 15, got {dist_near}"
    assert (
        dist_diff >= 20
    ), f"Expected distinct headline distance >= 20, got {dist_diff}"
    assert deduper.is_duplicate(h2, [h1], max_distance=15) is True


def test_news_ring_buffer():
    buf = NewsRingBuffer(capacity=5)
    item1 = {
        "title": "Apple announces new artificial intelligence iPhone chips",
        "url": "http://1",
    }
    item2 = {
        "title": "Apple announces new artificial intelligence iPhone chips today",
        "url": "http://2",
    }
    item3 = {
        "title": "Crude oil plunges 4% as OPEC increases global production quotas",
        "url": "http://3",
    }

    assert buf.append_news_item(item1) is True
    # Item 2 is near duplicate of item 1 -> should be rejected
    assert buf.append_news_item(item2) is False
    assert buf.append_news_item(item3) is True

    items = buf.get_latest()
    assert len(items) == 2


def test_materiality_scoring():
    res_fda = calculate_materiality_score(
        "FDA approves breakthrough cancer drug for Phase 3 oncology trial"
    )
    assert res_fda["is_highly_material"] is True
    assert res_fda["primary_event"] == "CLINICAL_FDA"

    res_sec = calculate_materiality_score(
        "SEC opens formal fraud investigation and issues subpoena"
    )
    assert res_sec["is_highly_material"] is True
    assert res_sec["primary_event"] == "REGULATORY_LEGAL"

    res_generic = calculate_materiality_score(
        "Stock market opens slightly higher on quiet Monday morning"
    )
    assert res_generic["is_highly_material"] is False


def test_sentiment_exhaustion_top():
    n = 20
    df = pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * (n - 1) + [101.0],  # Recent high was 105, now 101
            "Low": [98.0] * n,
            "Close": [100.0] * (n - 5)
            + [99.0, 99.5, 100.0, 99.0, 99.5],  # Flat/lagging returns
            "Volume": [1_000_000] * n,
        }
    )

    # Euphoric sentiment with flat price -> should trigger exhaustion top
    res = compute_sentiment_exhaustion_top("TEST", sentiment_score=0.85, df=df)
    assert res["is_exhaustion_top"] is True
    assert res["divergence_score"] > 50.0

    # Normal sentiment -> no exhaustion
    res_norm = compute_sentiment_exhaustion_top("TEST", sentiment_score=0.20, df=df)
    assert res_norm["is_exhaustion_top"] is False


def test_short_squeeze_probability():
    # Low short interest stock
    res_low = compute_short_squeeze_probability(
        ticker="LOW_SI",
        short_percent_of_float=2.0,
        short_ratio_days=1.5,
        sentiment_acceleration=0.0,
    )
    assert res_low["squeeze_regime"] == "LOW_SQUEEZE_RISK"
    assert res_low["ssps_score"] < 50.0

    # High squeeze candidate (35% SI, 7 days to cover, high sentiment velocity)
    res_high = compute_short_squeeze_probability(
        ticker="HIGH_SI",
        short_percent_of_float=35.0,
        short_ratio_days=7.5,
        sentiment_acceleration=0.6,
    )
    assert res_high["squeeze_regime"] == "HIGH_SQUEEZE_POTENTIAL"
    assert res_high["ssps_score"] >= 70.0
