import os
import pandas as pd
from unittest.mock import patch

from src.security_master import (
    SecurityMaster,
    TIER_1_MEGA,
    TIER_2_MID,
    TIER_3_SMALL,
    TIER_4_EXCLUDED,
    get_security_master,
)


def test_classify_liquidity_tier():
    sm = SecurityMaster(cache_file="dummy.csv")

    # Tier 4 ExCLUDED: penny stock (< $5)
    assert (
        sm.classify_liquidity_tier(ticker="PENY", price=3.50, adv_shares=1_000_000)
        == TIER_4_EXCLUDED
    )

    # Tier 4 Excluded: low dollar volume (< $2M)
    assert (
        sm.classify_liquidity_tier(ticker="LOWV", price=10.0, adv_shares=100_000)
        == TIER_4_EXCLUDED
    )

    # Tier 3 Small: $2M <= ADV < $10M
    # 250k shares * $20 = $5.0M
    assert (
        sm.classify_liquidity_tier(ticker="SML", price=20.0, adv_shares=250_000)
        == TIER_3_SMALL
    )

    # Tier 2 Mid: $10M <= ADV < $50M
    # 500k shares * $50 = $25.0M
    assert (
        sm.classify_liquidity_tier(ticker="MID", price=50.0, adv_shares=500_000)
        == TIER_2_MID
    )

    # Tier 1 Mega: ADV >= $50M
    # 1M shares * $100 = $100.0M
    assert (
        sm.classify_liquidity_tier(ticker="MEGA", price=100.0, adv_shares=1_000_000)
        == TIER_1_MEGA
    )


def test_load_fallback_universe(tmp_path):
    cache_path = str(tmp_path / "test_sec_universe.csv")
    sm = SecurityMaster(cache_file=cache_path)

    df = sm._load_fallback_universe()
    assert isinstance(df, pd.DataFrame)
    assert "ticker" in df.columns
    assert "company_name" in df.columns
    assert "is_etf" in df.columns
    assert len(df) >= 10
    assert "NVDA" in df["ticker"].values
    assert "SPY" in df["ticker"].values


def test_build_registry_and_caching(tmp_path):
    cache_path = str(tmp_path / "test_sec_universe.csv")
    sm = SecurityMaster(cache_file=cache_path)

    # Mock fetch_exchange_directory to return sample data
    sample_df = pd.DataFrame(
        {
            "ticker": ["NVDA", "AAPL", "SPY", "XYZ"],
            "company_name": [
                "NVIDIA Corp",
                "Apple Inc",
                "SPDR S&P 500",
                "XYZ Inc",
            ],
            "is_etf": ["N", "N", "Y", "N"],
        }
    )

    with patch.object(sm, "fetch_exchange_directory", return_value=sample_df):
        reg = sm.build_registry()
        assert len(reg) == 4
        assert "liquidity_tier" in reg.columns
        assert os.path.exists(cache_path)

        # Tradable tickers with TIER_3_SMALL should include all liquid tiers
        tradable = sm.get_tradable_tickers(min_tier=TIER_3_SMALL, include_etfs=True)
        assert "NVDA" in tradable
        assert "SPY" in tradable

        # Exclude ETFs
        tradable_no_etf = sm.get_tradable_tickers(
            min_tier=TIER_3_SMALL, include_etfs=False
        )
        assert "SPY" not in tradable_no_etf
        assert "NVDA" in tradable_no_etf


def test_get_security_master_factory(tmp_path):
    with patch(
        "src.security_master.SECURITY_MASTER_CACHE",
        str(tmp_path / "factory_cache.csv"),
    ):
        sm = get_security_master()
        assert isinstance(sm, SecurityMaster)
        assert sm.universe_df is not None
