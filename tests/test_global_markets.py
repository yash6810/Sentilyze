"""
Unit tests for Global Multi-Market Calendars & FX Normalization.
Validates session detection and currency conversions for US, Crypto, Japan, India, and China.
"""

from src.market_session import (
    get_regional_market_session,
    get_all_global_market_statuses,
    is_asset_market_open,
    is_crypto_asset,
)
from src.currency_fx import convert_to_usd, convert_from_usd, get_usd_fx_rate


def test_crypto_session_always_open():
    for ticker in ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE"]:
        assert is_crypto_asset(ticker) is True
        sess = get_regional_market_session(ticker)
        assert sess["is_open"] is True
        assert sess["market_code"] == "CRYPTO"
        assert is_asset_market_open(ticker) is True


def test_global_market_session_theatres():
    # Japan TSE
    sess_jp = get_regional_market_session("7203.T")
    assert sess_jp["market_code"] == "JP"
    assert sess_jp["currency"] == "JPY"
    assert "Tokyo Stock Exchange" in sess_jp["market_name"]

    # India NSE
    sess_in = get_regional_market_session("RELIANCE.NS")
    assert sess_in["market_code"] == "IN"
    assert sess_in["currency"] == "INR"
    assert "National Stock Exchange" in sess_in["market_name"]

    # Hong Kong HKEX
    sess_hk = get_regional_market_session("0700.HK")
    assert sess_hk["market_code"] == "HK"
    assert sess_hk["currency"] == "HKD"

    # China SSE
    sess_cn = get_regional_market_session("600519.SS")
    assert sess_cn["market_code"] == "CN"
    assert sess_cn["currency"] == "CNY"

    # US Equity Default
    sess_us = get_regional_market_session("AAPL")
    assert sess_us["market_code"] == "US"
    assert sess_us["currency"] == "USD"


def test_all_global_statuses():
    all_sess = get_all_global_market_statuses()
    assert "US" in all_sess
    assert "CRYPTO" in all_sess
    assert "JAPAN" in all_sess
    assert "INDIA" in all_sess
    assert "HONG_KONG" in all_sess
    assert "CHINA" in all_sess


def test_currency_fx_normalization():
    # USD identity
    assert get_usd_fx_rate("USD") == 1.0
    assert convert_to_usd(100.0, "USD") == 100.0

    # JPY conversion (e.g. 100,000 JPY is roughly $600-$700 USD)
    usd_val_jpy = convert_to_usd(100000.0, "JPY")
    assert 400.0 <= usd_val_jpy <= 1000.0

    # INR conversion (e.g. 10,000 INR is roughly $100-$130 USD)
    usd_val_inr = convert_to_usd(10000.0, "INR")
    assert 80.0 <= usd_val_inr <= 150.0

    # Two-way conversion consistency
    usd_orig = 500.0
    inr_val = convert_from_usd(usd_orig, "INR")
    back_to_usd = convert_to_usd(inr_val, "INR")
    assert abs(usd_orig - back_to_usd) < 0.01
