"""
Unit tests for Sentilyze Stock & Crypto Image & Brand Identity Engine.
=====================================================================
Tests URL resolution, CoinGecko routing, local caching, SVG fallback generation,
and network failure resilience.
"""

import os
import base64
from unittest.mock import patch, MagicMock

from src.stock_image_provider import (
    clean_ticker_symbol,
    is_crypto_symbol,
    resolve_crypto_logo_url,
    resolve_equity_logo_url,
    resolve_asset_image_url,
    generate_fallback_svg_badge,
    get_asset_logo_base64,
    get_asset_avatar_html,
    clear_logo_cache,
    STATIC_COINGECKO_MAP,
)


def test_clean_ticker_symbol():
    assert clean_ticker_symbol("BTC-USD") == "BTC"
    assert clean_ticker_symbol("eth/usd") == "ETH"
    assert clean_ticker_symbol("solusdt") == "SOL"
    assert clean_ticker_symbol(" nvda ") == "NVDA"
    assert clean_ticker_symbol("") == ""


def test_is_crypto_symbol():
    assert is_crypto_symbol("BTC") is True
    assert is_crypto_symbol("btc-usd") is True
    assert is_crypto_symbol("ETH") is True
    assert is_crypto_symbol("SOL/USD") is True
    assert is_crypto_symbol("NVDA") is False
    assert is_crypto_symbol("AAPL") is False
    assert is_crypto_symbol("SPY") is False
    assert is_crypto_symbol("") is False


def test_resolve_crypto_logo_url():
    url_btc = resolve_crypto_logo_url("BTC")
    assert url_btc == STATIC_COINGECKO_MAP["BTC"]
    assert "coin-images.coingecko.com" in url_btc

    url_eth = resolve_crypto_logo_url("ETH-USD")
    assert url_eth == STATIC_COINGECKO_MAP["ETH"]


def test_resolve_equity_logo_url():
    primary, secondary = resolve_equity_logo_url("NVDA")
    assert "assets.parqet.com/logos/symbol/NVDA" in primary
    assert "financialmodelingprep.com/image-stock/NVDA" in secondary


def test_resolve_asset_image_url_routing():
    equity_url = resolve_asset_image_url("AAPL")
    assert "parqet.com" in equity_url

    crypto_url = resolve_asset_image_url("BTC")
    assert "coingecko.com" in crypto_url


def test_generate_fallback_svg_badge():
    svg_uri = generate_fallback_svg_badge("NVDA", size=48)
    assert svg_uri.startswith("data:image/svg+xml;base64,")

    b64_part = svg_uri.replace("data:image/svg+xml;base64,", "")
    decoded_svg = base64.b64decode(b64_part).decode("utf-8")
    assert "<svg" in decoded_svg
    assert "NVDA" in decoded_svg
    assert "linearGradient" in decoded_svg


def test_get_asset_logo_base64_local_cache(tmp_path):
    clear_logo_cache()
    cache_dir = str(tmp_path / "logos")
    os.makedirs(cache_dir, exist_ok=True)

    # Place a dummy valid PNG file
    png_path = os.path.join(cache_dir, "MSFT.png")
    fake_png_bytes = b"\x89PNG\r\n\x1a\n" + b"0" * 120
    with open(png_path, "wb") as f:
        f.write(fake_png_bytes)

    # Should read directly from disk without any network call
    with patch("requests.get") as mock_get:
        uri = get_asset_logo_base64("MSFT", cache_dir=cache_dir)
        assert mock_get.call_count == 0
        assert uri.startswith("data:image/png;base64,")


def test_get_asset_logo_base64_download_and_cache(tmp_path):
    clear_logo_cache()
    cache_dir = str(tmp_path / "logos")

    fake_png_bytes = b"\x89PNG\r\n\x1a\n" + b"VALID_IMAGE_BYTES" * 10
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_png_bytes
    mock_resp.headers = {"content-type": "image/png"}

    with patch("requests.get", return_value=mock_resp):
        uri = get_asset_logo_base64("AMD", cache_dir=cache_dir, timeout=1.0)
        assert uri.startswith("data:image/png;base64,")

        # Verify it was saved to disk
        saved_file = os.path.join(cache_dir, "AMD.png")
        assert os.path.exists(saved_file)
        with open(saved_file, "rb") as f:
            assert f.read() == fake_png_bytes


def test_get_asset_logo_base64_network_failure_fallback(tmp_path):
    clear_logo_cache()
    cache_dir = str(tmp_path / "logos")

    with patch("requests.get", side_effect=Exception("Connection timed out")):
        uri = get_asset_logo_base64("NONEXISTENT", cache_dir=cache_dir, timeout=0.1)
        # Should smoothly fall back to dynamic SVG badge
        assert uri.startswith("data:image/svg+xml;base64,")


def test_get_asset_avatar_html():
    clear_logo_cache()
    html = get_asset_avatar_html("NVDA", size=36, border_radius="10px")
    assert "<img" in html
    assert 'alt="NVDA"' in html
    assert "width: 36px" in html
    assert "border-radius: 10px" in html
    assert "onerror=" in html
