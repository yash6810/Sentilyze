"""
Sentilyze Stock & Crypto Image & Brand Identity Engine.
=====================================================
Multi-tier, zero-cost image and corporate logo resolution engine for:
1. US Equities & ETFs (via Parqet CDN + Financial Modeling Prep fallback)
2. Cryptocurrencies & Digital Assets (via CoinGecko CDN & Search API)
3. Zero-Failure Institutional SVG Monogram Badges (dynamic sector-colored gradients)
4. Local Disk & Base64 In-Memory Caching for instant, zero-flicker UI rendering.
"""

import os
import base64
import hashlib
from typing import Optional, Dict, Tuple
import requests

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = os.path.join("data", "logos")

# Top crypto ticker identification
CRYPTO_SET = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "ADA",
    "AVAX",
    "DOT",
    "LINK",
    "BNB",
    "MATIC",
    "SHIB",
    "LTC",
    "NEAR",
    "UNI",
    "PEPE",
    "SUI",
    "APT",
    "RENDER",
    "TAO",
    "FET",
}

# Static verified CoinGecko CDN mappings for instant resolution
STATIC_COINGECKO_MAP = {
    "BTC": "https://coin-images.coingecko.com/coins/images/1/large/bitcoin.png",
    "ETH": "https://coin-images.coingecko.com/coins/images/279/large/ethereum.png",
    "SOL": "https://coin-images.coingecko.com/coins/images/4128/large/solana.png",
    "XRP": "https://coin-images.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png",
    "DOGE": "https://coin-images.coingecko.com/coins/images/5/large/dogecoin.png",
    "ADA": "https://coin-images.coingecko.com/coins/images/975/large/cardano.png",
    "AVAX": "https://coin-images.coingecko.com/coins/images/12559/large/Avalanche_Circle_RedWhite_Trans.png",
    "DOT": "https://coin-images.coingecko.com/coins/images/12171/large/polkadot.png",
    "LINK": "https://coin-images.coingecko.com/coins/images/877/large/chainlink-new-logo.png",
    "BNB": "https://coin-images.coingecko.com/coins/images/825/large/bnb-icon2_2x.png",
}

# Sector-specific gradient palettes for institutional SVG badges
SECTOR_GRADIENTS = {
    "Information Technology": ("#3B82F6", "#1D4ED8"),  # Royal Blue
    "Health Care": ("#10B981", "#047857"),  # Emerald
    "Financials": ("#F59E0B", "#B45309"),  # Amber / Gold
    "Energy": ("#EF4444", "#B91C1C"),  # Crimson
    "Consumer Discretionary": ("#8B5CF6", "#6D28D9"),  # Purple
    "Communication Services": ("#EC4899", "#BE185D"),  # Pink
    "Industrials": ("#64748B", "#334155"),  # Slate
    "Consumer Staples": ("#14B8A6", "#0F766E"),  # Teal
    "Utilities": ("#06B6D4", "#0E7490"),  # Cyan
    "Real Estate": ("#84CC16", "#4D7C0F"),  # Lime
    "Materials": ("#A855F7", "#7E22CE"),  # Violet
    "Cryptocurrency": ("#F59E0B", "#D97706"),  # Crypto Amber
    "DEFAULT": ("#2563EB", "#7C3AED"),  # Indigo gradient
}

# Process memory cache: ticker -> base64 data URI
_IN_MEMORY_CACHE: Dict[str, str] = {}


def clean_ticker_symbol(ticker: str) -> str:
    """Normalizes ticker symbol for lookup (e.g. BRK.B -> BRK-B, BTC-USD -> BTC)."""
    if not ticker:
        return ""
    t = ticker.strip().upper()
    if t.endswith("-USD") or t.endswith("/USD") or t.endswith("USDT"):
        t = t.replace("-USD", "").replace("/USD", "").replace("USDT", "")
    return t


def is_crypto_symbol(ticker: str) -> bool:
    """Detects whether a symbol represents a cryptocurrency."""
    if not ticker:
        return False
    raw = ticker.strip().upper()
    if raw.endswith("-USD") or raw.endswith("/USD") or raw.endswith("USDT"):
        return True
    base = clean_ticker_symbol(raw)
    return base in CRYPTO_SET


def resolve_crypto_logo_url(symbol: str) -> str:
    """Resolves CoinGecko CDN image URL for cryptocurrencies."""
    clean = clean_ticker_symbol(symbol)
    if clean in STATIC_COINGECKO_MAP:
        return STATIC_COINGECKO_MAP[clean]

    # Dynamic lookup via CoinGecko Search API if not in static map
    try:
        url = f"https://api.coingecko.com/api/v3/search?query={clean.lower()}"
        res = requests.get(url, timeout=2.0, headers={"User-Agent": "Sentilyze/1.0"})
        if res.status_code == 200:
            coins = res.json().get("coins", [])
            for c in coins:
                if c.get("symbol", "").upper() == clean:
                    large_img = c.get("large") or c.get("thumb")
                    if large_img:
                        return large_img
    except Exception as e:
        logger.debug(f"CoinGecko search notice for {symbol}: {e}")

    # Fallback to generic symbol logo on parqet
    return f"https://assets.parqet.com/logos/symbol/{clean}?format=png"


def resolve_equity_logo_url(ticker: str) -> Tuple[str, str]:
    """Returns primary (Parqet) and secondary (FMP) logo URLs for an equity or ETF."""
    clean = clean_ticker_symbol(ticker)
    primary = f"https://assets.parqet.com/logos/symbol/{clean}?format=png"
    secondary = f"https://financialmodelingprep.com/image-stock/{clean}.png"
    return primary, secondary


def resolve_asset_image_url(ticker: str) -> str:
    """Returns the best public CDN URL for an asset (equity, ETF, or crypto)."""
    if is_crypto_symbol(ticker):
        return resolve_crypto_logo_url(ticker)
    primary, _ = resolve_equity_logo_url(ticker)
    return primary


def get_sector_gradient(ticker: str) -> Tuple[str, str]:
    """Retrieves gradient colors for a ticker based on its GICS sector."""
    if is_crypto_symbol(ticker):
        return SECTOR_GRADIENTS["Cryptocurrency"]
    try:
        from src.security_master import get_ticker_sector

        sector = get_ticker_sector(clean_ticker_symbol(ticker))
        if sector in SECTOR_GRADIENTS:
            return SECTOR_GRADIENTS[sector]
    except Exception:
        pass
    # Deterministic fallback color based on ticker hash
    colors = [
        ("#3B82F6", "#1D4ED8"),
        ("#10B981", "#047857"),
        ("#8B5CF6", "#6D28D9"),
        ("#06B6D4", "#0E7490"),
        ("#F59E0B", "#B45309"),
    ]
    idx = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % len(colors)
    return colors[idx]


def generate_fallback_svg_badge(ticker: str, size: int = 40) -> str:
    """Generates an institutional SVG brand avatar with crisp monogram and sector gradient."""
    clean = clean_ticker_symbol(ticker)
    display_text = clean[:4] if len(clean) > 4 else clean
    c1, c2 = get_sector_gradient(ticker)
    font_size = int(size * 0.38)
    radius = max(4, int(size * 0.2))

    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f"<defs>"
        f'<linearGradient id="g_{clean}" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{c1}" />'
        f'<stop offset="100%" stop-color="{c2}" />'
        f"</linearGradient>"
        f"</defs>"
        f'<rect width="{size}" height="{size}" rx="{radius}" fill="url(#g_{clean})" />'
        f'<rect width="{size}" height="{size}" rx="{radius}" fill="none" '
        f'stroke="rgba(255,255,255,0.2)" stroke-width="1" />'
        f'<text x="50%" y="54%" dominant-baseline="middle" text-anchor="middle" fill="#FFFFFF" '
        f'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" '
        f'font-weight="800" font-size="{font_size}px" letter-spacing="-0.03em">{display_text}</text>'
        f"</svg>"
    )
    b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def get_asset_logo_base64(
    ticker: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    allow_download: bool = True,
    timeout: float = 1.5,
) -> str:
    """
    Returns base64 data URI for the asset logo.
    Retrieval priority:
    1. In-memory session cache (0ms)
    2. Local file cache in data/logos/{ticker}.png (0ms)
    3. Live CDN download (Parqet/FMP/CoinGecko) with timeout (saves to cache)
    4. Fail-safe dynamic SVG badge (0ms)
    """
    clean = clean_ticker_symbol(ticker)
    if not clean:
        return generate_fallback_svg_badge("?", size=40)

    # 1. In-Memory Cache
    if clean in _IN_MEMORY_CACHE:
        return _IN_MEMORY_CACHE[clean]

    # 2. Disk Cache
    os.makedirs(cache_dir, exist_ok=True)
    local_file = os.path.join(cache_dir, f"{clean}.png")
    if os.path.exists(local_file) and os.path.getsize(local_file) > 100:
        try:
            with open(local_file, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            uri = f"data:image/png;base64,{b64_data}"
            _IN_MEMORY_CACHE[clean] = uri
            return uri
        except Exception as e:
            logger.debug(f"Error reading cached logo file for {clean}: {e}")

    # 3. Live Download
    if allow_download:
        if is_crypto_symbol(ticker):
            urls_to_try = [resolve_crypto_logo_url(ticker)]
        else:
            primary, secondary = resolve_equity_logo_url(ticker)
            urls_to_try = [primary, secondary]

        for url in urls_to_try:
            try:
                resp = requests.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                if resp.status_code == 200 and resp.content and len(resp.content) > 100:
                    ct = resp.headers.get("content-type", "").lower()
                    if (
                        "image" in ct
                        or resp.content.startswith(b"\x89PNG")
                        or resp.content.startswith(b"\xff\xd8")
                    ):
                        with open(local_file, "wb") as f:
                            f.write(resp.content)
                        b64_data = base64.b64encode(resp.content).decode("utf-8")
                        uri = f"data:image/png;base64,{b64_data}"
                        _IN_MEMORY_CACHE[clean] = uri
                        return uri
            except Exception:
                continue

    # 4. Zero-Failure SVG Monogram Fallback
    svg_uri = generate_fallback_svg_badge(clean, size=48)
    _IN_MEMORY_CACHE[clean] = svg_uri
    return svg_uri


def get_asset_avatar_html(
    ticker: str,
    size: int = 40,
    border_radius: str = "8px",
    extra_style: str = "",
) -> str:
    """Generates an embeddable HTML string with responsive logo avatar."""
    clean = clean_ticker_symbol(ticker)
    data_uri = get_asset_logo_base64(ticker)
    fallback_uri = generate_fallback_svg_badge(clean, size=size)

    html = (
        f'<img src="{data_uri}" alt="{clean}" '
        f"onerror=\"this.onerror=null; this.src='{fallback_uri}';\" "
        f'style="width: {size}px; height: {size}px; border-radius: {border_radius}; '
        f"object-fit: contain; background: rgba(255, 255, 255, 0.05); padding: 2px; "
        f"border: 1px solid rgba(255, 255, 255, 0.12); vertical-align: middle; "
        f'display: inline-block; box-shadow: 0 2px 6px rgba(0,0,0,0.3); {extra_style}" />'
    )
    return html


def clear_logo_cache(ticker: Optional[str] = None) -> None:
    """Clears in-memory and optionally disk logo cache."""
    if ticker:
        clean = clean_ticker_symbol(ticker)
        _IN_MEMORY_CACHE.pop(clean, None)
    else:
        _IN_MEMORY_CACHE.clear()
