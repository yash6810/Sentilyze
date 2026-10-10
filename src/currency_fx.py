"""
Multi-Currency FX Normalizer & Exchange Rate Engine for Sentilyze.
Enables seamless execution of global assets (JPY, INR, HKD, CNY, EUR, GBP)
while normalizing all portfolio balances into the base USD ledger ($136,545.29).
"""

from typing import Dict, Optional
import time
from src.utils import get_logger

logger = get_logger(__name__)

# Fallback reference exchange rates (USD per 1 Local Currency unit)
# If local = JPY, 1 JPY = ~0.0066 USD; 1 USD = ~152 JPY
DEFAULT_USD_RATES: Dict[str, float] = {
    "USD": 1.0,
    "JPY": 0.00658,  # ~152 JPY per USD
    "INR": 0.01152,  # ~86.8 INR per USD
    "HKD": 0.1287,  # ~7.77 HKD per USD
    "CNY": 0.1381,  # ~7.24 CNY per USD
    "EUR": 1.0850,  # ~0.92 EUR per USD
    "GBP": 1.3020,  # ~0.77 GBP per USD
}

# In-memory cache for live FX rates: { "JPY": (rate, timestamp) }
_FX_CACHE: Dict[str, tuple[float, float]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour cache


def get_usd_fx_rate(currency: str) -> float:
    """
    Returns the real-time conversion rate from local currency to USD.
    Rate represents: USD amount per 1 unit of foreign currency.
    Example: get_usd_fx_rate("JPY") -> ~0.00658 ($0.00658 per 1 Yen)
    """
    curr_upper = (currency or "USD").upper().strip()
    if curr_upper == "USD":
        return 1.0

    now = time.time()
    if curr_upper in _FX_CACHE:
        rate, ts = _FX_CACHE[curr_upper]
        if now - ts < _CACHE_TTL_SECONDS:
            return rate

    # Fetch live from Yahoo Finance: e.g. "USDJPY=X" gives JPY per 1 USD
    try:
        import yfinance as yf

        pair_symbol = f"USD{curr_upper}=X"
        ticker = yf.Ticker(pair_symbol)
        fast_info = getattr(ticker, "fast_info", None)
        price = None
        if fast_info:
            price = fast_info.get("last_price")
        if not price:
            hist = ticker.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])

        if price and price > 0:
            # price is foreign currency per 1 USD (e.g. 152.5 JPY per USD)
            # So 1 JPY = 1 / 152.5 USD
            usd_rate = 1.0 / float(price)
            _FX_CACHE[curr_upper] = (usd_rate, now)
            logger.info(
                f"💱 Updated live FX rate for {curr_upper}: 1 {curr_upper} = ${usd_rate:.6f} USD"
            )
            return usd_rate
    except Exception as e:
        logger.debug(
            f"Live FX rate fetch for {curr_upper} skipped ({e}), using reference default."
        )

    fallback = DEFAULT_USD_RATES.get(curr_upper, 1.0)
    _FX_CACHE[curr_upper] = (fallback, now)
    return fallback


def convert_to_usd(amount_local: float, currency: str) -> float:
    """
    Converts any foreign currency amount to USD.
    amount_usd = amount_local * fx_rate
    """
    rate = get_usd_fx_rate(currency)
    return float(amount_local * rate)


def convert_from_usd(amount_usd: float, target_currency: str) -> float:
    """
    Converts a USD amount into foreign currency.
    amount_local = amount_usd / fx_rate
    """
    rate = get_usd_fx_rate(target_currency)
    if rate <= 0:
        return amount_usd
    return float(amount_usd / rate)
