"""
Unified US Stock Market (NYSE / NASDAQ) Session & Calendar Engine for Sentilyze.
Enforces that all execution workflows and price guardians respect US Eastern Time (EDT / EST),
regular trading hours (09:30 - 16:00 EDT), and market holidays.
"""

from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, Optional
import os

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from src.utils import get_logger

logger = get_logger(__name__)

# US Federal Exchange Market Holidays for 2025 - 2027
US_MARKET_HOLIDAYS = {
    # 2025
    date(2025, 1, 1),  # New Year's Day
    date(2025, 1, 20),  # MLK Jr. Day
    date(2025, 2, 17),  # Washington's Birthday (Presidents' Day)
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),  # Independence Day
    date(2025, 9, 1),  # Labor Day
    date(2025, 11, 27),  # Thanksgiving Day
    date(2025, 12, 25),  # Christmas Day
    # 2026
    date(2026, 1, 1),  # New Year's Day
    date(2026, 1, 19),  # MLK Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),  # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),  # Independence Day (Observed)
    date(2026, 9, 7),  # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
    # 2027
    date(2027, 1, 1),  # New Year's Day
    date(2027, 1, 18),  # MLK Jr. Day
    date(2027, 2, 15),  # Presidents' Day
    date(2027, 3, 26),  # Good Friday
    date(2027, 5, 31),  # Memorial Day
    date(2027, 6, 18),  # Juneteenth (Observed)
    date(2027, 7, 5),  # Independence Day (Observed)
    date(2027, 9, 6),  # Labor Day
    date(2027, 11, 25),  # Thanksgiving Day
    date(2027, 12, 24),  # Christmas Day (Observed)
}


def get_current_ny_time() -> datetime:
    """Returns the current precise timestamp in America/New_York (Eastern Time)."""
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # Fallback approximation (UTC - 4 hours during EDT)
        return datetime.now(timezone.utc) - timedelta(hours=4)


def get_us_market_session() -> Dict[str, Any]:
    """
    Computes the exact real-time US equity market session (NYSE / NASDAQ).

    Session Windows (Eastern Time):
    - Regular Trading Hours: 09:30 - 16:00 EDT (13:30 - 20:00 UTC)
    - Pre-Market Session: 04:00 - 09:30 EDT
    - After-Hours Session: 16:00 - 20:00 EDT
    - Weekend / Holiday: Closed
    """
    now_ny = get_current_ny_time()
    today_date = now_ny.date()
    weekday = now_ny.weekday()  # 0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday
    hour = now_ny.hour
    minute = now_ny.minute
    time_float = hour + (minute / 60.0)

    time_edt_str = now_ny.strftime("%I:%M:%S %p EDT")
    date_edt_str = now_ny.strftime("%A, %B %d, %Y")
    utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Check Weekend
    if weekday >= 5:
        return {
            "status": "MARKET_CLOSED_WEEKEND",
            "is_open": False,
            "can_execute": False,
            "session_name": "Weekend Market Break",
            "time_edt": time_edt_str,
            "date_edt": date_edt_str,
            "utc_time": utc_str,
            "badge_color": "#EF4444",
            "icon": "🔴",
            "message": f"US Exchanges (NYSE/NASDAQ) are closed for the weekend (Current: {time_edt_str}).",
        }

    # Check US Holiday
    if today_date in US_MARKET_HOLIDAYS:
        return {
            "status": "MARKET_CLOSED_HOLIDAY",
            "is_open": False,
            "can_execute": False,
            "session_name": "US Exchange Holiday",
            "time_edt": time_edt_str,
            "date_edt": date_edt_str,
            "utc_time": utc_str,
            "badge_color": "#EF4444",
            "icon": "🔴",
            "message": f"US Exchanges are closed today in observance of a federal holiday ({date_edt_str}).",
        }

    # Check Regular Trading Hours (09:30 - 16:00 EDT)
    if 9.5 <= time_float < 16.0:
        return {
            "status": "REGULAR_MARKET_OPEN",
            "is_open": True,
            "can_execute": True,
            "session_name": "Regular Trading Session (NYSE / NASDAQ)",
            "time_edt": time_edt_str,
            "date_edt": date_edt_str,
            "utc_time": utc_str,
            "badge_color": "#10B981",
            "icon": "🟢",
            "message": f"🟢 US Market is LIVE & OPEN for continuous execution (09:30 - 16:00 EDT). Current: {time_edt_str}.",
        }

    # Check Pre-Market (04:00 - 09:30 EDT)
    elif 4.0 <= time_float < 9.5:
        return {
            "status": "PRE_MARKET",
            "is_open": False,
            "can_execute": False,
            "session_name": "Early Pre-Market Session",
            "time_edt": time_edt_str,
            "date_edt": date_edt_str,
            "utc_time": utc_str,
            "badge_color": "#F59E0B",
            "icon": "🟡",
            "message": f"🟡 Early Pre-Market session active (04:00 - 09:30 EDT). Regular market opens at 09:30 EDT (in {round(9.5 - time_float, 1)}h).",
        }

    # Check After-Hours (16:00 - 20:00 EDT)
    elif 16.0 <= time_float < 20.0:
        return {
            "status": "AFTER_HOURS",
            "is_open": False,
            "can_execute": False,
            "session_name": "Post-Market / After-Hours",
            "time_edt": time_edt_str,
            "date_edt": date_edt_str,
            "utc_time": utc_str,
            "badge_color": "#8B5CF6",
            "icon": "🟣",
            "message": f"🟣 After-Hours trading active (16:00 - 20:00 EDT). Continuous auction closed at 16:00 EDT.",
        }

    # Overnight Closed
    else:
        return {
            "status": "OVERNIGHT_CLOSED",
            "is_open": False,
            "can_execute": False,
            "session_name": "Overnight Closed",
            "time_edt": time_edt_str,
            "date_edt": date_edt_str,
            "utc_time": utc_str,
            "badge_color": "#64748B",
            "icon": "⚪",
            "message": f"⚪ US Markets closed overnight. Pre-market opens at 04:00 EDT.",
        }


def check_market_hours_preflight(
    allow_force_execution: bool = False,
) -> bool:
    """
    Pre-flight sanity check for automated workflows.
    Returns True if execution should proceed, False if it should be skipped.
    """
    session = get_us_market_session()
    logger.info(
        f"🏛️ [US Market Pre-Flight] Session: {session['status']} | EDT: {session['time_edt']} | Live: {session['is_open']}"
    )

    if session["is_open"]:
        return True

    # If force flag is enabled (e.g. simulation or test override)
    force_env = os.getenv("FORCE_OFF_HOURS_EXECUTION", "false").lower() == "true"
    if allow_force_execution or force_env:
        logger.warning("⚠️ Off-hours execution forced via configuration flag.")
        return True

    logger.info(
        f"🛑 Execution skipped because US Market is closed: {session['message']}"
    )
    return False


CRYPTO_SYMBOLS = {
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "DOGE-USD",
    "ADA-USD",
    "AVAX-USD",
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "ADA",
    "AVAX",
}


def is_crypto_asset(ticker: str) -> bool:
    """Checks whether a ticker symbol represents a 24/7 cryptocurrency asset."""
    if not ticker or not isinstance(ticker, str):
        return False
    t_clean = ticker.upper().strip()
    return (
        t_clean in CRYPTO_SYMBOLS
        or t_clean.endswith("-USD")
        or t_clean.endswith("/USD")
        or t_clean.endswith("USDT")
        or t_clean.endswith("-PERP")
    )


GLOBAL_EXCHANGES = {
    "JP": {
        "name": "Tokyo Stock Exchange (TSE)",
        "suffixes": [".T"],
        "timezone": "Asia/Tokyo",
        "open_hour": 9.0,
        "close_hour": 15.5,
        "lunch_break": (11.5, 12.5),
        "currency": "JPY",
        "symbol_prefix": "¥",
    },
    "IN": {
        "name": "National Stock Exchange of India (NSE)",
        "suffixes": [".NS", ".BO"],
        "timezone": "Asia/Kolkata",
        "open_hour": 9.25,  # 09:15 IST
        "close_hour": 15.5,  # 15:30 IST
        "lunch_break": None,
        "currency": "INR",
        "symbol_prefix": "₹",
    },
    "HK": {
        "name": "Hong Kong Exchanges and Clearing (HKEX)",
        "suffixes": [".HK"],
        "timezone": "Asia/Hong_Kong",
        "open_hour": 9.5,  # 09:30 HKT
        "close_hour": 16.0,  # 16:00 HKT
        "lunch_break": (12.0, 13.0),
        "currency": "HKD",
        "symbol_prefix": "HK$",
    },
    "CN": {
        "name": "Shanghai & Shenzhen Stock Exchanges (SSE/SZSE)",
        "suffixes": [".SS", ".SZ"],
        "timezone": "Asia/Shanghai",
        "open_hour": 9.5,  # 09:30 CST
        "close_hour": 15.0,  # 15:00 CST
        "lunch_break": (11.5, 13.0),
        "currency": "CNY",
        "symbol_prefix": "¥",
    },
}


def get_regional_market_session(ticker: str) -> Dict[str, Any]:
    """
    Computes real-time exchange session for any asset across US, Crypto, Japan, India, or China.
    """
    t_clean = (ticker or "").upper().strip()

    # 1. 24/7 Crypto
    if is_crypto_asset(t_clean):
        now_utc = datetime.now(timezone.utc)
        return {
            "market_code": "CRYPTO",
            "market_name": "24/7 Digital Asset Market",
            "is_open": True,
            "status": "CRYPTO_24_7_OPEN",
            "currency": "USD",
            "symbol_prefix": "$",
            "local_time": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "badge_color": "#10B981",
            "icon": "🟢",
            "message": "🟢 24/7/365 Continuous Execution Active for Crypto.",
        }

    # 2. Check Global Suffixes (Japan, India, Hong Kong, China)
    for code, conf in GLOBAL_EXCHANGES.items():
        if any(t_clean.endswith(sfx) for sfx in conf["suffixes"]):
            try:
                tz = ZoneInfo(conf["timezone"])
            except Exception:
                tz = timezone.utc

            now_local = datetime.now(tz)
            weekday = now_local.weekday()
            time_float = now_local.hour + (now_local.minute / 60.0)
            time_str = now_local.strftime("%I:%M:%S %p")
            date_str = now_local.strftime("%Y-%m-%d")

            # Weekend check
            if weekday >= 5:
                return {
                    "market_code": code,
                    "market_name": conf["name"],
                    "is_open": False,
                    "status": "MARKET_CLOSED_WEEKEND",
                    "currency": conf["currency"],
                    "symbol_prefix": conf["symbol_prefix"],
                    "local_time": f"{date_str} {time_str}",
                    "badge_color": "#EF4444",
                    "icon": "🔴",
                    "message": f"🔴 {conf['name']} closed for weekend ({time_str}).",
                }

            # Lunch break check (TSE, HKEX, SSE)
            if conf["lunch_break"]:
                l_start, l_end = conf["lunch_break"]
                if l_start <= time_float < l_end:
                    return {
                        "market_code": code,
                        "market_name": conf["name"],
                        "is_open": False,
                        "status": "MARKET_LUNCH_BREAK",
                        "currency": conf["currency"],
                        "symbol_prefix": conf["symbol_prefix"],
                        "local_time": f"{date_str} {time_str}",
                        "badge_color": "#F59E0B",
                        "icon": "🟡",
                        "message": f"🟡 {conf['name']} in mid-day auction recess ({time_str}).",
                    }

            # Regular trading hours
            if conf["open_hour"] <= time_float < conf["close_hour"]:
                return {
                    "market_code": code,
                    "market_name": conf["name"],
                    "is_open": True,
                    "status": "REGULAR_MARKET_OPEN",
                    "currency": conf["currency"],
                    "symbol_prefix": conf["symbol_prefix"],
                    "local_time": f"{date_str} {time_str}",
                    "badge_color": "#10B981",
                    "icon": "🟢",
                    "message": f"🟢 {conf['name']} is LIVE & OPEN ({time_str}).",
                }

            return {
                "market_code": code,
                "market_name": conf["name"],
                "is_open": False,
                "status": "MARKET_CLOSED_OFF_HOURS",
                "currency": conf["currency"],
                "symbol_prefix": conf["symbol_prefix"],
                "local_time": f"{date_str} {time_str}",
                "badge_color": "#64748B",
                "icon": "⚪",
                "message": f"⚪ {conf['name']} closed off-hours ({time_str}).",
            }

    # 3. Default to US Session
    us_sess = get_us_market_session()
    us_sess["market_code"] = "US"
    us_sess["market_name"] = "NYSE / NASDAQ"
    us_sess["currency"] = "USD"
    us_sess["symbol_prefix"] = "$"
    return us_sess


def get_all_global_market_statuses() -> Dict[str, Any]:
    """Returns real-time session snapshots for all 5 global trading theatres."""
    return {
        "US": get_regional_market_session("AAPL"),
        "CRYPTO": get_regional_market_session("BTC-USD"),
        "JAPAN": get_regional_market_session("7203.T"),
        "INDIA": get_regional_market_session("RELIANCE.NS"),
        "HONG_KONG": get_regional_market_session("0700.HK"),
        "CHINA": get_regional_market_session("600519.SS"),
    }


def is_asset_market_open(ticker: str, allow_force: bool = False) -> bool:
    """
    Returns True if trading is active for the specified asset:
    - 24/7/365 continuous execution for Crypto.
    - Regular hours for US, Japan, India, or China depending on ticker.
    """
    session = get_regional_market_session(ticker)
    if session.get("is_open", False):
        return True

    force_env = os.getenv("FORCE_OFF_HOURS_EXECUTION", "false").lower() == "true"
    return bool(allow_force or force_env)
