"""
Unit tests for US Market Session & Calendar Engine.
"""

import pytest
from datetime import datetime, timezone, timedelta, date
from src.market_session import (
    get_us_market_session,
    check_market_hours_preflight,
    US_MARKET_HOLIDAYS,
)


def test_get_us_market_session_keys():
    session = get_us_market_session()
    assert "status" in session
    assert "is_open" in session
    assert "time_edt" in session
    assert "date_edt" in session
    assert "utc_time" in session
    assert isinstance(session["is_open"], bool)


def test_us_market_holidays_present():
    assert date(2026, 1, 1) in US_MARKET_HOLIDAYS
    assert date(2026, 12, 25) in US_MARKET_HOLIDAYS
    assert date(2026, 7, 3) in US_MARKET_HOLIDAYS


def test_preflight_check():
    # Forced execution should always return True
    res = check_market_hours_preflight(allow_force_execution=True)
    assert res is True
