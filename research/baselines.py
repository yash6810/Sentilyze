"""
Quantitative Baselines Comparison Experiment
Sentilyze Research Framework (Branch: research-baselines)
Author: Quant Research Team
Date: October 2026

Deterministic evaluation comparing:
a) SPY Buy-and-Hold
b) Equal-Weight Universe (monthly rebalance)
c) 12-1 Momentum (top 10 equal-weight, monthly rebalance)
d) SPY 200-Day Trend Filter (daily check)
e) Random Selection (10 stocks, 50 runs, seeds 42..91)
f) Existing ML Strategy (30 tickers, seed 42, zero sentiment)
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import requests
import numpy as np
import pandas as pd
from scipy import stats

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.purged_cv import compute_deflated_sharpe_ratio
from src.config import FEATURES
from src.feature_engineering import create_technical_indicators, create_features
from src.modeling import train_model
from src.backtesting import run_backtest

# Directories
RESEARCH_DIR = PROJECT_ROOT / "research"
CACHE_DIR = RESEARCH_DIR / "data_cache"
OUTPUT_DIR = RESEARCH_DIR / "outputs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed Parameters
START_DATE = "2018-01-01"
END_DATE = "2026-10-26"
IN_SAMPLE_END = "2022-12-31"
OUT_OF_SAMPLE_START = "2023-01-01"

FEE_PCT = 0.0010  # 0.10% broker fee
SLIPPAGE_PCT = 0.0005  # 0.05% slippage
TOTAL_FRICTION = FEE_PCT + SLIPPAGE_PCT  # 0.0015 (0.15% per traded notional)
INITIAL_CAPITAL = 10000.0
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# 1. Unit Tests for Max Drawdown
# -----------------------------------------------------------------------------
def calculate_max_drawdown(series: Any) -> float:
    """
    Computes maximum peak-to-trough drawdown of a value series.
    Returns a negative float (e.g. -0.50 for -50%).
    """
    s = pd.Series(series, dtype=float)
    if len(s) == 0:
        return 0.0
    cummax = s.cummax()
    dd = (s - cummax) / (cummax + 1e-12)
    return float(dd.min())


def run_max_drawdown_unit_tests(
    spy_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Runs unit tests for calculate_max_drawdown:
    1) [100, 50, 75] gives -50% (-0.50)
    2) SPY buy-and-hold up to 2020-03-23 gives about -34%
    """
    test1_arr = [100, 50, 75]
    res1 = calculate_max_drawdown(test1_arr)
    assert abs(res1 - (-0.50)) < 1e-6, f"Test 1 failed: expected -0.50, got {res1}"

    res2 = None
    if spy_series is not None:
        spy_covid = spy_series.loc[:"2020-03-23"]
        res2 = calculate_max_drawdown(spy_covid)
        assert (
            abs(res2 - (-0.3372)) < 0.015
        ), f"Test 2 failed: expected ~ -0.34, got {res2}"

    return {
        "test1_passed": True,
        "test1_value": res1,
        "test2_passed": res2 is not None,
        "test2_value": res2,
    }


# -----------------------------------------------------------------------------
# 2. Data Fetching & Caching
# -----------------------------------------------------------------------------
def fetch_yahoo_chart_data(
    ticker: str, session: requests.Session
) -> Optional[pd.DataFrame]:
    """
    Fetches 10 years of daily dividend/split-adjusted OHLCV data from Yahoo Chart API.
    Auto-adjusts Open, High, Low using ratio = AdjClose / Close.
    """
    clean_ticker = ticker.replace(".", "-")
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{clean_ticker}?range=10y&interval=1d"
    try:
        r = session.get(url, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        result = data.get("chart", {}).get("result")
        if not result or len(result) == 0:
            return None
        res = result[0]
        ts = res.get("timestamp")
        quote = res.get("indicators", {}).get("quote", [{}])[0]
        adj = res.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
        if not ts or not quote or not adj:
            return None

        dates = (
            pd.to_datetime(ts, unit="s", utc=True)
            .tz_convert("America/New_York")
            .tz_localize(None)
            .normalize()
        )
        df = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "AdjClose": adj,
                "Volume": quote.get("volume"),
            },
            index=dates,
        ).dropna()

        if df.empty or len(df) < 50:
            return None

        # Auto-adjust prices
        ratio = df["AdjClose"] / (df["Close"] + 1e-12)
        df_adj = pd.DataFrame(
            {
                "Open": df["Open"] * ratio,
                "High": df["High"] * ratio,
                "Low": df["Low"] * ratio,
                "Close": df["AdjClose"],
                "Volume": df["Volume"],
            },
            index=df.index,
        ).sort_index()

        # Remove duplicate index if any
        df_adj = df_adj[~df_adj.index.duplicated(keep="last")]
        return df_adj
    except Exception:
        return None


def get_or_fetch_ticker(
    ticker: str, session: requests.Session
) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{ticker.replace('.', '-')}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            pass

    df = fetch_yahoo_chart_data(ticker, session)
    if df is not None and not df.empty:
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass
    return df


import concurrent.futures


def build_universe(
    stocks_path: str = "stocks.txt",
) -> Tuple[List[str], List[str], Dict[str, pd.DataFrame]]:
    """
    Downloads and caches all universe tickers concurrently.
    Filters to tickers with continuous data from 2018-01-02 to 2026-10-26.
    Returns (included_tickers, excluded_tickers, data_dict).
    """
    manifest_path = CACHE_DIR / "universe_manifest.json"
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    # Fetch SPY and VIX first
    spy_df = get_or_fetch_ticker("SPY", session)
    if spy_df is None:
        raise RuntimeError("Failed to fetch SPY benchmark data.")
    vix_df = get_or_fetch_ticker("^VIX", session)

    # Read tickers from stocks.txt
    with open(PROJECT_ROOT / stocks_path, "r", encoding="utf-8") as f:
        raw_tickers = [
            line.strip().upper()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
    unique_tickers = sorted(list(set(raw_tickers)))

    print(f"Fetching and caching data for {len(unique_tickers)} tickers in parallel...")
    data_dict = {"SPY": spy_df}
    if vix_df is not None:
        data_dict["^VIX"] = vix_df

    def _fetch_one(t: str):
        # Dedicated session per thread or shared session
        return t, get_or_fetch_ticker(t, session)

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        fetched_pairs = list(executor.map(_fetch_one, unique_tickers))

    included = []
    excluded = []

    for t, df in fetched_pairs:
        if df is None:
            excluded.append((t, "fetch_failed_or_missing"))
            continue
        # Verify coverage from 2018-01-02 to 2026-10-26
        sub = df.loc[START_DATE:END_DATE]
        if (
            sub.empty
            or sub.index[0] > pd.Timestamp("2018-01-05")
            or sub.index[-1] < pd.Timestamp("2026-10-20")
        ):
            start_str = str(sub.index[0].date()) if not sub.empty else "None"
            end_str = str(sub.index[-1].date()) if not sub.empty else "None"
            excluded.append((t, f"insufficient_range_{start_str}_to_{end_str}"))
            continue
        # Need lookback before 2018 for momentum 12-1
        pre_2018 = df.loc[:"2017-12-31"]
        if len(pre_2018) < 250:
            excluded.append((t, f"insufficient_2017_lookback_{len(pre_2018)}_days"))
            continue

        included.append(t)
        data_dict[t] = df

    manifest = {
        "total_tickers_in_file": len(raw_tickers),
        "unique_tickers": len(unique_tickers),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "included": included,
        "excluded": excluded,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Ensure sector and multi-asset ETFs are in data_dict
    sector_etfs_all = [
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    ]
    multi_asset = ["SPY", "EFA", "EEM", "AGG", "GLD"]
    for etf in set(sector_etfs_all + multi_asset):
        if etf not in data_dict:
            df_etf = get_or_fetch_ticker(etf, session)
            if df_etf is not None:
                data_dict[etf] = df_etf

    return included, excluded, data_dict


# -----------------------------------------------------------------------------
# 3. Strategy Simulations
# -----------------------------------------------------------------------------
def simulate_spy_buy_and_hold(
    spy_df: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    SPY Buy-and-Hold: Buy at Day 1 Open with 0.15% friction, hold through end.
    """
    df = spy_df.loc[dates].copy()
    n = len(df)
    equity = np.zeros(n)
    cash = 0.0
    initial_cash = INITIAL_CAPITAL
    shares = (initial_cash * (1.0 - TOTAL_FRICTION)) / df["Open"].iloc[0]

    for i in range(n):
        equity[i] = shares * df["Close"].iloc[i]

    res = pd.DataFrame(
        {
            "equity": equity,
            "invested_pct": 100.0,
            "traded_notional": np.zeros(n),
            "trades": np.zeros(n),
        },
        index=dates,
    )
    res.iloc[0, res.columns.get_loc("traded_notional")] = initial_cash
    res.iloc[0, res.columns.get_loc("trades")] = 1
    return res


def simulate_spy_200_trend(
    spy_df: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    SPY 200-day trend filter: hold SPY when Close > SMA200 at t-1, otherwise cash.
    Trades execute at day t Open with 0.15% friction.
    """
    full_spy = spy_df.copy()
    full_spy["sma200"] = full_spy["Close"].rolling(200).mean()
    sub_spy = full_spy.loc[dates].copy()

    n = len(sub_spy)
    equity = np.zeros(n)
    invested_pct = np.zeros(n)
    traded_notional = np.zeros(n)
    trades = np.zeros(n)

    cash = INITIAL_CAPITAL
    shares = 0.0
    holding = False

    for i in range(n):
        curr_date = dates[i]
        curr_open = sub_spy["Open"].iloc[i]
        curr_close = sub_spy["Close"].iloc[i]

        # Check prior day close vs SMA200
        prev_idx = full_spy.index.get_loc(curr_date) - 1
        prev_close = full_spy["Close"].iloc[prev_idx]
        prev_sma = full_spy["sma200"].iloc[prev_idx]

        target_hold = (prev_close > prev_sma) if not np.isnan(prev_sma) else True

        if target_hold and not holding:
            # Buy at Open
            trade_val = cash
            friction = trade_val * TOTAL_FRICTION
            shares = (trade_val - friction) / curr_open
            cash = 0.0
            holding = True
            traded_notional[i] += trade_val
            trades[i] += 1
        elif not target_hold and holding:
            # Sell at Open
            proceeds = shares * curr_open
            friction = proceeds * TOTAL_FRICTION
            cash = proceeds - friction
            shares = 0.0
            holding = False
            traded_notional[i] += proceeds
            trades[i] += 1

        eq = cash + shares * curr_close
        equity[i] = eq
        invested_pct[i] = 100.0 if holding else 0.0

    return pd.DataFrame(
        {
            "equity": equity,
            "invested_pct": invested_pct,
            "traded_notional": traded_notional,
            "trades": trades,
        },
        index=dates,
    )


def simulate_equal_weight_universe(
    universe: List[str], data_dict: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Equal-weight all universe stocks, rebalanced monthly on 1st trading day at Open.
    """
    n_days = len(dates)
    n_stocks = len(universe)
    weight_target = 1.0 / n_stocks

    # Pre-extract Open and Close matrices
    open_df = pd.DataFrame(
        {t: data_dict[t]["Open"] for t in universe}, index=dates
    ).ffill()
    close_df = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in universe}, index=dates
    ).ffill()

    # Month-end masks
    month_ids = dates.to_period("M")
    is_new_month = np.concatenate([[True], month_ids[1:] != month_ids[:-1]])

    shares = np.zeros(n_stocks)
    cash = INITIAL_CAPITAL
    equity = np.zeros(n_days)
    traded_notional = np.zeros(n_days)
    trades = np.zeros(n_days)
    invested_pct = np.zeros(n_days)

    for i in range(n_days):
        curr_opens = open_df.iloc[i].values
        curr_closes = close_df.iloc[i].values

        if is_new_month[i]:
            # Current open value
            port_val = cash + np.sum(shares * curr_opens)
            # Allocate equity after reserving buffer for transaction friction
            investable_equity = port_val / (1.0 + TOTAL_FRICTION)
            target_val_per_stock = investable_equity * weight_target
            target_shares = target_val_per_stock / curr_opens
            delta_shares = target_shares - shares
            trade_vals = np.abs(delta_shares) * curr_opens
            total_trade = np.sum(trade_vals)
            friction = total_trade * TOTAL_FRICTION

            cash = port_val - np.sum(target_shares * curr_opens) - friction
            shares = target_shares.copy()
            traded_notional[i] = total_trade
            trades[i] = np.sum(trade_vals > 1e-4)

        holdings_val = np.sum(shares * curr_closes)
        eq = cash + holdings_val
        equity[i] = eq
        invested_pct[i] = (holdings_val / eq) * 100.0 if eq > 0 else 0.0

    return pd.DataFrame(
        {
            "equity": equity,
            "invested_pct": invested_pct,
            "traded_notional": traded_notional,
            "trades": trades,
        },
        index=dates,
    )


def simulate_12_1_momentum(
    universe: List[str], data_dict: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    12-1 momentum: at each month-end rank stocks by return from t-252 to t-21 trading days,
    hold the top 10 equal-weight, rebalance monthly at Open.
    """
    n_days = len(dates)
    # Pre-extract full close series for all tickers to calculate lookback returns
    spy_full_dates = data_dict["SPY"].loc[:END_DATE].index
    full_close = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in universe}, index=spy_full_dates
    ).ffill()
    open_sub = pd.DataFrame(
        {t: data_dict[t]["Open"] for t in universe}, index=dates
    ).ffill()
    close_sub = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in universe}, index=dates
    ).ffill()

    month_ids = dates.to_period("M")
    is_new_month = np.concatenate([[True], month_ids[1:] != month_ids[:-1]])

    current_holdings: Dict[str, float] = {}  # ticker -> shares
    cash = INITIAL_CAPITAL
    equity = np.zeros(n_days)
    traded_notional = np.zeros(n_days)
    trades = np.zeros(n_days)
    invested_pct = np.zeros(n_days)

    for i in range(n_days):
        curr_date = dates[i]

        if is_new_month[i]:
            # Locate decision day (previous trading day)
            loc_full = full_close.index.get_loc(curr_date)
            decision_idx = loc_full - 1
            t21_idx = decision_idx - 21
            t252_idx = decision_idx - 252

            if t252_idx >= 0 and t21_idx >= 0:
                c21 = full_close.iloc[t21_idx]
                c252 = full_close.iloc[t252_idx]
                mom_returns = ((c21 / c252) - 1.0).dropna()
                top10 = mom_returns.nlargest(10).index.tolist()
            else:
                top10 = (
                    list(current_holdings.keys()) if current_holdings else universe[:10]
                )

            # Valuation at today's open
            port_val = cash
            for t_held, s_held in current_holdings.items():
                port_val += s_held * open_sub.loc[curr_date, t_held]

            investable_equity = port_val / (1.0 + TOTAL_FRICTION)
            target_val_per_stock = investable_equity * 0.10
            new_holdings = {}
            total_traded = 0.0
            trade_count = 0

            # Sell positions not in top 10
            for t_held, s_held in current_holdings.items():
                if t_held not in top10:
                    val = s_held * open_sub.loc[curr_date, t_held]
                    total_traded += val
                    trade_count += 1
                    cash += val

            # Buy / adjust top 10 positions
            for t_top in top10:
                op = open_sub.loc[curr_date, t_top]
                tgt_s = target_val_per_stock / op
                curr_s = current_holdings.get(t_top, 0.0)
                diff_s = tgt_s - curr_s
                val = abs(diff_s) * op
                total_traded += val
                if val > 1e-4:
                    trade_count += 1
                new_holdings[t_top] = tgt_s

            friction = total_traded * TOTAL_FRICTION
            cash = port_val - (10 * target_val_per_stock) - friction
            current_holdings = new_holdings
            traded_notional[i] = total_traded
            trades[i] = trade_count

        # Daily closing valuation
        holdings_val = sum(
            s * close_sub.loc[curr_date, t] for t, s in current_holdings.items()
        )
        eq = cash + holdings_val
        equity[i] = eq
        invested_pct[i] = (holdings_val / eq) * 100.0 if eq > 0 else 0.0

    return pd.DataFrame(
        {
            "equity": equity,
            "invested_pct": invested_pct,
            "traded_notional": traded_notional,
            "trades": trades,
        },
        index=dates,
    )


def simulate_random_selection_runs(
    universe: List[str],
    data_dict: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    seeds: range = range(42, 92),
) -> List[pd.DataFrame]:
    """
    Random selection: each month hold 10 randomly chosen stocks equal-weight,
    50 runs with seeds 42..91.
    """
    cache_path = CACHE_DIR / "random_selection_runs.parquet"
    if cache_path.exists():
        try:
            full_df = pd.read_parquet(cache_path)
            res_list = []
            for s in seeds:
                cols = [f"equity_{s}", f"invested_{s}", f"notional_{s}", f"trades_{s}"]
                if all(c in full_df.columns for c in cols):
                    df_s = pd.DataFrame(
                        {
                            "equity": full_df[f"equity_{s}"],
                            "invested_pct": full_df[f"invested_{s}"],
                            "traded_notional": full_df[f"notional_{s}"],
                            "trades": full_df[f"trades_{s}"],
                        },
                        index=full_df.index,
                    )
                    res_list.append(df_s)
            if len(res_list) == len(seeds):
                print(f"Resumed {len(res_list)} random selection runs from cache.")
                return res_list
        except Exception:
            pass

    results = []
    n_days = len(dates)
    open_sub = pd.DataFrame(
        {t: data_dict[t]["Open"] for t in universe}, index=dates
    ).ffill()
    close_sub = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in universe}, index=dates
    ).ffill()
    month_ids = dates.to_period("M")
    is_new_month = np.concatenate([[True], month_ids[1:] != month_ids[:-1]])

    save_dict = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        current_holdings: Dict[str, float] = {}
        cash = INITIAL_CAPITAL
        equity = np.zeros(n_days)
        traded_notional = np.zeros(n_days)
        trades = np.zeros(n_days)
        invested_pct = np.zeros(n_days)

        for i in range(n_days):
            curr_date = dates[i]

            if is_new_month[i]:
                # Sample 10 random stocks from universe
                chosen10 = rng.choice(universe, size=10, replace=False).tolist()

                # Valuation at Open
                port_val = cash
                for t_held, s_held in current_holdings.items():
                    port_val += s_held * open_sub.loc[curr_date, t_held]

                investable_equity = port_val / (1.0 + TOTAL_FRICTION)
                target_val = investable_equity * 0.10
                new_holdings = {}
                total_traded = 0.0
                trade_count = 0

                for t_held, s_held in current_holdings.items():
                    if t_held not in chosen10:
                        val = s_held * open_sub.loc[curr_date, t_held]
                        total_traded += val
                        trade_count += 1
                        cash += val

                for t_ch in chosen10:
                    op = open_sub.loc[curr_date, t_ch]
                    tgt_s = target_val / op
                    curr_s = current_holdings.get(t_ch, 0.0)
                    diff_s = tgt_s - curr_s
                    val = abs(diff_s) * op
                    total_traded += val
                    if val > 1e-4:
                        trade_count += 1
                    new_holdings[t_ch] = tgt_s

                friction = total_traded * TOTAL_FRICTION
                cash = port_val - (10 * target_val) - friction
                current_holdings = new_holdings
                traded_notional[i] = total_traded
                trades[i] = trade_count

            holdings_val = sum(
                s * close_sub.loc[curr_date, t] for t, s in current_holdings.items()
            )
            eq = cash + holdings_val
            equity[i] = eq
            invested_pct[i] = (holdings_val / eq) * 100.0 if eq > 0 else 0.0

        df_run = pd.DataFrame(
            {
                "equity": equity,
                "invested_pct": invested_pct,
                "traded_notional": traded_notional,
                "trades": trades,
            },
            index=dates,
        )
        results.append(df_run)

        save_dict[f"equity_{seed}"] = equity
        save_dict[f"invested_{seed}"] = invested_pct
        save_dict[f"notional_{seed}"] = traded_notional
        save_dict[f"trades_{seed}"] = trades

    try:
        pd.DataFrame(save_dict, index=dates).to_parquet(cache_path)
    except Exception:
        pass

    return results


def simulate_sector_etf_momentum(
    data_dict: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, List[str], List[Tuple[str, str]]]:
    """
    (g) Sector ETF momentum: 11 SPDR sector ETFs, drop any lacking complete history,
    hold top 3 by 12-1 momentum monthly.
    """
    sector_etfs_all = [
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    ]
    dropped_sectors = []
    valid_sectors = []
    for t in sector_etfs_all:
        df = data_dict[t]
        if df.index[0] > pd.Timestamp("2018-01-05") or df.index[-1] < pd.Timestamp(
            "2026-10-20"
        ):
            dropped_sectors.append((t, f"Inception {df.index[0].date()}"))
        else:
            valid_sectors.append(t)

    n_days = len(dates)
    spy_full_dates = data_dict["SPY"].loc[:END_DATE].index
    full_close_sec = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in valid_sectors}, index=spy_full_dates
    ).ffill()
    open_sec = pd.DataFrame(
        {t: data_dict[t]["Open"] for t in valid_sectors}, index=dates
    ).ffill()
    close_sec = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in valid_sectors}, index=dates
    ).ffill()

    month_ids = dates.to_period("M")
    is_new_month = np.concatenate([[True], month_ids[1:] != month_ids[:-1]])

    current_holdings = {}
    cash = INITIAL_CAPITAL
    equity = np.zeros(n_days)
    invested = np.zeros(n_days)
    notional = np.zeros(n_days)
    trades = np.zeros(n_days)

    for i in range(n_days):
        curr_date = dates[i]
        if is_new_month[i]:
            loc_full = full_close_sec.index.get_loc(curr_date)
            decision_idx = loc_full - 1
            t21_idx = decision_idx - 21
            t252_idx = decision_idx - 252

            if t252_idx >= 0 and t21_idx >= 0:
                c21 = full_close_sec.iloc[t21_idx]
                c252 = full_close_sec.iloc[t252_idx]
                mom_returns = ((c21 / c252) - 1.0).dropna()
                top3 = mom_returns.nlargest(3).index.tolist()
            else:
                top3 = valid_sectors[:3]

            port_val = cash
            for t_h, s_h in current_holdings.items():
                port_val += s_h * open_sec.loc[curr_date, t_h]

            investable = port_val / (1.0 + TOTAL_FRICTION)
            target_val = investable / 3.0
            new_holdings = {}
            total_traded = 0.0
            trade_count = 0

            for t_h, s_h in current_holdings.items():
                if t_h not in top3:
                    val = s_h * open_sec.loc[curr_date, t_h]
                    total_traded += val
                    trade_count += 1
                    cash += val

            for t_top in top3:
                op = open_sec.loc[curr_date, t_top]
                tgt_s = target_val / op
                curr_s = current_holdings.get(t_top, 0.0)
                diff_s = tgt_s - curr_s
                val = abs(diff_s) * op
                total_traded += val
                if val > 1e-4:
                    trade_count += 1
                new_holdings[t_top] = tgt_s

            friction = total_traded * TOTAL_FRICTION
            cash = port_val - (3.0 * target_val) - friction
            current_holdings = new_holdings
            notional[i] = total_traded
            trades[i] = trade_count

        h_val = sum(
            s * close_sec.loc[curr_date, t] for t, s in current_holdings.items()
        )
        eq = cash + h_val
        equity[i] = eq
        invested[i] = (h_val / eq) * 100.0 if eq > 0 else 0.0

    df_g = pd.DataFrame(
        {
            "equity": equity,
            "invested_pct": invested,
            "traded_notional": notional,
            "trades": trades,
        },
        index=dates,
    )
    return df_g, valid_sectors, dropped_sectors


def simulate_multi_asset_trend(
    data_dict: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    (h) Multi-asset trend: SPY, EFA, EEM, AGG, GLD, each held at equal weight (20%)
    when above its 200-day SMA, otherwise cash.
    """
    multi_asset = ["SPY", "EFA", "EEM", "AGG", "GLD"]
    n_days = len(dates)
    spy_full_dates = data_dict["SPY"].loc[:END_DATE].index
    full_close_ma = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in multi_asset}, index=spy_full_dates
    ).ffill()
    open_ma = pd.DataFrame(
        {t: data_dict[t]["Open"] for t in multi_asset}, index=dates
    ).ffill()
    close_ma = pd.DataFrame(
        {t: data_dict[t]["Close"] for t in multi_asset}, index=dates
    ).ffill()

    sma200_ma = full_close_ma.rolling(200).mean()

    current_holdings = {t: 0.0 for t in multi_asset}
    cash = INITIAL_CAPITAL
    equity = np.zeros(n_days)
    invested = np.zeros(n_days)
    notional = np.zeros(n_days)
    trades = np.zeros(n_days)

    for i in range(n_days):
        curr_date = dates[i]
        loc_full = full_close_ma.index.get_loc(curr_date)
        decision_idx = loc_full - 1

        target_weights = {}
        for t in multi_asset:
            p_c = full_close_ma.loc[full_close_ma.index[decision_idx], t]
            p_sma = sma200_ma.loc[sma200_ma.index[decision_idx], t]
            target_weights[t] = 0.20 if (not np.isnan(p_sma) and p_c > p_sma) else 0.0

        port_val = cash
        for t in multi_asset:
            port_val += current_holdings[t] * open_ma.loc[curr_date, t]

        investable = port_val / (1.0 + TOTAL_FRICTION)
        total_traded = 0.0
        trade_count = 0
        new_holdings = {}

        for t in multi_asset:
            tgt_val = investable * target_weights[t]
            op = open_ma.loc[curr_date, t]
            tgt_s = tgt_val / op if op > 0 else 0.0
            curr_s = current_holdings[t]
            diff_s = tgt_s - curr_s
            val = abs(diff_s) * op
            total_traded += val
            if val > 1e-4:
                trade_count += 1
            new_holdings[t] = tgt_s

        friction = total_traded * TOTAL_FRICTION
        cash = (
            port_val
            - sum(new_holdings[t] * open_ma.loc[curr_date, t] for t in multi_asset)
            - friction
        )
        current_holdings = new_holdings
        notional[i] = total_traded
        trades[i] = trade_count

        h_val = sum(
            current_holdings[t] * close_ma.loc[curr_date, t] for t in multi_asset
        )
        eq = cash + h_val
        equity[i] = eq
        invested[i] = (h_val / eq) * 100.0 if eq > 0 else 0.0

    return pd.DataFrame(
        {
            "equity": equity,
            "invested_pct": invested,
            "traded_notional": notional,
            "trades": trades,
        },
        index=dates,
    )


# -----------------------------------------------------------------------------
# 4. Existing ML Strategy (Unchanged Code & Calls)
# -----------------------------------------------------------------------------
def run_existing_ml_strategy(
    universe: List[str], data_dict: Dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the existing ML strategy on 30 random tickers (seed 42) with zero sentiment.
    Uses UNCHANGED src/modeling.py train_model and src/backtesting.py run_backtest.
    Returns:
      (ml_per_ticker_df, ml_portfolio_combined, bh_portfolio_combined)
    """
    intermediate_path = CACHE_DIR / "ml_intermediate.json"
    rng = np.random.default_rng(RANDOM_SEED)
    selected_30 = sorted(rng.choice(sorted(universe), size=30, replace=False).tolist())
    print(f"Selected 30 tickers for ML strategy (seed 42): {selected_30}")

    intermediate_data = {}
    if intermediate_path.exists():
        try:
            with open(intermediate_path, "r", encoding="utf-8") as f:
                intermediate_data = json.load(f)
        except Exception:
            intermediate_data = {}

    ml_portfolios = {}
    bh_portfolios = {}
    rows = []

    # Get VIX data if available
    vix_df = data_dict.get("^VIX", None)

    for idx, ticker in enumerate(selected_30):
        ticker_cache_file = CACHE_DIR / f"ml_port_{ticker}.parquet"
        if ticker_cache_file.exists() and ticker in intermediate_data:
            print(f"[{idx+1}/30] Resuming cached ML model for {ticker}...")
            cached_port = pd.read_parquet(ticker_cache_file)
            ml_eq = cached_port["total"].loc[dates.intersection(cached_port.index)]
            bh_eq = cached_port["benchmark"].loc[dates.intersection(cached_port.index)]
            ml_portfolios[ticker] = ml_eq
            bh_portfolios[ticker] = bh_eq
            rows.append(intermediate_data[ticker])
            continue

        print(f"[{idx+1}/30] Processing ML model for {ticker}...")
        df = data_dict[ticker].copy()
        ph_ind = create_technical_indicators(df)

        # Set sentiment features to 0.0 (state plainly that no real news history exists)
        daily_sentiment = pd.DataFrame(
            {
                "mean_sentiment_score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            },
            index=df.index,
        )

        feats = (
            create_features(ph_ind, daily_sentiment, vix_data=vix_df)
            .dropna()
            .sort_index()
        )
        # Filter to target date range
        feats_sub = feats.loc[:END_DATE]
        X = pd.DataFrame(feats_sub[FEATURES])
        y = feats_sub["target"]

        # Call repo's UNCHANGED modeling.train_model
        model, train_metrics, oos_predictions = train_model(X, y)

        # Call repo's UNCHANGED backtesting.run_backtest
        test_price_history = pd.DataFrame(ph_ind.loc[oos_predictions.index])
        portfolio, backtest_metrics, _ = run_backtest(
            price_history=test_price_history, prediction_probs=oos_predictions
        )

        # Save individual portfolio for resume capability
        portfolio[["total", "benchmark"]].to_parquet(ticker_cache_file)

        ml_eq = portfolio["total"].loc[dates.intersection(portfolio.index)]
        bh_eq = portfolio["benchmark"].loc[dates.intersection(portfolio.index)]
        ml_portfolios[ticker] = ml_eq
        bh_portfolios[ticker] = bh_eq

        # Ticker level metrics
        ml_tot = (ml_eq.iloc[-1] / ml_eq.iloc[0]) - 1.0
        bh_tot = (bh_eq.iloc[-1] / bh_eq.iloc[0]) - 1.0
        n_days = len(ml_eq)
        ml_cagr = (1.0 + ml_tot) ** (252.0 / n_days) - 1.0
        bh_cagr = (1.0 + bh_tot) ** (252.0 / n_days) - 1.0
        ml_rets = ml_eq.pct_change().fillna(0)
        bh_rets = bh_eq.pct_change().fillna(0)
        ml_sr = (ml_rets.mean() / (ml_rets.std() + 1e-12)) * np.sqrt(252)
        bh_sr = (bh_rets.mean() / (bh_rets.std() + 1e-12)) * np.sqrt(252)
        ml_mdd = calculate_max_drawdown(ml_eq)
        bh_mdd = calculate_max_drawdown(bh_eq)

        row = {
            "Ticker": ticker,
            "ML Total Return": ml_tot,
            "B&H Total Return": bh_tot,
            "ML CAGR": ml_cagr,
            "B&H CAGR": bh_cagr,
            "ML Sharpe": ml_sr,
            "B&H Sharpe": bh_sr,
            "ML Max Drawdown": ml_mdd,
            "B&H Max Drawdown": bh_mdd,
            "ML Total Trades": backtest_metrics.get("total_trades", 0),
            "ML Win Rate": backtest_metrics.get("win_rate", 0.0),
        }
        rows.append(row)
        intermediate_data[ticker] = row
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_data, f, indent=2)

    ml_per_ticker_df = pd.DataFrame(rows)

    # Combine into 30-ticker equal-weighted equity curves
    ml_matrix = pd.DataFrame(ml_portfolios).loc[dates].ffill().bfill()
    bh_matrix = pd.DataFrame(bh_portfolios).loc[dates].ffill().bfill()

    ml_daily_returns = ml_matrix.pct_change().fillna(0).mean(axis=1)
    bh_daily_returns = bh_matrix.pct_change().fillna(0).mean(axis=1)

    ml_combined_eq = INITIAL_CAPITAL * (1.0 + ml_daily_returns).cumprod()
    bh_combined_eq = INITIAL_CAPITAL * (1.0 + bh_daily_returns).cumprod()

    ml_port = pd.DataFrame(
        {
            "equity": ml_combined_eq,
            "invested_pct": 100.0,  # or average of individual invested
            "traded_notional": np.zeros(len(dates)),
            "trades": np.zeros(len(dates)),
        },
        index=dates,
    )

    bh_port = pd.DataFrame(
        {
            "equity": bh_combined_eq,
            "invested_pct": 100.0,
            "traded_notional": np.zeros(len(dates)),
            "trades": np.zeros(len(dates)),
        },
        index=dates,
    )

    return ml_per_ticker_df, ml_port, bh_port


# -----------------------------------------------------------------------------
# 5. Metrics & Statistical Auditing (Bootstrap, OLS, DSR)
# -----------------------------------------------------------------------------
def block_bootstrap_sharpe(
    strat_returns: np.ndarray,
    spy_returns: np.ndarray,
    block_size: int = 21,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Stationary block bootstrap for Sharpe ratio and Sharpe difference vs SPY.
    Block size = 21 days, 2000 resamples, fixed seed 42.
    """
    T = len(strat_returns)
    K = int(math.ceil(T / block_size))
    rng = np.random.default_rng(seed)

    block_starts = rng.integers(0, T - block_size + 1, size=(n_boot, K))
    offsets = np.arange(block_size)
    indices = (block_starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[
        :, :T
    ]

    strat_samples = strat_returns[indices]
    spy_samples = spy_returns[indices]

    s_mean = np.mean(strat_samples, axis=1)
    s_std = np.std(strat_samples, axis=1, ddof=1)
    spy_mean = np.mean(spy_samples, axis=1)
    spy_std = np.std(spy_samples, axis=1, ddof=1)

    sr_strat = np.where(s_std > 1e-12, (s_mean / s_std) * np.sqrt(252), 0.0)
    sr_spy = np.where(spy_std > 1e-12, (spy_mean / spy_std) * np.sqrt(252), 0.0)
    sr_diff = sr_strat - sr_spy

    ci_strat = (
        float(np.percentile(sr_strat, 2.5)),
        float(np.percentile(sr_strat, 97.5)),
    )
    ci_diff = (float(np.percentile(sr_diff, 2.5)), float(np.percentile(sr_diff, 97.5)))
    return ci_strat, ci_diff


def compute_all_metrics(
    equity_series: pd.Series,
    spy_equity_series: pd.Series,
    invested_pct_series: pd.Series,
    traded_notional_series: pd.Series,
    trades_series: pd.Series,
    n_trials: int = 6,
    var_sharpe_trials: float = 0.05,
    benchmark_sharpe: float = 0.0,
) -> Dict[str, Any]:
    """
    Computes all audited performance, regression, bootstrap, and DSR metrics for a period.
    """
    n_days = len(equity_series)
    if n_days < 5:
        return {}

    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    cagr = (1.0 + total_return) ** (252.0 / n_days) - 1.0

    daily_returns = equity_series.pct_change().fillna(0).values
    spy_daily_returns = spy_equity_series.pct_change().fillna(0).values

    ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
    mean_ret = float(np.mean(daily_returns))
    sharpe = float((mean_ret / (np.std(daily_returns, ddof=1) + 1e-12)) * np.sqrt(252))

    max_dd = calculate_max_drawdown(equity_series)

    # Annual Turnover
    avg_equity = float(equity_series.mean())
    total_traded = float(traded_notional_series.sum())
    annual_turnover = (total_traded / 2.0 / (avg_equity + 1e-12)) * (252.0 / n_days)
    total_trades = int(trades_series.sum())
    avg_invested = float(invested_pct_series.mean())

    # Excess return vs SPY
    spy_tot_ret = (spy_equity_series.iloc[-1] / spy_equity_series.iloc[0]) - 1.0
    excess_return = total_return - spy_tot_ret

    # Regression Alpha and Beta
    reg = stats.linregress(spy_daily_returns, daily_returns)
    beta = float(reg.slope)
    beta_tstat = float(reg.slope / (reg.stderr + 1e-12))
    alpha_daily = float(reg.intercept)
    alpha_ann = float(alpha_daily * 252.0)
    alpha_tstat = float(reg.intercept / (reg.intercept_stderr + 1e-12))

    # Bootstrap 95% CIs
    ci_strat, ci_diff = block_bootstrap_sharpe(daily_returns, spy_daily_returns)

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe (rf=0)": sharpe,
        "Max Drawdown": max_dd,
        "Annual Turnover": annual_turnover,
        "Number of Trades": total_trades,
        "Average % Invested": avg_invested,
        "Excess Return vs SPY": excess_return,
        "Alpha (Ann)": alpha_ann,
        "Alpha t-stat": alpha_tstat,
        "Beta": beta,
        "Beta t-stat": beta_tstat,
        "Sharpe 95% CI Lower": ci_strat[0],
        "Sharpe 95% CI Upper": ci_strat[1],
        "Sharpe Diff 95% CI Lower": ci_diff[0],
        "Sharpe Diff 95% CI Upper": ci_diff[1],
    }


def format_summary_table(metrics_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats the raw dictionary of strategy metrics into a clean presentation DataFrame.
    """
    df = pd.DataFrame(metrics_dict).T
    return df


# -----------------------------------------------------------------------------
# 6. Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("SENTILYZE QUANTITATIVE BASELINES EXPERIMENT")
    print("Branch: research-baselines | Seed: 42")
    print("=" * 80)

    # Step 1: Unit Tests
    print("\n--- RUNNING UNIT TESTS FOR MAX DRAWDOWN ---")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    spy_full = get_or_fetch_ticker("SPY", session)
    if spy_full is None:
        raise RuntimeError("SPY download failed")
    ut_res = run_max_drawdown_unit_tests(spy_full["Close"])
    print(
        f"Test 1 [100, 50, 75]: {ut_res['test1_value']:.4f} (Expected: -0.5000) -> PASSED: {ut_res['test1_passed']}"
    )
    print(
        f"Test 2 SPY COVID 2020-03-23: {ut_res['test2_value']:.4f} (Expected: ~ -0.3372) -> PASSED: {ut_res['test2_passed']}"
    )

    # Step 2: Build Universe
    print("\n--- BUILDING UNIVERSE FROM stocks.txt ---")
    included_tickers, excluded_tickers, data_dict = build_universe("stocks.txt")
    print(
        f"Total Tickers in stocks.txt: {len(included_tickers) + len(excluded_tickers)}"
    )
    print(f"Included Tickers (Full 2018-2026 data): {len(included_tickers)}")
    print(f"Excluded Tickers: {len(excluded_tickers)}")
    print("\nSURVIVORSHIP BIAS STATEMENT:")
    print(
        "The universe is composed of today's constituents from stocks.txt, so survivorship bias"
    )
    print(
        "inflates all stock strategies (stocks that went bankrupt, were acquired, or were delisted"
    )
    print("between 2018 and 2026 are not represented).")

    # Step 3: Align Dates
    spy_sub = data_dict["SPY"].loc[START_DATE:END_DATE]
    trading_dates = spy_sub.index
    print(
        f"\nTrading Days: {len(trading_dates)} from {trading_dates[0].date()} to {trading_dates[-1].date()}"
    )

    # Step 4: Run Baseline Strategies
    print("\n--- SIMULATING STRATEGY A: SPY BUY-AND-HOLD ---")
    strat_a = simulate_spy_buy_and_hold(data_dict["SPY"], trading_dates)

    print("\n--- SIMULATING STRATEGY B: EQUAL-WEIGHT UNIVERSE (MONTHLY REBAL) ---")
    strat_b = simulate_equal_weight_universe(included_tickers, data_dict, trading_dates)

    print("\n--- SIMULATING STRATEGY C: 12-1 MOMENTUM (TOP 10, MONTHLY REBAL) ---")
    strat_c = simulate_12_1_momentum(included_tickers, data_dict, trading_dates)

    print("\n--- SIMULATING STRATEGY D: SPY 200-DAY TREND FILTER ---")
    strat_d = simulate_spy_200_trend(data_dict["SPY"], trading_dates)

    print("\n--- SIMULATING STRATEGY E: RANDOM SELECTION (50 RUNS, SEEDS 42..91) ---")
    strat_e_runs = simulate_random_selection_runs(
        included_tickers, data_dict, trading_dates
    )

    print("\n--- SIMULATING STRATEGY F: EXISTING ML STRATEGY (30 TICKERS, SEED 42) ---")
    ml_per_ticker_df, strat_f_ml, strat_f_bh = run_existing_ml_strategy(
        included_tickers, data_dict, trading_dates
    )

    print("\n--- SIMULATING STRATEGY G: SECTOR ETF MOMENTUM (TOP 3) ---")
    strat_g, valid_sectors, dropped_sectors = simulate_sector_etf_momentum(
        data_dict, trading_dates
    )
    print(
        f"Sector ETFs - Dropped: {dropped_sectors}, Valid ({len(valid_sectors)}): {valid_sectors}"
    )

    print("\n--- SIMULATING STRATEGY H: MULTI-ASSET TREND (5 ETFS) ---")
    strat_h = simulate_multi_asset_trend(data_dict, trading_dates)

    # Step 5: Save ML per ticker CSV
    ml_per_ticker_path = OUTPUT_DIR / "ml_per_ticker.csv"
    ml_per_ticker_df.to_csv(ml_per_ticker_path, index=False)
    print(f"\nSaved ML per-ticker results to: {ml_per_ticker_path}")

    # Step 6: Compute Multi-Period Metrics
    periods = {
        "full": (trading_dates[0], trading_dates[-1]),
        "in_sample": (pd.Timestamp(START_DATE), pd.Timestamp(IN_SAMPLE_END)),
        "out_of_sample": (pd.Timestamp(OUT_OF_SAMPLE_START), trading_dates[-1]),
    }

    summary_tables = {}

    for p_name, (p_start, p_end) in periods.items():
        p_dates = trading_dates[(trading_dates >= p_start) & (trading_dates <= p_end)]
        spy_eq_p = strat_a.loc[p_dates, "equity"]

        p_metrics = {}

        # a) SPY B&H
        p_metrics["SPY Buy & Hold"] = compute_all_metrics(
            strat_a.loc[p_dates, "equity"],
            spy_eq_p,
            strat_a.loc[p_dates, "invested_pct"],
            strat_a.loc[p_dates, "traded_notional"],
            strat_a.loc[p_dates, "trades"],
        )
        # b) Equal-Weight Universe
        p_metrics["Equal-Weight Universe"] = compute_all_metrics(
            strat_b.loc[p_dates, "equity"],
            spy_eq_p,
            strat_b.loc[p_dates, "invested_pct"],
            strat_b.loc[p_dates, "traded_notional"],
            strat_b.loc[p_dates, "trades"],
        )
        # c) 12-1 Momentum
        p_metrics["12-1 Momentum (Top 10)"] = compute_all_metrics(
            strat_c.loc[p_dates, "equity"],
            spy_eq_p,
            strat_c.loc[p_dates, "invested_pct"],
            strat_c.loc[p_dates, "traded_notional"],
            strat_c.loc[p_dates, "trades"],
        )
        # d) SPY 200-day Trend Filter
        p_metrics["SPY 200-Day Trend Filter"] = compute_all_metrics(
            strat_d.loc[p_dates, "equity"],
            spy_eq_p,
            strat_d.loc[p_dates, "invested_pct"],
            strat_d.loc[p_dates, "traded_notional"],
            strat_d.loc[p_dates, "trades"],
        )

        # e) Random Selection: evaluate all 50 runs for this period
        rand_metrics_list = [
            compute_all_metrics(
                run_df.loc[p_dates, "equity"],
                spy_eq_p,
                run_df.loc[p_dates, "invested_pct"],
                run_df.loc[p_dates, "traded_notional"],
                run_df.loc[p_dates, "trades"],
            )
            for run_df in strat_e_runs
        ]
        # Aggregate Random Selection into Mean, 5th, 95th pct
        keys = list(rand_metrics_list[0].keys())
        rand_mean = {}
        rand_p05 = {}
        rand_p95 = {}
        for k in keys:
            vals = [m[k] for m in rand_metrics_list if not np.isnan(m[k])]
            rand_mean[k] = float(np.mean(vals)) if vals else float("nan")
            rand_p05[k] = float(np.percentile(vals, 5)) if vals else float("nan")
            rand_p95[k] = float(np.percentile(vals, 95)) if vals else float("nan")

        p_metrics["Random Selection (Mean)"] = rand_mean
        p_metrics["Random Selection (5th pct)"] = rand_p05
        p_metrics["Random Selection (95th pct)"] = rand_p95

        # f) Existing ML Strategy (30-Ticker EW)
        p_metrics["ML Strategy (30-Ticker EW)"] = compute_all_metrics(
            strat_f_ml.loc[p_dates, "equity"],
            spy_eq_p,
            strat_f_ml.loc[p_dates, "invested_pct"],
            strat_f_ml.loc[p_dates, "traded_notional"],
            strat_f_ml.loc[p_dates, "trades"],
        )
        p_metrics["30-Ticker Buy & Hold (EW)"] = compute_all_metrics(
            strat_f_bh.loc[p_dates, "equity"],
            spy_eq_p,
            strat_f_bh.loc[p_dates, "invested_pct"],
            strat_f_bh.loc[p_dates, "traded_notional"],
            strat_f_bh.loc[p_dates, "trades"],
        )

        # g) Sector ETF Momentum (Top 3)
        p_metrics["Sector ETF Momentum (Top 3)"] = compute_all_metrics(
            strat_g.loc[p_dates, "equity"],
            spy_eq_p,
            strat_g.loc[p_dates, "invested_pct"],
            strat_g.loc[p_dates, "traded_notional"],
            strat_g.loc[p_dates, "trades"],
        )

        # h) Multi-Asset Trend (5 ETFs)
        p_metrics["Multi-Asset Trend (5 ETFs)"] = compute_all_metrics(
            strat_h.loc[p_dates, "equity"],
            spy_eq_p,
            strat_h.loc[p_dates, "invested_pct"],
            strat_h.loc[p_dates, "traded_notional"],
            strat_h.loc[p_dates, "trades"],
        )

        table_df = format_summary_table(p_metrics)
        summary_tables[p_name] = table_df

        csv_path = OUTPUT_DIR / f"summary_{p_name}.csv"
        table_df.to_csv(csv_path)
        print(f"Saved {p_name} summary to {csv_path}")
        print(f"Saved {p_name} summary to {csv_path}")

    # Step 7: Generate RESULTS.md from CSVs
    results_md_path = OUTPUT_DIR / "RESULTS.md"
    generate_results_md(
        summary_tables,
        ml_per_ticker_df,
        len(included_tickers),
        len(excluded_tickers),
        results_md_path,
    )
    print(f"\nGenerated RESULTS.md at: {results_md_path}")


def df_to_markdown_custom(df: pd.DataFrame, include_index: bool = True) -> str:
    lines = []
    if include_index:
        cols = ["Strategy"] + list(df.columns)
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for idx, row in df.iterrows():
            vals = [str(idx)] + [
                (
                    f"{v:.4f}"
                    if isinstance(v, (float, np.floating)) and not np.isnan(v)
                    else (
                        "NaN"
                        if isinstance(v, (float, np.floating)) and np.isnan(v)
                        else str(v)
                    )
                )
                for v in row
            ]
            lines.append("| " + " | ".join(vals) + " |")
    else:
        cols = list(df.columns)
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = [
                (
                    f"{v:.4f}"
                    if isinstance(v, (float, np.floating)) and not np.isnan(v)
                    else (
                        "NaN"
                        if isinstance(v, (float, np.floating)) and np.isnan(v)
                        else str(v)
                    )
                )
                for v in row
            ]
            lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_results_md(
    summary_tables: Dict[str, pd.DataFrame],
    ml_per_ticker_df: pd.DataFrame,
    included_count: int,
    excluded_count: int,
    output_path: Path,
):
    """
    Generates research/outputs/RESULTS.md purely from computed dataframes.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Quantitative Baselines Experiment Results\n\n")
        f.write("## 1. Methodology & Disclosures\n\n")
        f.write(
            "- **Universe Composition & Survivorship Bias**: Universe constructed from `stocks.txt` "
        )
        f.write(
            f"({included_count} tickers included with complete history from 2018-01-02 to 2026-10-26, "
        )
        f.write(f"{excluded_count} excluded due to post-2018 IPO or missing history). ")
        f.write(
            "**State plainly that the universe is today's constituents, so survivorship bias inflates all stock strategies.**\n"
        )
        f.write(
            "- **Zero-Sentiment Limitation**: Sentiment features (`mean_sentiment_score`, `positive`, `negative`, `neutral`) "
        )
        f.write(
            "were set to 0.0 for the ML strategy because no real news history exists for this historical period.\n"
        )
        f.write(
            "- **Execution Friction & Transaction Costs**: All trades execute at the NEXT trading day's Open price based on signals "
        )
        f.write(
            "generated at the decision day Close. Costs: **0.10% broker fee + 0.05% slippage = 0.15% (0.0015)** applied to every buy and sell traded notional.\n"
        )
        f.write(
            "- **Survivorship-Free Baselines**: (g) Sector ETF Momentum (top 3 of 10 SPDR sector ETFs, XLC dropped due to mid-2018 inception); (h) Multi-Asset Trend (SPY, EFA, EEM, AGG, GLD 200-day SMA trend filter, 20% max weight each).\n"
        )
        f.write(
            "- **Deflated Sharpe Ratio (DSR) Audit**: The DSR column was removed from primary performance tables because the required input `var_sharpe` cannot be justified as an identical strategy trial variance across heterogeneous baseline regimes (see Item 7 audit).\n\n"
        )

        f.write("## 2. Summary Table - Full Period (2018 - 2026)\n\n")
        f.write(df_to_markdown_custom(summary_tables["full"], include_index=True))
        f.write("\n\n")

        f.write("## 3. Summary Table - In-Sample (2018 - 2022)\n\n")
        f.write(df_to_markdown_custom(summary_tables["in_sample"], include_index=True))
        f.write("\n\n")

        f.write("## 4. Summary Table - Out-of-Sample (2023 - 2026)\n\n")
        f.write(
            df_to_markdown_custom(summary_tables["out_of_sample"], include_index=True)
        )
        f.write("\n\n")

        f.write(
            "## 5. Existing ML Strategy Per-Ticker Breakdown (30 Tickers, Seed 42)\n\n"
        )
        f.write(df_to_markdown_custom(ml_per_ticker_df, include_index=False))
        f.write("\n")


if __name__ == "__main__":
    main()
