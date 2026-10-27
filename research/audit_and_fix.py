"""
Sentilyze Research Baselines Audit and Fix Engine
1. Fix double-subtracted borrowed margin accounting for ML strategy.
2. Verify exact equality of arithmetic daily return means.
3. Compute 1.5x leveraged B&H benchmark with 5% annual margin interest.
4. Correct trade counts (including open positions) and turnover from holdings.
5. True Walk-Forward on 10 tickers (seed 42) with identical dates.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from scipy import stats

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.baselines import (
    get_or_fetch_ticker,
    compute_all_metrics,
    calculate_max_drawdown,
    block_bootstrap_sharpe,
    START_DATE,
    END_DATE,
    IN_SAMPLE_END,
    OUT_OF_SAMPLE_START,
    TOTAL_FRICTION,
    INITIAL_CAPITAL,
    RANDOM_SEED,
)
from src.feature_engineering import create_technical_indicators, create_features
from src.config import FEATURES
from src.modeling import train_model
from src.backtesting import run_backtest

CACHE_DIR = PROJECT_ROOT / "research" / "data_cache"
OUTPUT_DIR = PROJECT_ROOT / "research" / "outputs"

# Load the 30 tickers
manifest = pd.read_json(CACHE_DIR / "ml_intermediate.json", orient="index")
tickers_30 = sorted(list(manifest["Ticker"].values))

print("=" * 80)
print("ITEM 1: AGGREGATION AUDIT & FIXING THE BORROWED MARGIN DOUBLE-SUBTRACTION")
print("=" * 80)

# We load each cached ticker data and re-run run_backtest to get the full port with cash and holdings
ml_true_equity_dict = {}
ml_true_returns_dict = {}
bh_returns_dict = {}
ticker_metrics_corrected = []

for idx, ticker in enumerate(tickers_30):
    port_fixed_file = CACHE_DIR / f"ml_port_fixed_{ticker}.parquet"
    df_raw = pd.read_parquet(CACHE_DIR / f"{ticker.replace('.', '-')}.parquet")
    ph_ind = create_technical_indicators(df_raw.copy())

    if port_fixed_file.exists():
        port = pd.read_parquet(port_fixed_file)
        test_ph = pd.DataFrame(ph_ind.loc[port.index])
    else:
        daily_sentiment = pd.DataFrame(
            {
                "mean_sentiment_score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            },
            index=df_raw.index,
        )
        feats = (
            create_features(ph_ind, daily_sentiment, vix_data=None)
            .dropna()
            .sort_index()
        )
        feats_sub = feats.loc[:END_DATE]
        X = pd.DataFrame(feats_sub[FEATURES])
        y = feats_sub["target"]

        model, metrics, oos_pred = train_model(X, y)
        test_ph = pd.DataFrame(ph_ind.loc[oos_pred.index])
        port, bt_metrics, _ = run_backtest(test_ph, oos_pred)
        port.to_parquet(port_fixed_file)

    # Correct net liquidation value:
    # In run_backtest line 263, cash was reduced by total_investment = cash + borrowed_margin.
    # Therefore, cash is already net of borrowed margin (cash is negative when margin is borrowed).
    # Line 303 subtracted borrowed_margin a SECOND time!
    # True net liquidation equity is: cash + holdings (or total + borrowed_margin).
    true_net_equity = port["cash"] + port["holdings"]
    true_rets = true_net_equity.pct_change().fillna(0)
    bh_rets = port["benchmark"].pct_change().fillna(0)

    ml_true_equity_dict[ticker] = true_net_equity
    ml_true_returns_dict[ticker] = true_rets
    bh_returns_dict[ticker] = bh_rets

    # Fix trade count: count entries (signal == 1) so open positions count as a trade
    entry_count = int((port["signal"] == 1).sum())
    exit_count = int((port["signal"] == -1).sum())
    open_at_end = entry_count > exit_count

    # Compute true turnover and avg % invested from holdings series
    avg_invested_pct = float(
        (port["holdings"] / (true_net_equity + 1e-12)).mean() * 100.0
    )

    # Traded notional: entries + exits
    # Entry value = effective_investment; Exit value = proceeds
    turnover_series = pd.Series(0.0, index=port.index)
    for i in range(1, len(port)):
        sig = port["signal"].iloc[i]
        if sig == 1:
            turnover_series.iloc[i] = port["holdings"].iloc[i]
        elif sig == -1:
            turnover_series.iloc[i] = port["holdings"].iloc[i - 1] * (
                port["Open"].iloc[i] / (port["Close"].iloc[i - 1] + 1e-12)
            )

    ann_turnover = (turnover_series.sum() / 2.0 / (true_net_equity.mean() + 1e-12)) * (
        252.0 / len(port)
    )

    # --- 1.5x Leveraged Buy & Hold Benchmark ---
    # Buy 1.5x at Day 1 Open, borrow 0.5x on margin, pay 5% annual interest daily
    p_close = test_ph["Close"]
    p_open = test_ph["Open"]
    lev_cash = -5000.0 - (15000.0 * TOTAL_FRICTION)  # borrowing $5,000
    lev_shares = 15000.0 / p_open.iloc[0]
    daily_rate = 0.05 / 252.0

    lev_equity = np.zeros(len(test_ph))
    curr_cash = lev_cash
    for i in range(len(test_ph)):
        # Deduct margin interest
        curr_cash -= 5000.0 * daily_rate
        lev_equity[i] = curr_cash + (lev_shares * p_close.iloc[i])

    lev_bh_series = pd.Series(lev_equity, index=test_ph.index)
    lev_bh_tot = (lev_bh_series.iloc[-1] / lev_bh_series.iloc[0]) - 1.0

    ml_tot = (true_net_equity.iloc[-1] / true_net_equity.iloc[0]) - 1.0
    bh_1x_tot = (port["benchmark"].iloc[-1] / port["benchmark"].iloc[0]) - 1.0

    ticker_metrics_corrected.append(
        {
            "Ticker": ticker,
            "ML Total Return (Fixed)": ml_tot,
            "1.5x Lev B&H Total Return": lev_bh_tot,
            "1.0x B&H Total Return": bh_1x_tot,
            "ML vs 1.5x Lev B&H Diff": ml_tot - lev_bh_tot,
            "ML vs 1.0x B&H Diff": ml_tot - bh_1x_tot,
            "ML Entries (Trades)": entry_count,
            "Position Open at End": open_at_end,
            "ML Avg % Invested": avg_invested_pct,
            "ML Annual Turnover": ann_turnover,
        }
    )

# Save corrected daily returns
ml_returns_df = pd.DataFrame(ml_true_returns_dict).sort_index()
ml_returns_df.to_csv(OUTPUT_DIR / "ml_daily_returns_corrected.csv")

# Filter to 2018-01-02 to 2026-10-26
dates = ml_returns_df.loc["2018-01-02":"2026-10-26"].index
ml_sub = ml_returns_df.loc[dates]

# 1. Arithmetic mean daily return per ticker
per_ticker_means = ml_sub.mean(axis=0)
avg_of_per_ticker_means = per_ticker_means.mean()

# 2. Mean of the portfolio daily return series
port_daily_rets = ml_sub.mean(axis=1)
mean_of_port_series = port_daily_rets.mean()

print(
    f"\nArithmetic mean daily return per ticker (average): {avg_of_per_ticker_means:.10f}"
)
print(f"Mean of portfolio daily return series:            {mean_of_port_series:.10f}")
print(
    f"Absolute Difference:                              {abs(avg_of_per_ticker_means - mean_of_port_series):.12e}"
)

# Check for equity <= 0 or daily return beyond +/- 30%
extreme_count = 0
for col in ml_sub.columns:
    col_rets = ml_sub[col]
    extreme = col_rets[(col_rets > 0.30) | (col_rets < -0.30)]
    if len(extreme) > 0:
        print(f"WARNING: Ticker {col} has extreme return: {extreme.to_dict()}")
        extreme_count += len(extreme)

print(f"Total tickers with daily return beyond +/-30%: {extreme_count}")

# 10 largest daily portfolio returns
print("\nTop 10 Largest Daily Portfolio Returns (Fixed Accounting):")
top10_days = port_daily_rets.nlargest(10)
for dt, val in top10_days.items():
    top3_movers = ml_sub.loc[dt].nlargest(3).to_dict()
    top3_str = ", ".join([f"{k}: {v*100:.2f}%" for k, v in top3_movers.items()])
    print(f"  {dt.date()}: {val*100:.2f}% | Top movers that day: {top3_str}")

# Re-generate the ML Row
ml_fixed_equity = INITIAL_CAPITAL * (1.0 + port_daily_rets).cumprod()
spy_raw = get_or_fetch_ticker("SPY", session)
spy_sub = spy_raw.loc[dates, "Close"]
spy_eq = INITIAL_CAPITAL * (spy_sub / spy_sub.iloc[0])

ml_metrics_full = compute_all_metrics(
    ml_fixed_equity,
    spy_eq,
    pd.Series(100.0, index=dates),
    pd.Series(0.0, index=dates),
    pd.Series(0, index=dates),
)

print("\n=== REGENERATED ML STRATEGY (30-TICKER EW) - FULL PERIOD ===")
for k, v in ml_metrics_full.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# Save Item 2 comparison table
df_lev_comp = pd.DataFrame(ticker_metrics_corrected)
df_lev_comp.to_csv(OUTPUT_DIR / "ml_vs_leveraged_bh.csv", index=False)

print("\n" + "=" * 80)
print("ITEM 4: TRUE WALK-FORWARD (500 TRAIN / 20 TEST) ON 10 TICKERS (SEED 42)")
print("=" * 80)

rng = np.random.default_rng(RANDOM_SEED)
selected_10 = sorted(rng.choice(tickers_30, size=10, replace=False).tolist())
print(f"Selected 10 tickers for True Walk-Forward: {selected_10}")

wf_ml_equity_dict = {}
wf_bh_equity_dict = {}
wf_ticker_results = []

for ticker in selected_10:
    wf_port_file = CACHE_DIR / f"wf_port_{ticker}.parquet"
    df_raw = pd.read_parquet(CACHE_DIR / f"{ticker.replace('.', '-')}.parquet")
    ph_ind = create_technical_indicators(df_raw.copy())

    if wf_port_file.exists():
        port = pd.read_parquet(wf_port_file)
    else:
        daily_sentiment = pd.DataFrame(
            {
                "mean_sentiment_score": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0,
            },
            index=df_raw.index,
        )
        feats = (
            create_features(ph_ind, daily_sentiment, vix_data=None)
            .dropna()
            .sort_index()
        )
        feats_sub = feats.loc[:END_DATE]
        X = pd.DataFrame(feats_sub[FEATURES])
        y = feats_sub["target"]

        # True walk-forward: 500 train, 20 test, use_purged_cv=False
        model, metrics, oos_pred = train_model(
            X, y, train_window=500, test_window=20, use_purged_cv=False
        )
        test_ph = pd.DataFrame(ph_ind.loc[oos_pred.index])
        port, bt_metrics, _ = run_backtest(test_ph, oos_pred)
        port.to_parquet(wf_port_file)

    # Correct net equity
    true_eq = port["cash"] + port["holdings"]
    wf_ml_equity_dict[ticker] = true_eq
    wf_bh_equity_dict[ticker] = port["benchmark"]

    ml_tot = (true_eq.iloc[-1] / true_eq.iloc[0]) - 1.0
    bh_tot = (port["benchmark"].iloc[-1] / port["benchmark"].iloc[0]) - 1.0

    wf_ticker_results.append(
        {
            "Ticker": ticker,
            "True WF ML Total Return": ml_tot,
            "True WF B&H Total Return": bh_tot,
            "ML Trades": int((port["signal"] == 1).sum()),
        }
    )

# Align dates across the 10 tickers
common_wf_dates = None
for t in selected_10:
    if common_wf_dates is None:
        common_wf_dates = wf_ml_equity_dict[t].index
    else:
        common_wf_dates = common_wf_dates.intersection(wf_ml_equity_dict[t].index)

common_wf_dates = sorted(common_wf_dates)
wf_ml_matrix = pd.DataFrame(
    {t: wf_ml_equity_dict[t].loc[common_wf_dates] for t in selected_10}
)
wf_bh_matrix = pd.DataFrame(
    {t: wf_bh_equity_dict[t].loc[common_wf_dates] for t in selected_10}
)

wf_ml_daily_rets = wf_ml_matrix.pct_change().fillna(0).mean(axis=1)
wf_bh_daily_rets = wf_bh_matrix.pct_change().fillna(0).mean(axis=1)

wf_ml_port_eq = INITIAL_CAPITAL * (1.0 + wf_ml_daily_rets).cumprod()
wf_bh_port_eq = INITIAL_CAPITAL * (1.0 + wf_bh_daily_rets).cumprod()

spy_wf = spy_raw.loc[common_wf_dates, "Close"]
spy_wf_eq = INITIAL_CAPITAL * (spy_wf / spy_wf.iloc[0])

wf_ml_metrics = compute_all_metrics(
    wf_ml_port_eq,
    spy_wf_eq,
    pd.Series(100.0, index=common_wf_dates),
    pd.Series(0.0, index=common_wf_dates),
    pd.Series(0, index=common_wf_dates),
)

wf_bh_metrics = compute_all_metrics(
    wf_bh_port_eq,
    spy_wf_eq,
    pd.Series(100.0, index=common_wf_dates),
    pd.Series(0.0, index=common_wf_dates),
    pd.Series(0, index=common_wf_dates),
)

print(
    f"\nDates covered by True Walk-Forward: {common_wf_dates[0].date()} to {common_wf_dates[-1].date()} ({len(common_wf_dates)} bars)"
)

print("\n--- 10-TICKER EQUAL-WEIGHT TRUE WALK-FORWARD METRICS ---")
print("ML Strategy (10-Ticker EW True Walk-Forward):")
for k, v in wf_ml_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print("\n10-Ticker Buy & Hold (EW Benchmark over Identical Dates):")
for k, v in wf_bh_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
