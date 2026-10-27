"""
Survivorship-Free Baselines:
(g) Sector ETF Momentum: 11 SPDR Sector ETFs (drop any lacking history), hold top 3 by 12-1 momentum monthly.
(h) Multi-Asset Trend: SPY, EFA, EEM, AGG, GLD, each held at equal weight (20%) when above its 200-day SMA, else cash.
Same costs (0.15% friction) and next-open fills, same metrics.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.baselines import (
    get_or_fetch_ticker,
    compute_all_metrics,
    calculate_max_drawdown,
    START_DATE,
    END_DATE,
    IN_SAMPLE_END,
    OUT_OF_SAMPLE_START,
    TOTAL_FRICTION,
    INITIAL_CAPITAL,
)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

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

data_dict = {}
for t in set(sector_etfs_all + multi_asset):
    data_dict[t] = get_or_fetch_ticker(t, session)

# Check sector ETF dates
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

print(f"Dropped Sector ETFs: {dropped_sectors}")
print(f"Valid Sector ETFs ({len(valid_sectors)}): {valid_sectors}")

dates = data_dict["SPY"].loc[START_DATE:END_DATE].index
n_days = len(dates)
spy_full_dates = data_dict["SPY"].loc[:END_DATE].index

# -----------------------------------------------------------------------------
# (g) Sector ETF Momentum (Top 3 of 10)
# -----------------------------------------------------------------------------
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

current_holdings_g = {}
cash_g = INITIAL_CAPITAL
equity_g = np.zeros(n_days)
invested_g = np.zeros(n_days)
notional_g = np.zeros(n_days)
trades_g = np.zeros(n_days)

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

        port_val = cash_g
        for t_h, s_h in current_holdings_g.items():
            port_val += s_h * open_sec.loc[curr_date, t_h]

        investable = port_val / (1.0 + TOTAL_FRICTION)
        target_val = investable / 3.0
        new_holdings = {}
        total_traded = 0.0
        trade_count = 0

        for t_h, s_h in current_holdings_g.items():
            if t_h not in top3:
                val = s_h * open_sec.loc[curr_date, t_h]
                total_traded += val
                trade_count += 1
                cash_g += val

        for t_top in top3:
            op = open_sec.loc[curr_date, t_top]
            tgt_s = target_val / op
            curr_s = current_holdings_g.get(t_top, 0.0)
            diff_s = tgt_s - curr_s
            val = abs(diff_s) * op
            total_traded += val
            if val > 1e-4:
                trade_count += 1
            new_holdings[t_top] = tgt_s

        friction = total_traded * TOTAL_FRICTION
        cash_g = port_val - (3.0 * target_val) - friction
        current_holdings_g = new_holdings
        notional_g[i] = total_traded
        trades_g[i] = trade_count

    h_val = sum(s * close_sec.loc[curr_date, t] for t, s in current_holdings_g.items())
    eq = cash_g + h_val
    equity_g[i] = eq
    invested_g[i] = (h_val / eq) * 100.0 if eq > 0 else 0.0

df_g = pd.DataFrame(
    {
        "equity": equity_g,
        "invested_pct": invested_g,
        "traded_notional": notional_g,
        "trades": trades_g,
    },
    index=dates,
)

# -----------------------------------------------------------------------------
# (h) Multi-Asset Trend (SPY, EFA, EEM, AGG, GLD)
# -----------------------------------------------------------------------------
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

current_holdings_h = {t: 0.0 for t in multi_asset}
cash_h = INITIAL_CAPITAL
equity_h = np.zeros(n_days)
invested_h = np.zeros(n_days)
notional_h = np.zeros(n_days)
trades_h = np.zeros(n_days)

for i in range(n_days):
    curr_date = dates[i]
    loc_full = full_close_ma.index.get_loc(curr_date)
    decision_idx = loc_full - 1

    target_weights = {}
    for t in multi_asset:
        p_c = full_close_ma.loc[full_close_ma.index[decision_idx], t]
        p_sma = sma200_ma.loc[sma200_ma.index[decision_idx], t]
        target_weights[t] = 0.20 if (not np.isnan(p_sma) and p_c > p_sma) else 0.0

    port_val = cash_h
    for t in multi_asset:
        port_val += current_holdings_h[t] * open_ma.loc[curr_date, t]

    investable = port_val / (1.0 + TOTAL_FRICTION)
    total_traded = 0.0
    trade_count = 0
    new_holdings_h = {}

    for t in multi_asset:
        tgt_val = investable * target_weights[t]
        op = open_ma.loc[curr_date, t]
        tgt_s = tgt_val / op if op > 0 else 0.0
        curr_s = current_holdings_h[t]
        diff_s = tgt_s - curr_s
        val = abs(diff_s) * op
        total_traded += val
        if val > 1e-4:
            trade_count += 1
        new_holdings_h[t] = tgt_s

    friction = total_traded * TOTAL_FRICTION
    cash_h = (
        port_val
        - sum(new_holdings_h[t] * open_ma.loc[curr_date, t] for t in multi_asset)
        - friction
    )
    current_holdings_h = new_holdings_h
    notional_h[i] = total_traded
    trades_h[i] = trade_count

    h_val = sum(current_holdings_h[t] * close_ma.loc[curr_date, t] for t in multi_asset)
    eq = cash_h + h_val
    equity_h[i] = eq
    invested_h[i] = (h_val / eq) * 100.0 if eq > 0 else 0.0

df_h = pd.DataFrame(
    {
        "equity": equity_h,
        "invested_pct": invested_h,
        "traded_notional": notional_h,
        "trades": trades_h,
    },
    index=dates,
)

spy_eq = data_dict["SPY"].loc[dates, "Close"]
spy_eq = 10000.0 * (spy_eq / spy_eq.iloc[0])

periods = {
    "Full (2018-2026)": (dates[0], dates[-1]),
    "In-Sample (2018-2022)": (pd.Timestamp(START_DATE), pd.Timestamp(IN_SAMPLE_END)),
    "Out-of-Sample (2023-2026)": (pd.Timestamp(OUT_OF_SAMPLE_START), dates[-1]),
}

results_dict = {}

for p_name, (p_start, p_end) in periods.items():
    p_dates = dates[(dates >= p_start) & (dates <= p_end)]
    s_eq = spy_eq.loc[p_dates]
    m_g = compute_all_metrics(
        df_g.loc[p_dates, "equity"],
        s_eq,
        df_g.loc[p_dates, "invested_pct"],
        df_g.loc[p_dates, "traded_notional"],
        df_g.loc[p_dates, "trades"],
    )
    m_h = compute_all_metrics(
        df_h.loc[p_dates, "equity"],
        s_eq,
        df_h.loc[p_dates, "invested_pct"],
        df_h.loc[p_dates, "traded_notional"],
        df_h.loc[p_dates, "trades"],
    )
    results_dict[p_name] = {
        "Sector Momentum (Top 3)": m_g,
        "Multi-Asset Trend (5 ETFs)": m_h,
    }

for p_name, r in results_dict.items():
    print(f"\n==================== {p_name} ====================")
    for strat_name, m in r.items():
        print(f"\nStrategy: {strat_name}")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
