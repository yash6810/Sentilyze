"""
Sentilyze Alpaca Performance Report & Benchmark Verification Engine.
Connects directly to Alpaca Brokerage API (source of truth), extracts
real paper execution orders and verified daily portfolio equity history,
and compares against the benchmark SPY rebased to starting balance.
"""

import os
import sys
import json
import datetime
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def generate_alpaca_report():
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = (
        os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        .rstrip("/")
        .removesuffix("/v2")
    )

    if not api_key or not secret_key:
        print("ERROR: Alpaca credentials missing from environment.")
        return

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }

    # 1. Fetch all orders
    orders_url = f"{base_url}/v2/orders?status=all&limit=500"
    orders_res = requests.get(orders_url, headers=headers)
    if orders_res.status_code != 200:
        print(f"ERROR fetching orders: {orders_res.status_code} {orders_res.text}")
        return

    orders_data = orders_res.json()
    order_rows = []
    for o in orders_data:
        order_rows.append(
            {
                "order_id": o.get("id"),
                "client_order_id": o.get("client_order_id"),
                "created_at": o.get("created_at"),
                "submitted_at": o.get("submitted_at"),
                "filled_at": o.get("filled_at"),
                "status": o.get("status"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": o.get("qty"),
                "filled_qty": o.get("filled_qty"),
                "filled_avg_price": o.get("filled_avg_price"),
                "order_type": o.get("order_type"),
                "time_in_force": o.get("time_in_force"),
            }
        )

    orders_df = pd.DataFrame(order_rows)
    os.makedirs("results", exist_ok=True)
    os.makedirs("audit_outputs", exist_ok=True)
    orders_df.to_csv("results/alpaca_orders.csv", index=False)
    print(f"Saved {len(orders_df)} orders to results/alpaca_orders.csv")

    # 2. Fetch daily equity history (all time)
    history_url = (
        f"{base_url}/v2/account/portfolio/history?period=all&timeframe=1D"
    )
    hist_res = requests.get(history_url, headers=headers)
    if hist_res.status_code != 200:
        print(f"ERROR fetching portfolio history: {hist_res.status_code}")
        return

    hist_data = hist_res.json()
    timestamps = hist_data.get("timestamp", [])
    equities = hist_data.get("equity", [])
    profit_loss = hist_data.get("profit_loss", [])
    profit_loss_pct = hist_data.get("profit_loss_pct", [])
    base_value = float(hist_data.get("base_value", 100000.0))

    hist_rows = []
    for ts, eq, pl, plp in zip(timestamps, equities, profit_loss, profit_loss_pct):
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%d"
        )
        hist_rows.append(
            {
                "date": dt,
                "timestamp": ts,
                "equity": float(eq) if eq is not None else np.nan,
                "profit_loss": float(pl) if pl is not None else 0.0,
                "profit_loss_pct": float(plp) if plp is not None else 0.0,
            }
        )

    equity_df = pd.DataFrame(hist_rows)
    equity_df.dropna(subset=["equity"], inplace=True)
    equity_df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    equity_df.sort_values("date", inplace=True)
    equity_df.to_csv("results/alpaca_equity.csv", index=False)
    print(f"Saved {len(equity_df)} daily equity rows to results/alpaca_equity.csv")

    if equity_df.empty:
        print("ERROR: Alpaca equity history is empty.")
        return

    # 3. Download SPY benchmark over EXACT same dates
    start_date = equity_df["date"].iloc[0]
    end_date = equity_df["date"].iloc[-1]
    print(f"Fetching SPY benchmark data for {start_date} to {end_date}...")

    spy_df = None
    try:
        import duckdb
        db_path = os.path.join(str(PROJECT_ROOT), "data", "sentilyze_market.duckdb")
        if os.path.exists(db_path):
            con = duckdb.connect(db_path, read_only=True)
            sql = f"SELECT date, close as Close FROM daily_bars WHERE ticker = 'SPY' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date"
            res = con.execute(sql).df()
            if len(res) > 0:
                res["date"] = pd.to_datetime(res["date"]).dt.strftime("%Y-%m-%d")
                res.set_index("date", inplace=True)
                spy_df = res
    except Exception as d_err:
        print(f"DuckDB SPY fetch notice: {d_err}")

    if spy_df is None or spy_df.empty:
        try:
            from src.data_ingestion import _fetch_direct_yahoo_chart

            start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d") - datetime.timedelta(days=5)
            end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d") + datetime.timedelta(days=5)
            spy_df = _fetch_direct_yahoo_chart("SPY", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"Direct Yahoo chart fetch note: {e}")

    if spy_df is None or spy_df.empty:
        import yfinance as yf
        spy_raw = yf.download("SPY", start=start_date, auto_adjust=False, progress=False)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_raw.columns = spy_raw.columns.get_level_values(0)
        spy_df = spy_raw

    # Prepare SPY close series with date string index
    spy_df.index = pd.to_datetime(spy_df.index).strftime("%Y-%m-%d")
    spy_df = spy_df[~spy_df.index.duplicated(keep="first")]

    # 4. Align Alpaca Equity and SPY
    combined_rows = []
    alpaca_start_equity = equity_df["equity"].iloc[0]
    
    # Find initial SPY price
    initial_spy_price = None
    for d in equity_df["date"]:
        if d in spy_df.index:
            initial_spy_price = float(spy_df.loc[d, "Close"])
            break
    if initial_spy_price is None and len(spy_df) > 0:
        initial_spy_price = float(spy_df["Close"].iloc[0])

    for _, row in equity_df.iterrows():
        d = row["date"]
        alpaca_eq = row["equity"]
        spy_price = float(spy_df.loc[d, "Close"]) if d in spy_df.index else np.nan
        combined_rows.append(
            {
                "date": d,
                "alpaca_equity": alpaca_eq,
                "spy_close": spy_price,
            }
        )

    comp_df = pd.DataFrame(combined_rows)
    comp_df["spy_close"] = comp_df["spy_close"].ffill().bfill()
    if initial_spy_price is not None and initial_spy_price > 0:
        comp_df["spy_equity"] = alpaca_start_equity * (comp_df["spy_close"] / initial_spy_price)
    else:
        comp_df["spy_equity"] = alpaca_start_equity

    # Save results/alpaca_vs_spy.csv (columns: date, alpaca_equity, spy_equity)
    vs_df = comp_df[["date", "alpaca_equity", "spy_equity"]].copy()
    vs_df.to_csv("results/alpaca_vs_spy.csv", index=False)
    print("Saved comparison to results/alpaca_vs_spy.csv")

    # 5. Compute Metrics
    # Alpaca metrics
    alp_series = comp_df["alpaca_equity"]
    alp_ret = (alp_series.iloc[-1] / alp_series.iloc[0]) - 1.0
    alp_roll_max = alp_series.cummax()
    alp_dd = (alp_series / alp_roll_max) - 1.0
    alp_mdd = float(alp_dd.min())
    alp_daily_pct = alp_series.pct_change().dropna()
    alp_sharpe = float((alp_daily_pct.mean() / (alp_daily_pct.std() + 1e-10)) * np.sqrt(252)) if len(alp_daily_pct) > 1 else 0.0

    # SPY metrics
    spy_series = comp_df["spy_equity"]
    spy_ret = (spy_series.iloc[-1] / spy_series.iloc[0]) - 1.0
    spy_roll_max = spy_series.cummax()
    spy_dd = (spy_series / spy_roll_max) - 1.0
    spy_mdd = float(spy_dd.min())
    spy_daily_pct = spy_series.pct_change().dropna()
    spy_sharpe = float((spy_daily_pct.mean() / (spy_daily_pct.std() + 1e-10)) * np.sqrt(252)) if len(spy_daily_pct) > 1 else 0.0

    summary_text = f"""======================================================================
ALPACA PAPER TRADING VS SPY BENCHMARK PERFORMANCE REPORT
======================================================================
Account ID:           PA3NFG0UWP3R
Evaluation Period:    {start_date} to {end_date} ({len(comp_df)} trading days)
Starting Capital:     ${alp_series.iloc[0]:,.2f}
Total Orders Count:   {len(orders_df)}

METRIC                   ALPACA (SENTILYZE)       SPY BENCHMARK
----------------------------------------------------------------------
Ending Equity:           ${alp_series.iloc[-1]:,.2f}          ${spy_series.iloc[-1]:,.2f}
Total Return:            {alp_ret:+8.2%}                 {spy_ret:+8.2%}
Max Drawdown:            {alp_mdd:8.2%}                 {spy_mdd:8.2%}
Annualized Sharpe:       {alp_sharpe:8.2f}                 {spy_sharpe:8.2f}
======================================================================
"""
    print(summary_text)
    with open("audit_outputs/alpaca_vs_spy_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved summary to audit_outputs/alpaca_vs_spy_summary.txt")


if __name__ == "__main__":
    generate_alpaca_report()
