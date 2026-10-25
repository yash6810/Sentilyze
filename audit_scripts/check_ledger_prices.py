"""
Audit script: check_ledger_prices.py
Audits every trade in archive/invalid_simulation_2026/executed_trades.csv against
actual daily market OHLC (auto_adjust=False).
Writes results to audit_outputs/ledger_price_check.csv.
"""

import os
import sys
import pandas as pd
import requests

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import _fetch_direct_yahoo_chart

ARCHIVED_CSV = os.path.join("archive", "invalid_simulation_2026", "executed_trades.csv")
OUTPUT_CSV = os.path.join("audit_outputs", "ledger_price_check.csv")

def main():
    if not os.path.exists(ARCHIVED_CSV):
        print(f"ERROR: {ARCHIVED_CSV} does not exist!")
        sys.exit(1)

    df_trades = pd.read_csv(ARCHIVED_CSV)
    print(f"Loaded {len(df_trades)} trades from {ARCHIVED_CSV}")

    # Cache charts per ticker to avoid repetitive calls
    charts = {}
    records = []

    entry_inside = 0
    entry_outside = 0
    entry_missing = 0

    exit_inside = 0
    exit_outside = 0
    exit_missing = 0

    for idx, row in df_trades.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        e_date = str(row["entry_date"])[:10]
        x_date = str(row["exit_date"])[:10]
        e_price = float(row["entry_price"])
        x_price = float(row["exit_price"])

        if ticker not in charts:
            try:
                c = _fetch_direct_yahoo_chart(ticker, period="6mo")
                if not c.empty:
                    c.index = c.index.strftime("%Y-%m-%d")
                    charts[ticker] = c
                else:
                    charts[ticker] = None
            except Exception as ex:
                print(f"Failed to fetch {ticker}: {ex}")
                charts[ticker] = None

        chart = charts.get(ticker)

        # Check Entry Price
        if chart is not None and e_date in chart.index:
            day_low = float(chart.loc[e_date, "Low"])
            day_high = float(chart.loc[e_date, "High"])
            inside = (day_low <= e_price <= day_high)
            if inside:
                entry_inside += 1
            else:
                entry_outside += 1
        else:
            day_low = "UNKNOWN"
            day_high = "UNKNOWN"
            inside = "UNKNOWN"
            entry_missing += 1

        records.append({
            "trade_id": idx,
            "type": "ENTRY",
            "ticker": ticker,
            "date": e_date,
            "csv_price": e_price,
            "day_low": day_low,
            "day_high": day_high,
            "inside_range": inside
        })

        # Check Exit Price
        if chart is not None and x_date in chart.index:
            day_low_x = float(chart.loc[x_date, "Low"])
            day_high_x = float(chart.loc[x_date, "High"])
            inside_x = (day_low_x <= x_price <= day_high_x)
            if inside_x:
                exit_inside += 1
            else:
                exit_outside += 1
        else:
            day_low_x = "UNKNOWN"
            day_high_x = "UNKNOWN"
            inside_x = "UNKNOWN"
            exit_missing += 1

        records.append({
            "trade_id": idx,
            "type": "EXIT",
            "ticker": ticker,
            "date": x_date,
            "csv_price": x_price,
            "day_low": day_low_x,
            "day_high": day_high_x,
            "inside_range": inside_x
        })

    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved audit output to {OUTPUT_CSV}")

    print("\n--- AUDIT SUMMARY ---")
    print(f"Total Trades Evaluated: {len(df_trades)}")
    print(f"Entry Prices: Inside Range = {entry_inside}, Outside Range = {entry_outside}, Unknown/Date Missing = {entry_missing}")
    print(f"Exit Prices:  Inside Range = {exit_inside}, Outside Range = {exit_outside}, Unknown/Date Missing = {exit_missing}")

if __name__ == "__main__":
    main()
