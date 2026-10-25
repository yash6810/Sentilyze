"""
Audit script to detect duplicate buy entries within 24 hours in results/alpaca_orders.csv,
and document the structural root cause in src/autonomous_trader.py and src/paper_broker.py.
Outputs: audit_outputs/duplicate_entries_audit.txt
"""

import pandas as pd
import datetime

ORDERS_CSV = "results/alpaca_orders.csv"
OUTPUT_TXT = "audit_outputs/duplicate_entries_audit.txt"


def main():
    df = pd.read_csv(ORDERS_CSV)
    buys = df[df["side"] == "buy"].copy()
    buys["filled_at_dt"] = pd.to_datetime(buys["filled_at"])
    buys = buys.sort_values(["symbol", "filled_at_dt"])

    # Group by symbol and find orders within 24 hours of each other
    duplicates = {}
    for sym, group in buys.groupby("symbol"):
        group = group.reset_index(drop=True)
        dup_rows = []
        for i in range(len(group)):
            t_i = group.loc[i, "filled_at_dt"]
            # Check if any other order is within 24 hours
            has_close = False
            for j in range(len(group)):
                if i != j:
                    diff_hours = (
                        abs((t_i - group.loc[j, "filled_at_dt"]).total_seconds())
                        / 3600.0
                    )
                    if diff_hours <= 24.0:
                        has_close = True
                        break
            if has_close:
                dup_rows.append(group.loc[i])
        if dup_rows:
            duplicates[sym] = pd.DataFrame(dup_rows)

    lines = [
        "==================================================================",
        "AUDIT REPORT: DUPLICATE BUY ENTRIES WITHIN 24 HOURS",
        f"Source file: {ORDERS_CSV}",
        "==================================================================",
        "",
        "1. TICKERS WITH MULTIPLE BUY ORDERS WITHIN 24 HOURS",
        "------------------------------------------------------------------",
    ]

    for sym, dup_df in duplicates.items():
        lines.append(f"\nTicker: {sym} (Total Duplicate Buys: {len(dup_df)})")
        lines.append(
            f"{'Timestamp (UTC)':<32} | {'Qty':>8} | {'Price ($)':>12} | {'Status':<10}"
        )
        lines.append("-" * 68)
        for _, row in dup_df.iterrows():
            t = str(row["filled_at"])
            q = float(row["filled_qty"])
            p = float(row["filled_avg_price"])
            s = str(row["status"])
            lines.append(f"{t:<32} | {q:>8.1f} | {p:>12.4f} | {s:<10}")

    lines.extend(
        [
            "",
            "2. ROOT CAUSE ANALYSIS: 5 NVDA BUYS IN 6 MINUTES ON 2026-09-24",
            "------------------------------------------------------------------",
            "On 2026-09-24 between 18:57:49 UTC and 19:03:06 UTC (5 minutes, 17 seconds),",
            "5 separate buy orders for 40 shares each of NVDA (~$8,970 each, total $44,845)",
            "were submitted and filled on Alpaca paper markets:",
            "  - 2026-09-24 18:57:49 UTC: 40 shares @ $224.31525",
            "  - 2026-09-24 18:59:42 UTC: 40 shares @ $224.28900",
            "  - 2026-09-24 19:01:21 UTC: 40 shares @ $224.18800",
            "  - 2026-09-24 19:01:25 UTC: 40 shares @ $224.13000",
            "  - 2026-09-24 19:03:06 UTC: 40 shares @ $224.02000",
            "",
            "Code Path & Structural Vulnerabilities Identified:",
            "1. Missing Alpaca Live State Query:",
            "   In `src/autonomous_trader.py` (Phase B candidate screening, line 1077):",
            "   `unheld_tickers = [t for t in tickers_to_scan if t not in self.broker.state['open_positions']]`",
            "   The check evaluated ONLY the local JSON file (`results/paper_portfolio.json`).",
            "   It NEVER queried Alpaca (`AlpacaBrokerBridge.get_open_positions()` or `/v2/orders?status=open`).",
            "",
            "2. GitHub Actions Isolated Container State:",
            "   Each scheduled GitHub Actions container checkout pulled the latest `main` commit.",
            "   Because state was only committed at daily close (or failed push due to race conditions),",
            "   subsequent runner containers had an empty local `open_positions` dict.",
            "   Every run saw NVDA as unheld and triggered a fresh buy order.",
            "",
            "3. Missing Existing Position Guard in `PaperBroker.execute_buy()`:",
            "   In `src/paper_broker.py` line 1257:",
            "   `self.state['open_positions'][ticker] = { ... }`",
            "   The method unconditionally overwrote the position dictionary without checking whether",
            "   the ticker was already open, blindly accepting repeated buys.",
            "",
            "4. Missing Concentration Cap Enforcement in `PaperBroker.execute_buy()`:",
            "   `execute_buy()` did not verify whether total ticker exposure exceeded 10% of total equity,",
            "   nor whether total portfolio positions exceeded `MAX_CONCURRENT_POSITIONS` (10).",
            "",
            "Required Fix Architecture:",
            "1. In `AlpacaBrokerBridge`: add `get_open_orders(status='open')` and `has_open_position_or_order(ticker)`.",
            "2. In `PaperBroker.execute_buy()`:",
            "   - Query Alpaca for open positions and open orders; reject buy if ticker exists.",
            "   - Reject if ticker is already in `self.state['open_positions']`.",
            "   - Enforce `MAX_CONCURRENT_POSITIONS` (default 10) across paper and Alpaca active symbols.",
            "   - Enforce strict 10% maximum equity concentration cap per ticker across existing and new orders.",
            "3. In `AutonomousTradingEngine._execute_cycle_body()`: query Alpaca before submitting orders.",
        ]
    )

    report_text = "\n".join(lines)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote audit report to {OUTPUT_TXT}")
    print(report_text)


if __name__ == "__main__":
    main()
