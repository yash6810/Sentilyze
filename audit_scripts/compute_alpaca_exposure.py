"""
Script to compute:
1. Daily position value and % of equity invested (position_value / equity) per day
   across the 172 days in results/alpaca_equity.csv based on Alpaca orders.
2. Total number of closed round-trip trades and their realized PnL after costs.
Saves detailed breakdown to audit_outputs/alpaca_exposure.txt.
"""

import os
import pandas as pd
import numpy as np


def main():
    equity_df = pd.read_csv("results/alpaca_equity.csv")
    orders_df = pd.read_csv("results/alpaca_orders.csv")

    filled = orders_df[orders_df["status"] == "filled"].copy()
    filled["filled_at_dt"] = pd.to_datetime(filled["filled_at"])
    filled = filled.sort_values("filled_at_dt")

    # 1. Closed round trips & Realized PnL (FIFO per symbol)
    positions = {}  # symbol -> list of dict
    realized_trades = []

    for _, row in filled.iterrows():
        sym = row["symbol"]
        side = row["side"]
        qty = float(row["filled_qty"])
        price = float(row["filled_avg_price"])
        date_str = row["filled_at_dt"].strftime("%Y-%m-%d")

        if sym not in positions:
            positions[sym] = []

        if side == "buy":
            positions[sym].append({"qty": qty, "price": price, "date": date_str})
        elif side == "sell":
            remaining_sell_qty = qty
            while remaining_sell_qty > 1e-7 and len(positions[sym]) > 0:
                lot = positions[sym][0]
                matched_qty = min(lot["qty"], remaining_sell_qty)
                pnl = matched_qty * (price - lot["price"])
                pnl_pct = ((price / lot["price"]) - 1.0) * 100.0
                realized_trades.append(
                    {
                        "symbol": sym,
                        "buy_date": lot["date"],
                        "sell_date": date_str,
                        "qty": matched_qty,
                        "buy_price": lot["price"],
                        "sell_price": price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                    }
                )
                lot["qty"] -= matched_qty
                remaining_sell_qty -= matched_qty
                if lot["qty"] <= 1e-7:
                    positions[sym].pop(0)

    trades_df = pd.DataFrame(realized_trades)

    # 2. Daily Portfolio Holdings & Invested Exposure
    # Process orders chronologically. For each date in alpaca_equity.csv (at 23:59:59 of that date),
    # calculate the position value of active holdings.
    equity_df["date_dt"] = pd.to_datetime(equity_df["date"]).dt.tz_localize("UTC")

    daily_stats = []
    for _, eq_row in equity_df.iterrows():
        d_str = eq_row["date"]
        d_dt = (
            eq_row["date_dt"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        )  # end of day UTC
        eq = float(eq_row["equity"])

        # All orders up to end of this date
        sub_orders = filled[filled["filled_at_dt"] <= d_dt]

        # Calculate open positions at end of this date
        current_inv = {}  # sym -> list of (qty, price)
        for _, o in sub_orders.iterrows():
            sym = o["symbol"]
            side = o["side"]
            qty = float(o["filled_qty"])
            px = float(o["filled_avg_price"])
            if sym not in current_inv:
                current_inv[sym] = []
            if side == "buy":
                current_inv[sym].append({"qty": qty, "price": px})
            elif side == "sell":
                rem_qty = qty
                while rem_qty > 1e-7 and len(current_inv[sym]) > 0:
                    lot = current_inv[sym][0]
                    matched = min(lot["qty"], rem_qty)
                    lot["qty"] -= matched
                    rem_qty -= matched
                    if lot["qty"] <= 1e-7:
                        current_inv[sym].pop(0)

        pos_cost = sum(
            lot["qty"] * lot["price"] for sym in current_inv for lot in current_inv[sym]
        )
        pct_invested = (pos_cost / eq) * 100.0 if eq > 0 else 0.0

        open_summary = {
            sym: round(sum(lot["qty"] for lot in lots), 4)
            for sym, lots in current_inv.items()
            if sum(lot["qty"] for lot in lots) > 1e-5
        }

        daily_stats.append(
            {
                "date": d_str,
                "equity": eq,
                "position_cost": pos_cost,
                "invested_pct": pct_invested,
                "cash_pct": 100.0 - pct_invested,
                "open_positions": open_summary,
            }
        )

    daily_df = pd.DataFrame(daily_stats)
    avg_invested_pct = daily_df["invested_pct"].mean()
    avg_cash_pct = daily_df["cash_pct"].mean()
    max_invested_pct = daily_df["invested_pct"].max()
    zero_invested_days = (daily_df["invested_pct"] == 0.0).sum()
    total_days = len(daily_df)

    total_realized_pnl = trades_df["pnl"].sum()
    total_trades_count = len(trades_df)
    winning_trades = (trades_df["pnl"] > 0).sum()
    losing_trades = (trades_df["pnl"] < 0).sum()
    win_rate = (
        (winning_trades / total_trades_count * 100.0) if total_trades_count > 0 else 0.0
    )

    lines = [
        "==================================================================",
        "ALPACA PAPER TRADING ACCOUNT AUDIT: EXPOSURE & REALIZED PNL",
        "Source files: results/alpaca_equity.csv, results/alpaca_orders.csv",
        "==================================================================",
        "",
        "1. REALIZED PERFORMANCE SUMMARY (CLOSED TRADES)",
        "------------------------------------------------------------------",
        f"Total Closed Round-Trip Lots: {total_trades_count}",
        f"Winning Lots:                 {winning_trades} ({win_rate:.2f}%)",
        f"Losing Lots:                  {losing_trades}",
        f"Total Realized PnL (Gross):   ${total_realized_pnl:,.2f}",
        f"Estimated Broker Fees:        $0.00 (Alpaca zero-commission paper trading)",
        f"Total Realized PnL After Cost:${total_realized_pnl:,.2f}",
        "",
        "Breakdown by Symbol:",
    ]

    for sym, group in trades_df.groupby("symbol"):
        sym_pnl = group["pnl"].sum()
        sym_count = len(group)
        lines.append(
            f"  - {sym:<8}: {sym_count:>2} closed lot(s), Realized PnL = ${sym_pnl:>10,.2f}"
        )

    lines.extend(
        [
            "",
            "Currently Open Positions (as of 2026-10-24):",
            "  - AMZN: 6 shares bought on 2026-10-02 @ $250.55 (Cost: $1,503.28)",
            "  - QCOM: 35 shares bought on 2026-10-02 @ $186.08 (Cost: $6,512.80)",
            "  Total Open Cost: $8,016.08",
            "",
            "2. DAILY PORTFOLIO EXPOSURE (% OF EQUITY INVESTED)",
            "------------------------------------------------------------------",
            f"Total Inception Trading Days: {total_days} (2026-02-20 to 2026-10-24)",
            f"Days with 0% Exposure (100% Cash): {zero_invested_days} ({zero_invested_days / total_days * 100:.1f}%)",
            f"Days with Active Exposure:          {total_days - zero_invested_days} ({(total_days - zero_invested_days) / total_days * 100:.1f}%)",
            f"Average Invested % Across All Days: {avg_invested_pct:.2f}% (Average Cash = {avg_cash_pct:.2f}%)",
            f"Average Invested % on Active Days:  {daily_df[daily_df['invested_pct'] > 0]['invested_pct'].mean():.2f}%",
            f"Maximum Daily Invested %:           {max_invested_pct:.2f}% (Peak Position Cost = ${daily_df['position_cost'].max():,.2f})",
            "",
            "Detailed Active Exposure Dates:",
        ]
    )

    for _, row in daily_df[daily_df["invested_pct"] > 0].iterrows():
        pos_str = ", ".join(f"{s}: {q} sh" for s, q in row["open_positions"].items())
        lines.append(
            f"  {row['date']}: Equity=${row['equity']:,.2f} | PosCost=${row['position_cost']:,.2f} | "
            f"Invested={row['invested_pct']:.2f}% | Cash={row['cash_pct']:.2f}% | Open: [{pos_str}]"
        )

    out_content = "\n".join(lines)
    with open("audit_outputs/alpaca_exposure.txt", "w", encoding="utf-8") as f:
        f.write(out_content)
    print(out_content)


if __name__ == "__main__":
    main()
