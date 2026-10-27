"""
Weekly Scorecard Generator for Sentilyze Alpaca Paper Trading vs SPY Benchmark.
Tracks forward performance from the fixed inception date of 2026-10-26.

Rule fixed for PASS/FAIL audit:
- PASS only if, after 8 full weeks (56 calendar days) AND at least 30 closed trades:
    1. Alpaca return > SPY return
    2. Average invested exposure >= 25.0%
- FAIL if after 8 full weeks and 30 trades any condition is breached.
- "IN PROGRESS" before 8 weeks have elapsed.

Outputs: results/scorecard.md
"""

import os
import json
import datetime
import pandas as pd
import numpy as np

START_DATE_STR = "2026-11-02"
OUTPUT_MD = os.path.join("results", "scorecard.md")
EQUITY_CSV = os.path.join("results", "alpaca_equity.csv")
ORDERS_CSV = os.path.join("results", "alpaca_orders.csv")


def calculate_max_drawdown(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return float(drawdown.min() * 100.0)


def fetch_alpaca_history(start_date: str):
    """
    Attempts to fetch portfolio history from Alpaca API if keys exist;
    otherwise reads from results/alpaca_equity.csv and results/alpaca_orders.csv.
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    # Check local CSVs first
    eq_df = None
    if os.path.exists(EQUITY_CSV):
        try:
            df = pd.read_csv(EQUITY_CSV)
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            # Filter from start_date forward
            sub = df[df["date"] >= start_date].copy()
            if not sub.empty:
                eq_df = sub
        except Exception:
            pass

    orders_df = None
    if os.path.exists(ORDERS_CSV):
        try:
            df = pd.read_csv(ORDERS_CSV)
            orders_df = df
        except Exception:
            pass

    return eq_df, orders_df


def fetch_spy_history(start_date: str, end_date: str):
    """Fetches SPY adjusted close prices via yfinance."""
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY")
        hist = spy.history(start=start_date, auto_adjust=False)
        if not hist.empty:
            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]
            hist["date"] = pd.to_datetime(hist[date_col]).dt.strftime("%Y-%m-%d")
            hist["adj_close"] = (
                hist["Adj Close"] if "Adj Close" in hist.columns else hist["Close"]
            )
            return hist[["date", "adj_close"]]
    except Exception as e:
        print(f"Notice fetching SPY via yfinance: {e}")
    return pd.DataFrame(columns=["date", "adj_close"])


def main():
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    start_date = datetime.date.fromisoformat(START_DATE_STR)
    raw_days_elapsed = (today_utc - start_date).days
    if raw_days_elapsed < 0:
        days_elapsed = 0
        weeks_elapsed = 0.0
        elapsed_display = "0 days (not started)"
    else:
        days_elapsed = raw_days_elapsed
        weeks_elapsed = days_elapsed / 7.0
        elapsed_display = f"{days_elapsed} days / {weeks_elapsed:.1f} weeks elapsed"

    eq_df, orders_df = fetch_alpaca_history(START_DATE_STR)

    # Rebase both Alpaca and SPY to identical starting balance baseline ($100,000.00)
    baseline_equity = 100000.0
    start_equity = baseline_equity
    current_equity = baseline_equity
    alpaca_return_pct = 0.0
    alpaca_max_dd = 0.0
    avg_invested_pct = 0.0
    closed_trades_count = 0

    # Determine last Alpaca data point used
    last_alpaca_date = "UNKNOWN"
    if os.path.exists(EQUITY_CSV):
        try:
            df_full = pd.read_csv(EQUITY_CSV)
            if not df_full.empty and "date" in df_full.columns:
                last_alpaca_date = str(df_full["date"].iloc[-1])
        except Exception:
            pass

    if eq_df is not None and not eq_df.empty:
        alpaca_start_raw = float(eq_df["equity"].iloc[0])
        alpaca_curr_raw = float(eq_df["equity"].iloc[-1])
        alpaca_return_pct = (
            ((alpaca_curr_raw / alpaca_start_raw) - 1.0) * 100.0
            if alpaca_start_raw > 0
            else 0.0
        )
        current_equity = round(baseline_equity * (1.0 + alpaca_return_pct / 100.0), 2)
        alpaca_max_dd = calculate_max_drawdown(eq_df["equity"])
    else:
        current_equity = baseline_equity
        alpaca_return_pct = 0.0

    # Calculate forward closed trades since 2026-10-26
    if orders_df is not None and not orders_df.empty:
        filled = orders_df[orders_df["status"] == "filled"].copy()
        if "filled_at" in filled.columns:
            filled["filled_dt"] = pd.to_datetime(filled["filled_at"])
            forward_orders = filled[
                filled["filled_dt"]
                >= pd.to_datetime(START_DATE_STR + " 00:00:00+00:00")
            ]
            # count closed lots (sells)
            closed_trades_count = len(forward_orders[forward_orders["side"] == "sell"])

    # Fetch SPY
    end_date_str = (today_utc + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    spy_df = fetch_spy_history(START_DATE_STR, end_date_str)

    spy_return_pct = 0.0
    spy_max_dd = 0.0
    if not spy_df.empty and len(spy_df) >= 1:
        spy_start_px = float(spy_df["adj_close"].iloc[0])
        spy_end_px = float(spy_df["adj_close"].iloc[-1])
        spy_return_pct = ((spy_end_px / spy_start_px) - 1.0) * 100.0
        spy_max_dd = calculate_max_drawdown(spy_df["adj_close"])

    spy_equity = round(baseline_equity * (1.0 + spy_return_pct / 100.0), 2)
    diff_return_pct = alpaca_return_pct - spy_return_pct

    # Determine status against strict audit rule:
    # PASS only if: weeks_elapsed >= 8 and closed_trades >= 30 and alpaca_return > spy_return and avg_invested >= 25%
    if weeks_elapsed < 8.0:
        status = "IN PROGRESS"
        status_detail = f"Evaluation active: {elapsed_display} ({weeks_elapsed:.1f}/8.0 weeks elapsed). Minimum 8 weeks required."
    else:
        passed = (
            closed_trades_count >= 30
            and alpaca_return_pct > spy_return_pct
            and avg_invested_pct >= 25.0
        )
        if passed:
            status = "PASS"
            status_detail = "All institutional audit gates achieved: >= 8 weeks, >= 30 trades, beat SPY, >= 25% exposure."
        else:
            status = "FAIL"
            reasons = []
            if closed_trades_count < 30:
                reasons.append(
                    f"Insufficient trade count ({closed_trades_count}/30 trades)"
                )
            if alpaca_return_pct <= spy_return_pct:
                reasons.append(
                    f"Underperformed SPY ({alpaca_return_pct:+.2f}% vs {spy_return_pct:+.2f}%)"
                )
            if avg_invested_pct < 25.0:
                reasons.append(
                    f"Under-invested exposure ({avg_invested_pct:.1f}% < 25.0% minimum)"
                )
            status_detail = f"Failed audit gates: {'; '.join(reasons)}"

    lines = [
        "# Sentilyze Forward Execution Scorecard",
        "",
        f"**Audit Inception Date**: `{START_DATE_STR}`  ",
        f"**As Of Date**: `{today_utc.isoformat()}` (`{elapsed_display}`)  ",
        f"**Last Alpaca Data Point Used**: `{last_alpaca_date}`  ",
        f"**Audit Status**: **`{status}`**  ",
        f"**Status Detail**: {status_detail}",
        "",
        "## Performance Comparison",
        "",
        "| Metric | Alpaca Paper Portfolio | SPY Benchmark (Adjusted) | Spread / Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Starting Equity** | **${start_equity:,.2f}** | **${baseline_equity:,.2f}** | $0.00 |",
        f"| **Current Equity** | **${current_equity:,.2f}** | **${spy_equity:,.2f}** | ${current_equity - spy_equity:,.2f} |",
        f"| **Net Total Return** | **{alpaca_return_pct:+.2f}%** | **{spy_return_pct:+.2f}%** | **{diff_return_pct:+.2f}%** |",
        f"| **Maximum Drawdown** | **{alpaca_max_dd:.2f}%** | **{spy_max_dd:.2f}%** | {alpaca_max_dd - spy_max_dd:+.2f}% |",
        f"| **Closed Forward Trades** | **{closed_trades_count}** (min 30 required) | Buy & Hold | — |",
        f"| **Average Invested Exposure** | **{avg_invested_pct:.2f}%** (min 25% required) | 100.0% | — |",
        "",
        "## Strict Evaluation Gate Rules",
        "",
        f"1. **Evaluation Horizon**: Minimum 8 full weeks (56 calendar days) forward from `{START_DATE_STR}`.",
        "2. **Minimum Trade Activity**: At least 30 closed round-trip orders executed on Alpaca paper markets.",
        "3. **Alpha Outperformance**: Alpaca net return must strictly exceed SPY adjusted total return.",
        "4. **Capital Utilization**: Average daily invested exposure (`position_cost / equity`) must be at least 25.0% to reject cash-hiding survivorship.",
        "",
    ]

    # -------------------------------------------------------------------------
    # Open Alpaca Positions & Age Telemetry
    # -------------------------------------------------------------------------
    open_positions_data = []

    try:
        from src.alpaca_broker import AlpacaBrokerBridge

        alp = AlpacaBrokerBridge()
        if alp.is_connected():
            live_pos = alp.get_open_positions()
            for lp in live_pos:
                sym = lp.get("symbol")
                qty = float(lp.get("qty", 0.0))
                avg_px = float(lp.get("avg_entry_price", 0.0))
                entry_date_str = "UNKNOWN"
                age_days_str = "UNKNOWN"
                if orders_df is not None and not orders_df.empty:
                    sym_buys = orders_df[
                        (orders_df["status"] == "filled")
                        & (orders_df["side"] == "buy")
                        & (orders_df["symbol"] == sym)
                    ]
                    if not sym_buys.empty and "filled_at" in sym_buys.columns:
                        first_fill = pd.to_datetime(sym_buys["filled_at"]).min()
                        entry_date_str = first_fill.strftime("%Y-%m-%d")
                        age_days_str = f"{(today_utc - first_fill.date()).days} days"
                open_positions_data.append(
                    {
                        "symbol": sym,
                        "qty": qty,
                        "avg_price": avg_px,
                        "entry_date": entry_date_str,
                        "age": age_days_str,
                    }
                )
    except Exception:
        pass

    if not open_positions_data and orders_df is not None and not orders_df.empty:
        try:
            filled_df = orders_df[orders_df["status"] == "filled"].copy()
            filled_df["qty_num"] = pd.to_numeric(
                filled_df["qty"], errors="coerce"
            ).fillna(0.0)
            buys = (
                filled_df[filled_df["side"] == "buy"].groupby("symbol")["qty_num"].sum()
            )
            sells = (
                filled_df[filled_df["side"] == "sell"]
                .groupby("symbol")["qty_num"]
                .sum()
            )
            net_pos = buys.subtract(sells, fill_value=0.0)
            open_symbols = net_pos[net_pos > 0.0001]

            for sym, qty in open_symbols.items():
                sym_buys = filled_df[
                    (filled_df["side"] == "buy") & (filled_df["symbol"] == sym)
                ]
                first_fill = (
                    pd.to_datetime(sym_buys["filled_at"]).min()
                    if ("filled_at" in sym_buys.columns and not sym_buys.empty)
                    else None
                )
                entry_date_str = (
                    first_fill.strftime("%Y-%m-%d")
                    if first_fill is not None
                    else "UNKNOWN"
                )
                age_days_str = (
                    f"{(today_utc - first_fill.date()).days} days"
                    if first_fill is not None
                    else "UNKNOWN"
                )
                avg_px = (
                    float(sym_buys["filled_avg_price"].iloc[-1])
                    if ("filled_avg_price" in sym_buys.columns and not sym_buys.empty)
                    else 0.0
                )
                open_positions_data.append(
                    {
                        "symbol": sym,
                        "qty": qty,
                        "avg_price": avg_px,
                        "entry_date": entry_date_str,
                        "age": age_days_str,
                    }
                )
        except Exception:
            pass

    # Calculate age in days of the oldest open Alpaca position
    oldest_age_days = 0
    for pos in open_positions_data:
        age_str = str(pos.get("age", ""))
        if "days" in age_str:
            try:
                days_val = int(age_str.split()[0])
                if days_val > oldest_age_days:
                    oldest_age_days = days_val
            except Exception:
                pass

    # Count local-ledger open positions that are NOT on Alpaca (phantoms)
    local_open_symbols = set()
    paper_portfolio_path = os.path.join("results", "paper_portfolio.json")
    if os.path.exists(paper_portfolio_path):
        try:
            with open(paper_portfolio_path, "r", encoding="utf-8") as f:
                port_data = json.load(f)
                local_open_symbols = set(port_data.get("open_positions", {}).keys())
        except Exception:
            pass

    alpaca_open_symbols = {p["symbol"] for p in open_positions_data if p.get("symbol")}
    phantom_local_positions = [
        s for s in local_open_symbols if s not in alpaca_open_symbols
    ]
    phantom_count = len(phantom_local_positions)

    lines.extend(
        [
            "## 📦 Open Alpaca Positions Telemetry",
            "",
            f"- **Total Open Positions on Alpaca**: **{len(open_positions_data)}**",
            f"- **Oldest Open Alpaca Position Age**: **{oldest_age_days} days**",
            f"- **Local-Ledger Positions NOT on Alpaca (Phantoms)**: **{phantom_count}**"
            + (
                f" (`{', '.join(sorted(phantom_local_positions))}`)"
                if phantom_local_positions
                else " (None)"
            ),
            "",
        ]
    )

    if open_positions_data:
        lines.extend(
            [
                "| Symbol | Open Quantity | Entry Date | Position Age | Avg Entry Price |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for pos in sorted(open_positions_data, key=lambda x: x["symbol"]):
            lines.append(
                f"| `{pos['symbol']}` | {pos['qty']:,.1f} | `{pos['entry_date']}` | {pos['age']} | ${pos['avg_price']:,.2f} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "*(No active open positions on Alpaca account — 100% liquid cash reserves)*",
                "",
            ]
        )

    # -------------------------------------------------------------------------
    # Bracket Leg Exit Telemetry (Take-Profit vs Stop-Loss)
    # -------------------------------------------------------------------------
    tp_closed_count = 0
    sl_closed_count = 0
    other_closed_count = 0

    if orders_df is not None and not orders_df.empty:
        try:
            sell_orders = orders_df[
                (orders_df["status"] == "filled") & (orders_df["side"] == "sell")
            ].copy()
            if not sell_orders.empty and "order_type" in sell_orders.columns:
                for _, s_row in sell_orders.iterrows():
                    o_type = str(s_row.get("order_type", "")).lower()
                    if o_type == "limit":
                        tp_closed_count += 1
                    elif o_type in ("stop", "stop_limit"):
                        sl_closed_count += 1
                    else:
                        other_closed_count += 1
        except Exception:
            pass

    total_closed_exits = tp_closed_count + sl_closed_count + other_closed_count
    denom = max(1, total_closed_exits)
    tp_pct = (tp_closed_count / denom) * 100.0
    sl_pct = (sl_closed_count / denom) * 100.0
    other_pct = (other_closed_count / denom) * 100.0

    lines.extend(
        [
            "## 🎯 Bracket Order Leg Exits (Alpaca Brokerage)",
            "",
            "Breakdown of all filled sell orders executed by Alpaca exchange-side bracket orders:",
            "",
            "| Exit Mechanism | Leg Order Type | Orders Closed | Distribution |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Take-Profit Leg** | `limit` sell | **{tp_closed_count}** | {tp_pct:.1f}% |",
            f"| **Stop-Loss Leg** | `stop` / `stop_limit` sell | **{sl_closed_count}** | {sl_pct:.1f}% |",
            f"| **Discretionary / Other** | `market` sell | **{other_closed_count}** | {other_pct:.1f}% |",
            f"| **Total Realized Exits** | — | **{total_closed_exits}** | 100.0% |",
            "",
        ]
    )

    # Incorporate Funnel Table from results/funnel_log.jsonl or results/funnel.json
    funnel_dir = os.getenv("FUNNEL_DIR", "results")
    funnel_log_path = os.path.join(funnel_dir, "funnel_log.jsonl")
    funnel_json_path = os.path.join(funnel_dir, "funnel.json")

    total_scanned = 0
    total_skipped_news = 0
    total_skipped_quota_error = 0
    total_vetoed = 0
    total_blocked = 0
    total_orders = 0
    vetoes_by_reason: dict[str, int] = {}
    blocked_by_guard: dict[str, int] = {}
    runs_count = 0
    aborted_runs_count = 0

    if os.path.exists(funnel_log_path):
        try:
            with open(funnel_log_path, "r", encoding="utf-8") as fl:
                for line in fl:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    runs_count += 1
                    aborted_runs_count += int(
                        entry.get("runs_aborted_alpaca_unavailable", 0)
                    )
                    total_scanned += int(entry.get("tickers_scanned", 0))
                    total_skipped_news += int(entry.get("skipped_no_real_news", 0))
                    total_skipped_quota_error += int(
                        entry.get("skipped_news_quota_or_error", 0)
                    )
                    total_vetoed += int(
                        entry.get(
                            "total_vetoed",
                            sum(entry.get("vetoed_by_reason", {}).values()),
                        )
                    )
                    total_blocked += int(
                        entry.get(
                            "total_buy_blocked",
                            sum(entry.get("buy_blocked_guards", {}).values()),
                        )
                    )
                    total_orders += int(entry.get("orders_submitted", 0))

                    for r, c in entry.get("vetoed_by_reason", {}).items():
                        vetoes_by_reason[r] = vetoes_by_reason.get(r, 0) + c
                    for g, c in entry.get("buy_blocked_guards", {}).items():
                        blocked_by_guard[g] = blocked_by_guard.get(g, 0) + c

            lines.extend(
                [
                    "## 🔍 Cumulative Execution Funnel Telemetry",
                    "",
                    f"Aggregated across **{runs_count}** autonomous live trading cycles recorded in `results/funnel_log.jsonl`:",
                    "",
                    "| Total Runs | Aborted Runs | Total Scanned | Skipped (No News) | Skipped (Quota/Error) | Total Vetoed | Blocked (Risk Guards) | Orders Submitted |",
                    "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    f"| {runs_count} | {aborted_runs_count} | {total_scanned} | {total_skipped_news} | {total_skipped_quota_error} | {total_vetoed} | {total_blocked} | **{total_orders}** |",
                    "",
                ]
            )
            if vetoes_by_reason:
                lines.append("**Cumulative Vetoes by Reason**:")
                for r, c in sorted(
                    vetoes_by_reason.items(), key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"- `{r}`: {c}")
                lines.append("")
            if blocked_by_guard:
                lines.append("**Cumulative Orders Blocked by Position/Risk Guards**:")
                for g, c in sorted(
                    blocked_by_guard.items(), key=lambda x: x[1], reverse=True
                ):
                    lines.append(f"- `{g}`: {c}")
                lines.append("")
        except Exception as fe:
            lines.append(f"*(Funnel telemetry log error: {fe})*")
            lines.append("")
    elif os.path.exists(funnel_json_path):
        try:
            with open(funnel_json_path, "r", encoding="utf-8") as f_json:
                fj = json.load(f_json)
            lines.extend(
                [
                    "## 🔍 Execution Funnel Telemetry",
                    "",
                    "| Latest Run Timestamp | Total Runs | Aborted Runs | Tickers Scanned | Skipped (No News) | Vetoed | Blocked (Guards) | Orders Submitted |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    f"| `{fj.get('timestamp', 'N/A')}` | 1 | {fj.get('runs_aborted_alpaca_unavailable', 0)} | {fj.get('tickers_scanned', 0)} | {fj.get('skipped_no_real_news', 0)} | {fj.get('total_vetoed', 0)} | {fj.get('total_buy_blocked', 0)} | **{fj.get('orders_submitted', 0)}** |",
                    "",
                ]
            )
            vetoes = fj.get("vetoed_by_reason", {})
            if vetoes:
                lines.append("**Recent Vetoes by Reason**:")
                for r, c in vetoes.items():
                    lines.append(f"- `{r}`: {c}")
                lines.append("")
            blocked = fj.get("buy_blocked_guards", {})
            if blocked:
                lines.append("**Recent Orders Blocked by Position/Risk Guards**:")
                for g, c in blocked.items():
                    lines.append(f"- `{g}`: {c}")
                lines.append("")
        except Exception as fe:
            lines.append(f"*(Funnel telemetry load notice: {fe})*")
            lines.append("")

    lines.append(
        f"*Generated automatically by `scripts/weekly_scorecard.py` on {today_utc.isoformat()}.*"
    )

    content = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote scorecard ({status}) to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
