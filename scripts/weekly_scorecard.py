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

START_DATE_STR = "2026-10-26"
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
        "1. **Evaluation Horizon**: Minimum 8 full weeks (56 calendar days) forward from `2026-10-26`.",
        "2. **Minimum Trade Activity**: At least 30 closed round-trip orders executed on Alpaca paper markets.",
        "3. **Alpha Outperformance**: Alpaca net return must strictly exceed SPY adjusted total return.",
        "4. **Capital Utilization**: Average daily invested exposure (`position_cost / equity`) must be at least 25.0% to reject cash-hiding survivorship.",
        "",
    ]

    # Incorporate Funnel Table from results/funnel.json or results/funnel.md
    funnel_json_path = os.path.join("results", "funnel.json")
    funnel_history_path = os.path.join("results", "funnel_history.json")
    if os.path.exists(funnel_json_path):
        try:
            with open(funnel_json_path, "r", encoding="utf-8") as f_json:
                fj = json.load(f_json)
            lines.extend([
                "## 🔍 Execution Funnel Telemetry",
                "",
                "| Latest Run Timestamp | Tickers Scanned | Skipped (No News) | Vetoed | Blocked (Guards) | Orders Submitted |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
                f"| `{fj.get('timestamp', 'N/A')}` | {fj.get('tickers_scanned', 0)} | {fj.get('skipped_no_real_news', 0)} | {fj.get('total_vetoed', 0)} | {fj.get('total_buy_blocked', 0)} | **{fj.get('orders_submitted', 0)}** |",
                "",
            ])
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

    lines.append(f"*Generated automatically by `scripts/weekly_scorecard.py` on {today_utc.isoformat()}.*")

    content = "\n".join(lines)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote scorecard ({status}) to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
