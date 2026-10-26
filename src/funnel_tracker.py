"""
Execution Funnel Telemetry Tracker for Sentilyze Autonomous Trading.
Tracks and audits:
1. Tickers scanned
2. Tickers skipped for no real news
3. Candidates vetoed (categorized by veto reason)
4. Buy orders blocked by guards (duplicate, max positions, concentration, etc.)
5. Orders submitted

Outputs:
- results/funnel.json (latest run telemetry)
- results/funnel.md (daily aggregated execution funnel log)
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)

FUNNEL_JSON_PATH = os.path.join("results", "funnel.json")
FUNNEL_MD_PATH = os.path.join("results", "funnel.md")


def record_funnel_run(
    tickers_scanned: int,
    skipped_no_real_news: int,
    vetoed_by_reason: Dict[str, int],
    buy_blocked_guards: Dict[str, int],
    orders_submitted: int,
    json_path: str = FUNNEL_JSON_PATH,
    md_path: str = FUNNEL_MD_PATH,
) -> Dict[str, Any]:
    """
    Records cycle funnel metrics to results/funnel.json and accumulates daily total in results/funnel.md.
    """
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    timestamp_str = now_utc.isoformat()

    run_record = {
        "timestamp": timestamp_str,
        "date": today_str,
        "tickers_scanned": int(tickers_scanned),
        "skipped_no_real_news": int(skipped_no_real_news),
        "total_vetoed": int(sum(vetoed_by_reason.values())),
        "vetoed_by_reason": vetoed_by_reason,
        "total_buy_blocked": int(sum(buy_blocked_guards.values())),
        "buy_blocked_guards": buy_blocked_guards,
        "orders_submitted": int(orders_submitted),
    }

    # 1. Save results/funnel.json (latest run)
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2)
    except Exception as e:
        logger.warning(f"Notice saving {json_path}: {e}")

    # 2. Update results/funnel.md (accumulate daily totals)
    try:
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        daily_records: Dict[str, Dict[str, Any]] = {}
        history_file = os.path.join(os.path.dirname(md_path), "funnel_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as hf:
                    daily_records = json.load(hf)
            except Exception:
                daily_records = {}

        if today_str not in daily_records:
            daily_records[today_str] = {
                "date": today_str,
                "tickers_scanned": 0,
                "skipped_no_real_news": 0,
                "vetoed_by_reason": {},
                "buy_blocked_guards": {},
                "orders_submitted": 0,
                "runs_count": 0,
            }

        day = daily_records[today_str]
        day["runs_count"] += 1
        day["tickers_scanned"] += run_record["tickers_scanned"]
        day["skipped_no_real_news"] += run_record["skipped_no_real_news"]
        day["orders_submitted"] += run_record["orders_submitted"]

        for k, v in vetoed_by_reason.items():
            day["vetoed_by_reason"][k] = day["vetoed_by_reason"].get(k, 0) + v
        for k, v in buy_blocked_guards.items():
            day["buy_blocked_guards"][k] = day["buy_blocked_guards"].get(k, 0) + v

        with open(history_file, "w", encoding="utf-8") as hf:
            json.dump(daily_records, hf, indent=2)

        # Render Markdown Table
        lines = [
            "# 🔍 Sentilyze Execution Funnel Telemetry",
            "",
            "Tracks the multi-stage elimination pipeline per trading session: Scanned → News Check → Committee Veto → Guards → Execution.",
            "",
            "## Daily Execution Totals",
            "",
            "| Date | Runs | Scanned | Skipped (No News) | Vetoed | Blocked by Guards | Orders Submitted |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for d_key in sorted(daily_records.keys(), reverse=True):
            d = daily_records[d_key]
            v_total = sum(d.get("vetoed_by_reason", {}).values())
            b_total = sum(d.get("buy_blocked_guards", {}).values())
            lines.append(
                f"| `{d['date']}` | {d['runs_count']} | {d['tickers_scanned']} | {d['skipped_no_real_news']} | {v_total} | {b_total} | **{d['orders_submitted']}** |"
            )

        lines.extend([
            "",
            f"### Today's Detailed Breakdown (`{today_str}`)",
            "",
            "#### Vetoes by Reason",
        ])
        today_v = daily_records[today_str].get("vetoed_by_reason", {})
        if today_v:
            for reason, count in sorted(today_v.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- **`{reason}`**: {count}")
        else:
            lines.append("- *(No candidate vetoes recorded today)*")

        lines.extend([
            "",
            "#### Orders Blocked by Position & Risk Guards",
        ])
        today_b = daily_records[today_str].get("buy_blocked_guards", {})
        if today_b:
            for guard, count in sorted(today_b.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- **`{guard}`**: {count}")
        else:
            lines.append("- *(No guard blocks recorded today)*")

        lines.append("")
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write("\n".join(lines))

    except Exception as me:
        logger.warning(f"Notice updating {md_path}: {me}")

    return run_record
