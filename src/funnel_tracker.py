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

FUNNEL_DIR = os.getenv("FUNNEL_DIR", "results")
FUNNEL_JSON_PATH = os.path.join(FUNNEL_DIR, "funnel.json")
FUNNEL_MD_PATH = os.path.join(FUNNEL_DIR, "funnel.md")
FUNNEL_LOG_PATH = os.path.join(FUNNEL_DIR, "funnel_log.jsonl")


def record_funnel_run(
    tickers_scanned: int,
    skipped_no_real_news: int,
    vetoed_by_reason: Dict[str, int],
    buy_blocked_guards: Dict[str, int],
    orders_submitted: int,
    skipped_news_quota_or_error: int = 0,
    json_path: Optional[str] = None,
    md_path: Optional[str] = None,
    log_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Records cycle funnel metrics to results/funnel.json, appends to results/funnel_log.jsonl,
    and accumulates daily total in results/funnel.md.
    """
    funnel_dir = os.getenv("FUNNEL_DIR", "results")
    if json_path is None:
        json_path = os.path.join(funnel_dir, "funnel.json")
    if md_path is None:
        md_path = os.path.join(funnel_dir, "funnel.md")
    if log_path is None:
        log_path = os.path.join(funnel_dir, "funnel_log.jsonl")
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    timestamp_str = now_utc.isoformat()

    run_record = {
        "timestamp": timestamp_str,
        "date": today_str,
        "tickers_scanned": int(tickers_scanned),
        "skipped_no_real_news": int(skipped_no_real_news),
        "skipped_news_quota_or_error": int(skipped_news_quota_or_error),
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

    # 2. Append to results/funnel_log.jsonl (persistent log across ephemeral runs)
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(run_record) + "\n")
    except Exception as le:
        logger.warning(f"Notice appending {log_path}: {le}")

    # 3. Update results/funnel.md (accumulate daily totals from persistent log)
    try:
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        daily_records: Dict[str, Dict[str, Any]] = {}

        # If log_path exists, aggregate all runs by date
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        d_str = entry.get("date") or entry.get("timestamp", "")[:10]
                        if not d_str:
                            continue
                        if d_str not in daily_records:
                            daily_records[d_str] = {
                                "date": d_str,
                                "tickers_scanned": 0,
                                "skipped_no_real_news": 0,
                                "skipped_news_quota_or_error": 0,
                                "vetoed_by_reason": {},
                                "buy_blocked_guards": {},
                                "orders_submitted": 0,
                                "runs_count": 0,
                            }
                        day = daily_records[d_str]
                        day["runs_count"] += 1
                        day["tickers_scanned"] += int(entry.get("tickers_scanned", 0))
                        day["skipped_no_real_news"] += int(
                            entry.get("skipped_no_real_news", 0)
                        )
                        day["skipped_news_quota_or_error"] += int(
                            entry.get("skipped_news_quota_or_error", 0)
                        )
                        day["orders_submitted"] += int(entry.get("orders_submitted", 0))

                        for k, v in entry.get("vetoed_by_reason", {}).items():
                            day["vetoed_by_reason"][k] = (
                                day["vetoed_by_reason"].get(k, 0) + v
                            )
                        for k, v in entry.get("buy_blocked_guards", {}).items():
                            day["buy_blocked_guards"][k] = (
                                day["buy_blocked_guards"].get(k, 0) + v
                            )
                    except Exception:
                        continue
        else:
            # Fallback to single run record
            daily_records[today_str] = {
                "date": today_str,
                "tickers_scanned": run_record["tickers_scanned"],
                "skipped_no_real_news": run_record["skipped_no_real_news"],
                "skipped_news_quota_or_error": run_record[
                    "skipped_news_quota_or_error"
                ],
                "vetoed_by_reason": dict(vetoed_by_reason),
                "buy_blocked_guards": dict(buy_blocked_guards),
                "orders_submitted": run_record["orders_submitted"],
                "runs_count": 1,
            }

        # Render Markdown Table
        lines = [
            "# 🔍 Sentilyze Execution Funnel Telemetry",
            "",
            "Tracks the multi-stage elimination pipeline per trading session: Scanned → News Check → Committee Veto → Guards → Execution.",
            "",
            "## Daily Execution Totals",
            "",
            "| Date | Runs | Scanned | Skipped (No News) | Skipped (Quota/Error) | Vetoed | Blocked by Guards | Orders Submitted |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for d_key in sorted(daily_records.keys(), reverse=True):
            d = daily_records[d_key]
            v_total = sum(d.get("vetoed_by_reason", {}).values())
            b_total = sum(d.get("buy_blocked_guards", {}).values())
            lines.append(
                f"| `{d['date']}` | {d['runs_count']} | {d['tickers_scanned']} | {d['skipped_no_real_news']} | {d.get('skipped_news_quota_or_error', 0)} | {v_total} | {b_total} | **{d['orders_submitted']}** |"
            )

        lines.extend(
            [
                "",
                f"### Today's Detailed Breakdown (`{today_str}`)",
                "",
                "#### Vetoes by Reason",
            ]
        )
        today_v = daily_records[today_str].get("vetoed_by_reason", {})
        if today_v:
            for reason, count in sorted(
                today_v.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"- **`{reason}`**: {count}")
        else:
            lines.append("- *(No candidate vetoes recorded today)*")

        lines.extend(
            [
                "",
                "#### Orders Blocked by Position & Risk Guards",
            ]
        )
        today_b = daily_records[today_str].get("buy_blocked_guards", {})
        if today_b:
            for guard, count in sorted(
                today_b.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"- **`{guard}`**: {count}")
        else:
            lines.append("- *(No guard blocks recorded today)*")

        lines.append("")
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write("\n".join(lines))

    except Exception as me:
        logger.warning(f"Notice updating {md_path}: {me}")

    return run_record
