"""
Generates audit_outputs/workflow_changes.md containing an audit table of
all 12 GitHub Actions workflows in .github/workflows.
"""

import os
import glob
import re

WORKFLOW_DIR = ".github/workflows"
OUTPUT_FILE = "audit_outputs/workflow_changes.md"

WORKFLOWS_METADATA = [
    {
        "file": ".github/workflows/agent_committee_audit.yml",
        "cron": "`0 2 * * 1-5` (02:00 UTC Mon-Fri)",
        "commits": "`results/committee_resolutions.json`",
        "concurrent": "Yes (runs concurrently with any workflow active at 02:00 UTC)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/autonomous_live_trader.yml",
        "cron": "`*/5 13-20 * * 1-5` (Every 5 min during US session)",
        "commits": "`results/paper_portfolio.json`, `results/executed_trades.csv`, `results/autonomous_execution_log.json`, `results/committee_resolutions.json` (daily session close only, >=20:00 UTC)",
        "concurrent": "No (within its own workflow via `concurrency: group: trading-state, cancel-in-progress: false`). Yes with other workflows lacking concurrency groups.",
        "changed_vs_main": "1. Added `concurrency: group: trading-state`. 2. Restricted git commit to session close (>=20:00 UTC) or manual dispatch instead of every 5 min. 3. Removed `|| true` and `-X ours`; replaced with 3-attempt `--rebase` retry loop. 4. Removed `models/*.json` from commit path.",
    },
    {
        "file": ".github/workflows/ci.yml",
        "cron": "None (push to `main`, PRs to `main`, `workflow_dispatch`)",
        "commits": "None (Read-only CI pipeline: Black format check, Bandit security scan, pip-audit, Pytest)",
        "concurrent": "Yes (can run concurrently with any scheduled workflow)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/daily_signals.yml",
        "cron": "`30 2 * * 1-6`, `30 12 * * 1-5`",
        "commits": "`results/` (pre-computed model signals & predictions)",
        "concurrent": "Yes (runs concurrently with any workflow active at 02:30 or 12:30 UTC)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/intraday_market_tracker.yml",
        "cron": "`*/5 13-21 * * 1-5` (Every 5 min Mon-Fri)",
        "commits": "None (Stripped)",
        "concurrent": "Yes (runs every 5 minutes during trading hours)",
        "changed_vs_main": "Removed `git commit` and `git push` steps completely. It previously committed `results/paper_portfolio.json` and `results/executed_trades.csv`, causing high-frequency race conditions with `autonomous_live_trader.yml`. It now operates strictly read-only for Discord price alerts.",
    },
    {
        "file": ".github/workflows/nightly_data_sync.yml",
        "cron": "`0 0 * * *` (Daily midnight UTC)",
        "commits": "`results/` (historical price parquet files & DuckDB sync)",
        "concurrent": "Yes (runs concurrently with any workflow active at 00:00 UTC)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/opening_range_breakout.yml",
        "cron": "`35 13 * * 1-5` (13:35 UTC Mon-Fri)",
        "commits": "`results/` (`opening_range_latest.json`)",
        "concurrent": "Yes (runs concurrently with any workflow active at 13:35 UTC)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/portfolio_stress_audit.yml",
        "cron": "`0 5 * * 0` (05:00 UTC Sunday)",
        "commits": "`results/` (CVaR stress test and sensitivity matrices)",
        "concurrent": "Yes (runs concurrently with Sunday scheduled workflows)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/premarket_briefing.yml",
        "cron": "`0 13 * * 1-5` (13:00 UTC Mon-Fri)",
        "commits": "`results/morning_briefing_latest.json`",
        "concurrent": "Yes (runs concurrently with any workflow active at 13:00 UTC)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/repo_issue_auto_triage.yml",
        "cron": "`0 6 * * *` (06:00 UTC daily)",
        "commits": "None (Creates GitHub issues for untriaged topics, no repo file commits)",
        "concurrent": "Yes (can run concurrently with any scheduled workflow)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
    {
        "file": ".github/workflows/sentilyze_247_cloud.yml",
        "cron": "`*/15 13-20 * * 1-5`, `0 0-12,21-23 * * 1-5`, `0 * * * 0,6`",
        "commits": "`results/pilot_crypto_portfolio.json`, `results/pilot_crypto_trades.csv`, `results/lakehouse/`",
        "concurrent": "Yes (runs every 15 min during market hours and hourly off-hours)",
        "changed_vs_main": "Removed `results/paper_portfolio.json`, `results/executed_trades.csv`, `results/autonomous_execution_log.json`, `results/committee_resolutions.json`, and `models/` from commit path. Prevents crypto daemon from overwriting equity ledger.",
    },
    {
        "file": ".github/workflows/weekly_retrain.yml",
        "cron": "`30 4 * * 0` (04:30 UTC Sunday)",
        "commits": "`models/`, `results/` (Walk-forward retraining models and backtest metrics)",
        "concurrent": "Yes (runs concurrently with Sunday scheduled workflows)",
        "changed_vs_main": "Unchanged vs `main`.",
    },
]


def main():
    lines = [
        "# GitHub Actions Workflows Audit Table (Branch: cleanup-honest-results)",
        "",
        "This table documents every workflow located in `.github/workflows/`, its trigger schedule, target files committed to the repository, concurrency behavior, and the precise hardening changes implemented on branch `cleanup-honest-results` relative to `main`.",
        "",
        "| Workflow File | Cron Schedule | Files Committed | Can Run Concurrently? | What Changed vs `main` |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in WORKFLOWS_METADATA:
        lines.append(
            f"| `{item['file']}` | {item['cron']} | {item['commits']} | {item['concurrent']} | {item['changed_vs_main']} |"
        )

    lines.append("")
    lines.append("## Raw Git Diff vs Main")
    lines.append("")
    lines.append("Raw diff output is saved to `audit_outputs/workflow_diff.txt`.")
    lines.append("")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(WORKFLOWS_METADATA)} workflow records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
