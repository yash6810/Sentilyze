# GitHub Actions Workflows Audit Table (Branch: cleanup-honest-results)

This table documents every workflow located in `.github/workflows/`, its trigger schedule, target files committed to the repository, concurrency behavior, and the precise hardening changes implemented on branch `cleanup-honest-results` relative to `main`.

| Workflow File | Cron Schedule | Files Committed | Can Run Concurrently? | What Changed vs `main` |
| :--- | :--- | :--- | :--- | :--- |
| `.github/workflows/agent_committee_audit.yml` | `0 2 * * 1-5` (02:00 UTC Mon-Fri) | `results/committee_resolutions.json` | Yes (runs concurrently with any workflow active at 02:00 UTC) | Unchanged vs `main`. |
| `.github/workflows/autonomous_live_trader.yml` | `*/5 13-20 * * 1-5` (Every 5 min during US session) | `results/paper_portfolio.json`, `results/executed_trades.csv`, `results/autonomous_execution_log.json`, `results/committee_resolutions.json` (daily session close only, >=20:00 UTC) | No (within its own workflow via `concurrency: group: trading-state, cancel-in-progress: false`). Yes with other workflows lacking concurrency groups. | 1. Added `concurrency: group: trading-state`. 2. Restricted git commit to session close (>=20:00 UTC) or manual dispatch instead of every 5 min. 3. Removed `|| true` and `-X ours`; replaced with 3-attempt `--rebase` retry loop. 4. Removed `models/*.json` from commit path. |
| `.github/workflows/ci.yml` | None (push to `main`, PRs to `main`, `workflow_dispatch`) | None (Read-only CI pipeline: Black format check, Bandit security scan, pip-audit, Pytest) | Yes (can run concurrently with any scheduled workflow) | Unchanged vs `main`. |
| `.github/workflows/daily_signals.yml` | `30 2 * * 1-6`, `30 12 * * 1-5` | `results/` (pre-computed model signals & predictions) | Yes (runs concurrently with any workflow active at 02:30 or 12:30 UTC) | Unchanged vs `main`. |
| `.github/workflows/intraday_market_tracker.yml` | `*/5 13-21 * * 1-5` (Every 5 min Mon-Fri) | None (Stripped) | Yes (runs every 5 minutes during trading hours) | Removed `git commit` and `git push` steps completely. It previously committed `results/paper_portfolio.json` and `results/executed_trades.csv`, causing high-frequency race conditions with `autonomous_live_trader.yml`. It now operates strictly read-only for Discord price alerts. |
| `.github/workflows/nightly_data_sync.yml` | `0 0 * * *` (Daily midnight UTC) | `results/` (historical price parquet files & DuckDB sync) | Yes (runs concurrently with any workflow active at 00:00 UTC) | Unchanged vs `main`. |
| `.github/workflows/opening_range_breakout.yml` | `35 13 * * 1-5` (13:35 UTC Mon-Fri) | `results/` (`opening_range_latest.json`) | Yes (runs concurrently with any workflow active at 13:35 UTC) | Unchanged vs `main`. |
| `.github/workflows/portfolio_stress_audit.yml` | `0 5 * * 0` (05:00 UTC Sunday) | `results/` (CVaR stress test and sensitivity matrices) | Yes (runs concurrently with Sunday scheduled workflows) | Unchanged vs `main`. |
| `.github/workflows/premarket_briefing.yml` | `0 13 * * 1-5` (13:00 UTC Mon-Fri) | `results/morning_briefing_latest.json` | Yes (runs concurrently with any workflow active at 13:00 UTC) | Unchanged vs `main`. |
| `.github/workflows/repo_issue_auto_triage.yml` | `0 6 * * *` (06:00 UTC daily) | None (Creates GitHub issues for untriaged topics, no repo file commits) | Yes (can run concurrently with any scheduled workflow) | Unchanged vs `main`. |
| `.github/workflows/sentilyze_247_cloud.yml` | `*/15 13-20 * * 1-5`, `0 0-12,21-23 * * 1-5`, `0 * * * 0,6` | `results/pilot_crypto_portfolio.json`, `results/pilot_crypto_trades.csv`, `results/lakehouse/` | Yes (runs every 15 min during market hours and hourly off-hours) | Removed `results/paper_portfolio.json`, `results/executed_trades.csv`, `results/autonomous_execution_log.json`, `results/committee_resolutions.json`, and `models/` from commit path. Prevents crypto daemon from overwriting equity ledger. |
| `.github/workflows/weekly_retrain.yml` | `30 4 * * 0` (04:30 UTC Sunday) | `models/`, `results/` (Walk-forward retraining models and backtest metrics) | Yes (runs concurrently with Sunday scheduled workflows) | Unchanged vs `main`. |

## Raw Git Diff vs Main

Raw diff output is saved to `audit_outputs/workflow_diff.txt`.
