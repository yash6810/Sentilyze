# GitHub Actions Workflows Audit & Hardening Report

## 1. Inventory of Workflows Writing to the Repository (Before)

| Workflow File | Cron Schedule | Files Committed | Concurrent Run Conflict Risk |
| :--- | :--- | :--- | :--- |
| `.github/workflows/autonomous_live_trader.yml` | `*/5 13-20 * * 1-5` (every 5 min) | `results/paper_portfolio.json`, `results/executed_trades.csv`, `results/autonomous_execution_log.json`, `results/committee_resolutions.json`, `results/agent_learning_memory.json`, `results/agent_memory/`, `results/*_retrain_audit.json`, `models/*.json` | **HIGH**: Collides with `intraday_market_tracker.yml` and `sentilyze_247_cloud.yml` on the same 5-minute interval. Uses `-X ours` and `|| true` masking conflicts. |
| `.github/workflows/intraday_market_tracker.yml` | `*/5 13-21 * * 1-5` (every 5 min) | `results/paper_portfolio.json`, `results/executed_trades.csv` | **HIGH**: Collides directly with `autonomous_live_trader.yml`. Multiple runners overwriting ledger state simultaneously. |
| `.github/workflows/sentilyze_247_cloud.yml` | `*/15 13-20 * * 1-5`, `0 0-12,21-23 * * 1-5`, `0 * * * 0,6` (hourly weekends) | `results/paper_portfolio.json`, `results/executed_trades.csv`, `results/pilot_crypto_portfolio.json`, `results/pilot_crypto_trades.csv`, `results/lakehouse/`, `results/autonomous_execution_log.json`, `results/committee_resolutions.json`, `results/agent_memory/`, `models/` | **HIGH**: Continuous weekend runs and weekday overlap pushing ledger files. |
| `.github/workflows/daily_signals.yml` | `0 21 * * 1-5` (daily close) | `results/` | **MEDIUM**: Overwrites entire results/ directory. |
| `.github/workflows/agent_committee_audit.yml` | `30 20 * * 1-5` | `results/committee_resolutions.json` | **LOW-MEDIUM**: Pushes resolutions. |
| `.github/workflows/opening_range_breakout.yml` | `45 13 * * 1-5` | `results/` | **MEDIUM**: Overwrites results/ directory during trading session. |
| `.github/workflows/portfolio_stress_audit.yml` | `0 22 * * 5` (weekly) | `results/` | **LOW**: Friday close. |
| `.github/workflows/weekly_retrain.yml` | `0 2 * * 0` (weekly Sunday) | `models/`, `results/` | **LOW**: Sunday batch retrain. |
| `.github/workflows/premarket_briefing.yml` | `0 13 * * 1-5` | `results/morning_briefing_latest.json` | **LOW**: Premarket JSON update. |

---

## 2. Structural Root Causes for State Corruption & Overwrites

1. **Dual Ledger Writers**: Both `autonomous_live_trader.yml` and `intraday_market_tracker.yml` were scheduled every 5 minutes during market hours, both running `git add results/paper_portfolio.json results/executed_trades.csv`.
2. **Conflict Masking**: The git push step executed `git pull origin main --no-rebase -X ours || true` followed by `git push origin main || echo "No changes or push skipped"`. The `-X ours` flag discarded remote changes in favor of local changes without error, overwriting real trades with stale container states.
3. **High-Frequency Model Commits**: Committing `models/*.json` every 5 minutes led to bloated git history and unnecessary merge conflicts.

---

## 3. Implemented Hardening & Architecture Changes (After)

1. **Single Source of State**:
   - `.github/workflows/autonomous_live_trader.yml` is designated as the **EXACT ONE** workflow authorized to persist trading state.
   - Added concurrency lock:
     ```yaml
     concurrency:
       group: trading-state
       cancel-in-progress: false
     ```
   - Commit & push step modified to remove `|| true` and `-X ours`. Implemented 3-attempt linear rebase retry:
     ```bash
     for i in 1 2 3; do
       git pull --rebase origin main && git push origin main && break || {
         if [ $i -eq 3 ]; then echo "Push failed after 3 attempts"; exit 1; fi
         sleep 5
       }
     done
     ```
   - Removed `models/*.json` from high-frequency commits. Model serialization is isolated to weekly retraining.
   - State committed at end of day or session close rather than raw file dumping every 5 minutes.

2. **Removal of State Commits from Secondary Workflows**:
   - `.github/workflows/intraday_market_tracker.yml`: Removed `git commit` step for `results/paper_portfolio.json` and `results/executed_trades.csv`. The tracker monitors price movements and delivers Discord alerts without mutating git history.
   - `.github/workflows/sentilyze_247_cloud.yml`: Disabled (commented out) the weekend 24/7 crypto hourly job (`0 * * * 0,6`). Removed equity ledger commits.
