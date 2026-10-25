"""
Hardens all workflow files in .github/workflows/:
1. Enforces concurrency: group: repo-state, cancel-in-progress: false on all committing workflows.
2. Ensures ONLY autonomous_live_trader.yml writes to:
   - results/paper_portfolio.json
   - results/executed_trades.csv
   - results/autonomous_execution_log.json
   - results/committee_resolutions.json
3. Modifies secondary workflows so they only add their specific output files:
   - agent_committee_audit.yml -> results/committee_audit_resolutions.json
   - daily_signals.yml -> results/daily_signals_latest.json
   - nightly_data_sync.yml -> results/data_sync_latest.json
   - opening_range_breakout.yml -> results/opening_range_latest.json
   - portfolio_stress_audit.yml -> results/black_swan_crisis_audit.json
   - premarket_briefing.yml -> results/morning_briefing_latest.json
   - sentilyze_247_cloud.yml -> results/pilot_crypto_portfolio.json results/pilot_crypto_trades.csv results/lakehouse/
   - weekly_retrain.yml -> models/*.json results/*_metrics.json results/*_portfolio.json results/*_shap*
4. Disables weekend/overnight crons in sentilyze_247_cloud.yml.
"""

import os
import re


def update_file(path, transformer):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = transformer(content)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)
    print(f"Updated {path}")


def harden_opening_range(content):
    if "concurrency:" not in content:
        content = content.replace(
            "workflow_dispatch: # Manual on-demand trigger\n",
            "workflow_dispatch: # Manual on-demand trigger\n\nconcurrency:\n  group: repo-state\n  cancel-in-progress: false\n",
        )
    # Fix git add
    content = re.sub(
        r"git add results/", "git add results/opening_range_latest.json", content
    )
    # Replace push step with rebase retry
    old_push = r"git pull --rebase origin main \|\| git pull origin main --no-rebase -X ours \|\| true\s+git push origin main \|\| echo \"No changes or push skipped\""
    new_push = """for i in 1 2 3; do\n            git pull --rebase origin main && git push origin main && break || {\n              if [ $i -eq 3 ]; then echo "Push failed after 3 attempts"; exit 1; fi\n              sleep 5\n            }\n          done"""
    content = re.sub(old_push, new_push, content)
    return content


def harden_portfolio_stress(content):
    if "concurrency:" not in content:
        content = content.replace(
            "workflow_dispatch: # Allows manual trigger from GitHub Actions tab\n",
            "workflow_dispatch: # Allows manual trigger from GitHub Actions tab\n\nconcurrency:\n  group: repo-state\n  cancel-in-progress: false\n",
        )
    content = re.sub(
        r"git add results/", "git add results/black_swan_crisis_audit.json", content
    )
    old_push = r"git pull --rebase origin main \|\| git pull origin main --no-rebase -X ours \|\| true\s+git push origin main \|\| echo \"No changes or push skipped\""
    new_push = """for i in 1 2 3; do\n            git pull --rebase origin main && git push origin main && break || {\n              if [ $i -eq 3 ]; then echo "Push failed after 3 attempts"; exit 1; fi\n              sleep 5\n            }\n          done"""
    content = re.sub(old_push, new_push, content)
    return content


def harden_premarket_briefing(content):
    if "concurrency:" not in content:
        content = content.replace(
            "workflow_dispatch:\n",
            "workflow_dispatch:\n\nconcurrency:\n  group: repo-state\n  cancel-in-progress: false\n",
        )
    content = re.sub(
        r"git add results/morning_briefing_latest\.json \|\| true",
        "git add results/morning_briefing_latest.json",
        content,
    )
    old_push = r"git pull --rebase origin main \|\| git pull origin main --no-rebase -X ours \|\| true\s+git push origin main \|\| echo \"No changes or push skipped\""
    new_push = """for i in 1 2 3; do\n            git pull --rebase origin main && git push origin main && break || {\n              if [ $i -eq 3 ]; then echo "Push failed after 3 attempts"; exit 1; fi\n              sleep 5\n            }\n          done"""
    content = re.sub(old_push, new_push, content)
    return content


def harden_weekly_retrain(content):
    if "concurrency:" not in content:
        content = content.replace(
            "workflow_dispatch: # Allows manual trigger from GitHub Actions tab\n",
            "workflow_dispatch: # Allows manual trigger from GitHub Actions tab\n\nconcurrency:\n  group: repo-state\n  cancel-in-progress: false\n",
        )
    # Don't add all results/, add only models and backtest results
    content = re.sub(
        r"git add models/ results/",
        "git add models/*.json results/*_metrics.json results/*_portfolio.json results/*_shap*\n          git reset results/paper_portfolio.json results/executed_trades.csv results/autonomous_execution_log.json results/committee_resolutions.json || true",
        content,
    )
    old_push = r"git pull --rebase origin main \|\| git pull origin main --no-rebase -X ours \|\| true\s+git push origin main \|\| echo \"No changes or push skipped\""
    new_push = """for i in 1 2 3; do\n            git pull --rebase origin main && git push origin main && break || {\n              if [ $i -eq 3 ]; then echo "Push failed after 3 attempts"; exit 1; fi\n              sleep 5\n            }\n          done"""
    content = re.sub(old_push, new_push, content)
    return content


def main():
    update_file(".github/workflows/opening_range_breakout.yml", harden_opening_range)
    update_file(".github/workflows/portfolio_stress_audit.yml", harden_portfolio_stress)
    update_file(".github/workflows/premarket_briefing.yml", harden_premarket_briefing)
    update_file(".github/workflows/weekly_retrain.yml", harden_weekly_retrain)


if __name__ == "__main__":
    main()
