"""
Generates STRATEGY_FREEZE.md and tests/test_strategy_freeze.py.
Computes SHA-256 hashes of all frozen files across:
- All src/*.py modules
- Key scripts: scripts/run_247_daemon.py, scripts/weekly_scorecard.py
- Live trader workflow: .github/workflows/autonomous_live_trader.yml
- Pinned dependency files: requirements.txt, requirements-dev.txt
- Persistent learned state files:
  * results/agent_learning_memory.json
  * results/agent_memory/committee_weights.json
  * results/agent_memory/semantic_rules.json
  * results/agent_memory/session_episodic_memory.json
  * results/agent_memory/trade_postmortems.jsonl
- All trained XGBoost production models in models/*.json
"""

import hashlib
import glob
import json
import os

LEARNED_STATE_FILES = [
    "results/agent_learning_memory.json",
    "results/agent_memory/committee_weights.json",
    "results/agent_memory/semantic_rules.json",
    "results/agent_memory/session_episodic_memory.json",
    "results/agent_memory/trade_postmortems.jsonl",
]

CONFIG_AND_WORKFLOW_FILES = [
    ".github/workflows/autonomous_live_trader.yml",
    "requirements.txt",
    "requirements-dev.txt",
    "scripts/run_247_daemon.py",
    "scripts/weekly_scorecard.py",
]

RUNTIME_INPUT_FILES = [
    "stocks.txt",
]

NON_JSON_MODELS = sorted(
    glob.glob("models/*.pt")
    + glob.glob("models/drl_policy/*.pt")
    + glob.glob("models/finbert-fine-tuned/*.*")
)

SRC_FILES = sorted(glob.glob("src/*.py"))
MODELS = sorted(glob.glob("models/*.json"))

ALL_FROZEN_PATHS = sorted(
    SRC_FILES
    + CONFIG_AND_WORKFLOW_FILES
    + LEARNED_STATE_FILES
    + RUNTIME_INPUT_FILES
    + NON_JSON_MODELS
    + MODELS
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    hashes = {}
    for f in ALL_FROZEN_PATHS:
        normalized_path = f.replace("\\", "/")
        hashes[normalized_path] = sha256_file(f)

    # Save manifest JSON for programmatic verification
    manifest_path = "results/strategy_freeze_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)
    print(f"Saved manifest of {len(hashes)} files to {manifest_path}")

    # Generate STRATEGY_FREEZE.md
    lines = [
        "# Sentilyze Quantitative Strategy Freeze Protocol",
        "",
        "**Effective Freeze Date**: `2026-11-02`  ",
        "**Baseline Fix Commit Hash**: `GIT_COMMIT_HASH_PLACEHOLDER`  ",
        "**Freeze Expiration**: When `results/scorecard.md` outputs `PASS` or `FAIL` (minimum 8 full weeks and 30 closed trades).  ",
        "**Enforcement Mechanism**: Automated CI test [`tests/test_strategy_freeze.py`](tests/test_strategy_freeze.py).  ",
        "",
        "## 1. Freeze Mandate",
        "",
        "From **November 02, 2026 forward**, no changes, adjustments, re-calibrations, or hotfixes are permitted to:",
        f"1. **Core Strategy & Risk Source Modules ({len(SRC_FILES)} files)**: All modules in [`src/*.py`](src/)",
        f"2. **Production Model Weights ({len(MODELS)} models)**: All XGBoost JSON models in [`models/*.json`](models/)",
        "3. **Persistent Learned State & Memory**: [`results/agent_learning_memory.json`](results/agent_learning_memory.json) and all files in [`results/agent_memory/`](results/agent_memory/)",
        "4. **Live Execution Workflow & Scripts**: [`.github/workflows/autonomous_live_trader.yml`](.github/workflows/autonomous_live_trader.yml), [`scripts/run_247_daemon.py`](scripts/run_247_daemon.py), [`scripts/weekly_scorecard.py`](scripts/weekly_scorecard.py)",
        "5. **Pinned Dependency Requirements**: [`requirements.txt`](requirements.txt) & [`requirements-dev.txt`](requirements-dev.txt)",
        "",
        "> [!IMPORTANT]",
        "> **Any change to a frozen file restarts the 8-week clock and must be logged in `CHANGELOG.md` with date and reason.**",
        "",
        "Any modification to strategy parameters, thresholds, risk gates, learning state, dependencies, or model weights will cause the continuous integration test `tests/test_strategy_freeze.py` to immediately fail.",
        "",
        "## 2. Learned State & Pinned Configuration Hashes (SHA-256)",
        "",
        "| File Path | SHA-256 Digest |",
        "| :--- | :--- |",
    ]

    for p in CONFIG_AND_WORKFLOW_FILES + LEARNED_STATE_FILES:
        norm = p.replace("\\", "/")
        lines.append(f"| `{norm}` | `{hashes[norm]}` |")

    lines.extend([
        "",
        "## 3. Core Strategy Modules Sample (SHA-256)",
        "",
        "| Core Module | SHA-256 Digest |",
        "| :--- | :--- |",
        f"| `src/agent_committee.py` | `{hashes['src/agent_committee.py']}` |",
        f"| `src/autonomous_trader.py` | `{hashes['src/autonomous_trader.py']}` |",
        f"| `src/paper_broker.py` | `{hashes['src/paper_broker.py']}` |",
        f"| `src/alpaca_broker.py` | `{hashes['src/alpaca_broker.py']}` |",
        f"| `src/post_mortem_learner.py` | `{hashes['src/post_mortem_learner.py']}` |",
        f"| `src/agent_memory.py` | `{hashes['src/agent_memory.py']}` |",
        "",
        f"## 4. Production Model Weights ({len(MODELS)} Models)",
        "",
        f"All {len(MODELS)} XGBoost production models in `models/*.json` are cryptographically frozen. The complete list of SHA-256 signatures is tracked in [`results/strategy_freeze_manifest.json`](results/strategy_freeze_manifest.json). Below is a sample of key active universe models:",
        "",
        "| Model File | SHA-256 Digest |",
        "| :--- | :--- |",
    ])

    sample_tickers = [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "SPY",
        "QQQ",
        "WMT",
    ]
    for t in sample_tickers:
        m_path = f"models/{t}_model.json"
        if m_path in hashes:
            lines.append(f"| `{m_path}` | `{hashes[m_path]}` |")

    lines.append("")
    lines.append("## 5. Runtime Inputs & Neural Tensorial Weights (SHA-256)")
    lines.append("")
    lines.append("Every non-.py file read on the live execution path is cryptographically signed:")
    lines.append("")
    lines.append("| File Path | SHA-256 Digest |")
    lines.append("| :--- | :--- |")
    for r in RUNTIME_INPUT_FILES + NON_JSON_MODELS:
        norm_r = r.replace("\\", "/")
        lines.append(f"| `{norm_r}` | `{hashes[norm_r]}` |")

    lines.append("")
    lines.append("## 6. Exit Behavior & Brokerage Execution Model")
    lines.append("")
    lines.append("- **Alpaca Live Account**: Positions on the Alpaca paper brokerage account (`PA3NFG0UWP3R`) exit **exclusively through native entry-time exchange-side OCO bracket legs** (`take_profit.limit_price` and `stop_loss.stop_price`). Alpaca's matching engine executes either the limit sell or stop sell automatically; when one leg fills, Alpaca server-side cancels the other leg.")
    lines.append("- **Simulation Ledger**: Intraday discretionary exits, committee veto exits, and dynamic trailing stops evaluated in `AutonomousTradingEngine._evaluate_portfolio_exits()` apply **strictly to the local simulation ledger** (`results/paper_portfolio.json`). They do not submit market or limit sell orders to Alpaca.")
    lines.append("")
    lines.append("---")
    lines.append(
        "*Enforced by Sentilyze Audit Protocol and `tests/test_strategy_freeze.py`.*"
    )

    with open("STRATEGY_FREEZE.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote STRATEGY_FREEZE.md")

    # Generate tests/test_strategy_freeze.py
    test_code = '''"""
Regression Test: Quantitative Strategy Freeze Enforcement.
Fails if any frozen core strategy file, workflow, pinned dependency,
learned-state file, or production model has been modified from its
cryptographic SHA-256 hash recorded on 2026-11-02.
"""

import os
import json
import hashlib
import pytest

MANIFEST_PATH = "results/strategy_freeze_manifest.json"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_strategy_freeze_hashes_unaltered():
    assert os.path.exists(MANIFEST_PATH), f"Missing freeze manifest at {MANIFEST_PATH}"
    assert os.path.exists("STRATEGY_FREEZE.md"), "Missing STRATEGY_FREEZE.md protocol document"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mismatches = []
    missing_files = []

    for file_path, expected_hash in manifest.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            continue

        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            mismatches.append({
                "file": file_path,
                "expected": expected_hash,
                "actual": actual_hash,
            })

    assert not missing_files, f"Frozen strategy files missing: {missing_files}"
    assert not mismatches, (
        f"STRATEGY FREEZE VIOLATION! The following files have been modified since 2026-11-02: "
        f"{mismatches}"
    )
'''
    with open("tests/test_strategy_freeze.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    print("Wrote tests/test_strategy_freeze.py")


if __name__ == "__main__":
    main()
