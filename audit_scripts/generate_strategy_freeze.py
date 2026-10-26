"""
Generates STRATEGY_FREEZE.md and tests/test_strategy_freeze.py.
Computes SHA-256 hashes of all frozen files:
- src/agent_committee.py (decision & veto rules)
- src/autonomous_trader.py (sizing & execution cycle)
- models/*.json (all trained XGBoost production model weights)
"""

import hashlib
import glob
import json
import os

FROZEN_FILES = [
    "src/agent_committee.py",
    "src/autonomous_trader.py",
    "results/agent_learning_memory.json",
    "results/agent_memory/committee_weights.json",
]
MODELS = sorted(glob.glob("models/*.json"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    hashes = {}
    for f in FROZEN_FILES + MODELS:
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
        "1. **Entry Rules & Multi-Agent Deliberation**: [`src/agent_committee.py`](src/agent_committee.py)",
        "2. **Position Sizing & Execution Schedulers**: [`src/autonomous_trader.py`](src/autonomous_trader.py)",
        "3. **Persistent Agent Learning Memory**: [`results/agent_learning_memory.json`](results/agent_learning_memory.json) & [`results/agent_memory/committee_weights.json`](results/agent_memory/committee_weights.json)",
        "4. **Production Model Weights**: All XGBoost JSON models in [`models/*.json`](models/)",
        "",
        "Any modification to strategy parameters, thresholds, risk gates, learning state, or model weights will cause the continuous integration test `tests/test_strategy_freeze.py` to immediately fail.",
        "",
        "## 2. Core Execution Engine & Memory Cryptographic Hashes (SHA-256)",
        "",
        "| File Path | SHA-256 Digest |",
        "| :--- | :--- |",
        f"| `src/agent_committee.py` | `{hashes['src/agent_committee.py']}` |",
        f"| `src/autonomous_trader.py` | `{hashes['src/autonomous_trader.py']}` |",
        f"| `results/agent_learning_memory.json` | `{hashes['results/agent_learning_memory.json']}` |",
        f"| `results/agent_memory/committee_weights.json` | `{hashes['results/agent_memory/committee_weights.json']}` |",
        "",
        f"## 3. Production Model Weights ({len(MODELS)} Models)",
        f"| `src/autonomous_trader.py` | `{hashes['src/autonomous_trader.py']}` |",
        "",
        f"## 3. Production Model Weights ({len(MODELS)} Models)",
        "",
        f"All {len(MODELS)} XGBoost production models in `models/*.json` are cryptographically frozen. The complete list of SHA-256 signatures is tracked in [`results/strategy_freeze_manifest.json`](results/strategy_freeze_manifest.json). Below is a sample of key active universe models:",
        "",
        "| Model File | SHA-256 Digest |",
        "| :--- | :--- |",
    ]

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
Fails if any frozen core strategy file (src/agent_committee.py, src/autonomous_trader.py,
results/agent_learning_memory.json, results/agent_memory/committee_weights.json,
or models/*.json) has been modified from its cryptographic SHA-256 hash recorded on 2026-11-02.
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
