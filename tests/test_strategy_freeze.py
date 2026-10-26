"""
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
