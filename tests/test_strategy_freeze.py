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
    with open(path, "rb") as f:
        raw = f.read()

    # 1. Git LFS pointer: extract oid sha256
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if line.startswith("oid sha256:"):
                return line.split("oid sha256:")[1].strip()

    # 2. Text files: normalize CRLF to LF
    text_exts = {
        ".py",
        ".json",
        ".jsonl",
        ".txt",
        ".yml",
        ".yaml",
        ".md",
        ".toml",
        ".ini",
        ".cfg",
        ".csv",
    }
    ext = os.path.splitext(path)[1].lower()
    if ext in text_exts or (
        ext not in {".pt", ".safetensors", ".bin", ".png", ".jpg", ".npy"}
        and 0 not in raw[:4096]
    ):
        raw = raw.replace(bytes([13, 10]), bytes([10]))

    # 3. Hash raw or normalized bytes
    return hashlib.sha256(raw).hexdigest()


def test_strategy_freeze_hashes_unaltered():
    assert os.path.exists(MANIFEST_PATH), f"Missing freeze manifest at {MANIFEST_PATH}"
    assert os.path.exists(
        "STRATEGY_FREEZE.md"
    ), "Missing STRATEGY_FREEZE.md protocol document"

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
            mismatches.append(
                {
                    "file": file_path,
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )

    assert not missing_files, f"Frozen strategy files missing: {missing_files}"
    assert not mismatches, (
        f"STRATEGY FREEZE VIOLATION! The following files have been modified since 2026-11-02: "
        f"{mismatches}"
    )
