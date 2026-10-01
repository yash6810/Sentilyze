"""
Unit tests for Cryptographic Merkle Trade Ledger (Sprint 1, Module 1.7)
"""

import pytest
import os
from src.merkle_ledger import MerkleTradeLedger, hash_payload, sha256_hash


def test_hash_determinism():
    payload1 = {"ticker": "NVDA", "shares": 10, "price": 220.0}
    payload2 = {"price": 220.0, "shares": 10, "ticker": "NVDA"}  # different key order
    assert hash_payload(payload1) == hash_payload(payload2)


def test_append_and_root(tmp_path):
    ledger_file = tmp_path / "mock_merkle_ledger.json"
    ledger = MerkleTradeLedger(ledger_path=str(ledger_file))

    h0 = ledger.append_entry("TRADE_EXECUTION", {"ticker": "NVDA", "pnl": 500.0})
    assert len(ledger.leaves) == 1
    assert ledger.root is not None
    assert len(ledger.root) == 64

    h1 = ledger.append_entry("TRADE_EXECUTION", {"ticker": "AAPL", "pnl": -150.0})
    assert len(ledger.leaves) == 2
    assert ledger.root != h0


def test_merkle_proof_verification(tmp_path):
    ledger_file = tmp_path / "mock_merkle_ledger_proof.json"
    ledger = MerkleTradeLedger(ledger_path=str(ledger_file))

    # Add 5 leaves to test odd number branching
    leaf_hashes = []
    for i in range(5):
        h = ledger.append_entry(
            "COMMITTEE_RESOLUTION", {"ticker": f"TICKER_{i}", "action": "BUY"}
        )
        leaf_hashes.append(h)

    current_root = ledger.root

    for i in range(5):
        proof = ledger.get_merkle_proof(i)
        is_valid = MerkleTradeLedger.verify_proof(leaf_hashes[i], proof, current_root)
        assert is_valid is True

    # Tampered leaf hash must fail verification
    tampered_hash = sha256_hash("fake_content")
    proof_0 = ledger.get_merkle_proof(0)
    assert MerkleTradeLedger.verify_proof(tampered_hash, proof_0, current_root) is False


def test_verify_entire_ledger_tamper_detection(tmp_path):
    ledger_file = tmp_path / "mock_merkle_tamper.json"
    ledger = MerkleTradeLedger(ledger_path=str(ledger_file))

    ledger.append_entry("TRADE", {"ticker": "NVDA", "shares": 10})
    ledger.append_entry("TRADE", {"ticker": "MSFT", "shares": 20})

    audit_clean = ledger.verify_entire_ledger()
    assert audit_clean["is_valid"] is True
    assert audit_clean["status"] == "LEDGER_INTEGRITY_VERIFIED"

    # Deliberately mutate a record payload in memory
    ledger.leaves[0]["record"]["payload"]["shares"] = 999999
    audit_tampered = ledger.verify_entire_ledger()
    assert audit_tampered["is_valid"] is False
    assert audit_tampered["status"] == "TAMPER_DETECTED"
