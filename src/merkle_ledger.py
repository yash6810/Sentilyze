"""
Cryptographic Merkle Trade Ledger (Sprint 1, Module 1.7 / Idea 42)

Provides an immutable, tamper-evident cryptographic audit trail for:
1. Agent Committee deliberations and votes
2. Paper broker order fills and position transitions
3. Model checkpoints and prediction provenance

Implements binary SHA-256 Merkle tree construction, audit path proofs,
and zero-dependency ledger integrity verification.
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

logger = logging.getLogger("Sentilyze.MerkleLedger")
logging.basicConfig(level=logging.INFO)

DEFAULT_LEDGER_PATH = os.path.join("results", "merkle_audit_ledger.json")


def sha256_hash(data: str) -> str:
    """Computes standard hex SHA-256 digest."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_payload(payload: Dict[str, Any]) -> str:
    """Produces deterministic canonical SHA-256 hash of a JSON-serializable dictionary."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_hash(canonical_json)


class MerkleTradeLedger:
    """
    Cryptographic Merkle tree ledger maintaining an append-only verifiable log.
    """

    def __init__(self, ledger_path: str = DEFAULT_LEDGER_PATH):
        self.ledger_path = ledger_path
        self.leaves: List[Dict[str, Any]] = []
        self.tree_levels: List[List[str]] = []
        self.root: Optional[str] = None
        self.load_ledger()

    def append_entry(
        self,
        entry_type: str,
        payload: Dict[str, Any],
        author: str = "SentilyzeCouncil",
    ) -> str:
        """
        Appends an event to the ledger and recalculates the Merkle root.
        entry_type can be 'COMMITTEE_RESOLUTION', 'TRADE_EXECUTION', 'MODEL_AUDIT', etc.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        canonical_content = {
            "entry_index": len(self.leaves),
            "entry_type": entry_type,
            "author": author,
            "timestamp": timestamp,
            "payload": payload,
        }
        leaf_hash = hash_payload(canonical_content)

        leaf_record = {
            "index": len(self.leaves),
            "leaf_hash": leaf_hash,
            "record": canonical_content,
        }
        self.leaves.append(leaf_record)
        self.build_tree()
        self.save_ledger()
        return leaf_hash

    def build_tree(self) -> str:
        """
        Constructs the binary Merkle tree from current leaf hashes.
        Returns the top Merkle Root hex digest.
        """
        if not self.leaves:
            self.root = None
            self.tree_levels = []
            return ""

        current_level = [leaf["leaf_hash"] for leaf in self.leaves]
        self.tree_levels = [current_level]

        while len(current_level) > 1:
            next_level = []
            # If odd count of nodes, duplicate the last node
            if len(current_level) % 2 != 0:
                current_level.append(current_level[-1])

            for i in range(0, len(current_level), 2):
                combined = current_level[i] + current_level[i + 1]
                parent_hash = sha256_hash(combined)
                next_level.append(parent_hash)

            self.tree_levels.append(next_level)
            current_level = next_level

        self.root = current_level[0]
        return self.root

    def get_merkle_proof(self, index: int) -> List[Dict[str, str]]:
        """
        Generates an inclusion proof (sibling hashes and directions) for leaf at index.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError("Leaf index out of bounds for Merkle proof.")

        proof: List[Dict[str, str]] = []
        idx = index

        for level in self.tree_levels[:-1]:
            # Ensure level length is even for pairing
            lvl = list(level)
            if len(lvl) % 2 != 0:
                lvl.append(lvl[-1])

            is_right_child = idx % 2 == 1
            sibling_idx = idx - 1 if is_right_child else idx + 1

            if sibling_idx < len(lvl):
                proof.append(
                    {
                        "sibling_hash": lvl[sibling_idx],
                        "position": "left" if is_right_child else "right",
                    }
                )
            idx = idx // 2

        return proof

    @staticmethod
    def verify_proof(
        leaf_hash: str, proof: List[Dict[str, str]], expected_root: str
    ) -> bool:
        """
        Verifies that leaf_hash belongs to the Merkle tree with expected_root using proof.
        """
        current_hash = leaf_hash
        for step in proof:
            sibling = step["sibling_hash"]
            if step["position"] == "left":
                current_hash = sha256_hash(sibling + current_hash)
            else:
                current_hash = sha256_hash(current_hash + sibling)

        return current_hash == expected_root

    def verify_entire_ledger(self) -> Dict[str, Any]:
        """
        Full cryptographic validation:
        1. Checks every leaf hash matches canonical payload content
        2. Recalculates Merkle tree and validates root consistency
        """
        if not self.leaves:
            return {"status": "EMPTY_LEDGER", "is_valid": True, "entries_checked": 0}

        for idx, leaf in enumerate(self.leaves):
            recomputed_hash = hash_payload(leaf["record"])
            if recomputed_hash != leaf["leaf_hash"]:
                return {
                    "status": "TAMPER_DETECTED",
                    "is_valid": False,
                    "compromised_index": idx,
                    "message": f"Leaf at index {idx} has invalid hash.",
                }

        original_root = self.root
        recomputed_root = self.build_tree()
        is_valid = original_root == recomputed_root

        return {
            "status": "LEDGER_INTEGRITY_VERIFIED" if is_valid else "ROOT_MISMATCH",
            "is_valid": is_valid,
            "merkle_root": recomputed_root,
            "entries_checked": len(self.leaves),
        }

    def save_ledger(self) -> None:
        """Persists ledger state to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
            data = {
                "merkle_root": self.root,
                "total_leaves": len(self.leaves),
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "leaves": self.leaves,
            }
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist Merkle ledger: {e}")

    def load_ledger(self) -> None:
        """Loads existing ledger if present."""
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.leaves = data.get("leaves", [])
                if self.leaves:
                    self.build_tree()
            except Exception as e:
                logger.error(f"Error loading Merkle ledger: {e}")
                self.leaves = []


def get_merkle_ledger() -> MerkleTradeLedger:
    return MerkleTradeLedger()
