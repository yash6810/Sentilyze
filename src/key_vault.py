"""
Multi-Party Computation (MPC) Key Vault & Shamir Secret Sharing
(Sprint 4, Module 4.9 / Idea 50)

Implements (k, n) threshold Shamir Secret Sharing with information-theoretic security
over a 256-bit prime Galois field to protect broker keys, API secrets, and signing credentials.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import secrets
from src.utils import get_logger

logger = get_logger(__name__)

# Standard 256-bit Mersenne-like prime: 2^256 - 189 (large enough to fit 32-byte chunks)
PRIME_256 = 2**256 - 189


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm: returns (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = _extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


def _mod_inverse(k: int, p: int) -> int:
    """Computes modular multiplicative inverse k^-1 mod p."""
    k = k % p
    gcd, x, _ = _extended_gcd(k, p)
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    return (x % p + p) % p


def _eval_polynomial(coeffs: List[int], x: int, p: int) -> int:
    """Evaluates polynomial sum_j coeffs[j] * x^j mod p using Horner's method."""
    result = 0
    for coeff in reversed(coeffs):
        result = (result * x + coeff) % p
    return result


def split_secret_into_shares(
    secret: str, n_shares: int = 5, threshold_k: int = 3, prime: int = PRIME_256
) -> List[str]:
    """
    Splits an arbitrary secret string into n shares such that any k shares reconstruct it.
    Returns human-readable share strings in format: 'MPC-<x_idx>-<hex_y>'.
    """
    if threshold_k > n_shares:
        raise ValueError("Threshold k cannot exceed total shares n")
    if threshold_k < 2:
        raise ValueError("Threshold k must be at least 2")

    secret_bytes = secret.encode("utf-8")
    secret_int = int.from_bytes(secret_bytes, byteorder="big")
    if secret_int >= prime:
        raise ValueError("Secret too large for 256-bit prime field")

    # Generate random polynomial of degree k - 1: f(0) = secret_int
    coeffs = [secret_int] + [secrets.randbelow(prime) for _ in range(threshold_k - 1)]

    shares = []
    for x in range(1, n_shares + 1):
        y = _eval_polynomial(coeffs, x, prime)
        shares.append(f"MPC-{x}-{hex(y)[2:]}")

    return shares


def reconstruct_secret_from_shares(shares: List[str], prime: int = PRIME_256) -> str:
    """
    Reconstructs the original secret string from any k distinct shares using Lagrange interpolation.
    """
    if len(shares) < 2:
        raise ValueError("Need at least 2 shares to reconstruct secret")

    points = []
    seen_x = set()
    for s in shares:
        parts = s.strip().split("-")
        if len(parts) != 3 or parts[0] != "MPC":
            raise ValueError(f"Invalid share format: {s}")
        x = int(parts[1])
        y = int(parts[2], 16)
        if x in seen_x:
            continue  # Deduplicate duplicate shares
        seen_x.add(x)
        points.append((x, y))

    k = len(points)
    secret_int = 0

    # Lagrange Interpolation at x = 0: S = sum_j y_j * prod_{m != j} (-x_m / (x_j - x_m)) mod p
    for j in range(k):
        x_j, y_j = points[j]
        numerator = 1
        denominator = 1

        for m in range(k):
            if m == j:
                continue
            x_m, _ = points[m]
            numerator = (numerator * (-x_m)) % prime
            denominator = (denominator * (x_j - x_m)) % prime

        lagrange_coeff = (numerator * _mod_inverse(denominator, prime)) % prime
        secret_int = (secret_int + y_j * lagrange_coeff) % prime

    # Convert integer back to UTF-8 string
    byte_len = (secret_int.bit_length() + 7) // 8
    secret_bytes = secret_int.to_bytes(max(byte_len, 1), byteorder="big")
    try:
        return secret_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            "Decoded bytes are not valid UTF-8: insufficient threshold shares or corrupted share values."
        )


class MultiPartyKeyVault:
    """
    Institutional MPC Key Management Vault.
    Guarantees no single agent or server holds the complete trading API private keys.
    """

    def __init__(self, n_custodians: int = 5, threshold: int = 3):
        self.n = n_custodians
        self.k = threshold
        self.vault_dir = os.path.join("data", "mpc_vault")
        os.makedirs(self.vault_dir, exist_ok=True)

    def provision_broker_credentials(
        self, key_id: str, api_secret: str
    ) -> Dict[str, Any]:
        """
        Shreds the secret across n custodian shares and returns the distributed shares.
        """
        shares = split_secret_into_shares(
            api_secret, n_shares=self.n, threshold_k=self.k
        )
        return {
            "key_id": key_id,
            "threshold_k": self.k,
            "total_custodians_n": self.n,
            "custodian_shares": shares,
            "status": "PROVISIONED_SECURELY",
        }

    def unlock_and_sign_order(
        self, key_id: str, submitted_shares: List[str]
    ) -> Dict[str, Any]:
        """
        Reconstructs the ephemeral signing secret in-memory only when >= k threshold shares are presented.
        """
        if len(submitted_shares) < self.k:
            return {
                "status": "INSUFFICIENT_SHARES",
                "shares_presented": len(submitted_shares),
                "threshold_required": self.k,
                "message": f"Security Policy Rejected: Requires {self.k} custodian signatures, got {len(submitted_shares)}.",
            }

        try:
            secret = reconstruct_secret_from_shares(submitted_shares[: self.k])
            return {
                "status": "AUTHENTICATED",
                "key_id": key_id,
                "reconstructed_secret_masked": f"{secret[:4]}***{secret[-3:]}",
                "unlocked": True,
            }
        except Exception as e:
            return {
                "status": "CORRUPTED_SHARES",
                "error": str(e),
                "unlocked": False,
            }
