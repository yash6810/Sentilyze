import pytest
from src.key_vault import (
    split_secret_into_shares,
    reconstruct_secret_from_shares,
    MultiPartyKeyVault,
)


def test_shamir_split_and_reconstruct():
    secret = "Alpaca_API_Secret_99x#Z"
    shares = split_secret_into_shares(secret, n_shares=5, threshold_k=3)

    assert len(shares) == 5
    for s in shares:
        assert s.startswith("MPC-")

    # Reconstruct with any 3 shares (e.g. shares 0, 2, 4)
    sub_shares = [shares[0], shares[2], shares[4]]
    recovered = reconstruct_secret_from_shares(sub_shares)
    assert recovered == secret

    # Another subset (e.g. shares 1, 2, 3)
    sub_shares_2 = [shares[1], shares[2], shares[3]]
    recovered_2 = reconstruct_secret_from_shares(sub_shares_2)
    assert recovered_2 == secret


def test_shamir_insufficient_shares():
    secret = "Top_Secret_Key_123"
    shares = split_secret_into_shares(secret, n_shares=5, threshold_k=4)

    # Only 3 shares provided when threshold is 4
    sub_shares = [shares[0], shares[1], shares[2]]
    # Either raises ValueError (due to invalid UTF-8) or does not match original secret
    try:
        recovered = reconstruct_secret_from_shares(sub_shares)
        assert recovered != secret
    except ValueError:
        pass  # Expected security rejection


def test_multi_party_key_vault_signing():
    vault = MultiPartyKeyVault(n_custodians=5, threshold=3)
    prov = vault.provision_broker_credentials("ALPACA_LIVE", "live_secret_key_8888")

    assert prov["status"] == "PROVISIONED_SECURELY"
    shares = prov["custodian_shares"]

    # Attempt with 2 shares -> Insufficient
    auth_fail = vault.unlock_and_sign_order("ALPACA_LIVE", shares[:2])
    assert auth_fail["status"] == "INSUFFICIENT_SHARES"
    assert auth_fail["unlocked"] is False if "unlocked" in auth_fail else True

    # Attempt with 3 shares -> Authenticated
    auth_success = vault.unlock_and_sign_order("ALPACA_LIVE", shares[:3])
    assert auth_success["status"] == "AUTHENTICATED"
    assert auth_success["unlocked"] is True
