"""
Pre-flight Alpaca connectivity check for autonomous live trading workflow.
Calls GET /v2/account with environment keys.
Prints ONLY the HTTP status code and whether keys are present.
Exits non-zero if the status isn't 200.
NEVER prints or logs API keys, secrets, or request headers.
"""

import os
import sys
import requests


def check_alpaca_connectivity() -> int:
    api_key = (
        os.getenv("ALPACA_API_KEY", "") or os.getenv("APCA_API_KEY_ID", "")
    ).strip()
    secret_key = (
        os.getenv("ALPACA_SECRET_KEY", "") or os.getenv("APCA_API_SECRET_KEY", "")
    ).strip()
    raw_base_url = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    ).strip()
    base_url = raw_base_url.rstrip("/")

    api_key_present = bool(api_key)
    secret_key_present = bool(secret_key)

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=6)
        status_code = response.status_code
        print(
            f"Alpaca Pre-flight Check: api_key_present={api_key_present}, "
            f"secret_key_present={secret_key_present}, status_code={status_code}"
        )
        return 0 if status_code == 200 else 1
    except Exception:
        print(
            f"Alpaca Pre-flight Check: api_key_present={api_key_present}, "
            f"secret_key_present={secret_key_present}, status_code=None"
        )
        return 1


if __name__ == "__main__":
    sys.exit(check_alpaca_connectivity())
