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
    ak = (os.getenv("ALPACA_API_KEY", "") or os.getenv("APCA_API_KEY_ID", "")).strip()
    sk = (
        os.getenv("ALPACA_SECRET_KEY", "") or os.getenv("APCA_API_SECRET_KEY", "")
    ).strip()
    raw_base_url = os.getenv(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    ).strip()
    base_url = raw_base_url.rstrip("/")

    # Explicit non-sensitive booleans (no taint propagation)
    has_k = True if len(ak) > 0 else False
    has_s = True if len(sk) > 0 else False

    hdrs = {
        "APCA-API-KEY-ID": ak,
        "APCA-API-SECRET-KEY": sk,
        "Content-Type": "application/json",
    }

    try:
        response = requests.get(f"{base_url}/v2/account", headers=hdrs, timeout=6)
        status_code = response.status_code
        # codeql[py/clear-text-logging-sensitive-data]
        print(
            f"Alpaca Pre-flight Check: api_key_present={has_k}, "
            f"secret_key_present={has_s}, status_code={status_code}"
        )
        return 0 if status_code == 200 else 1
    except Exception:
        # codeql[py/clear-text-logging-sensitive-data]
        print(
            f"Alpaca Pre-flight Check: api_key_present={has_k}, "
            f"secret_key_present={has_s}, status_code=None"
        )
        return 1


if __name__ == "__main__":
    sys.exit(check_alpaca_connectivity())
