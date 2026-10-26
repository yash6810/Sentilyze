with open("src/alpaca_broker.py", "r", encoding="utf-8") as f:
    alp_code = f.read()

# Update get_account_summary to raise on error if requests fails
target_gas = """        try:
            res = requests.get(
                f"{self.base_url}/v2/account", headers=self.headers, timeout=6
            )
            if res.status_code == 200:
                data = res.json()
                return {
                    "status": "CONNECTED",
                    "equity": float(data.get("equity", 0)),
                    "cash": float(data.get("cash", 0)),
                    "buying_power": float(data.get("buying_power", 0)),
                    "currency": data.get("currency", "USD"),
                    "mode": "ALPACA PAPER" if self.is_paper else "ALPACA LIVE",
                    "account_number": data.get("account_number", "N/A"),
                }
        except Exception as e:
            logger.error(f"Alpaca account fetch error: {e}")

        return {"status": "ERROR"}"""

replacement_gas = """        try:
            res = requests.get(
                f"{self.base_url}/v2/account", headers=self.headers, timeout=6
            )
            res.raise_for_status()
            data = res.json()
            return {
                "status": "CONNECTED",
                "equity": float(data.get("equity", 0)),
                "cash": float(data.get("cash", 0)),
                "buying_power": float(data.get("buying_power", 0)),
                "currency": data.get("currency", "USD"),
                "mode": "ALPACA PAPER" if self.is_paper else "ALPACA LIVE",
                "account_number": data.get("account_number", "N/A"),
            }
        except Exception as e:
            logger.error(f"Alpaca account fetch error: {e}")
            raise"""

assert target_gas in alp_code, "target_gas not found in alpaca_broker.py"
alp_code = alp_code.replace(target_gas, replacement_gas, 1)

with open("src/alpaca_broker.py", "w", encoding="utf-8") as f:
    f.write(alp_code)
print("Updated src/alpaca_broker.py")
