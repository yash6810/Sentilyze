with open("src/alpaca_broker.py", "r", encoding="utf-8") as f:
    c = f.read()

old = """    def get_open_positions(self) -> List[Dict[str, Any]]:
        \"\"\"Fetches active positions from Alpaca brokerage.\"\"\"
        if not self.is_connected():
            return []
        try:
            res = requests.get(
                f\"{self.base_url}/v2/positions\", headers=self.headers, timeout=6
            )
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.error(f\"Alpaca positions error: {e}\")
        return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        \"\"\"Fetches active/pending orders from Alpaca brokerage.\"\"\"
        if not self.is_connected():
            return []
        try:
            res = requests.get(
                f\"{self.base_url}/v2/orders?status=open\",
                headers=self.headers,
                timeout=6,
            )
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.error(f\"Alpaca open orders error: {e}\")
        return []"""

new = """    def get_open_positions(self) -> List[Dict[str, Any]]:
        \"\"\"Fetches active positions from Alpaca brokerage.\"\"\"
        if not self.is_connected():
            return []
        try:
            res = requests.get(
                f\"{self.base_url}/v2/positions\", headers=self.headers, timeout=6
            )
            if res.status_code == 200:
                return res.json()
            res.raise_for_status()
        except Exception as e:
            logger.error(f\"Alpaca positions error: {e}\")
            raise
        return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        \"\"\"Fetches active/pending orders from Alpaca brokerage.\"\"\"
        if not self.is_connected():
            return []
        try:
            res = requests.get(
                f\"{self.base_url}/v2/orders?status=open\",
                headers=self.headers,
                timeout=6,
            )
            if res.status_code == 200:
                return res.json()
            res.raise_for_status()
        except Exception as e:
            logger.error(f\"Alpaca open orders error: {e}\")
            raise
        return []"""

assert old in c, "Target text not matched!"
c = c.replace(old, new, 1)

with open("src/alpaca_broker.py", "w", encoding="utf-8") as f:
    f.write(c)

print("Updated get_open_positions and get_open_orders to raise!")
