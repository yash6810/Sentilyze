import os, json
from dotenv import load_dotenv
load_dotenv()

from src.alpaca_broker import AlpacaBrokerBridge

bridge = AlpacaBrokerBridge()
print(f"Connected: {bridge.is_connected()}")
print(f"Base URL: {bridge.base_url}")
print(f"Is Paper: {bridge.is_paper}")

acc = bridge.get_account_summary()
print("\n--- ACCOUNT SUMMARY ---")
for k, v in acc.items():
    print(f"  {k}: {v}")

positions = bridge.get_open_positions()
print(f"\n--- OPEN POSITIONS ({len(positions)} total) ---")
for p in positions:
    print(
        f"Symbol: {p.get('symbol')} | Qty: {p.get('qty')} | Cost Basis: ${float(p.get('cost_basis', 0)):,.2f} | "
        f"Avg Entry Price: ${float(p.get('avg_entry_price', 0)):,.2f} | Market Value: ${float(p.get('market_value', 0)):,.2f} | "
        f"Unrealized PnL: ${float(p.get('unrealized_pl', 0)):,.2f}"
    )

orders = bridge.get_open_orders()
print(f"\n--- OPEN ORDERS ({len(orders)} total) ---")
for o in orders:
    legs = o.get("legs") or []
    legs_str = f" [Legs: {len(legs)}]" if legs else ""
    print(
        f"Order ID: {o.get('id')} | Symbol: {o.get('symbol')} | Side: {o.get('side')} | "
        f"Type: {o.get('type')} | Qty: {o.get('qty')} | Status: {o.get('status')} | "
        f"Limit: {o.get('limit_price')} | Stop: {o.get('stop_price')}{legs_str}"
    )
    for leg in legs:
        print(
            f"   └─ Leg ID: {leg.get('id')} | Type: {leg.get('type')} | Side: {leg.get('side')} | "
            f"Status: {leg.get('status')} | Limit: {leg.get('limit_price')} | Stop: {leg.get('stop_price')}"
        )
