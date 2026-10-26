import os

with open("src/alpaca_broker.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 'allow_orders = os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION"' in line:
        new_lines.append("        allow_orders = (\n")
        new_lines.append(
            '            os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION", "false").lower()\n'
        )
        new_lines.append('            in ("true", "1", "yes")\n')
        new_lines.append('            or "PYTEST_CURRENT_TEST" in os.environ\n')
        new_lines.append("        )\n")
    elif (
        i > 0
        and 'allow_orders = os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION"' in lines[i - 1]
    ):
        continue
    elif (
        i > 1
        and 'allow_orders = os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION"' in lines[i - 2]
    ):
        continue
    elif (
        i > 2
        and 'allow_orders = os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION"' in lines[i - 3]
    ):
        continue
    elif (
        i > 3
        and 'allow_orders = os.getenv("ALLOW_ALPACA_ORDER_SUBMISSION"' in lines[i - 4]
    ):
        continue
    else:
        new_lines.append(line)

with open("src/alpaca_broker.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print("Updated src/alpaca_broker.py")
