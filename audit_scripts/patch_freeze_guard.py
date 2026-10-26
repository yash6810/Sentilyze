with open("src/autonomous_trader.py", "r", encoding="utf-8") as f:
    at_code = f.read()

target_at = '''        # Enforce Strategy Freeze: Learning and adaptive weight updates frozen during evaluation
        if os.getenv("LEARNING_FROZEN", "false").lower() in ("true", "1", "yes"):'''

replacement_at = '''        # Enforce Strategy Freeze: Learning and adaptive weight updates frozen during evaluation
        if (
            os.getenv("LEARNING_FROZEN", "false").lower() in ("true", "1", "yes")
            or "PYTEST_CURRENT_TEST" in os.environ
            or getattr(self.paper_broker, "portfolio_file", "") != os.path.join("results", "paper_portfolio.json")
        ):'''

assert target_at in at_code, "Target in autonomous_trader not found"
at_code = at_code.replace(target_at, replacement_at, 1)

with open("src/autonomous_trader.py", "w", encoding="utf-8") as f:
    f.write(at_code)
print("Patched src/autonomous_trader.py")

with open("src/post_mortem_learner.py", "r", encoding="utf-8") as f:
    pml_code = f.read()

target_pml = '''        # Enforce Strategy Freeze: Weight updates frozen during evaluation
        if os.getenv("LEARNING_FROZEN", "false").lower() in ("true", "1", "yes"):'''

replacement_pml = '''        # Enforce Strategy Freeze: Weight updates frozen during evaluation
        if (
            os.getenv("LEARNING_FROZEN", "false").lower() in ("true", "1", "yes")
            or ("PYTEST_CURRENT_TEST" in os.environ and self.memory_path == MEMORY_FILE)
        ):'''

assert target_pml in pml_code, "Target in post_mortem_learner not found"
pml_code = pml_code.replace(target_pml, replacement_pml, 1)

with open("src/post_mortem_learner.py", "w", encoding="utf-8") as f:
    f.write(pml_code)
print("Patched src/post_mortem_learner.py")
