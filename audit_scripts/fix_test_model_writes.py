import os

# 1. Update src/acpm_trainer.py
with open("src/acpm_trainer.py", "r", encoding="utf-8") as f:
    c = f.read()

c = c.replace(
    "benchmark_returns: Optional[pd.Series] = None,\n    ) -> Tuple[Any, Dict[str, Any], pd.Series]:",
    'benchmark_returns: Optional[pd.Series] = None,\n        model_dir: str = "models",\n    ) -> Tuple[Any, Dict[str, Any], pd.Series]:',
)
c = c.replace(
    'os.makedirs("models", exist_ok=True)\n        model_path = f"models/{ticker}_model.json"',
    'os.makedirs(model_dir, exist_ok=True)\n        model_path = os.path.join(model_dir, f"{ticker}_model.json")',
)

with open("src/acpm_trainer.py", "w", encoding="utf-8") as f:
    f.write(c)

# 2. Update tests/test_acpm_trainer.py
with open("tests/test_acpm_trainer.py", "r", encoding="utf-8") as f:
    t = f.read()

t = t.replace(
    "def test_acpm_walk_forward_pipeline_execution():",
    "def test_acpm_walk_forward_pipeline_execution(tmp_path):",
)
t = t.replace(
    "benchmark_returns=spy,\n    )",
    "benchmark_returns=spy,\n        model_dir=str(tmp_path),\n    )",
)
t = t.replace(
    'assert os.path.exists("models/TEST_TICKER_model.json")',
    'assert (tmp_path / "TEST_TICKER_model.json").exists()',
)
t = t.replace(
    'if os.path.exists("models/TEST_TICKER_model.json"):\n        os.remove("models/TEST_TICKER_model.json")',
    "# Cleaned up automatically by tmp_path fixture",
)

with open("tests/test_acpm_trainer.py", "w", encoding="utf-8") as f:
    f.write(t)

print("Successfully updated acpm_trainer and test_acpm_trainer!")
