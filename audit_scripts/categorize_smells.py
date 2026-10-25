import os
import re

with open("audit_outputs/fake_data_smells_raw.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

output_lines = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    # Format: filepath:lineno [pattern] content
    match = re.match(r"^([^:]+):(\d+)\s+\[([^\]]+)\]\s+(.*)$", line)
    if not match:
        continue
    filepath, lineno, pat, content = match.groups()
    filepath_lower = filepath.lower()
    content_lower = content.lower()

    # Determine verdict
    if "data_ingestion.py" in filepath_lower and (
        "_generate_dummy_news" in content_lower
        or "dummy" in content_lower
        or "random.choice" in content_lower
        or "random.randint" in content_lower
    ):
        verdict = "silent fabrication - Generates biased synthetic news headlines and persists them to cache when API fails"
    elif "cross_asset_matrix.py" in filepath_lower and (
        "calibrated synthetic fallback" in content_lower
        or "np.random.normal" in content_lower
        or "np.random.seed" in content_lower
    ):
        verdict = "flagged fallback - Synthetic geometric Brownian motion market data generated when API is unavailable"
    elif "drl_policy_agent.py" in filepath_lower and "np.random.normal" in content_lower:
        verdict = "flagged fallback - Fallback synthetic market trajectory when historical series missing"
    elif "microstructure_engine.py" in filepath_lower and "np.random" in content_lower:
        verdict = "flagged fallback - Synthetic tick jumps generated when orderbook stream is dry"
    elif "insider_signals.py" in filepath_lower and "placeholder" in content_lower:
        verdict = "flagged fallback - Mock insider transactions when SEC EDGAR fails"
    elif "strategy_incubator.py" in filepath_lower:
        verdict = "legitimate - Genetic algorithm mutation and crossover exploration in incubator"
    elif "credit_risk.py" in filepath_lower and "synthetic" in content_lower:
        verdict = "legitimate - Merton model structural synthetic CDS spread formula (ISDA standard terminology)"
    elif "etf_arbitrage.py" in filepath_lower and "synthetic" in content_lower:
        verdict = "legitimate - Industry standard terminology for synthetic constituent basket and iNAV"
    elif "adaptive_execution_router.py" in filepath_lower and "jitter" in content_lower:
        verdict = "legitimate - Anti-sniping execution timing jitter to prevent adverse selection"
    elif "backtesting.py" in filepath_lower and "default_rng" in content_lower:
        verdict = "legitimate - Permutation Monte Carlo statistical significance test seed"
    elif "counterfactual_simulator.py" in filepath_lower and "synthetic" in content_lower:
        verdict = "legitimate - Synthetic barrier replay for stress-testing counterfactual drawdowns"
    elif "conformal_prediction.py" in filepath_lower and "sim_residuals" in content_lower:
        verdict = "flagged fallback - Synthetic residual sampling when calibration set is unpopulated"
    elif "attribution_analysis.py" in filepath_lower and "np.random" in content_lower:
        verdict = "legitimate - Permutation feature shuffling for Shapley attribution analysis"
    elif "paper_broker.py" in filepath_lower and "simulated" in content_lower:
        verdict = "legitimate - Logging prefix to distinguish simulated paper broker from live bridge"
    elif "fast_finbert.py" in filepath_lower and "random" in content_lower:
        verdict = "legitimate - Batch shuffling"
    elif "hedging_engine.py" in filepath_lower and "synthetic" in content_lower:
        verdict = "legitimate - Synthetic option position replication terminology"
    else:
        if "dummy" in content_lower or "synthetic" in content_lower or "placeholder" in content_lower:
            verdict = "flagged fallback - Fallback data structure or mock object"
        elif "random" in content_lower:
            verdict = "legitimate - Mathematical randomized calculation, perturbation, or seed"
        else:
            verdict = "legitimate - Neutral variable naming"

    output_lines.append(f"{filepath}:{lineno} [{pat}] {content} -> {verdict}")

with open("audit_outputs/fake_data_smells.txt", "w", encoding="utf-8") as out:
    for item in output_lines:
        out.write(item + "\n")

print(f"Categorized {len(output_lines)} smells into audit_outputs/fake_data_smells.txt")
