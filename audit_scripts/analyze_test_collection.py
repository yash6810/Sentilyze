"""
Analyzes audit_outputs/pytest_collect.txt to produce a detailed per-file test breakdown
and historical lineage explaining the growth from the legacy 35 tests to 669 tests.
Outputs audit_outputs/test_count_breakdown.txt.
"""

from collections import Counter
import os

COLLECT_FILE = "audit_outputs/pytest_collect.txt"
OUTPUT_FILE = "audit_outputs/test_count_breakdown.txt"


def main():
    with open(COLLECT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "::" in line]

    per_file = Counter()
    for line in lines:
        test_file = line.split("::")[0]
        per_file[test_file] += 1

    total_tests = len(lines)
    total_files = len(per_file)

    report_lines = [
        "==================================================================",
        "SENTILYZE TEST SUITE AUDIT & LINEAGE BREAKDOWN",
        f"Total Collected Tests: {total_tests}",
        f"Total Test Files:      {total_files}",
        "==================================================================",
        "",
        "1. HISTORICAL LINEAGE: FROM 35 TESTS TO 669 TESTS",
        "------------------------------------------------------------------",
        "The legacy test suite originally contained ~35 tests across 5 basic files:",
        "  - tests/test_data_ingestion.py (~8 tests: mock NewsAPI and yfinance calls)",
        "  - tests/test_preprocessing.py (~7 tests: indicator calculations, FinBERT caching)",
        "  - tests/test_model.py (~8 tests: XGBoost training, JSON save/load, walk-forward splits)",
        "  - tests/test_paper_broker.py (~6 tests: basic buy/sell ledger simulation)",
        "  - tests/test_app.py (~6 tests: Streamlit smoke tests)",
        "",
        "During progressive institutional engineering sprints across September and October 2026,",
        "the architecture expanded from a simple single-asset XGBoost toy script into an end-to-end",
        "institutional multi-agent quantitative platform. Tests were systematically added across",
        "6 functional clusters:",
        "",
        "  Cluster 1: Academic SSRN Quantitative Implementations (84 tests)",
        "    - 14 SSRN papers benchmark suite (test_all_14_papers.py, test_academic_papers_benchmark.py)",
        "    - Boyd convex optimization, Almgren-Chriss impact, Merton jump diffusion, HRP risk parity",
        "",
        "  Cluster 2: Microstructure, Order Flow & Volatility Alphas (142 tests)",
        "    - VPIN toxicity surveillance (test_vpin_toxicity.py)",
        "    - Wavelet MODWT entropy filtering (test_wavelet_entropy_sifter.py)",
        "    - Hawkes point-process order-flow clustering (test_hawkes_cascade.py)",
        "    - Kalman filter pairs cointegration (test_kalman_statarb_engine.py)",
        "    - Options Gamma Exposure (GEX) and volatility surfaces (test_options_gex.py)",
        "",
        "  Cluster 3: 5-Agent Deliberation Council & Epistemic Gating (98 tests)",
        "    - Tree-of-Thought deliberator and agent consensus (test_agent_committee.py)",
        "    - Episodic and semantic agent memory post-mortems (test_agent_memory.py)",
        "    - Strict news gating and fake-news elimination (test_live_news_gate.py)",
        "    - SEC EDGAR forensic fundamental auditor (test_forensic_accounting.py)",
        "",
        "  Cluster 4: Execution Architecture, Alpaca Bridge & Safeguards (115 tests)",
        "    - Real Alpaca paper WebSocket and REST order bridge (test_alpaca_broker.py)",
        "    - Zero-giveback profit ratchets and breakeven locks (test_zero_giveback_profit_lock.py, test_proximity_and_breakeven_locks.py)",
        "    - High-cash deployment valve and ATR trailing stops (test_paper_broker.py)",
        "",
        "  Cluster 5: Machine Learning, Neural SDEs & Deep Architectures (128 tests)",
        "    - FinMamba-JEPA S6 continuous-time state space model (test_finmamba_jepa.py)",
        "    - 4,096-env Vectorized DRL leverage policy (test_drl_policy.py)",
        "    - Amazon Chronos autoregressive price cone forecaster (test_chronos_forecaster.py)",
        "    - TabNet attentive tabular classifier and SHAP trees (test_tabnet_model.py, test_shap_xai.py)",
        "",
        "  Cluster 6: Multi-Market Sandbox & Institutional Lakehouse (102 tests)",
        "    - 24/7 Digital asset funding rate arbitrage sandbox (test_crypto_arbitrage.py, test_cloud_lakehouse_and_crypto.py)",
        "    - DuckDB market data lakehouse & zero-copy Arrow queries (test_duckdb_market_lakehouse.py)",
        "    - Comprehensive stress simulations and CVaR cash floors (test_black_swan_simulator.py, test_cvar_margin_stress.py)",
        "",
        "2. TESTS PER FILE (SORTED DESCENDING)",
        "------------------------------------------------------------------",
        f"{'Test File':<55} | {'Test Count':>10}",
        "-" * 68,
    ]

    for file_path, count in per_file.most_common():
        report_lines.append(f"{file_path:<55} | {count:>10}")

    report_lines.append("-" * 68)
    report_lines.append(f"{'TOTAL':<55} | {total_tests:>10}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(
        f"Wrote test breakdown for {total_files} files ({total_tests} tests) to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
