"""
Understand-Domain Generator for Sentilyze.
Maps quantitative trading codebase into business domains, flows, and steps.
Writes to .ua/domain-graph.json.
"""

import json
import os
from datetime import datetime, timezone

UA_DIR = ".ua"
os.makedirs(UA_DIR, exist_ok=True)

domain_graph = {
    "version": "1.0.0",
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "project": {
        "name": "Sentilyze Trading OS",
        "description": "Institutional Quantitative Trading & MLOps Platform"
    },
    "domains": [
        {
            "id": "domain:alpha_ingestion",
            "name": "Market Data & Alpha Generation",
            "description": "Real-time quote streaming, news sentiment ingestion via FinBERT, technical indicators, and Order Flow Volume Profile.",
            "components": [
                "src/data_ingestion.py",
                "src/realtime_tracker.py",
                "src/orderflow_chart.py",
                "src/sec_crawler.py"
            ]
        },
        {
            "id": "domain:agent_council",
            "name": "Multi-Agent Deliberation & Quorum",
            "description": "8 specialized market agents deliberate on incoming alpha signals with dynamic memory weights and CRO veto power.",
            "components": [
                "src/agent_committee.py",
                "src/autonomous_trader.py"
            ]
        },
        {
            "id": "domain:market_simulation",
            "name": "MiroFish Multi-Agent Market Simulation",
            "description": "5 market actors (Smart Money, HFT, Retail FOMO, Contrarian Short, CRO) simulate emergent order flow offloaded to Gemini Cloud.",
            "components": [
                "src/cloud_market_simulator.py",
                "src/strategy_incubator.py"
            ]
        },
        {
            "id": "domain:academic_benchmarks",
            "name": "14 Academic Research Paper Suite",
            "description": "Empirical implementations of Hazan ONS, Boyd SOCP, Almgren-Chriss SOR, HRP, and Triple-Barrier ATR corridors.",
            "components": [
                "src/academic_papers_benchmark.py",
                "src/convex_optimizer.py",
                "src/online_newton_step.py",
                "src/triple_barrier.py"
            ]
        },
        {
            "id": "domain:execution_ledger",
            "name": "Execution Engine & Capital Preservation",
            "description": "Smart Order Routing (SOR), Kyle Lambda market impact, and strict paper portfolio preservation ($159,199.62 cash buffer).",
            "components": [
                "src/paper_broker.py",
                "src/hedge_agents.py",
                "src/almgren_chriss_execution.py"
            ]
        }
    ],
    "flows": [
        {
            "id": "flow:live_trade_lifecycle",
            "name": "Live Signal-to-Execution Flow",
            "description": "End-to-end path from ticker selection to filled order in the paper broker.",
            "steps": [
                {
                    "stepNumber": 1,
                    "name": "Asset Ingestion & Feature Engineering",
                    "component": "src/data_ingestion.py",
                    "action": "Fetch price history, news headlines, and compute 14-day RSI/ATR corridors."
                },
                {
                    "stepNumber": 2,
                    "name": "8-Agent Council Deliberation",
                    "component": "src/agent_committee.py",
                    "action": "Technical Alpha, Sentiment, Valuation, and CRO vote with memory-calibrated weights."
                },
                {
                    "stepNumber": 3,
                    "name": "Regime Risk Gate & Sizing",
                    "component": "src/agent_committee.py",
                    "action": "Chief Risk Officer checks drawdown limits and sizes position using Fractional Kelly (Quarter-Kelly)."
                },
                {
                    "stepNumber": 4,
                    "name": "Broker Execution & Ledger Audit",
                    "component": "src/paper_broker.py",
                    "action": "Executes order, logs to executed_trades.csv, and preserves live cash ledger ($159,199.62)."
                }
            ]
        },
        {
            "id": "flow:mirofish_cloud_simulation",
            "name": "MiroFish Cloud Flight Simulation",
            "description": "Stress-tests strategies against synthetic multi-round agent order books.",
            "steps": [
                {
                    "stepNumber": 1,
                    "name": "Snapshot & Scenario Injection",
                    "component": "src/cloud_market_simulator.py",
                    "action": "Load S&P 100 spot metrics and select shock (Fed Rate Shock, Liquidity Crunch, Short Squeeze)."
                },
                {
                    "stepNumber": 2,
                    "name": "Gemini Cloud Deliberation Turn",
                    "component": "src/cloud_market_simulator.py",
                    "action": "Offload 5 agent persona theses and order volumes to Google Gemini 3.6 Flash REST API."
                },
                {
                    "stepNumber": 3,
                    "name": "Kyle Lambda Clearing Price Calculation",
                    "component": "src/cloud_market_simulator.py",
                    "action": "Arbitrator reconciles order imbalance and computes emergent price trajectory."
                },
                {
                    "stepNumber": 4,
                    "name": "Academic Paper Strategy Benchmarking",
                    "component": "src/academic_papers_benchmark.py",
                    "action": "Evaluate Boyd SOCP, Triple-Barrier, Hazan ONS, and Kelly sizing against the emergent price path."
                }
            ]
        }
    ]
}

with open(os.path.join(UA_DIR, "domain-graph.json"), "w", encoding="utf-8") as f:
    json.dump(domain_graph, f, indent=2)

print("[SUCCESS] Generated .ua/domain-graph.json: 5 domains, 2 business flows, 8 process steps.")
