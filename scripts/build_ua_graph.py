"""
Understand-Anything Knowledge Graph Synthesizer for Sentilyze.
Transforms project AST and graphify metadata into full .ua/knowledge-graph.json.
"""

import os
import json
import time
from datetime import datetime, timezone
import glob

UA_DIR = ".ua"
os.makedirs(UA_DIR, exist_ok=True)
os.makedirs(os.path.join(UA_DIR, "intermediate"), exist_ok=True)

git_commit = "3db0f9000cf94bfc965746e6257fe6e172f05d2f"
now_iso = datetime.now(timezone.utc).isoformat()

# Load base graphify graph
with open("graphify-out/graph.json", "r", encoding="utf-8") as f:
    raw_g = json.load(f)

nodes = []
edges = []
node_id_map = {}

# File summaries lookup
FILE_SUMMARIES = {
    "app.py": "Main Streamlit entry point hosting 6 Mission Control Cockpits with global portfolio ribbon and session routing.",
    "src/cloud_market_simulator.py": "MiroFish-inspired 5-agent cloud market simulator benchmarking academic paper strategies on S&P 100/SPY.",
    "src/academic_papers_benchmark.py": "Empirical benchmark suite testing all 14 academic paper quantitative methodologies.",
    "src/agent_committee.py": "Deliberation engine for the 8-Agent Committee with fractional Kelly sizing and CRO vetoes.",
    "src/autonomous_trader.py": "24/7 background autonomous paper trading worker scanning and executing signals.",
    "src/paper_broker.py": "Strict paper portfolio broker maintaining live cash balances ($159,199.62) and trade execution audit logs.",
    "src/data_ingestion.py": "Unified financial price history and news headline ingestion with in-memory caching.",
    "src/fundamental_valuation.py": "Fundamental analysis engine computing Altman Z, Piotroski F-score, and DCF intrinsic fair value.",
    "src/orderflow_chart.py": "Volume Profile Point of Control (PoC) and Fair Value Gap (FVG) detection.",
    "src/strategy_incubator.py": "Genetic algorithm evolutionary strategy breeder with 3-zone out-of-sample robustness testing.",
    "src/hedge_agents.py": "Dynamic beta-neutral and delta-hedging engine balancing long exposure against market shocks.",
    "src/triple_barrier.py": "Marcos Lopez de Prado Triple-Barrier labeling method and Deflated Sharpe Ratio calculation.",
    "src/online_newton_step.py": "Hazan et al. Online Newton Step O(d^2) logarithmic regret gradient optimization.",
    "src/convex_optimizer.py": "Boyd et al. Stanford convex multi-period portfolio optimizer with transaction slippage penalties.",
    "src/moment_sos_portfolio.py": "Higher-order moment polynomial portfolio optimization using sum-of-squares.",
    "src/almgren_chriss_execution.py": "Almgren-Chriss optimal execution shortfall trajectory calculator.",
    "src/fx_arbitrage_graph.py": "Bellman-Ford negative cycle triangular arbitrage detector on currency graphs.",
    "src/gnn_supply_chain.py": "Graph neural network model predicting economic shock spillovers across supply chain nodes.",
    "src/realtime_tracker.py": "Real-time ticker quote fetcher with multi-thread batch pooling.",
    "src/ui/cockpit_trading.py": "Cockpit 1: Live Trading & Directional Alpha Desk with 6 main tabs.",
    "src/ui/cockpit_smart_money.py": "Cockpit 2: Smart Money, Dark Pool Flow & Alternative Data Radar.",
    "src/ui/cockpit_portfolio.py": "Cockpit 3: Master Fund Portfolio, Beta-Neutral Hedging & Cloud Lakehouse.",
    "src/ui/cockpit_backtest.py": "Cockpit 4: Rolling Walk-Forward Optimization, Trade Autopsy & Tearsheet Factsheet.",
    "src/ui/cockpit_deep_quant.py": "Cockpit 5: Deep Quant, SHAP XAI, GNN Contagion & MiroFish Simulator.",
    "src/ui/ws_strategy_incubator.py": "Workspace for Genetic Breeding, Strategy Vault, and MiroFish Multi-Agent Simulator.",
}

# 1. Transform Nodes
for raw_n in raw_g.get("nodes", []):
    raw_id = raw_n["id"]
    source_file = raw_n.get("source_file", "")
    label = raw_n.get("label", raw_id)
    is_class = raw_n.get("_callable_class", False)
    is_callable = raw_n.get("_callable", False)

    if is_class:
        node_type = "class"
        prefix = f"class:{source_file}:{label}"
    elif is_callable:
        node_type = "function"
        prefix = f"function:{source_file}:{label}"
    else:
        node_type = "file" if source_file.endswith(".py") else "module"
        prefix = f"file:{source_file}"

    node_id_map[raw_id] = prefix

    # Tags & Layer
    tags = ["quant", "python"]
    if "ui" in source_file:
        tags.extend(["ui", "streamlit"])
    if "simulator" in source_file:
        tags.extend(["simulation", "mirofish", "multi-agent"])
    if "agent" in source_file:
        tags.extend(["agent", "committee"])
    if "broker" in source_file or "order" in source_file:
        tags.extend(["execution", "broker"])
    if "benchmark" in source_file or "optimizer" in source_file:
        tags.extend(["academic-paper", "alpha"])

    complexity = "high" if is_class and any(k in label.lower() for k in ["optimizer", "simulator", "committee", "allocator", "network"]) else "medium"

    summary = FILE_SUMMARIES.get(source_file, f"{label} component defined in {source_file}.")

    nodes.append({
        "id": prefix,
        "type": node_type,
        "name": label,
        "filePath": source_file,
        "summary": summary,
        "tags": list(set(tags)),
        "complexity": complexity,
        "languageNotes": "Pure Python implementation optimized for low-latency in-memory execution."
    })

# Add primary file-level nodes if missing
existing_file_ids = {n["id"] for n in nodes if n["type"] == "file"}
for py_file in glob.glob("src/**/*.py", recursive=True) + ["app.py"]:
    norm_path = py_file.replace("\\", "/")
    f_id = f"file:{norm_path}"
    if f_id not in existing_file_ids:
        summary = FILE_SUMMARIES.get(norm_path, f"Python module {os.path.basename(norm_path)}.")
        nodes.append({
            "id": f_id,
            "type": "file",
            "name": os.path.basename(norm_path),
            "filePath": norm_path,
            "summary": summary,
            "tags": ["quant", "python"],
            "complexity": "medium",
            "languageNotes": "Pure Python"
        })

# 2. Transform Edges
for link in raw_g.get("links", []):
    src_raw = link["source"]
    tgt_raw = link["target"]
    if src_raw in node_id_map and tgt_raw in node_id_map:
        rel = link.get("relation", "calls")
        edge_type = "imports" if "import" in rel.lower() else "calls" if "call" in rel.lower() else "depends_on"
        edges.append({
            "source": node_id_map[src_raw],
            "target": node_id_map[tgt_raw],
            "type": edge_type,
            "direction": "outgoing",
            "weight": round(float(link.get("weight", 1.0)), 2)
        })

# 3. Define Architectural Layers
all_node_ids = {n["id"] for n in nodes}

layers = [
    {
        "id": "layer:ui",
        "name": "User Interface & Cockpit Layer",
        "description": "Streamlit frontend workspaces and mission control stations",
        "nodeIds": [nid for nid in all_node_ids if "src/ui/" in nid or nid == "file:app.py"]
    },
    {
        "id": "layer:market_sim",
        "name": "MiroFish Multi-Agent Cloud Market Simulator",
        "description": "5-Agent emergent market simulation engine offloaded to Google Gemini Cloud",
        "nodeIds": [nid for nid in all_node_ids if "cloud_market_simulator" in nid]
    },
    {
        "id": "layer:committee",
        "name": "Multi-Agent Deliberation & Council",
        "description": "8-Agent Committee, Autonomous Trader, and memory-calibrated voting",
        "nodeIds": [nid for nid in all_node_ids if "agent_committee" in nid or "autonomous_trader" in nid]
    },
    {
        "id": "layer:academic_papers",
        "name": "Academic Paper Benchmark Suite (14 Papers)",
        "description": "Empirical implementations of Hazan ONS, Boyd SOCP, Almgren-Chriss, HRP, and Triple-Barrier",
        "nodeIds": [nid for nid in all_node_ids if any(k in nid for k in ["academic_papers", "newton", "convex", "moment_sos", "almgren", "triple_barrier", "hedge_agents"])]
    },
    {
        "id": "layer:data_microstructure",
        "name": "Data Ingestion & Microstructure",
        "description": "yfinance, NewsAPI, SEC EDGAR, Order Flow Volume Profile, and Real-Time Tracking",
        "nodeIds": [nid for nid in all_node_ids if any(k in nid for k in ["data_ingestion", "orderflow", "realtime", "sec_crawler", "social_sentiment"])]
    },
    {
        "id": "layer:execution_broker",
        "name": "Execution & Strict Portfolio Broker",
        "description": "Paper broker, Kyle's Lambda clearance, cash preservation ($159,199.62), and order scheduling",
        "nodeIds": [nid for nid in all_node_ids if any(k in nid for k in ["paper_broker", "execution_engine", "order_routing"])]
    }
]

# 4. Guided Architectural Tour
tour = [
    {
        "order": 1,
        "title": "Application Entry & 6-Cockpit Navigation",
        "description": "Explore how app.py initialises themes, the top portfolio ribbon, and routes to the 6 Cockpits.",
        "nodeIds": ["file:app.py", "file:src/ui/cockpit_trading.py", "file:src/ui/cockpit_deep_quant.py"]
    },
    {
        "order": 2,
        "title": "The 8-Agent Quantitative Council",
        "description": "Walk through agent_committee.py where Technical Alpha, Sentiment, CRO, and Memory agents deliberate.",
        "nodeIds": [nid for nid in all_node_ids if "agent_committee" in nid][:3]
    },
    {
        "order": 3,
        "title": "MiroFish Multi-Agent Cloud Market Simulator",
        "description": "Discover cloud_market_simulator.py: 5 market actors simulate order flow and benchmark paper strategies in seconds.",
        "nodeIds": [nid for nid in all_node_ids if "cloud_market_simulator" in nid][:3]
    },
    {
        "order": 4,
        "title": "The 14 Academic Research Papers Suite",
        "description": "Examine academic_papers_benchmark.py implementing Hazan ONS, Boyd SOCP, and Triple-Barrier ATR corridors.",
        "nodeIds": [nid for nid in all_node_ids if "academic_papers_benchmark" in nid][:3]
    },
    {
        "order": 5,
        "title": "Paper Broker & Live Cash Preservation",
        "description": "Inspect paper_broker.py ensuring strict protection of the $159,199.62 cash ledger and 83.0% win rate.",
        "nodeIds": [nid for nid in all_node_ids if "paper_broker" in nid][:3]
    }
]

# 5. Assemble Complete Knowledge Graph
knowledge_graph = {
    "project": {
        "name": "Sentilyze",
        "description": "Institutional Algorithmic Trading & MLOps Platform Combining NLP Sentiment Analysis, XGBoost Machine Learning, and Multi-Agent Market Simulations.",
        "languages": ["Python"],
        "frameworks": ["Streamlit", "XGBoost", "FinBERT", "PyTorch", "Pandas", "Scikit-Learn"],
        "analyzedAt": now_iso,
        "gitCommitHash": git_commit
    },
    "nodes": nodes,
    "edges": edges,
    "layers": layers,
    "tour": tour
}

# Write knowledge-graph.json
kg_path = os.path.join(UA_DIR, "knowledge-graph.json")
with open(kg_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_graph, f, indent=2)

# Write meta.json
meta = {
    "version": "2.9.7",
    "analyzedAt": now_iso,
    "gitCommitHash": git_commit,
    "projectRoot": os.path.abspath("."),
    "totalNodes": len(nodes),
    "totalEdges": len(edges),
    "layersCount": len(layers),
    "tourStepsCount": len(tour)
}
with open(os.path.join(UA_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

# Write config.json
config = {
    "autoUpdate": True,
    "outputLanguage": "en"
}
with open(os.path.join(UA_DIR, "config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(f"[SUCCESS] Generated .ua/knowledge-graph.json: {len(nodes)} nodes, {len(edges)} edges, {len(layers)} layers, {len(tour)} tour steps.")
