---
name: agent-memory
description: Persistent episodic and semantic memory system across agent sessions. Retains trading post-mortems, committee vote calibrations, and historical market regime learnings.
---

# Agent Memory — Persistent Multi-Session Memory Skill

Use this skill when an agent needs to store, index, or retrieve context, trading heuristics, agent performance calibrations, and post-mortems across independent sessions.

## 1. Storage Architecture
- **Episodic Memory**: Records specific historical events (e.g. '2026-09-04 FOMC surprise caused +150 bps shift on TSLA; Sentiment Catalyst agent overweighted bullish news').
- **Semantic Memory**: Generalized rules distilled over time (e.g. 'When VIX > 30, technical momentum models lose predictive power; dynamic leverage must be capped at 1.0x').
- **Format**: Structured JSON/JSONL stored in `results/agent_memory/` with ISO timestamps, agent tags, and confidence scores.

## 2. Integration Pattern
- Before running daily committee deliberations, query `results/agent_memory/committee_history.json` for recent agent win rates and prediction biases.
- Dynamically calibrate committee voting weights based on empirical historical accuracy.
