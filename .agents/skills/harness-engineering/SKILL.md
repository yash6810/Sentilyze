---
name: harness-engineering
description: Agent orchestration patterns, evaluation harnesses, deterministic feedback loops, and state isolation architectures to ensure reliable multi-agent systems.
---

# Harness Engineering Skill

Use this skill to build robust, predictable, and self-correcting agent execution harnesses.

## 1. Core Principles
- **State Isolation**: Agents never mutate shared global state directly; all proposals pass through explicit message contracts and validation schemas.
- **Evaluation Gates**: Every agent decision must be validated against quantitative sanity bounds (e.g. max position size, regime alignment, stop-loss presence).
- **Red-Team Veto**: Independent adversarial agent reviews proposals to actively find flaws before execution.
- **Deterministic Replayability**: All inputs, prompts, temperature settings, and outputs are logged for exact post-trade replay and auditing.
