# Edge Routing Decision Guide

How to pick the right routing/layout option for draw.io diagrams.

## Default Behavior (Built-in Router)

draw.io's built-in router is intentionally basic:

- Draws each edge as a **straight line** or **simple right-angle path** between source and target
- **No obstacle avoidance** — a connector runs straight through any shape between its endpoints
- No server-side post-processing
- Fine when connected nodes have open space between them

## routing: libavoid

**What it does:** Obstacle-avoiding orthogonal edge routing.

- Vertices stay **exactly where you placed them** — only connectors are recomputed
- Wires route in clean right-angle segments **around** shapes
- Parallel edges spread apart automatically
- Runs client-side after the diagram renders
- Available for **XML diagrams only** (parameter on `open_drawio_xml`)

**When to use:**
- Architecture diagrams
- Network topology
- Deployment diagrams
- UML diagrams
- Floor plans
- Swimlanes
- Any densely connected diagram where edges would cross through shapes

**When NOT to use:**
- Sparse layouts where the basic router works fine
- When you also want vertex repositioning (use ELK instead)

## postLayout: elk

**What it does:** Full re-layout using ELK `layered` algorithm.

- Vertices **animate (morph)** from your positions to canonical hierarchical positions
- Edges are routed as part of the layout — ELK has decent built-in routing
- Best for directional/hierarchical diagrams

**When to use:**
- Flowcharts
- Process diagrams
- State diagrams
- Decision flows
- Pipelines
- Any directional/hierarchical diagram

**When NOT to use:**
- User asked for specific positions (swimlanes with exact lanes)
- Diagram relies on spatial arrangement encoding information
- Architecture diagrams where containment/grouping matters

**Direction options:**
- XML: set `direction` field — `"vertical"` (default) or `"horizontal"`
- Mermaid: direction is read from flowchart code (`TD`/`TB` vs `LR`/`RL`) — `direction` field is ignored

## The 4-Combination Table

| `postLayout` | `routing` | Result |
|---|---|---|
| — | — | Basic built-in router (straight / simple right-angle, no obstacle avoidance); your positions kept |
| — | `libavoid` | Your positions kept; wires re-routed orthogonally *around* shapes |
| `elk` | — | ELK places vertices **and** routes edges (decent routing built in) |
| `elk` | `libavoid` | Rarely worth it — ELK already routes; only add if ELK's routing specifically comes out poor |

**They are essentially alternatives, not a stack.** Pick ONE.

## Decision Flowchart

```
Is the layout sparse with open space between connected nodes?
  YES → Use NEITHER (default basic router)
  NO ↓

Do you want to keep your hand-placed positions?
  YES → routing: "libavoid"
  NO ↓

Is the diagram directional/hierarchical (flowchart, pipeline, state)?
  YES → postLayout: "elk"
        Set direction: "horizontal" for LR flow
  NO ↓

Is it a dense architecture/network with precise positions?
  YES → routing: "libavoid"
  NO → postLayout: "elk"
```

## For Mermaid Diagrams

### When to set `postLayout: "elk"`

| Condition | Set ELK? |
|-----------|----------|
| ≥ ~20 nodes | Yes |
| ≥ 3 decision diamonds | Yes |
| Has feedback/back edges | Yes |
| ≥ 3 terminal endpoints | Yes |
| Simple flowchart (< 20 nodes, linear) | No |
| Non-flowchart types (sequence, class, ER, sankey, etc.) | No |

### Mermaid Direction Handling

- ELK reads direction from the flowchart code: `flowchart TD/TB` → vertical, `flowchart LR/RL` → horizontal
- The `direction` parameter is **ignored** for Mermaid — direction comes from the code itself

## Critical Rules

### Every edge needs an mxGeometry child

```xml
<mxCell id="e1" edge="1" parent="1" source="a" target="b"
        style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;">
  <mxGeometry relative="1" as="geometry" />
</mxCell>

<mxCell id="e1" edge="1" parent="1" source="a" target="b" style="..." />
```

### Don't hand-route

- Do **NOT** add `<Array as="points">` waypoints
- Do **NOT** add `<mxPoint>` elements
- Do **NOT** set `exitX`/`exitY`/`entryX`/`entryY` connection-point overrides (unless specific geometric intent)
- Just declare `source` and `target` — the router handles the rest

### Don't add waypoints

Whether using basic routing, libavoid, or ELK — you never add waypoints by hand. All three compute connector paths automatically.

### Edge labels

- Set `value` directly on the edge `mxCell`
- Keep labels short: 1–3 words (`Yes`, `async`, `reads`)
- Drop labels that restate obvious actions

### Visual consistency

- Use one edge style per diagram (orthogonal, straight, curved, etc.)
- Apply `dashed=1`, `strokeColor`, `strokeWidth` consistently for one meaning
- Add a small legend node if mixing styles

## Quick Reference: Edge Style × Routing Compatibility

| Edge Style | Works with libavoid | Works with ELK |
|------------|-------------------|----------------|
| `edgeStyle=orthogonalEdgeStyle` | ✅ (primary use case) | ✅ |
| Straight (no edgeStyle) | ✅ | ✅ |
| `edgeStyle=entityRelationEdgeStyle` | ✅ | ✅ |
| `curved=1` | ✅ | ✅ |
| `edgeStyle=elbowEdgeStyle` | ✅ | ✅ |

All edge styles are honored by both routing options. The router respects the style family when computing paths.
