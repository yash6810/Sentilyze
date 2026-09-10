# Domain Expert Reference Template

> This is the canonical template for all draw.io domain expert reference documents.
> Copy this file and replace all `{PLACEHOLDERS}` with domain-specific content.
> Sections marked `(OPTIONAL)` may be omitted if not applicable to the domain.

---

```
IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any draw.io-{DOMAIN}-validation tasks.
```

## [Project Context]

```
domain:{domain-name}|env:draw.io-plugin|role:domain-expert-reviewer-agent|task:validate-{domain}-topology+enforce-rules|loop:detect-errors→output-corrections→trigger-redraw
```

- **domain**: The specific diagram domain (e.g., `gcp-well-architected-architecture`, `entity-relationship-diagrams(ERD)`, `kubernetes-topology-diagrams`).
- **role**: The expert persona the agent assumes (e.g., `database-schema-architect`, `network-infrastructure-architect`).
- **task**: The validation scope — what the agent checks and enforces.

## [Docs Index]

```
{domain}-reference:{comma-separated-doc-paths}
plugin-src:{comma-separated-source-file-paths}
*always-read-{domain}-specs-before-validating-graph*
```

- List all relevant specification documents and plugin source files the agent should read before performing validation.
- The `*always-read*` directive ensures retrieval-led reasoning is applied.

## [Domain Rules + Patterns]

Compressed pipe-delimited rules covering the domain's fundamental constraints. Each line encodes one rule using the format:

```
rule-name:constraint-description|enforce:required-behavior|prevent:disallowed-behavior
```

Examples of rule categories:
- **Hierarchy / containment**: What containers nest inside what (e.g., `Region>VPC>Subnet>Zone`).
- **Service placement**: Which services go in which container level.
- **Flow direction**: Canonical flow path through the architecture.
- **Physics / invariants**: Domain-specific physical or logical constraints (e.g., separation physics, normalization forms, network segmentation).

## [Project Conventions]

Layout and rendering conventions specific to this domain:

```
topology:{layout-rules — where-nodes-go, flow-direction, stacking-order}
routing:{edge-routing-conventions — orthogonal, curved, libavoid}
arrows:{edge-type-semantics — solid=sync, dashed=async, etc.}
labels:{label-placement-and-formatting-rules}
```

- **topology**: Defines spatial arrangement (e.g., `Client:top|Services:middle|Data:bottom`).
- **routing**: How edges are routed between nodes (e.g., `orthogonal-only`, `lines-terminate-at-boundary`).
- **arrows**: Semantic meaning of edge styles (e.g., `solid=synchronous`, `dashed=asynchronous`).
- **labels**: Where labels are placed and how they are formatted.

## [Anti-Patterns]

One anti-pattern per line using the format:

```
anti-pattern-id|correction:description-of-fix
```

Each entry identifies a common mistake and provides a concrete correction action. Anti-patterns should cover:
- Misplaced resources or nodes
- Missing intermediate components
- Incorrect edge directions or styles
- Orphaned or unconnected elements
- Violations of domain invariants

## [Visual Styling]

Domain-specific visual conventions:

```
icon-style:{icon-shape-conventions — which-shape-libraries-to-use}
edge-style:{edge-rendering-rules — orthogonal, curved, stroke-width}
color-palette:{domain-specific-colors-for-containers-and-nodes}
spacing:{recommended-node-spacing, container-padding, grid-dimensions}
```

## [Domain-Specific Catalog / Patterns] (OPTIONAL)

For domains with a fixed vocabulary of components (e.g., equipment catalogs, resource hierarchies, device inventories), list them here with their properties, port maps, or configuration details.

## [Sequential Execution Pairing] (OPTIONAL)

For domains where certain drawing actions must be performed in pairs:

```
pairing:{rule-name}|trigger:{triggering-action}|action:{simultaneous-required-action}
```

## [XML Graph DOM Rules] (OPTIONAL)

For domains that require specific XML containment or edge validation at the DOM level:

```
xml-containment:{parent-child-constraints}
xml-edges:{edge-semantics-and-constraints}
xml-anti-patterns:{DOM-level-violation-patterns}
```

## [Validator Rules Reference & Troubleshooting]

Numbered list of validator rule IDs. Each entry follows:

```
### N. `RULE_ID`
- **Trigger**: Condition that fires the rule.
- **Troubleshooting**: Step-by-step fix for the violation.
```

---

## Section Checklist

| Section | Required? | Notes |
|---|---|---|
| `[Project Context]` | ✅ Yes | Always include domain, env, role, task, loop |
| `[Docs Index]` | ✅ Yes | Reference docs + plugin source files |
| `[Domain Rules + Patterns]` | ✅ Yes | Core invariants and constraints |
| `[Project Conventions]` | ✅ Yes | topology, routing, arrows, labels |
| `[Anti-Patterns]` | ✅ Yes | At least 5 domain-specific anti-patterns |
| `[Visual Styling]` | ✅ Yes | icon-style, edge-style, color-palette, spacing |
| `[Domain-Specific Catalog]` | ⚪ Optional | Equipment catalogs, resource hierarchies |
| `[Sequential Execution Pairing]` | ⚪ Optional | Paired drawing actions |
| `[XML Graph DOM Rules]` | ⚪ Optional | DOM-level validation |
| `[Validator Rules Reference]` | ✅ Yes | Numbered rule IDs with triggers + troubleshooting |
