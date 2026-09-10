# Layout Patterns Guide

Patterns for swimlanes, nested containers, cross-functional flowcharts, and when to use each.

## 1. Flat Swimlanes (BPMN-style Flowcharts)

Use flat swimlanes at `parent="1"`, stacked vertically. One row of nodes per lane.

### Fixed Values — Do Not Compute

| Element | Value |
|---------|-------|
| Lane x | `0` |
| Lane y | `lane_index * 150` |
| Lane width | `CANVAS_W` (= `max_col * 180 + 300`) |
| Lane height | `150` (always) |
| Lane style | `swimlane;horizontal=0;startSize=110;fillColor=<pastel>;html=1;` |
| Child node x | `120 + col * 180` |
| Child node y | `45` (always) |
| Child node size | `140 × 60` (rectangles), `140 × 80` (diamonds) |
| Cross-lane edges | `parent="1"` (not inside a lane) |

### Lane Color Sequence

Use in order: `#f5f5f5`, `#e8f4f8`, `#fff0e6`, `#e8f5e9`, `#fff9e6`, `#fce4ec`

### Example

```xml
<mxCell id="lane1" value="Customer"
        style="swimlane;horizontal=0;startSize=110;fillColor=#f5f5f5;html=1;"
        vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="1800" height="150" as="geometry"/>
</mxCell>
<mxCell id="n1" value="Place Order" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="lane1">
  <mxGeometry x="120" y="45" width="140" height="60" as="geometry"/>
</mxCell>

<mxCell id="lane2" value="System"
        style="swimlane;horizontal=0;startSize=110;fillColor=#e8f4f8;html=1;"
        vertex="1" parent="1">
  <mxGeometry x="0" y="150" width="1800" height="150" as="geometry"/>
</mxCell>
<mxCell id="n2" value="Validate" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="lane2">
  <mxGeometry x="300" y="45" width="140" height="60" as="geometry"/>
</mxCell>

<mxCell id="e1" edge="1" parent="1" source="n1" target="n2"
        style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;">
  <mxGeometry relative="1" as="geometry"/>
</mxCell>
```

### Critical Rules

- Do **NOT** nest lanes inside a pool
- Do **NOT** vary lane heights
- Do **NOT** compute title-area offset — it is always `startSize=110`, children start at `x=120`
- `horizontal=0` makes the lane header on the left side

## 2. Nested Architecture Containers

For diagrams with nested groupings: VPC → AZ → Instance, Datacenter → Rack → Server, Region → Environment → Service.

### Pattern

Every container is a `swimlane` with `startSize=24`.

| Rule | Detail |
|------|--------|
| Container style | `swimlane;startSize=24;fillColor=#hex;strokeColor=#hex;html=1;` |
| Child `parent` | Set to container's id |
| Child coordinates | **Relative** to parent (origin 0,0 = parent's top-left, below title) |
| Cross-container edges | `parent="1"` — never inside a container |
| Industry icons | Use `search_shapes` for AWS/Azure/GCP — container structure stays the same |

### Example: VPC → AZ → Instances

```xml
<mxCell id="vpc" value="VPC"
        style="swimlane;startSize=24;fillColor=#dae8fc;strokeColor=#6c8ebf;html=1;"
        vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="720" height="360" as="geometry"/>
</mxCell>

<mxCell id="az1" value="AZ us-east-1a"
        style="swimlane;startSize=24;fillColor=#fff2cc;strokeColor=#d6b656;html=1;"
        vertex="1" parent="vpc">
  <mxGeometry x="20" y="36" width="320" height="300" as="geometry"/>
</mxCell>

<mxCell id="web1" value="web-1" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="az1">
  <mxGeometry x="30" y="40" width="120" height="60" as="geometry"/>
</mxCell>
<mxCell id="db1" value="db-1" style="shape=cylinder3;whiteSpace=wrap;html=1;"
        vertex="1" parent="az1">
  <mxGeometry x="180" y="40" width="100" height="70" as="geometry"/>
</mxCell>

<mxCell id="az2" value="AZ us-east-1b"
        style="swimlane;startSize=24;fillColor=#fff2cc;strokeColor=#d6b656;html=1;"
        vertex="1" parent="vpc">
  <mxGeometry x="360" y="36" width="340" height="300" as="geometry"/>
</mxCell>
<mxCell id="web2" value="web-2" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="az2">
  <mxGeometry x="30" y="40" width="120" height="60" as="geometry"/>
</mxCell>

<mxCell id="e1" edge="1" parent="1" source="web1" target="web2"
        style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;">
  <mxGeometry relative="1" as="geometry"/>
</mxCell>
```

### Nesting Tips

- `startSize=24` gives a compact title bar at the top
- First child y starts at ~36–40 to clear the title area
- Leave 20px padding from container edges
- Containers at the same level should have consistent fillColor

## 3. Cross-Functional Flowcharts (Actor × Phase Grid)

Use draw.io's `table` shape with `childLayout=tableLayout` for two-dimensional grids (actors as rows, phases as columns).

### Structure

```
table (shape=table;childLayout=tableLayout)
 └── row 0 (headers): shape=tableRow
 │    ├── empty corner cell
 │    ├── "Phase 1" header
 │    └── "Phase 2" header
 ├── row 1 (actor): shape=tableRow
 │    ├── "Customer" label cell
 │    ├── cell (contains process nodes)
 │    └── cell (contains process nodes)
 └── row 2 (actor): shape=tableRow
      ├── "System" label cell
      ├── cell
      └── cell
```

### Key Properties

| Element | Style |
|---------|-------|
| Table | `shape=table;childLayout=tableLayout;startSize=0;collapsible=0;fillColor=none;` |
| Row | `shape=tableRow;horizontal=0;startSize=0;collapsible=0;` |
| Header cell | `text;align=center;fontStyle=1;fillColor=#e8e8e8;` |
| Actor label | `fillColor=#dae8fc;fontStyle=1;` |
| Content cell | `fillColor=none;` |

### Process Node Placement

- Process nodes are children of their intersection cell (`parent="cell_id"`)
- Coordinates are relative to the cell
- Cross-cell edges use `parent="1"`

### Example

```xml
<mxCell id="tbl" style="shape=table;childLayout=tableLayout;startSize=0;collapsible=0;fillColor=none;"
        vertex="1" parent="1">
  <mxGeometry x="0" y="0" width="900" height="320" as="geometry"/>
</mxCell>

<mxCell id="r0" style="shape=tableRow;horizontal=0;startSize=0;collapsible=0;"
        vertex="1" parent="tbl">
  <mxGeometry width="900" height="40" as="geometry"/>
</mxCell>
<mxCell id="h0" style="text;html=1;" vertex="1" parent="r0">
  <mxGeometry width="140" height="40" as="geometry"/>
</mxCell>
<mxCell id="h1" value="Order" style="text;align=center;fontStyle=1;fillColor=#e8e8e8;"
        vertex="1" parent="r0">
  <mxGeometry x="140" width="380" height="40" as="geometry"/>
</mxCell>
<mxCell id="h2" value="Fulfill" style="text;align=center;fontStyle=1;fillColor=#e8e8e8;"
        vertex="1" parent="r0">
  <mxGeometry x="520" width="380" height="40" as="geometry"/>
</mxCell>

<mxCell id="r1" style="shape=tableRow;horizontal=0;startSize=0;collapsible=0;"
        vertex="1" parent="tbl">
  <mxGeometry y="40" width="900" height="140" as="geometry"/>
</mxCell>
<mxCell id="a1" value="Customer" style="fillColor=#dae8fc;fontStyle=1;"
        vertex="1" parent="r1">
  <mxGeometry width="140" height="140" as="geometry"/>
</mxCell>
<mxCell id="c1" style="fillColor=none;" vertex="1" parent="r1">
  <mxGeometry x="140" width="380" height="140" as="geometry"/>
</mxCell>
<mxCell id="t1" value="Place Order" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="c1">
  <mxGeometry x="120" y="40" width="140" height="60" as="geometry"/>
</mxCell>
```

### Critical Rules

- Do **NOT** nest swimlanes inside table rows
- Do **NOT** set `startSize` on rows or cells
- Cell widths don't need to sum exactly — `tableLayout` normalizes them

## 4. When to Use What

| Pattern | Axes | When to Use |
|---------|------|-------------|
| **Flat Swimlanes** | 1 (actors only) | BPMN, process flows showing "who does what" in sequence |
| **Cross-functional Table** | 2 (actors × phases) | Both actor AND process stage matter for every step |
| **Nested Containers** | Hierarchy (parent-child) | Architecture, infrastructure, any spatial containment |
| **Group** (invisible) | None | Logically group nodes without visual border; no connections to the group itself |

### Decision Guide

```
Need visual containment hierarchy?
  YES → Nested swimlane containers (startSize=24)
  NO ↓
Process flow with actors?
  YES → How many dimensions?
    1 dimension (just actors) → Flat swimlanes
    2 dimensions (actors + phases) → Cross-functional table
  NO ↓
Need invisible grouping?
  YES → Group (style="group;")
  NO → Plain layout (no containers)
```

## Container Key Rules Summary

| Rule | Detail |
|------|--------|
| Parent-child | Set `parent="containerId"` on children |
| Relative coords | Children use coordinates relative to container origin |
| Cross-container edges | Always `parent="1"` |
| `pointerEvents=0` | Add to containers that shouldn't capture connections |
| Don't omit `pointerEvents=0` | Unless the container itself must be connectable (use `swimlane` for that) |
| Edges crossing boundaries | This is correct — don't add waypoints to avoid it |
