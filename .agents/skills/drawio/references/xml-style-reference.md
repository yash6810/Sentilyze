# draw.io XML Style Reference

Quick-reference for all style properties, shapes, colors, labels, edges, containers, layers, tags, metadata, and the rigid grid system.

## Rigid Grid System

Every XML diagram uses this grid. No exceptions.

| Element | Formula | Examples |
|---------|---------|----------|
| Column x | `col_index * 180 + 40` | col 0 = 40, col 1 = 220, col 2 = 400 |
| Row y | `row_index * 120 + 40` | row 0 = 40, row 1 = 160, row 2 = 280 |
| Rectangle | `140 × 60` | — |
| Diamond | `140 × 80` | — |
| Circle | `60 × 60` | — |
| Document | `120 × 80` | — |
| Cylinder | `100 × 70` | — |

Pick `(col, row)` per node — ELK/libavoid handles the rest.

## Shape Types

| Shape | Style keyword(s) | Notes |
|-------|------------------|-------|
| Rectangle | *(default)* | `rounded=1` for rounded corners |
| Diamond | `rhombus` | Decision nodes |
| Circle/Oval | `ellipse` | Start/end nodes |
| Cylinder | `shape=cylinder3` | Databases, tanks |
| Document | `shape=mxgraph.flowchart.document` | Document shapes |
| Hexagon | `shape=hexagon` | — |
| Parallelogram | `shape=parallelogram` | I/O |
| UML Lifeline | `shape=umlLifeline;perimeter=lifelinePerimeter;size=16` | Sequence diagrams |

## Common Style Properties

| Property | Values | Use |
|----------|--------|-----|
| `rounded=1` | `0` or `1` | Rounded corners |
| `whiteSpace=wrap` | `wrap` | Text wrapping |
| `html=1` | `0` or `1` | Enable HTML rendering in labels |
| `fillColor=#hex` | Hex color | Background color |
| `strokeColor=#hex` | Hex color | Border color |
| `fontColor=#hex` | Hex color | Text color |
| `fontSize=14` | Number | Font size in pt |
| `fontStyle=N` | Bitmask | 1=bold, 2=italic, 4=underline (OR to combine: 3=bold+italic) |
| `dashed=1` | `0` or `1` | Dashed border/line |
| `opacity=50` | 0–100 | Transparency |
| `container=1` | `0` or `1` | Enable container behavior |
| `pointerEvents=0` | `0` or `1` | Prevent container from capturing child connections |
| `swimlane` | style keyword | Swimlane container |
| `group` | style keyword | Invisible container |

## Color Properties

**Default behavior:** `strokeColor`, `fillColor`, `fontColor` default to `"default"` — black in light theme, white in dark.

**Explicit colors:** Specify the light-mode hex (`fillColor=#DAE8FC`). Dark mode is auto-computed (RGB inversion at 93%, hue rotated 180°).

**`light-dark()` function:** Explicit control of both modes:
```
fontColor=light-dark(#7EA6E0,#FF0000)
```
First arg = light mode, second = dark mode.

**Enable dark mode adaptation:** Add `adaptiveColors="auto"` on `<mxGraphModel>`.

## HTML Labels

**Always include `html=1`** in the style. Without it, HTML tags render as literal text.

### Escaping Rules (XML attribute context)

| Character | Escape |
|-----------|--------|
| `<` | `&lt;` |
| `>` | `&gt;` |
| `&` | `&amp;` |
| `"` | `&quot;` |

### Line Breaks

| Method | Context |
|--------|---------|
| `&#xa;` | Works with `html=0` and `html=1` |
| `&lt;br&gt;` | Requires `html=1` |
| `\n` | **NEVER** — renders as literal `\n` |

### fontStyle vs HTML Tags

| Scenario | Use |
|----------|-----|
| Entire label bold/italic/underline | `fontStyle=N` in style string |
| Partial formatting (bold title + normal body) | HTML tags `<b>`, `<i>`, `<u>` in value |
| **Never** | Combine both for same effect |

### Example

```xml
<mxCell value="&lt;b&gt;Title&lt;/b&gt;&lt;br&gt;Description"
        style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="120" height="60" as="geometry"/>
</mxCell>
```

## Edge Styles

### When to Use Each Style

| Style | Syntax | Best for |
|-------|--------|----------|
| **Orthogonal** | `edgeStyle=orthogonalEdgeStyle` | Flowcharts, architecture, network, BPMN — right-angle connectors |
| **Straight** | *(no `edgeStyle`)* | UML class/sequence, direct point-to-point. Add `endSize=6;startSize=6;` for sequence messages |
| **Entity Relation** | `edgeStyle=entityRelationEdgeStyle` | ER diagrams — perpendicular stubs at both ends |
| **Curved** | `curved=1` | Mind maps, informal diagrams |
| **Elbow** | `edgeStyle=elbowEdgeStyle;elbow=vertical;` | Rarely needed — only for simple 1-bend linear flows |

**Rule:** Use one consistent edge style per diagram.

### Edge Attributes

| Attribute | Values | Effect |
|-----------|--------|--------|
| `rounded=1` | `0` or `1` | Rounded corners at bends (recommended for orthogonal) |
| `endArrow=classic` | `classic`, `none`, `block`, `open`, `diamond` | Arrowhead type |
| `dashed=1` | `0` or `1` | Dashed line |
| `strokeColor=#hex` | Hex | Edge color |
| `strokeWidth=2` | Number | Line thickness |
| `value="label"` | Text | Edge label (set directly on edge mxCell) |

### Critical Edge Rules

```xml
<mxCell id="e1" edge="1" parent="1" source="a" target="b" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;">
  <mxGeometry relative="1" as="geometry" />
</mxCell>
```

Self-closing edge cells will **not render**:
```xml
<mxCell id="e1" edge="1" parent="1" source="a" target="b" style="..." />
```

**Never:**
- Add `<mxPoint>` waypoints
- Set `exitX`/`exitY`/`entryX`/`entryY` unless specific geometric intent
- Route around obstacles manually

## Container Types

| Type | Style | When to use |
|------|-------|-------------|
| **Group** (invisible) | `group;` | No visual border, container has no connections. Implies `pointerEvents=0` |
| **Swimlane** (titled) | `swimlane;startSize=30;` | Needs visible title bar, or container itself has connections |
| **Custom container** | Add `container=1;pointerEvents=0;` to any shape | Any shape acting as container without its own connections |

### Key Containment Rules

- Children set `parent="containerId"` and use **relative coordinates**
- Edges crossing container boundaries are correct and expected
- Always add `pointerEvents=0;` unless the container itself is connectable
- Swimlane headers remain connectable by default (client area is transparent)
- Cross-container edges must use `parent="1"` (not inside a container)

### Group Example

```xml
<mxCell id="grp1" value="" style="group;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="300" height="200" as="geometry"/>
</mxCell>
<mxCell id="c1" value="Component" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="grp1">
  <mxGeometry x="10" y="10" width="120" height="60" as="geometry"/>
</mxCell>
```

### Swimlane Example

```xml
<mxCell id="svc1" value="User Service"
        style="swimlane;startSize=30;fillColor=#dae8fc;strokeColor=#6c8ebf;html=1;"
        vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="300" height="200" as="geometry"/>
</mxCell>
<mxCell id="api1" value="REST API" style="rounded=1;whiteSpace=wrap;html=1;"
        vertex="1" parent="svc1">
  <mxGeometry x="20" y="40" width="120" height="60" as="geometry"/>
</mxCell>
```

## Layers

Layers control visibility and z-order. Cell `id="0"` is root, `id="1"` is default layer.

```xml
<mxGraphModel>
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="2" value="Annotations" parent="0"/>
    <mxCell id="10" value="Server" style="rounded=1;html=1;"
            vertex="1" parent="1">
      <mxGeometry x="100" y="100" width="120" height="60" as="geometry"/>
    </mxCell>
    <mxCell id="20" value="Note" style="text;" vertex="1" parent="2">
      <mxGeometry x="100" y="170" width="120" height="30" as="geometry"/>
    </mxCell>
  </root>
</mxGraphModel>
```

- A layer: `mxCell` with `parent="0"`, no `vertex`/`edge` attribute
- Later layers render on top (higher z-order)
- `visible="0"` hides a layer by default

## Tags

Tags are per-element visibility filters (unlike layers, one element can have multiple tags). Require `<object>` wrapper:

```xml
<object id="2" label="Auth Service" tags="critical v2">
  <mxCell style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="1">
    <mxGeometry x="100" y="100" width="120" height="60" as="geometry"/>
  </mxCell>
</object>
```

- `label` on `<object>` replaces `value` on `mxCell`
- Tags are space-separated
- Viewers filter via Edit → Tags

## Metadata and Placeholders

Custom key-value properties on shapes via `<object>` attributes. Enable `%key%` substitution with `placeholders="1"`:

```xml
<object id="2"
        label="&lt;b&gt;%component%&lt;/b&gt;&lt;br&gt;Owner: %owner%"
        placeholders="1" component="Auth Service" owner="Team Backend">
  <mxCell style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="1">
    <mxGeometry x="100" y="100" width="160" height="80" as="geometry"/>
  </mxCell>
</object>
```

### Predefined Placeholders (no custom properties needed)

`%id%`, `%width%`, `%height%`, `%date%`, `%time%`, `%timestamp%`, `%page%`, `%pagenumber%`, `%pagecount%`, `%filename%`

- Placeholders resolve up the containment hierarchy (shape → parent → layer → root)
- Use `%%` for a literal percent sign
- Tags, metadata, and placeholders can combine on the same `<object>`

## XML Well-formedness

- **NEVER** include XML comments (`<!-- -->`)
- Escape special characters in attributes
- Use unique `id` values for every `mxCell`
