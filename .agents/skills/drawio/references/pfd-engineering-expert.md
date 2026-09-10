IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any draw.io-PFD-validation tasks.

## [Project Context]
domain:process-engineering-flow-diagrams(PFD)|industries:{oil+gas,mining,chemical,water-treatment}|env:draw.io-plugin|role:domain-expert-reviewer-agent|task:validate-graph-topology+enforce-physics|loop:detect-errors→output-corrections→trigger-redraw

## [Docs Index]
plugin-src:{src/graph-parser.ts,src/validation-engine.ts,src/auto-layout.ts}
engineering-specs:{docs/API-14C-safety.md,docs/PFD-symbology.md,docs/phase-separation-rules.md}
*always-read-engineering-specs-before-validating-graph*

## [Domain Rules + Patterns]
separation-physics:gravity-based→light-phase(gas,froth,fines):top+heavy-phase(water,tailings,coarse):bot
3-phase-separator(V):{inlet:side-upper|light-out:absolute-top-center|mid-out:side-mid-elevation|heavy-out:absolute-bottom}
equipment-ports:pump(P)→inlet:center/side+discharge:top|compressor(K)→inlet:side/center+discharge:top/bottom|cyclone(CY)→inlet:side+overflow:top+underflow:bot
piping-topology:single-source→single-dest|no-dead-ends|no-loops|no-shared-multi-phase-discharge-lines
instrumentation:bubbles-are-sensors-not-fittings|prevent:process-flow-through-instrument-bubble|signal-lines:dashed-only
valves:{LCV,PCV,FCV}→align-with-flow-vector

## [Project Conventions]
topology:flow:left→right|gravity:top→down
routing:lines-terminate-at-equip-boundary(exterior-shell-only)|prevent:internal-vessel-piping
intersections:orthogonal-only|minimize-crossings|inlet-never-intersects-outlet-at-equip
arrows:strict-flow-indication|prevent:opposing-arrows-on-same-stream(flow-contradiction)
labels:snap-to-edge|phase-text→must-anchor-to-specific-connector|prevent:floating-text|prevent:duplicate-labels

## [Anti-Patterns]
process-flow-through-instrument-bubble|correction:remove-instrument-from-pipe→attach-to-vessel-shell-via-dashed-line
dangling-or-looping-process-lines|correction:enforce-single-direct-path→delete-redundant-branches
shared-multi-phase-outlet-piping|correction:enforce-dedicated-nozzle-per-phase→mid-phase:side+heavy-phase:bot
internal-vessel-line-drawing|correction:truncate-all-lines-at-exterior-shell-boundary→no-lines-inside-shapes
compressor-inlet-at-bottom|correction:route-gas-inlet→side-or-center-port-of-compressor
raw-wellhead→compressor|correction:route-wellhead→separator-side+route-gas-top→compressor
light-phase-from-vessel-side|correction:route-light-phase(gas/froth)→absolute-top
heavy-phase-from-vessel-side|correction:route-heavy-phase(water/tailings)→absolute-bottom
LT-on-inlet-pipe|correction:move-LT-anchor→vessel-shell-side
intersecting-inlet+outlet-lines|correction:reroute-orthogonally→prevent-crossing-near-equip
opposing-arrows-at-node|correction:unify-direction→ensure-source→dest-logic
floating-or-duplicate-labels|correction:snap-text-node→nearest-process-edge+delete-duplicates
libavoid-routing-on-pfd|correction:never-pass-routing=libavoid-to-mcp-tool→it-destroys-physics-based-nozzle-routing

## [Equipment Catalog & Port Maps]

### 1. Distillation Column (`distillation_column`)
- **Tray Column** (`tray`) / **Packed Column** (`packed`)
- **Suction/Feed Inlet**: Center-Left (`entryX=0;entryY=0.5`)
- **Overhead Vapor Outlet**: Absolute Top (`exitX=0.5;exitY=0`)
- **Bottoms Liquid Outlet**: Absolute Bottom (`exitX=0.5;exitY=1`)

### 2. Pump (`pump`)
- **Centrifugal** (`centrifugal`) / **Positive Displacement** (`positive_displacement`)
- **Suction Inlet**: Center-Left (`entryX=0;entryY=0.5`)
- **Discharge Outlet**: Center-Right (`exitX=1;exitY=0.5`)

### 3. Compressor (`compressor`)
- **Centrifugal** (`centrifugal`) / **Reciprocating** (`reciprocating`)
- **Suction Inlet**: Absolute Bottom (`entryX=0.5;entryY=1`)
- **Discharge Outlet**: Center-Right (`exitX=1;exitY=0.5`)

### 4. Separator (`separator`)
- **2-Phase** (`2-phase`) / **3-Phase** (`3-phase`)
- **Mixed Inlet**: Center-Left (`entryX=0;entryY=0.5`)
- **Vapor/Light Outlet**: Absolute Top (`exitX=0.5;exitY=0`)
- **Liquid/Heavy Outlet**: Absolute Bottom (`exitX=0.5;exitY=1`)

### 5. Heat Exchanger (`heat_exchanger`)
- **Shell-and-Tube** (`shell-and-tube`) / **Plate** (`plate`)
- **Process Fluid Inlet**: Center-Left (`entryX=0;entryY=0.5`)
- **Process Fluid Outlet**: Center-Right (`exitX=1;exitY=0.5`)
- **Utility Fluid Inlet**: Absolute Top (`entryX=0.5;entryY=0`)
- **Utility Fluid Outlet**: Absolute Bottom (`exitX=0.5;exitY=1`)

### 6. Reactor (`reactor`)
- **CSTR** (`CSTR`) / **PFR** (`PFR`)
- **Reactant Inlet**: Center-Left (`entryX=0;entryY=0.5`)
- **Product Outlet**: Center-Right (`exitX=1;exitY=0.5`)

## [Stream Line Conventions]
- **Process Stream** (`style=process`): Heavy solid line (strokeWidth=3, solid) for primary chemical/process paths.
- **Utility Stream** (`style=utility`): Medium dashed line (strokeWidth=1.5, dashed=1) for steam, cooling water, fuel gas, and oil utilities.
- **Instrument Signal** (`style=instrument`): Thin dotted/dashed line (strokeWidth=1, dashed=1, dashPattern=1 3) for sensor, control, transmitter, and logic connections.

## [Validator Rules Reference & Troubleshooting]

### 1. `PHASE_PORT_VIOLATION`
- **Trigger**: Vapor streams exiting bottom ports or liquid/bottoms/heavy streams exiting top ports.
- **Troubleshooting**: Check the stream label and ensure its `exitPort` matches the phase physical behavior. Vapor goes up (top), liquid goes down (bottom).

### 2. `DEAD_END_STREAM`
- **Trigger**: Active process equipment (vessels, columns, pumps, reactors) with no feed streams or no discharge streams.
- **Troubleshooting**: Exclude boundary pumps/tanks by adding keywords like `feed`, `product`, `boundary`, `utility`, `slop` to their labels. Fully connect internal units with stream lines.

### 3. `OPPOSING_FLOW`
- **Trigger**: Primary process streams flowing from right to left (`source.x > target.x + 100`).
- **Troubleshooting**: Normal process flow is left-to-right. If this is an intentional recycle or return loop, add `recycle` or `return` to the stream's label to suppress the warning.

### 4. `GRAVITY_VIOLATION`
- **Trigger**: Liquid/heavy/slurry streams flowing uphill (`target.y < source.y - 50`) without a pump or compressor pushing them.
- **Troubleshooting**: Insert a pump or compressor between the source vessel and the higher destination node to satisfy pressure flow requirements.

### 5. `INSTRUMENT_IN_PROCESS_LINE`
- **Trigger**: Instrument signal lines modeled using solid/heavy process line styles.
- **Troubleshooting**: Update the connection style to `instrument` or `utility` for control loop edges.

### 6. `COMPRESSOR_INLET_AT_BOTTOM`
- **Trigger**: Compressor inlet streams entering from the side or top port instead of the bottom port.
- **Troubleshooting**: Change the compressor incoming edge's target port to bottom (`entryY=1` or `entryPort=bottom`).