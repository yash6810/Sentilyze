IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any draw.io-ERD-validation tasks.

## [Project Context]
domain:entity-relationship-diagrams(ERD)|env:draw.io-plugin|role:database-schema-architect|task:validate-database-schemas+enforce-normalization-rules|loop:detect-errors→output-corrections→trigger-redraw

## [Docs Index]
erd-reference:{docs/erd-database-expert.md}
plugin-src:{src/graph-parser.ts,src/validation-engine.ts,src/auto-layout.ts}
*always-read-database-specs-before-validating-graph*

## [Domain Rules + Patterns]
normalization-1NF:all-attributes-atomic|no-repeating-groups-or-arrays|unique-PK-required
normalization-2NF:meets-1NF+no-partial-key-dependencies|composite-PK→every-non-key-depends-on-entire-key
normalization-3NF:meets-2NF+no-transitive-dependencies|non-key-columns-depend-only-on-PK("the-key-the-whole-key-nothing-but-the-key")
notation-crowsfoot:1:1→endArrow=ERone;startArrow=ERone|1:N→endArrow=ERmany;startArrow=ERone|N:M→decompose-into-two-1:N-via-junction-table
junction-tables:resolve-M:N|composite-PK-from-two-FKs-referencing-parent-tables
polymorphic-associations:entity-belongs-to-multiple-types|uses-{type}_type+{type}_id-columns
self-referential-hierarchies:nullable-self-FK(e.g.,-employees.manager_id→employees.id)

## [Project Conventions]
topology:tables-flow-left→right-for-primary-relationships|junction-tables-placed-between-parent-tables|3-column-grid-layout
routing:crow's-foot-edges-use-orthogonal-routing|labels-at-edge-midpoint|lines-terminate-at-table-card-boundary
arrows:solid=relationship-edge(1:1,1:N)|dashed=optional-or-derived-relationship|prevent:floating-edges
labels:relationship-cardinality-at-both-ends|FK-column-name-as-edge-label-when-ambiguous
spacing:tables-spaced-260px-horizontally|220px-vertically|3-column-grid-alignment
alignment:related-tables-horizontally-aligned|parent-tables-left-of-child-tables

## [Anti-Patterns]
orphan-table|correction:table-with-no-relationships→connect-to-parent-or-mark-as-lookup-table
missing-junction|correction:M:N-relationship-without-decomposition→add-junction-table-with-composite-PK
circular-fk|correction:circular-foreign-key-chains→review-schema-design-and-break-cycle
over-normalization|correction:excessive-table-splitting→consider-denormalization-for-read-heavy-workloads
missing-audit-columns|correction:no-created_at/updated_at→add-temporal-audit-columns
composite-pk-without-index|correction:composite-PK-without-covering-index→add-covering-index
fk-without-target-edge|correction:FK-column-exists-but-no-relationship-edge→connect-to-parent-PK-table
duplicate-pk|correction:multiple-PK-columns-without-composite-declaration→unmark-duplicates-or-declare-composite-key
self-reference-missing|correction:hierarchical-column(parent_id/manager_id)-without-self-referencing-edge→add-self-loop

## [Visual Styling]
icon-style:tables-rendered-as-structured-cards|header-row+column-list|shape=table-or-entity-from-UML-library
edge-style:enforce-orthogonal|edgeStyle=orthogonalEdgeStyle|crow's-foot-terminators-for-cardinality
color-palette:table-header=#333333(dark-bg)+white-text|PK-columns=bold+key-icon-prefix(🔑)|FK-columns=italic+arrow-prefix(→)|nullable-columns=lighter-font-color(#999999)
spacing:260px-column-width|220px-row-height|grid-aligned-3-column-layout

## [Notation Reference]
table-card-structure:
  - header:table-name(bold,dark-background)
  - columns:PK|FK|column-name|data-type(INT,VARCHAR(255))|nullability(NULL,NOT-NULL)
crowsfoot-edges:
  - 1:1→endArrow=ERone;startArrow=ERone
  - 1:N→endArrow=ERmany;startArrow=ERone
  - N:M→decompose-to-junction-table

## [Validator Rules Reference & Troubleshooting]

### 1. `FK_WITHOUT_TARGET`
- **Trigger**: A column labeled as a foreign key (`FK`) exists in a table card, but no relationship edge connects this table card to the parent primary key table card.
- **Troubleshooting**: Connect the table card containing the FK to the parent table card using a solid `1:N` or `1:1` connector line.

### 2. `ORPHAN_TABLE`
- **Trigger**: A table card (`type === 'table'`) has no relationships or connected edges. (Generates a warning).
- **Troubleshooting**: Connect the table to related tables, mark as a standalone lookup table, or delete if obsolete.

### 3. `DUPLICATE_PK`
- **Trigger**: Multiple columns within the same table card are marked as `PK` without being configured as a composite primary key, or there are duplicate column names.
- **Troubleshooting**: Unmark duplicate PK fields or consolidate them into a declared composite key.

### 4. `SELF_REFERENCE_MISSING`
- **Trigger**: A column is named `parent_id`, `manager_id`, or similar hierarchical name, indicating a self-referencing tree relation, but no self-referencing relationship edge connects the table to itself.
- **Troubleshooting**: Add a self-referencing connection edge from the table card to itself.

### 5. `MISSING_JUNCTION`
- **Trigger**: An M:N relationship edge is detected between two tables without an intermediate junction table decomposing the relationship.
- **Troubleshooting**: Create a junction table with a composite primary key composed of foreign keys referencing both parent tables. Replace the M:N edge with two 1:N edges through the junction.

### 6. `CIRCULAR_FK`
- **Trigger**: Foreign key chains form a cycle (Table A → Table B → Table C → Table A).
- **Troubleshooting**: Review the schema design to break the circular dependency. Consider nullable FKs or restructuring the hierarchy.
