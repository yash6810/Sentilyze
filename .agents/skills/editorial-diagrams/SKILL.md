---
name: editorial-diagrams
description: Publication-grade editorial diagram design and clean SVG/HTML generation for financial architecture, strategy flowcharts, and risk topology without messy Mermaid markup.
---

# Editorial Diagram Design Skill

Use this skill to design crisp, minimalist, high-contrast diagrams using pure SVG and self-contained HTML/CSS.

## 1. Design Rules
- **No Generic Mermaid Slop**: Use pure semantic SVG or canvas with custom fonts, clean rounded corners (`rx="8"`), and crisp vector paths.
- **Institutional Dark Theme Palette**:
  - Background: `#0e1117` / `#1a1c24`
  - Borders: `#2a2d3d` / `#3b3f54`
  - Accents: Neon Emerald (`#00f59b`), Cyan (`#00d4ff`), Amber (`#ffb800`), Rose Red (`#ff3366`)
- **Typography**: Clean monospace / system sans-serif (`Inter`, `JetBrains Mono`, `Roboto`).
- **Responsive Embed**: Wrap in responsive `<svg viewBox="0 0 W H" width="100%">` for clean scaling in Streamlit tabs.
