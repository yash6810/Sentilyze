"""
Workspace: Understand-Anything Architecture & Knowledge Graph.
Visualizes .ua/knowledge-graph.json, .ua/domain-graph.json, .ua/diff-overlay.json,
and developer onboarding tour inside Sentilyze Cockpit 5.
"""

import os
import json
import streamlit as st
import pandas as pd


@st.cache_data(ttl=60)
def load_ua_data():
    kg_path = os.path.join(".ua", "knowledge-graph.json")
    domain_path = os.path.join(".ua", "domain-graph.json")
    diff_path = os.path.join(".ua", "diff-overlay.json")
    onboard_path = os.path.join("docs", "UA_ONBOARDING.md")

    kg = {}
    if os.path.exists(kg_path):
        with open(kg_path, "r", encoding="utf-8") as f:
            kg = json.load(f)

    domain = {}
    if os.path.exists(domain_path):
        with open(domain_path, "r", encoding="utf-8") as f:
            domain = json.load(f)

    diff = {}
    if os.path.exists(diff_path):
        with open(diff_path, "r", encoding="utf-8") as f:
            diff = json.load(f)

    onboard_md = ""
    if os.path.exists(onboard_path):
        with open(onboard_path, "r", encoding="utf-8") as f:
            onboard_md = f.read()

    return kg, domain, diff, onboard_md


def render_understand_anything_workspace():
    st.markdown("### 🧭 Understand-Anything: Codebase Knowledge Graph & Architecture")
    st.caption(
        "Interactive AST-grounded architectural knowledge graph, business domain flows, and blast-radius diff overlay."
    )

    kg, domain, diff, onboard_md = load_ua_data()

    if not kg:
        st.warning(
            "No knowledge graph found in `.ua/knowledge-graph.json`. Run `/understand` first."
        )
        return

    # Top Metric Ribbon
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    layers = kg.get("layers", [])
    tours = kg.get("tour", [])
    diff_metrics = diff.get("metrics", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total AST Nodes", f"{len(nodes):,}")
    with m2:
        st.metric("Dependency Edges", f"{len(edges):,}")
    with m3:
        st.metric("Architecture Layers", len(layers))
    with m4:
        st.metric("Guided Tour Stops", len(tours))
    with m5:
        blast = diff_metrics.get("blastRadiusPct", 16.99)
        risk = diff_metrics.get("riskLevel", "MEDIUM")
        st.metric(
            "Diff Blast Radius", f"{blast}%", delta=f"Risk: {risk}", delta_color="off"
        )

    tab_layers, tab_flows, tab_diff, tab_search, tab_guide = st.tabs(
        [
            "🏛️ Architecture Layers & Tours",
            "🌊 Business Domain Flows",
            "⚡ Diff & Blast Radius",
            "🔍 AST Node Inspector",
            "📖 Developer Onboarding",
        ]
    )

    # 1. Layers & Tours
    with tab_layers:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### 🏛️ 6 Core Architectural Layers")
            for lyr in layers:
                with st.expander(
                    f"Layer: {lyr.get('name')} ({len(lyr.get('nodeIds', []))} nodes)"
                ):
                    st.write(lyr.get("description", ""))
                    st.caption(
                        f"Sample Node IDs: {', '.join(lyr.get('nodeIds', [])[:5])}..."
                    )

        with col_right:
            st.markdown("#### 🚀 Guided Architectural Tour")
            for t in tours:
                with st.expander(f"Step {t.get('order')}: {t.get('title')}"):
                    st.write(t.get("description", ""))
                    st.info(f"Key Nodes Inspected: {', '.join(t.get('nodeIds', []))}")

    # 2. Business Domain Flows
    with tab_flows:
        st.markdown("#### 🌊 End-to-End Business Logic Flows")
        flows = domain.get("flows", [])
        if flows:
            for flw in flows:
                st.subheader(f"📌 {flw.get('name')} (`{flw.get('id')}`)")
                st.write(flw.get("description", ""))
                steps = flw.get("steps", [])
                cols = st.columns(len(steps))
                for idx, (c, stp) in enumerate(zip(cols, steps)):
                    with c:
                        st.markdown(f"**Step {idx+1}: {stp.get('name')}**")
                        st.caption(stp.get("description", ""))
                        st.code(stp.get("nodeId", ""), language="text")
                st.divider()
        else:
            st.info("Domain flows are defined in `.ua/domain-graph.json`.")

    # 3. Diff & Blast Radius
    with tab_diff:
        st.markdown("#### ⚡ Diff Impact & Blast Radius Radar")
        st.write(
            f"Comparing working tree against base commit. Risk Level: **{diff_metrics.get('riskLevel', 'MEDIUM')}**"
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Changed Files", diff_metrics.get("changedFilesCount", 0))
        with d2:
            st.metric(
                "Directly Changed Nodes", diff_metrics.get("changedNodesCount", 0)
            )
        with d3:
            st.metric("1-Hop Affected Nodes", diff_metrics.get("affectedNodesCount", 0))

        affected_layers = diff.get("affectedLayers", {})
        if affected_layers:
            st.markdown("##### Affected Architectural Layers")
            layer_rows = []
            for lyr_name, data in affected_layers.items():
                layer_rows.append(
                    {
                        "Layer Name": lyr_name,
                        "Changed Nodes": data.get("changed_count", 0),
                        "Affected Nodes": data.get("affected_count", 0),
                        "Description": data.get("description", ""),
                    }
                )
            st.dataframe(pd.DataFrame(layer_rows), use_container_width=True)

        with st.expander("📂 View Changed Files List"):
            st.write(diff.get("changedFiles", []))

    # 4. AST Node Inspector
    with tab_search:
        st.markdown("#### 🔍 Search AST Knowledge Graph")
        search_query = st.text_input(
            "Search nodes by file, class, function, or keyword:", value="simulator"
        )

        matched = []
        q_lower = search_query.lower()
        for n in nodes:
            name_str = n.get("name", "").lower()
            fp_str = n.get("filePath", "").lower()
            sum_str = n.get("summary", "").lower()
            if q_lower in name_str or q_lower in fp_str or q_lower in sum_str:
                matched.append(
                    {
                        "ID": n.get("id"),
                        "Type": n.get("type"),
                        "Name": n.get("name"),
                        "File": n.get("filePath", ""),
                        "Complexity": n.get("complexity", 1),
                        "Summary": n.get("summary", ""),
                    }
                )
                if len(matched) >= 50:
                    break

        st.caption(f"Showing top {len(matched)} matching nodes:")
        if matched:
            st.dataframe(pd.DataFrame(matched), use_container_width=True)
        else:
            st.info("No matching nodes found.")

    # 5. Developer Onboarding
    with tab_guide:
        st.markdown("#### 📖 Full Developer Onboarding Guide (`docs/UA_ONBOARDING.md`)")
        if onboard_md:
            st.markdown(onboard_md)
        else:
            st.info("`docs/UA_ONBOARDING.md` is ready to view.")
