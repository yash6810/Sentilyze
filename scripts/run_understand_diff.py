"""
Understand-Diff Engine for Sentilyze.
Performs graph-aware blast radius and diff impact analysis against .ua/knowledge-graph.json.
Generates .ua/diff-overlay.json for dashboard visualization.
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone


def run_diff():
    ua_path = os.path.join(".ua", "knowledge-graph.json")
    if not os.path.exists(ua_path):
        print("[ERROR] .ua/knowledge-graph.json not found. Run understand first.")
        sys.exit(1)

    print("[INFO] Loading knowledge graph...")
    with open(ua_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    layers = graph.get("layers", [])

    # Index nodes
    node_by_id = {n["id"]: n for n in nodes}
    nodes_by_file = {}
    for n in nodes:
        fp = n.get("filePath")
        if fp:
            norm_fp = fp.replace("\\", "/")
            nodes_by_file.setdefault(norm_fp, []).append(n["id"])

    # 1. Get changed files from git
    git_cmd = ["git", "status", "--porcelain"]
    res = subprocess.run(git_cmd, capture_output=True, text=True)
    changed_files = set()
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[1].strip()
        path = path.replace("\\", "/")
        # Ignore git ignored / non-source / temp folders
        if path.startswith(".ua/") or path.startswith(".understand-anything/"):
            continue
        if path.endswith((".py", ".json", ".toml", ".txt", ".md", ".csv")):
            changed_files.add(path)

    changed_files_list = sorted(list(changed_files))
    print(f"[INFO] Detected {len(changed_files_list)} changed project files.")

    # 2. Find matching changed node IDs
    changed_node_ids = set()
    for fp in changed_files_list:
        if fp in nodes_by_file:
            changed_node_ids.update(nodes_by_file[fp])
        # Also match file:path directly
        file_node_id = f"file:{fp}"
        if file_node_id in node_by_id:
            changed_node_ids.add(file_node_id)

    changed_node_ids = sorted(list(changed_node_ids))
    print(f"[INFO] Found {len(changed_node_ids)} directly changed nodes.")

    # 3. Traverse 1-hop connected edges
    affected_node_ids = set()
    changed_set = set(changed_node_ids)

    upstream_callers = {}
    downstream_deps = {}

    for e in edges:
        s = e.get("source")
        t = e.get("target")
        e_type = e.get("type", "related")

        if s in changed_set and t not in changed_set:
            affected_node_ids.add(t)
            downstream_deps.setdefault(s, []).append((t, e_type))
        elif t in changed_set and s not in changed_set:
            affected_node_ids.add(s)
            upstream_callers.setdefault(t, []).append((s, e_type))

    affected_node_ids = sorted(list(affected_node_ids))
    print(f"[INFO] Discovered {len(affected_node_ids)} 1-hop affected components.")

    # 4. Map to Layers
    layer_map = {}
    for lyr in layers:
        l_name = lyr.get("name", lyr.get("id"))
        l_nodes = set(lyr.get("nodeIds", []))
        touched_changed = l_nodes.intersection(changed_set)
        touched_affected = l_nodes.intersection(set(affected_node_ids))
        if touched_changed or touched_affected:
            layer_map[l_name] = {
                "changed_count": len(touched_changed),
                "affected_count": len(touched_affected),
                "description": lyr.get("description", ""),
            }

    # 5. Risk Assessment
    def parse_complexity(c):
        if isinstance(c, (int, float)):
            return float(c)
        if isinstance(c, str):
            c_lower = c.lower()
            if "high" in c_lower:
                return 4.0
            if "med" in c_lower:
                return 2.5
            if "low" in c_lower:
                return 1.0
            try:
                return float(c)
            except ValueError:
                return 1.0
        return 1.0

    complexities = [
        parse_complexity(node_by_id[nid].get("complexity", 1))
        for nid in changed_node_ids
        if nid in node_by_id
    ]
    avg_complexity = sum(complexities) / len(complexities) if complexities else 1.0
    max_complexity = max(complexities) if complexities else 1.0

    blast_radius_ratio = (len(changed_node_ids) + len(affected_node_ids)) / max(
        len(nodes), 1
    )
    if blast_radius_ratio > 0.4 or max_complexity >= 5:
        risk_level = "HIGH"
    elif blast_radius_ratio > 0.15 or max_complexity >= 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 6. Write diff overlay JSON
    overlay = {
        "version": "1.0.0",
        "baseBranch": "main",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "changedFiles": changed_files_list,
        "changedNodeIds": changed_node_ids,
        "affectedNodeIds": affected_node_ids,
        "metrics": {
            "changedFilesCount": len(changed_files_list),
            "changedNodesCount": len(changed_node_ids),
            "affectedNodesCount": len(affected_node_ids),
            "totalGraphNodes": len(nodes),
            "blastRadiusPct": round(blast_radius_ratio * 100, 2),
            "avgComplexity": round(avg_complexity, 2),
            "maxComplexity": max_complexity,
            "riskLevel": risk_level,
        },
        "affectedLayers": layer_map,
    }

    out_path = os.path.join(".ua", "diff-overlay.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(overlay, f, indent=2)

    print(f"[SUCCESS] Written diff overlay to {out_path}")
    print(
        f"Risk Level: {risk_level} | Blast Radius: {overlay['metrics']['blastRadiusPct']}%"
    )
    print(
        f"Changed Nodes: {len(changed_node_ids)} | Affected Nodes: {len(affected_node_ids)}"
    )


if __name__ == "__main__":
    run_diff()
