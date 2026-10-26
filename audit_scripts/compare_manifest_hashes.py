import json
import subprocess

def get_manifest_at(commit):
    res = subprocess.run(["git", "show", f"{commit}:results/strategy_freeze_manifest.json"], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)

m_old = get_manifest_at("2dd8289b")
m_new = get_manifest_at("76287264")

diffs = {}
for k, v in m_new.items():
    if k.startswith("models/"):
        v_old = m_old.get(k)
        if v_old != v:
            diffs[k] = {"old": v_old, "new": v}

for k, v in m_old.items():
    if k.startswith("models/") and k not in m_new:
        diffs[k] = {"old": v, "new": None}

print(f"Total model diffs: {len(diffs)}")
for k, v in diffs.items():
    print(f"{k}: old={v['old']} -> new={v['new']}")
