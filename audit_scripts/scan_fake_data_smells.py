import os
import re
import sys

patterns = [
    r"\bdummy\b",
    r"\bsynthetic\b",
    r"\bplaceholder\b",
    r"\bEphemeral_Key\b",
    r"\bnp\.random\b",
    r"\brandom\.",
]

compiled = [re.compile(p, re.IGNORECASE) for p in patterns]

matches = []

for root, _, files in os.walk("src"):
    for file in files:
        if not file.endswith(".py"):
            continue
        filepath = os.path.join(root, file)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for idx, line in enumerate(lines, 1):
                for p_idx, regex in enumerate(compiled):
                    if regex.search(line):
                        matches.append((filepath, idx, patterns[p_idx], line.strip()))
                        break
        except Exception as e:
            pass

print(f"Total smell matches found: {len(matches)}")
with open("audit_outputs/fake_data_smells_raw.txt", "w", encoding="utf-8") as out:
    for fpath, lnum, pat, text in matches:
        out.write(f"{fpath}:{lnum} [{pat}] {text}\n")
