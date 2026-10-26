import importlib.metadata
import re

def pin_requirements_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pinned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            pinned_lines.append(line)
            continue

        # Extract package name (before >=, ==, <=, ~, etc.)
        pkg_name = re.split(r"[><=~]", stripped)[0].strip()
        try:
            ver = importlib.metadata.version(pkg_name)
            pinned_lines.append(f"{pkg_name}=={ver}\n")
        except importlib.metadata.PackageNotFoundError:
            # Try normalizing hyphens/underscores
            alt_name = pkg_name.replace("-", "_") if "-" in pkg_name else pkg_name.replace("_", "-")
            try:
                ver = importlib.metadata.version(alt_name)
                pinned_lines.append(f"{pkg_name}=={ver}\n")
            except Exception:
                print(f"Could not find version for {pkg_name}, keeping original: {stripped}")
                pinned_lines.append(line)

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(pinned_lines)
    print(f"Pinned {filepath}")

pin_requirements_file("requirements.txt")
pin_requirements_file("requirements-dev.txt")
