"""
Clean orphaned MLflow models from mlruns/0/models that no longer belong
to any active runs, freeing the remaining 1.22 GB.
"""

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    exp0 = Path("mlruns/0")
    models_dir = exp0 / "models"
    if not models_dir.exists():
        print("mlruns/0/models does not exist.")
        return

    active_runs = [p for p in exp0.iterdir() if p.is_dir() and len(p.name) == 32]
    print(f"Active runs in mlruns/0: {len(active_runs)}")

    active_model_ids = set()
    for r in active_runs:
        for f in r.glob("**/*"):
            if f.is_file():
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                        for line in fp:
                            if "m-" in line:
                                for token in line.split():
                                    if "m-" in token:
                                        clean = (
                                            token.split("/")[-1]
                                            .replace('"', "")
                                            .replace("'", "")
                                        )
                                        if clean.startswith("m-"):
                                            active_model_ids.add(clean)
                except Exception:
                    pass

    print(f"Active model references found: {len(active_model_ids)}")
    print(f"Sample active models: {list(active_model_ids)[:5]}")

    all_models = [m.name for m in models_dir.iterdir() if m.is_dir()]
    print(f"Total model directories in mlruns/0/models: {len(all_models)}")

    orphan_models = [m for m in all_models if m not in active_model_ids]
    print(f"Orphan models to prune: {len(orphan_models)}")

    # Parallel delete
    def delete_dir(name):
        p = models_dir / name
        try:
            shutil.rmtree(p, ignore_errors=True)
            return True
        except Exception:
            return False

    t0 = time.time()
    pruned = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        for idx, success in enumerate(executor.map(delete_dir, orphan_models)):
            if success:
                pruned += 1
            if (idx + 1) % 1500 == 0:
                print(
                    f"  Pruned {idx + 1} / {len(orphan_models)} orphan model directories ({time.time() - t0:.1f}s)...",
                    flush=True,
                )

    print(f"Pruned {pruned} orphan models in {time.time() - t0:.1f}s.", flush=True)
    remaining_models = len(list(models_dir.iterdir()))
    print(f"Remaining models in mlruns/0/models: {remaining_models}")


if __name__ == "__main__":
    main()
