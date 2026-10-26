import os
import shutil
import tempfile
import json
from src.funnel_tracker import record_funnel_run


def simulate_funnel_persistence():
    base_dir = tempfile.mkdtemp(prefix="sentilyze_funnel_persist_")
    print(f"Simulation root: {base_dir}")

    # Shared remote git repository simulation
    remote_repo = os.path.join(base_dir, "remote_git_origin")
    os.makedirs(os.path.join(remote_repo, "results"), exist_ok=True)
    shared_log = os.path.join(remote_repo, "results", "funnel_log.jsonl")

    # --- Run 1: Fresh container 1 ---
    c1 = os.path.join(base_dir, "container_1")
    os.makedirs(os.path.join(c1, "results"), exist_ok=True)
    c1_log = os.path.join(c1, "results", "funnel_log.jsonl")
    c1_json = os.path.join(c1, "results", "funnel.json")
    c1_md = os.path.join(c1, "results", "funnel.md")

    # Fresh checkout from remote: copy shared_log if it exists (empty initially)
    if os.path.exists(shared_log):
        shutil.copy(shared_log, c1_log)

    record_funnel_run(
        tickers_scanned=25,
        skipped_no_real_news=10,
        vetoed_by_reason={"VALUATION_VETO": 5, "DOUBLE_CRASH_VETO": 2},
        buy_blocked_guards={},
        orders_submitted=0,
        json_path=c1_json,
        md_path=c1_md,
        log_path=c1_log,
    )
    # Rebase & push step: pushes c1_log back to remote_repo
    shutil.copy(c1_log, shared_log)

    with open(c1_md, "r", encoding="utf-8") as f:
        print("\n=== CONTAINER 1 funnel.md (Run 1 Completed) ===")
        print(f.read().strip())

    # --- Run 2: Fresh container 2 ---
    c2 = os.path.join(base_dir, "container_2")
    os.makedirs(os.path.join(c2, "results"), exist_ok=True)
    c2_log = os.path.join(c2, "results", "funnel_log.jsonl")
    c2_json = os.path.join(c2, "results", "funnel.json")
    c2_md = os.path.join(c2, "results", "funnel.md")

    # Fresh checkout from remote: checks out shared_log pushed by Run 1
    if os.path.exists(shared_log):
        shutil.copy(shared_log, c2_log)

    record_funnel_run(
        tickers_scanned=25,
        skipped_no_real_news=8,
        vetoed_by_reason={"VALUATION_VETO": 4, "DOUBLE_CRASH_VETO": 3},
        buy_blocked_guards={"DUPLICATE_POSITION": 1},
        orders_submitted=0,
        json_path=c2_json,
        md_path=c2_md,
        log_path=c2_log,
    )
    # Rebase & push step: pushes c2_log back to remote_repo
    shutil.copy(c2_log, shared_log)

    with open(c2_md, "r", encoding="utf-8") as f:
        print("\n=== CONTAINER 2 funnel.md (Cumulative 2 Runs) ===")
        print(f.read().strip())

    # --- Run 3: Fresh container 3 ---
    c3 = os.path.join(base_dir, "container_3")
    os.makedirs(os.path.join(c3, "results"), exist_ok=True)
    c3_log = os.path.join(c3, "results", "funnel_log.jsonl")
    c3_json = os.path.join(c3, "results", "funnel.json")
    c3_md = os.path.join(c3, "results", "funnel.md")

    # Fresh checkout from remote: checks out shared_log pushed by Run 2
    if os.path.exists(shared_log):
        shutil.copy(shared_log, c3_log)

    record_funnel_run(
        tickers_scanned=25,
        skipped_no_real_news=12,
        vetoed_by_reason={"VALUATION_VETO": 6},
        buy_blocked_guards={},
        orders_submitted=1,
        json_path=c3_json,
        md_path=c3_md,
        log_path=c3_log,
    )
    # Rebase & push step: pushes c3_log back to remote_repo
    shutil.copy(c3_log, shared_log)

    with open(c3_md, "r", encoding="utf-8") as f:
        print("\n=== CONTAINER 3 funnel.md (Cumulative 3 Runs) ===")
        print(f.read().strip())

    # Also check the JSONL entries count
    with open(shared_log, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
        print(f"\nTotal lines in persistent results/funnel_log.jsonl: {len(entries)}")

    shutil.rmtree(base_dir)


if __name__ == "__main__":
    simulate_funnel_persistence()
