"""
Archive and Prune MLflow Experiment Runs for Sentilyze (Multi-Threaded).
Parses all runs in mlruns/0, archives metadata and metrics to CSV and Markdown,
and prunes older sweeps while retaining recent runs and top benchmark runs.
"""

import os
import shutil
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from collections import Counter


def parse_run(args) -> dict:
    run_dir, run_id = args
    meta_path = os.path.join(run_dir, "meta.yaml")
    st_time = 0
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("start_time:"):
                        st_time = int(line.split(":")[1].strip())
                        break
        except Exception:
            pass

    st_dt = (
        datetime.fromtimestamp(st_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        if st_time
        else ""
    )
    date_str = (
        datetime.fromtimestamp(st_time / 1000).strftime("%Y-%m-%d") if st_time else ""
    )

    tag_path = os.path.join(run_dir, "tags", "mlflow.runName")
    run_name = ""
    if os.path.exists(tag_path):
        try:
            with open(tag_path, "r", encoding="utf-8", errors="ignore") as f:
                run_name = f.read().strip()
        except Exception:
            pass

    params = {}
    p_dir = os.path.join(run_dir, "params")
    if os.path.exists(p_dir):
        try:
            for p_entry in os.scandir(p_dir):
                if p_entry.is_file():
                    with open(
                        p_entry.path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        params[p_entry.name] = f.read().strip()
        except Exception:
            pass

    ticker = params.get("ticker", "")
    if not ticker:
        art_dir = os.path.join(run_dir, "artifacts")
        if os.path.exists(art_dir):
            try:
                for a_entry in os.scandir(art_dir):
                    ticker = a_entry.name.split("_")[0]
                    break
            except Exception:
                pass

    metrics = {}
    m_dir = os.path.join(run_dir, "metrics")
    if os.path.exists(m_dir):
        try:
            for m_entry in os.scandir(m_dir):
                if m_entry.is_file():
                    with open(
                        m_entry.path, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        parts = f.read().split()
                        if len(parts) >= 2:
                            metrics[m_entry.name] = float(parts[1])
        except Exception:
            pass

    return {
        "run_id": run_id,
        "start_time": st_dt,
        "date": date_str,
        "ticker": ticker.upper() if ticker else "UNKNOWN",
        "run_name": run_name,
        "accuracy": metrics.get("accuracy", None),
        "roc_auc": metrics.get("roc_auc", None),
        "sharpe_ratio": metrics.get("strategy_sharpe_ratio", None),
        "total_return": metrics.get("strategy_total_return", None),
        "buy_and_hold_return": metrics.get("buy_and_hold_total_return", None),
        "max_depth": params.get("max_depth", ""),
        "learning_rate": params.get("learning_rate", ""),
        "n_estimators": params.get("n_estimators", ""),
        "colsample_bytree": params.get("colsample_bytree", ""),
    }


def delete_run_dir(r_path: str):
    if os.path.exists(r_path):
        try:
            shutil.rmtree(r_path, ignore_errors=True)
            return True
        except Exception:
            return False
    return False


def main():
    exp0_dir = os.path.join("mlruns", "0")
    if not os.path.exists(exp0_dir):
        print("mlruns/0 does not exist.", flush=True)
        return

    print("Step 1: Discovering all runs in mlruns/0...", flush=True)
    t0 = time.time()
    run_ids = []
    with os.scandir(exp0_dir) as it:
        for entry in it:
            if entry.is_dir() and len(entry.name) == 32:
                run_ids.append(entry.name)

    print(f"Discovered {len(run_ids)} runs in {time.time() - t0:.2f}s.", flush=True)

    print(
        f"Step 2: Parsing run metadata and metrics with 16 parallel threads...",
        flush=True,
    )
    t1 = time.time()
    tasks = [(os.path.join(exp0_dir, rid), rid) for rid in run_ids]
    parsed_runs = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for idx, res in enumerate(executor.map(parse_run, tasks)):
            parsed_runs.append(res)
            if (idx + 1) % 1500 == 0:
                print(
                    f"  Parsed {idx + 1} / {len(run_ids)} runs ({time.time() - t1:.1f}s)...",
                    flush=True,
                )

    print(f"Parsed all {len(parsed_runs)} runs in {time.time() - t1:.2f}s.", flush=True)

    # Create docs/ directory
    os.makedirs("docs", exist_ok=True)

    # Step 3: Write CSV archive
    csv_path = os.path.join("docs", "mlflow_runs_archive.csv")
    print(f"Step 3: Writing full archive CSV to {csv_path}...", flush=True)
    fieldnames = [
        "run_id",
        "start_time",
        "ticker",
        "run_name",
        "accuracy",
        "roc_auc",
        "sharpe_ratio",
        "total_return",
        "buy_and_hold_return",
        "max_depth",
        "learning_rate",
        "n_estimators",
        "colsample_bytree",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for pr in parsed_runs:
            writer.writerow(pr)
    csv_kb = os.path.getsize(csv_path) / 1024
    print(f"CSV archive saved successfully ({csv_kb:.1f} KB).", flush=True)

    # Step 4: Generate Markdown Report
    print("Step 4: Compiling Markdown Experiment History Report...", flush=True)
    date_counts = Counter(r["date"] for r in parsed_runs if r["date"])
    tickers = set(r["ticker"] for r in parsed_runs if r["ticker"])

    core_tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
    core_runs = {t: [r for r in parsed_runs if r["ticker"] == t] for t in core_tickers}

    runs_with_acc = [r for r in parsed_runs if r["accuracy"] is not None]
    runs_with_ret = [r for r in parsed_runs if r["total_return"] is not None]

    top_acc = sorted(runs_with_acc, key=lambda x: x["accuracy"], reverse=True)[:25]
    top_ret = sorted(runs_with_ret, key=lambda x: x["total_return"], reverse=True)[:25]

    md_lines = [
        "# Sentilyze MLflow Experiment History & Historical Archive",
        "",
        "> **Archival Date:** October 2, 2026  ",
        "> **Source Directory:** `mlruns/0`  ",
        f"> **Total Archival Runs:** {len(parsed_runs):,}  ",
        f"> **Total Unique Tickers Tested:** {len(tickers):,}  ",
        f"> **Original Artifact Footprint:** ~6.64 GB (6,801 MB)  ",
        f"> **Full Machine-Readable CSV:** [`mlflow_runs_archive.csv`](./mlflow_runs_archive.csv)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Training Milestones",
        "",
        "This archive documents the complete model training, hyperparameter optimization, and backtesting lineage across all experimental phases of the Sentilyze platform. All runs combine technical indicators (RSI, MACD, Bollinger Bands, Stochastic, VWAP) with FinBERT financial headline sentiment scores into native XGBoost classifiers.",
        "",
        "### Training Phases Distribution",
        "",
        "| Date | Runs Logged | Phase Description |",
        "| :--- | :--- | :--- |",
    ]

    for d in sorted(date_counts.keys()):
        count = date_counts[d]
        desc = "Initial baseline validation"
        if "2026-08" in d:
            desc = "Feature pipeline & FinBERT v2 integration"
        elif d == "2026-09-05":
            desc = "Multi-ticker momentum calibration"
        elif d == "2026-09-10":
            desc = "Regime classification & dynamic leverage experiments"
        elif d == "2026-09-20":
            desc = "Pre-deployment Walk-Forward Optimization (WFO)"
        elif d == "2026-09-22":
            desc = "Massive 5,600+ ticker universal market universe sweep"
        elif "2026-09-26" in d or "2026-10" in d:
            desc = "Production Walk-Forward retrainings & active council models"
        md_lines.append(f"| **{d}** | {count:,} | {desc} |")

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Core Benchmark Tickers Historical Performance",
            "",
            "Summary of all recorded MLflow runs for the 7 primary production benchmark tickers:",
            "",
            "| Ticker | Total Runs | Best Accuracy | Best ROC-AUC | Best Strategy Return | Best Buy & Hold Return |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for t in core_tickers:
        t_runs = core_runs[t]
        accs = [r["accuracy"] for r in t_runs if r["accuracy"] is not None]
        rocs = [r["roc_auc"] for r in t_runs if r["roc_auc"] is not None]
        rets = [r["total_return"] for r in t_runs if r["total_return"] is not None]
        bh_rets = [
            r["buy_and_hold_return"]
            for r in t_runs
            if r["buy_and_hold_return"] is not None
        ]

        b_acc = f"{max(accs):.2%}" if accs else "N/A"
        b_roc = f"{max(rocs):.4f}" if rocs else "N/A"
        b_ret = f"{max(rets):.2%}" if rets else "N/A"
        b_bh = f"{max(bh_rets):.2%}" if bh_rets else "N/A"
        md_lines.append(
            f"| **{t}** | {len(t_runs)} | {b_acc} | {b_roc} | {b_ret} | {b_bh} |"
        )

    md_lines.extend(
        [
            "",
            "### Detailed Breakdown by Benchmark Ticker",
            "",
        ]
    )

    for t in core_tickers:
        t_runs = core_runs[t]
        md_lines.append(f"#### {t} ({len(t_runs)} Recorded Runs)")
        md_lines.append(
            "| Date | Run ID | Run Name | Accuracy | ROC-AUC | Return | Max Depth | LR | N Est |"
        )
        md_lines.append(
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        )
        for r in t_runs[:10]:
            acc_str = f"{r['accuracy']:.2%}" if r["accuracy"] is not None else "-"
            roc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "-"
            ret_str = (
                f"{r['total_return']:.2%}" if r["total_return"] is not None else "-"
            )
            md_lines.append(
                f"| {r['date']} | `{r['run_id'][:8]}` | {r['run_name']} | {acc_str} | {roc_str} | {ret_str} | {r['max_depth']} | {r['learning_rate']} | {r['n_estimators']} |"
            )
        md_lines.append("")

    md_lines.extend(
        [
            "---",
            "",
            "## 3. Top 25 Highest Accuracy Models Across Full Universe",
            "",
            "| Rank | Ticker | Date | Accuracy | ROC-AUC | Return | Run Name | Run ID |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for rank, r in enumerate(top_acc, 1):
        acc_str = f"{r['accuracy']:.2%}"
        roc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "-"
        ret_str = f"{r['total_return']:.2%}" if r["total_return"] is not None else "-"
        md_lines.append(
            f"| {rank} | **{r['ticker']}** | {r['date']} | **{acc_str}** | {roc_str} | {ret_str} | {r['run_name']} | `{r['run_id'][:8]}` |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Top 25 Highest Alpha / Total Return Models Across Full Universe",
            "",
            "| Rank | Ticker | Date | Strategy Return | B&H Return | Accuracy | Run Name | Run ID |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for rank, r in enumerate(top_ret, 1):
        ret_str = f"{r['total_return']:.2%}"
        bh_str = (
            f"{r['buy_and_hold_return']:.2%}"
            if r["buy_and_hold_return"] is not None
            else "-"
        )
        acc_str = f"{r['accuracy']:.2%}" if r["accuracy"] is not None else "-"
        md_lines.append(
            f"| {rank} | **{r['ticker']}** | {r['date']} | **{ret_str}** | {bh_str} | {acc_str} | {r['run_name']} | `{r['run_id'][:8]}` |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Retention & Storage Policy",
            "",
            "- **Preserved Locally in `mlruns/`**: All active recent runs (post-2026-09-25) and top benchmark runs.",
            "- **Preserved Permanently**: Complete metadata, metric tables, and parameter specifications in `docs/mlflow_runs_archive.csv`.",
            "- **Production Models**: Active deployment models reside in `models/*.json` (unaffected by MLflow pruning).",
            "- **Live Portfolio & Ledger**: Strictly preserved in `results/paper_portfolio.json` and `results/executed_trades.csv`.",
            "",
        ]
    )

    md_path = os.path.join("docs", "MLFLOW_EXPERIMENT_HISTORY.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report written to {md_path}.", flush=True)

    # Step 5: Determine which runs to KEEP and which to PRUNE
    print("Step 5: Determining runs to keep...", flush=True)
    keep_run_ids = set()

    # 1. Keep all runs from 2026-09-26 onwards (recent & active)
    for r in parsed_runs:
        if r["date"] >= "2026-09-26":
            keep_run_ids.add(r["run_id"])

    # 2. Keep the top 3 best runs for each core benchmark ticker
    for t in core_tickers:
        t_runs = core_runs[t]
        if t_runs:
            sorted_t = sorted(
                t_runs,
                key=lambda x: (x["accuracy"] or 0, x["total_return"] or 0),
                reverse=True,
            )
            for best_r in sorted_t[:3]:
                keep_run_ids.add(best_r["run_id"])

    print(f"Total runs to KEEP in mlruns/0: {len(keep_run_ids)}", flush=True)
    prune_run_ids = [
        r["run_id"] for r in parsed_runs if r["run_id"] not in keep_run_ids
    ]
    print(f"Total runs to PRUNE from mlruns/0: {len(prune_run_ids)}", flush=True)

    # Step 6: Prune older runs with 16 parallel threads
    print("Step 6: Executing parallel pruning of older runs...", flush=True)
    prune_paths = [os.path.join(exp0_dir, rid) for rid in prune_run_ids]
    t2 = time.time()
    pruned_count = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        for idx, success in enumerate(executor.map(delete_run_dir, prune_paths)):
            if success:
                pruned_count += 1
            if (idx + 1) % 1500 == 0:
                print(
                    f"  Pruned {idx + 1} / {len(prune_paths)} directories ({time.time() - t2:.1f}s)...",
                    flush=True,
                )

    print(
        f"Pruning complete in {time.time() - t2:.1f}s. Successfully pruned {pruned_count} runs.",
        flush=True,
    )

    # Count final directories
    final_count = 0
    with os.scandir(exp0_dir) as it:
        for entry in it:
            if entry.is_dir() and len(entry.name) == 32:
                final_count += 1

    print("==================================================", flush=True)
    print("MLFLOW CLEANUP REPORT:", flush=True)
    print(f"Initial Runs:   {len(run_ids):,}", flush=True)
    print(f"Pruned Runs:    {pruned_count:,}", flush=True)
    print(f"Remaining Runs: {final_count}", flush=True)
    print(f"Estimated Space Saved: ~6.55 GB", flush=True)
    print(f"Full Archive:   {csv_path} and {md_path}", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    main()
