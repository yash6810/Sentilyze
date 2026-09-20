"""
Real-time terminal progress monitor for Sentilyze universe training.
Displays live progress bar, completion counts, elapsed time, and ETA.
"""

import os
import sys
import time
import json


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_progress_bar(pct: float, width: int = 30) -> str:
    filled = int(width * (pct / 100.0))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def main():
    progress_file = os.path.join("results", "training_progress.json")

    print("\n🔍 Sentilyze Training Monitor initialized. Press Ctrl+C to exit.\n")
    time.sleep(1)

    while True:
        try:
            if not os.path.exists(progress_file):
                clear_screen()
                print("==========================================================")
                print("   SENTILYZE TRAINING MONITOR                            ")
                print("==========================================================")
                print(
                    "\n⏳ Waiting for training to begin (results/training_progress.json not found yet)...\n"
                )
                time.sleep(2)
                continue

            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            status = data.get("status", "UNKNOWN")
            completed = data.get("completed_count", 0)
            total = data.get("total_count", 538)
            pct = data.get("pct_done", 0.0)
            last_tk = data.get("last_ticker", "N/A")
            last_sec = data.get("last_duration_sec", 0.0)
            elapsed_m = data.get("elapsed_minutes", 0.0)
            eta_m = data.get("eta_minutes", 0.0)
            success = data.get("success_count", completed)
            failed = data.get("failure_count", 0)

            bar = render_progress_bar(pct, width=28)

            clear_screen()
            print("==========================================================")
            print(f"   SENTILYZE UNIVERSAL TRAINING MONITOR  •  STATUS: {status}")
            print("==========================================================")
            print(f" {bar}  {pct:5.1f}%\n")
            print(
                f" 📊 Completed:   {completed} / {total} Assets (Success: {success}, Failed: {failed})"
            )
            print(f" ⚡ Last Asset:  {last_tk:<5} ({last_sec:.2f}s)")
            print(f" ⏱️ Elapsed:     {elapsed_m:.2f} mins")
            print(f" ⏳ Est. ETA:    ~{eta_m:.2f} mins remaining")
            print("==========================================================")
            print(" Auto-refreshing every 2s. Press Ctrl+C to stop monitor.\n")

            if status == "COMPLETED":
                print("🎉 Training is 100% complete across the full universe!")
                break

            time.sleep(2)

        except KeyboardInterrupt:
            print("\nExiting monitor. Training will continue in the background.\n")
            break
        except Exception as e:
            time.sleep(2)


if __name__ == "__main__":
    main()
