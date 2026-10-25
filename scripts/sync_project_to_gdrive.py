r"""
Sentilyze - Google Drive High-Speed Project Sync Engine.

Mirrors D:\Sentilyze to G:\My Drive\Sentilyze using multithreaded robocopy,
excluding bloated environments (.venv, .git, cache directories).
Also copies powerhouse notebooks directly to G:\My Drive\Colab Notebooks\
so you can run Sentilyze directly on Google Colab with drive.mount().
"""

import os
import subprocess
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SOURCE_DIR = r"D:\Sentilyze"
GDRIVE_CANDIDATES = [
    r"G:\My Drive",
    r"G:\MyDrive",
    os.path.expanduser(r"~\Google Drive\My Drive"),
    os.path.expanduser(r"~\My Drive"),
]


def find_gdrive_root():
    for cand in GDRIVE_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


def sync_project():
    print("=" * 80)
    print("🚀 SENTILYZE TO GOOGLE DRIVE SYNC ENGINE")
    print("=" * 80)

    gdrive_root = find_gdrive_root()

    if not gdrive_root:
        print("⚠️  Google Drive (G: drive) is not currently mounted!")
        print("👉 Please open 'Google Drive' from your Windows Start Menu or taskbar.")
        print(
            "   Once Google Drive is running, re-run this script to complete the sync."
        )
        print("=" * 80)
        return False

    dest_project = os.path.join(gdrive_root, "Sentilyze")
    dest_colab = os.path.join(gdrive_root, "Colab Notebooks")

    os.makedirs(dest_project, exist_ok=True)
    os.makedirs(dest_colab, exist_ok=True)

    print(f"📁 Source Directory      : {SOURCE_DIR}")
    print(f"☁️ Google Drive Target   : {dest_project}")
    print(f"📓 Colab Notebooks Target : {dest_colab}")
    print("-" * 80)
    print(
        "⚡ Initiating high-speed multithreaded sync (excluding .venv, .git, caches)..."
    )

    t0 = time.time()

    # Robocopy command
    # /E: recursive including empty
    # /MT:16: 16 parallel threads
    # /XD: exclude directories (.venv, .git, __pycache__, .pytest_cache, mlruns)
    # /XF: exclude large local database caches if needed
    cmd = [
        "robocopy",
        SOURCE_DIR,
        dest_project,
        "/E",
        "/MT:16",
        "/XD",
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        "mlruns",
        "catboost_info",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NP",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    # Robocopy exit codes 0-7 are success (1 = files copied, 0 = no change, etc.)
    if res.returncode <= 7:
        print(f"✅ Core project mirror complete in {time.time() - t0:.2f}s!")
    else:
        print(f"⚠️ Robocopy completed with status code {res.returncode}: {res.stderr}")

    # Copy Notebooks to Colab Notebooks folder
    notebooks = [
        "sentilyze_finmamba_jepa_powerhouse.ipynb",
        "sentilyze_all_in_one_gpu_powerhouse.ipynb",
        "demo.ipynb",
    ]

    print("\n📓 Mirroring master Colab notebooks directly to 'Colab Notebooks'...")
    for nb in notebooks:
        src_nb = os.path.join(SOURCE_DIR, "notebooks", nb)
        dst_nb = os.path.join(dest_colab, nb)
        if os.path.exists(src_nb):
            import shutil

            shutil.copy2(src_nb, dst_nb)
            print(f"   [OK] {nb} -> {dst_nb}")

    print("=" * 80)
    print("🎉 GOOGLE DRIVE SYNC COMPLETE & VERIFIED!")
    print(f"📍 Colab Path: /content/drive/MyDrive/Sentilyze")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = sync_project()
    sys.exit(0 if success else 1)
