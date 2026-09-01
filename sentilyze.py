"""
Root entry point for Sentilyze CLI.
Run directly with:
    python sentilyze.py NVDA
    python sentilyze.py audit AAPL
    python sentilyze.py portfolio
"""
import sys
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.cli import main

if __name__ == "__main__":
    main()
