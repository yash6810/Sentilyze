"""
Root entry point for Sentilyze CLI.
Usage:
    python sentilyze.py audit NVDA
    python sentilyze.py portfolio
"""

from src.cli import main

if __name__ == "__main__":
    main()
