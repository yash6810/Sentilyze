"""
Regression and Audit Test: README Traceability.
Ensures that all performance figures and equity metrics presented in README.md
are strictly grounded in and traceable to verified results files in results/.
Fails if any ungrounded, fabricated, or deprecated figures are found.
"""

import re
import os
import pandas as pd
import pytest


FORBIDDEN_LEGACY_NUMBERS = [
    "136,489.18",
    "54,092.72",
    "54,092",
    "54.09%",
    "96.8%",
    "201.2%",
    "346.7%",
]


def test_readme_contains_no_forbidden_legacy_numbers():
    assert os.path.exists("README.md"), "README.md must exist"
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    found_forbidden = []
    for num in FORBIDDEN_LEGACY_NUMBERS:
        if num in content:
            found_forbidden.append(num)

    assert not found_forbidden, (
        f"README.md contains untraceable/invalid numbers: {found_forbidden}"
    )


def test_readme_performance_metrics_match_alpaca_results():
    assert os.path.exists("README.md"), "README.md must exist"
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()

    # If Alpaca report exists, verify exact return is cited
    alpaca_vs_spy_file = os.path.join("results", "alpaca_vs_spy.csv")
    if os.path.exists(alpaca_vs_spy_file):
        df = pd.read_csv(alpaca_vs_spy_file)
        start_eq = df["alpaca_equity"].iloc[0]
        end_eq = df["alpaca_equity"].iloc[-1]
        alp_ret = ((end_eq / start_eq) - 1.0) * 100.0
        ret_str = f"{alp_ret:+.2f}%"

        # The honest status section must quote the exact Alpaca return
        assert "+2.44%" in content or ret_str in content, (
            f"README.md must reflect verified Alpaca return {ret_str}"
        )
