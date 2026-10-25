"""
Generates the FinBERT committee ablation markdown table for all 5 tickers
from results/committee_ablation_study.json.
"""

import json


def main():
    with open("results/committee_ablation_study.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for ticker, item in data.items():
        days = item.get("evaluation_days", "UNKNOWN")
        full = item["configurations"]["full_committee"]
        minus = item["configurations"]["minus_sentiment"]
        ret_w = full["total_return_pct"]
        sh_w = full["sharpe_ratio"]
        ret_wo = minus["total_return_pct"]
        sh_wo = minus["sharpe_ratio"]
        diff_ret = ret_w - ret_wo
        diff_sh = sh_w - sh_wo
        rows.append(
            {
                "ticker": ticker,
                "days": days,
                "ret_with": ret_w,
                "sh_with": sh_w,
                "ret_without": ret_wo,
                "sh_without": sh_wo,
                "diff_ret": diff_ret,
                "diff_sh": diff_sh,
            }
        )

    avg_ret_w = sum(r["ret_with"] for r in rows) / len(rows)
    avg_sh_w = sum(r["sh_with"] for r in rows) / len(rows)
    avg_ret_wo = sum(r["ret_without"] for r in rows) / len(rows)
    avg_sh_wo = sum(r["sh_without"] for r in rows) / len(rows)
    avg_diff_ret = sum(r["diff_ret"] for r in rows) / len(rows)
    avg_diff_sh = sum(r["diff_sh"] for r in rows) / len(rows)

    lines = [
        "| Ticker | Eval Days | Return (With FinBERT) | Sharpe (With) | Return (Without FinBERT) | Sharpe (Without) | Delta Return | Delta Sharpe |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in rows:
        lines.append(
            f"| **{r['ticker']}** | {r['days']} | {r['ret_with']:+.2f}% | {r['sh_with']:.2f} | {r['ret_without']:+.2f}% | {r['sh_without']:.2f} | **{r['diff_ret']:+.2f}%** | {r['diff_sh']:+.2f} |"
        )

    lines.append(
        f"| **Average** | — | {avg_ret_w:+.2f}% | {avg_sh_w:.2f} | {avg_ret_wo:+.2f}% | {avg_sh_wo:.2f} | **{avg_diff_ret:+.2f}%** | {avg_diff_sh:+.2f} |"
    )

    table_md = "\n".join(lines)
    print(table_md)
    return table_md


if __name__ == "__main__":
    main()
