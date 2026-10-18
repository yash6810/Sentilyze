"""
Sentilyze - Tier-1 Asymmetric Alpha Radar Generator.
Ranks all 1,974 assets by combining:
1. TimesFM Corridors: Reward-to-Risk ratio (TP1 - Spot) / (Spot - SL).
2. GPU XGBoost Out-of-Sample Performance: Accuracy, ROC-AUC, Sharpe ratio.
3. Liquidity & Capital Safety Gates.
"""

import json
import os
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_radar(top_n: int = 50):
    tfm_path = "results/timesfm_forecasts.json"
    sum_path = "results/universe_summary.json"

    if not os.path.exists(tfm_path) or not os.path.exists(sum_path):
        print("❌ Missing required result files.")
        return

    with open(tfm_path, "r", encoding="utf-8") as f:
        tfm = json.load(f)

    with open(sum_path, "r", encoding="utf-8") as f:
        summary = {row["ticker"]: row for row in json.load(f)}

    rankings = []
    for sym, data in tfm.items():
        spot = float(data.get("spot_price", 0.0))
        sl = float(data.get("recommended_sl", 0.0))
        tp1 = float(data.get("recommended_tp1", 0.0))
        tp2 = float(data.get("recommended_tp2", 0.0))

        risk = spot - sl
        reward1 = tp1 - spot
        reward2 = tp2 - spot

        if risk > 0 and reward1 > 0 and spot >= 2.0:
            rr1 = reward1 / risk
            rr2 = reward2 / risk
            metrics = summary.get(sym, {})
            acc = float(metrics.get("accuracy", 0.5))
            sh = float(metrics.get("sharpe_ratio", 0.0))
            ret = float(metrics.get("total_return", 0.0))
            mdd = float(metrics.get("max_drawdown", 0.0))

            # Quality gate: minimum 70% accuracy, minimum Sharpe 2.0
            if acc >= 0.70 and sh >= 2.0:
                composite_score = rr1 * (sh**0.5) * (acc**2)
                rankings.append(
                    {
                        "ticker": sym,
                        "spot_price": round(spot, 2),
                        "stop_loss_p10": round(sl, 2),
                        "target1_p50": round(tp1, 2),
                        "runner_p90": round(tp2, 2),
                        "reward_to_risk": round(rr1, 2),
                        "runner_reward_to_risk": round(rr2, 2),
                        "validation_accuracy": round(acc, 4),
                        "strategy_sharpe": round(sh, 2),
                        "strategy_return_pct": round(ret, 2),
                        "max_drawdown_pct": round(mdd, 2),
                        "composite_alpha_score": round(composite_score, 2),
                    }
                )

    rankings.sort(key=lambda x: x["composite_alpha_score"], reverse=True)
    top_candidates = rankings[:top_n]

    os.makedirs("results", exist_ok=True)
    output_path = "results/tier1_asymmetric_radar.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_candidates, f, indent=2)

    print(f"✅ Generated {len(top_candidates)} Tier-1 Asymmetric Alpha candidates.")
    print(f"📁 Saved to: {output_path}")


if __name__ == "__main__":
    generate_radar(top_n=50)
