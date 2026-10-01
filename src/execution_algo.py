"""
Institutional Execution Algorithms & Transient Market Impact Decay (Sprint 2, Module 2.10 / Idea 18)

Implements the Bouchaud-Gatheral Propagator Model of Transient Market Impact:
G(tau) = Gamma0 * (tau + tau0)^(-alpha)

Calculates:
1. Power-law transient price impact decay over time lag tau
2. Optimal child-order cooldown intervals Delta tau* to prevent self-impact clustering
3. Almgren-Chriss / Propagator child order execution trajectory
4. Total execution slippage and cost minimization
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("Sentilyze.ExecutionAlgo")
logging.basicConfig(level=logging.INFO)


def transient_impact_kernel(
    tau: np.ndarray,
    gamma0: float = 0.01,
    tau0: float = 1.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Computes power-law market impact propagator kernel:
    G(tau) = Gamma0 * (tau + tau0)^(-alpha)
    """
    tau_arr = np.maximum(np.asarray(tau, dtype=float), 0.0)
    t0 = max(float(tau0), 1e-4)
    a = float(alpha)
    g0 = float(gamma0)

    kernel = g0 * ((tau_arr + t0) / t0) ** (-a)
    return kernel


def compute_optimal_cooldown(
    decay_target_pct: float = 0.20,
    tau0_seconds: float = 2.0,
    alpha: float = 0.5,
) -> float:
    """
    Computes the minimum cooldown interval Delta tau* required for impact to decay to eta:
    G(Delta tau*) / G(0) <= eta  ==>  Delta tau* = tau0 * (eta^(-1/alpha) - 1)
    """
    eta = max(float(decay_target_pct), 0.01)
    t0 = max(float(tau0_seconds), 0.1)
    a = max(float(alpha), 0.1)

    cooldown = t0 * (eta ** (-1.0 / a) - 1.0)
    return round(float(max(cooldown, 1.0)), 1)


def simulate_child_order_schedule(
    total_shares: float,
    slice_count: int = 5,
    interval_seconds: float = 15.0,
    adv: float = 1000000.0,
    spot_price: float = 100.0,
    volatility: float = 0.02,
) -> Dict[str, Any]:
    """
    Simulates cumulative transient and permanent price impact across child orders.
    Permanent impact: I_perm = 0.5 * sigma * sqrt(Q / ADV)
    Transient impact: I_trans(t) = sum_j q * G(t - t_j)
    """
    n_slices = max(int(slice_count), 1)
    q_slice = float(total_shares) / n_slices
    dt = float(interval_seconds)

    # Base instantaneous impact parameter Gamma0
    gamma0 = float(volatility * spot_price * np.sqrt(q_slice / max(adv, 1.0)))

    impact_history = []
    total_impact = 0.0
    slippage_cost = 0.0

    times = np.arange(n_slices) * dt

    for k in range(n_slices):
        t_curr = times[k]
        # Accumulated transient impact from prior orders up to now
        lags = t_curr - times[: k + 1]
        kernel_vals = transient_impact_kernel(lags, gamma0=gamma0, tau0=2.0, alpha=0.5)
        # Total impact on current child order
        curr_impact = float(np.sum(kernel_vals))
        impact_history.append(round(curr_impact, 4))
        total_impact += curr_impact
        slippage_cost += curr_impact * q_slice

    avg_impact = total_impact / n_slices
    avg_slippage_bps = (avg_impact / max(spot_price, 1e-4)) * 10000.0

    # Optimal recommended cooldown
    opt_cooldown = compute_optimal_cooldown(
        decay_target_pct=0.25, tau0_seconds=2.0, alpha=0.5
    )

    is_overclustering = dt < opt_cooldown

    return {
        "total_shares": total_shares,
        "slice_count": n_slices,
        "shares_per_slice": round(q_slice, 1),
        "execution_interval_seconds": dt,
        "recommended_cooldown_seconds": opt_cooldown,
        "is_impact_overclustering": is_overclustering,
        "average_impact_dollars": round(avg_impact, 4),
        "average_slippage_bps": round(avg_slippage_bps, 2),
        "total_dollar_slippage": round(slippage_cost, 2),
        "impact_trajectory": impact_history,
    }


def optimize_order_execution_trajectory(
    total_shares: float,
    urgency: str = "MEDIUM",
    adv: float = 1000000.0,
    spot_price: float = 100.0,
) -> Dict[str, Any]:
    """
    Calculates institutional optimal child order slicing policy.
    Urgency presets:
    - HIGH: 3 slices, 10s cooldown
    - MEDIUM: 6 slices, 25s cooldown
    - LOW: 12 slices, 60s cooldown
    """
    presets = {
        "HIGH": {"slices": 3, "interval": 10.0},
        "MEDIUM": {"slices": 6, "interval": 25.0},
        "LOW": {"slices": 12, "interval": 60.0},
    }
    cfg = presets.get(urgency.upper(), presets["MEDIUM"])

    sim = simulate_child_order_schedule(
        total_shares=total_shares,
        slice_count=cfg["slices"],
        interval_seconds=cfg["interval"],
        adv=adv,
        spot_price=spot_price,
    )

    return {
        "status": "SUCCESS",
        "urgency_level": urgency.upper(),
        "optimal_policy": {
            "slice_count": cfg["slices"],
            "interval_seconds": cfg["interval"],
            "shares_per_slice": sim["shares_per_slice"],
            "expected_slippage_bps": sim["average_slippage_bps"],
            "total_execution_time_minutes": round(
                (cfg["slices"] * cfg["interval"]) / 60.0, 1
            ),
        },
        "simulation_details": sim,
    }


# ==============================================================================
# Sprint 3, Module 3.8 / Idea 10: Decision Transformer Order Slicer
# ==============================================================================
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DecisionTransformerOrderSlicer(nn.Module):
        """
        Decision Transformer for Optimal Execution Scheduling (Idea 10).
        Conditions sequence of child order actions on target return-to-go (RTG)
        and execution state (time fraction, remaining share inventory, volatility, spread).
        """

        def __init__(
            self,
            state_dim: int = 4,
            d_model: int = 32,
            n_heads: int = 2,
            max_len: int = 30,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.d_model = d_model
            self.max_len = max_len

            # Linear token embeddings
            self.embed_rtg = nn.Linear(1, d_model)
            self.embed_state = nn.Linear(state_dim, d_model)
            self.embed_action = nn.Linear(1, d_model)

            # Positional embeddings
            self.pos_emb = nn.Embedding(max_len * 3, d_model)

            # Causal Transformer Layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.0,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

            # Action prediction head (outputs fraction in [0.05, 0.60] per slice)
            self.action_head = nn.Sequential(
                nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
            )

        def forward(
            self,
            rtgs: torch.Tensor,  # (batch, seq_len, 1)
            states: torch.Tensor,  # (batch, seq_len, state_dim)
            actions: torch.Tensor,  # (batch, seq_len, 1)
        ) -> torch.Tensor:
            batch_size, seq_len, _ = states.shape

            # Embed each modality
            e_rtg = self.embed_rtg(rtgs)
            e_state = self.embed_state(states)
            e_act = self.embed_action(actions)

            # Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
            # We construct a sequence of length 3 * seq_len
            tokens = torch.zeros(
                batch_size, seq_len * 3, self.d_model, device=states.device
            )
            tokens[:, 0::3, :] = e_rtg
            tokens[:, 1::3, :] = e_state
            tokens[:, 2::3, :] = e_act

            # Add positional embeddings
            positions = (
                torch.arange(seq_len * 3, device=states.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            tokens = tokens + self.pos_emb(positions)

            # Causal mask so each token only attends to past tokens
            total_len = seq_len * 3
            causal_mask = nn.Transformer.generate_square_subsequent_mask(total_len).to(
                states.device
            )

            out = self.transformer(tokens, mask=causal_mask, is_causal=True)

            # Action is predicted from state token positions: 1, 4, 7, ... (1::3)
            state_out = out[:, 1::3, :]
            pred_action = self.action_head(state_out)
            return pred_action


def generate_decision_transformer_schedule(
    total_shares: float,
    horizon_steps: int = 5,
    target_return_to_go: float = 0.05,
    volatility: float = 0.02,
    spread_bps: float = 5.0,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Autoregressively rolls out a Decision Transformer to generate child order slice sizes
    conditioned on a positive target return-to-go (e.g. execution benchmark cost savings).

    Guarantees terminal completeness: sum(slices) == total_shares.
    """
    total_q = float(max(total_shares, 1.0))
    steps = max(int(horizon_steps), 2)

    if not TORCH_AVAILABLE:
        # Graceful fallback: TWAP / VWAP U-shape approximation
        u_weights = np.array([0.25, 0.15, 0.20, 0.15, 0.25][:steps])
        u_weights = u_weights / np.sum(u_weights)
        slices = [round(float(w * total_q), 1) for w in u_weights]
        return {
            "status": "FALLBACK_TWAP",
            "total_shares": total_q,
            "horizon_steps": steps,
            "target_return_to_go": target_return_to_go,
            "slices": slices,
            "cumulative_executed": list(np.cumsum(slices)),
            "remaining_shares": [round(total_q - c, 1) for c in np.cumsum(slices)],
            "terminal_fill_pct": 1.0,
        }

    if model is None:
        model = DecisionTransformerOrderSlicer(
            state_dim=4, d_model=32, n_heads=2, max_len=steps + 5
        )
    model.eval()

    q_rem = total_q
    rtg = float(target_return_to_go)
    slices: List[float] = []
    cumulative: List[float] = []
    rtg_trajectory: List[float] = [round(rtg, 4)]

    # Maintain sequence history for autoregressive conditioning
    hist_rtg: List[float] = []
    hist_state: List[List[float]] = []
    hist_action: List[float] = []

    for t in range(steps):
        time_frac = float(t) / float(steps)
        rem_frac = float(q_rem) / float(total_q)
        state_t = [time_frac, rem_frac, float(volatility), float(spread_bps) / 100.0]

        hist_rtg.append(rtg)
        hist_state.append(state_t)
        # Dummy action for the current step to build tensor
        hist_action.append(0.0)

        # Build tensors
        t_rtg = torch.tensor(hist_rtg, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        t_state = torch.tensor(hist_state, dtype=torch.float32).unsqueeze(0)
        t_act = (
            torch.tensor(hist_action, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        )

        with torch.no_grad():
            pred_actions = model(t_rtg, t_state, t_act)
            raw_action = float(pred_actions[0, -1, 0].item())

        # Scale raw action: base target slice fraction between 10% and 40% of remaining
        slice_fraction = 0.10 + 0.40 * raw_action

        if t == steps - 1:
            # Terminal step: execute all remaining inventory
            q_slice = round(q_rem, 1)
        else:
            q_slice = round(max(1.0, min(q_rem - 1.0, total_q * slice_fraction)), 1)

        q_rem -= q_slice
        executed = total_q - q_rem

        slices.append(q_slice)
        cumulative.append(round(executed, 1))

        # Reward: positive return-to-go decumulation
        reward_t = (target_return_to_go / float(steps)) * (
            1.0 + 0.1 * (raw_action - 0.5)
        )
        rtg = max(0.0, rtg - reward_t)
        rtg_trajectory.append(round(rtg, 4))

        # Update last action in history
        hist_action[-1] = q_slice / float(total_q)

    # Sanity check sum
    diff = round(total_q - sum(slices), 2)
    if diff != 0.0:
        slices[-1] = round(slices[-1] + diff, 1)
        cumulative[-1] = round(total_q, 1)

    return {
        "status": "SUCCESS",
        "total_shares": total_q,
        "horizon_steps": steps,
        "target_return_to_go": target_return_to_go,
        "slices": slices,
        "cumulative_executed": cumulative,
        "remaining_shares": [round(total_q - c, 1) for c in cumulative],
        "rtg_trajectory": rtg_trajectory,
        "terminal_fill_pct": round(cumulative[-1] / total_q, 4),
    }
