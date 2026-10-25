"""
FinMamba-JEPA: Multimodal Neural Liquidity Manifold Model for Institutional Trading.

Fuses:
1. Continuous-Time Selective State-Space Recurrence (S6/Mamba) with adaptive delta_t
2. FiLM (Feature-wise Linear Modulation) News/Catalyst Gating
3. Non-Generative Energy-Based JEPA Latent Manifold Predictor
4. Physics-Informed Differentiable Sharpe & Turnover Loss Function
5. Evidential Deep Learning Head (P10/P50/P90, Epistemic Uncertainty, Half-Kelly Fraction)
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousStateSpaceBlock(nn.Module):
    """Continuous-Time Selective State-Space Block (S6/Mamba inspired).

    Computes continuous linear recurrence with adaptive time discretization delta_t.
    """

    def __init__(self, d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Continuous transition matrix A (initialized negative for mathematical stability)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
            )
        )
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, dt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        x: [batch_size, seq_len, d_model]
        dt: optional [batch_size, seq_len, 1] elapsed continuous time
        """
        batch_size, seq_len, _ = x.shape
        A = -torch.exp(self.A_log)  # [d_model, d_state]

        if dt is None:
            delta = F.softplus(self.delta_proj(x))
        else:
            delta = F.softplus(self.delta_proj(x)) * dt

        B = self.B_proj(x)
        y_out = []
        h = torch.zeros(
            batch_size,
            self.d_model,
            self.d_state,
            device=x.device,
            dtype=x.dtype,
        )

        for t in range(seq_len):
            d_t = delta[:, t, :].unsqueeze(-1)  # [batch, d_model, 1]
            A_bar = torch.exp(d_t * A.unsqueeze(0))
            b_t = B[:, t, :].unsqueeze(1)  # [batch, 1, d_state]
            B_bar = d_t * b_t  # [batch, d_model, d_state]
            u_t = x[:, t, :].unsqueeze(-1)  # [batch, d_model, 1]

            h = A_bar * h + B_bar * u_t
            y_t = self.C_proj(h.mean(dim=1)) + self.D * x[:, t, :]
            y_out.append(y_t.unsqueeze(1))

        y = torch.cat(y_out, dim=1)
        return self.norm(x + y)


class FiLMCatalystModulator(nn.Module):
    """Feature-wise Linear Modulation (FiLM) Catalyst Gate.

    Transforms text catalyst embeddings into affine scale (gamma) and shift (beta) vectors.
    """

    def __init__(self, d_catalyst: int = 16, d_model: int = 64):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(d_catalyst, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(d_catalyst, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h_state: torch.Tensor, catalyst: torch.Tensor) -> torch.Tensor:
        """h_state: [batch_size, d_model], catalyst: [batch_size, d_catalyst]."""
        gamma = 1.0 + self.gamma_net(catalyst)
        beta = self.beta_net(catalyst)
        return gamma * h_state + beta


class LatentManifoldJEPA(nn.Module):
    """Energy-Based Joint Embedding Predictive Architecture.

    Maps multi-asset state representations to latent manifold z and predicts next manifold state.
    """

    def __init__(self, d_model: int = 64, d_latent: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_latent),
            nn.LayerNorm(d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, d_latent),
        )

    def forward(
        self, h_curr: torch.Tensor, h_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_curr = self.encoder(h_curr)
        z_next_pred = self.predictor(z_curr)
        with torch.no_grad():
            z_next_true = self.encoder(h_next)
        energy_distance = F.mse_loss(z_next_pred, z_next_true, reduction="none").mean(
            dim=-1
        )
        return z_curr, z_next_pred, energy_distance


class EvidentialOutputHead(nn.Module):
    """Evidential Bayesian Head (Monte Carlo Dropout & Conformal Uncertainty).

    Outputs tri-quantiles (P10, P50, P90), epistemic uncertainty, and half-Kelly fraction.
    """

    def __init__(self, d_model: int = 64, dropout_rate: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = nn.Linear(d_model, 2)  # [return, scale]

    def forward(self, h: torch.Tensor, n_mc: int = 5) -> Dict[str, torch.Tensor]:
        x1 = F.gelu(self.fc1(h))
        base_out = self.head(x1)
        gamma = torch.tanh(base_out[:, 0]) * 0.15
        base_scale = F.softplus(base_out[:, 1]) + 1e-4

        # Monte Carlo Dropout for Epistemic Uncertainty (Gal & Ghahramani, 2016)
        # Compute variance on pre-saturation logits to avoid saturation collapse
        self.dropout.train()
        mc_samples = []
        for _ in range(n_mc):
            h_dropped = self.dropout(x1)
            raw_logit = self.head(h_dropped)[:, 0]
            mc_samples.append(raw_logit)

        mc_stack = torch.stack(mc_samples, dim=0)
        sigma_epistemic = mc_stack.var(dim=0) + 1e-6
        sigma_total = sigma_epistemic + base_scale

        std = torch.sqrt(sigma_total)
        p10 = gamma - 1.282 * std
        p50 = gamma
        p90 = gamma + 1.282 * std

        kelly_f = 0.5 * (gamma / (sigma_total + 1e-6))
        kelly_f = torch.clamp(kelly_f, -1.0, 1.0)

        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "sigma_epistemic": sigma_epistemic,
            "kelly_fraction": kelly_f,
        }


class FinMambaJEPA(nn.Module):
    """Master FinMamba-JEPA Multimodal Architecture."""

    def __init__(
        self,
        n_features: int = 15,
        n_catalyst: int = 4,
        d_model: int = 64,
        d_state: int = 16,
        d_latent: int = 128,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.ssm_block = ContinuousStateSpaceBlock(d_model=d_model, d_state=d_state)
        self.catalyst_modulator = FiLMCatalystModulator(
            d_catalyst=n_catalyst, d_model=d_model
        )
        self.jepa_block = LatentManifoldJEPA(d_model=d_model, d_latent=d_latent)
        self.output_head = EvidentialOutputHead(
            d_model=d_model, dropout_rate=dropout_rate
        )

    def forward(
        self,
        x: torch.Tensor,
        catalyst: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
        x_next: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h_seq = self.input_proj(x)
        h_ssm = self.ssm_block(h_seq, dt)
        h_final = h_ssm[:, -1, :]

        # FiLM catalyst modulation
        h_modulated = self.catalyst_modulator(h_final, catalyst)

        # JEPA Latent Manifold Prediction
        z_curr = self.jepa_block.encoder(h_modulated)
        z_pred = self.jepa_block.predictor(z_curr)

        preds = self.output_head(h_modulated)
        preds["h_final"] = h_modulated
        preds["z_curr"] = z_curr
        preds["z_pred"] = z_pred

        if x_next is not None:
            h_next_seq = self.input_proj(x_next)
            h_next_ssm = self.ssm_block(h_next_seq, dt)
            h_next_final = h_next_ssm[:, -1, :]
            _, _, energy_dist = self.jepa_block(h_modulated, h_next_final)
            preds["energy_distance"] = energy_dist
        else:
            # Latent stability energy distance
            preds["energy_distance"] = F.mse_loss(
                z_pred, z_curr.detach(), reduction="none"
            ).mean(dim=-1)

        return preds


class DifferentiableSharpeLoss(nn.Module):
    """Loss function directly optimizing annualized Sharpe Ratio while penalizing turnover friction and manifold energy."""

    def __init__(
        self,
        lambda_turnover: float = 0.05,
        lambda_energy: float = 0.01,
        transaction_fee: float = 0.001,
    ):
        super().__init__()
        self.lambda_turnover = lambda_turnover
        self.lambda_energy = lambda_energy
        self.fee = transaction_fee

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        energy_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pnl = weights * returns
        turnover = (
            torch.abs(weights[1:] - weights[:-1]).mean()
            if len(weights) > 1
            else torch.tensor(0.0, device=weights.device)
        )
        net_pnl = pnl - self.fee * torch.abs(weights)

        mean_r = torch.mean(net_pnl)
        std_r = torch.std(net_pnl) + 1e-6
        annualized_sharpe = (mean_r / std_r) * math.sqrt(252.0)
        loss = -annualized_sharpe + self.lambda_turnover * turnover
        if energy_dist is not None:
            loss = loss + self.lambda_energy * energy_dist.mean()
        return loss
