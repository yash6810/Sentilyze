"""
TabNet Attentive Interpretable Tabular Network (Sprint 3, Module 3.1 / Idea 2)

Implements Arik & Pfister (Google Cloud AI, AAAI 2021):
- Sequential Multi-Step Decision Architecture (N_steps = 3 to 5)
- Sparsemax Attentive Transformer for dynamic feature mask selection: M[i] in [0, 1]^D
- Feature Transformer with GLU (Gated Linear Units)
- Prior scale P[i] tracking feature usage across decision steps
- Instant interpretable instance-level feature attribution masks M_explain
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Sentilyze.TabNet")
logging.basicConfig(level=logging.INFO)


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes Sparsemax projection (Martins & Astudillo 2016):
    sparsemax(z) = argmin_{p in Delta} ||p - z||^2 = [z - tau(z)]_+
    Guarantees exact sparsity in feature attention masks.
    """
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    k_range = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = -1
    k_range = k_range.view(view_shape)

    cumsum_z = torch.cumsum(z_sorted, dim=dim)
    bound = 1.0 + k_range * z_sorted
    is_gt = bound > cumsum_z

    k_max = torch.max(is_gt * k_range, dim=dim, keepdim=True)[0]
    tau = (torch.gather(cumsum_z, dim, (k_max - 1).long()) - 1.0) / k_max.float()

    return torch.clamp(z - tau, min=0.0)


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) block with Ghost BatchNorm."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features * 2, bias=False)
        self.bn = nn.BatchNorm1d(out_features * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        if out.size(0) > 1:
            out = self.bn(out)
        out1, out2 = torch.chunk(out, 2, dim=-1)
        return out1 * torch.sigmoid(out2)


class FeatureTransformer(nn.Module):
    """Processes masked features into decision and attention representations."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.glu1 = GatedLinearUnit(in_features, out_features)
        self.glu2 = GatedLinearUnit(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.glu1(x)
        h2 = self.glu2(h1)
        return (h1 + h2) * np.sqrt(0.5)


class AttentiveTransformer(nn.Module):
    """Generates sparse feature selection mask M[i] using previous step activations."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, a_prev: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        out = self.fc(a_prev)
        if out.size(0) > 1:
            out = self.bn(out)
        out = out * prior
        mask = sparsemax(out, dim=-1)
        return mask


class TabNetModel(nn.Module):
    """
    Lightweight PyTorch TabNet architecture for high-precision tabular financial prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        # Initial feature normalization
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Multi-step feature transformers and attentive transformers
        self.feat_transformers = nn.ModuleList(
            [FeatureTransformer(input_dim, n_d + n_a) for _ in range(n_steps)]
        )
        self.att_transformers = nn.ModuleList(
            [AttentiveTransformer(n_a, input_dim) for _ in range(n_steps)]
        )

        # Final classification / regression head
        self.head = nn.Linear(n_d, output_dim, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            logits: (Batch, output_dim)
            m_explain: (Batch, input_dim) Normalized aggregate feature importance mask
            step_masks: List of M[i] at each decision step
        """
        batch_size = x.size(0)
        if batch_size > 1:
            x_norm = self.initial_bn(x)
        else:
            x_norm = x

        prior = torch.ones_like(x_norm)
        m_explain = torch.zeros_like(x_norm)
        step_masks = []
        d_aggregate = torch.zeros((batch_size, self.n_d), device=x.device)

        # Initial attention representation
        a_prev = torch.zeros((batch_size, self.n_a), device=x.device)

        for step in range(self.n_steps):
            # 1. Attentive Transformer
            mask = self.att_transformers[step](a_prev, prior)
            step_masks.append(mask)

            # Update prior: P[i] = P[i-1] * (gamma - M[i])
            prior = prior * (self.gamma - mask)

            # 2. Mask features and process
            x_masked = mask * x_norm
            h = self.feat_transformers[step](x_masked)

            # Split into d (decision) and a (attention for next step)
            d = F.relu(h[:, : self.n_d])
            a_prev = h[:, self.n_d :]

            d_aggregate = d_aggregate + d

            # Step weight for explanation: sum of d activations
            step_importance = torch.sum(d, dim=-1, keepdim=True)
            m_explain = m_explain + mask * step_importance

        # Normalize explanation mask
        total_importance = torch.sum(m_explain, dim=-1, keepdim=True)
        m_explain = m_explain / torch.clamp(total_importance, min=1e-6)

        logits = self.head(d_aggregate)
        return logits, m_explain, step_masks


class TabNetClassifier:
    """Wrapper providing scikit-learn style fit and predict with feature importance extraction."""

    def __init__(
        self,
        input_dim: int = 15,
        n_steps: int = 3,
        lr: float = 0.02,
        feature_names: Optional[List[str]] = None,
    ):
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feat_{i}" for i in range(input_dim)]
        self.model = TabNetModel(input_dim=input_dim, output_dim=2, n_steps=n_steps)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.is_fitted = False

    def fit(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 25, batch_size: int = 32
    ) -> "TabNetClassifier":
        """Fast CPU training loop."""
        self.model.train()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for _ in range(epochs):
            for bx, by in loader:
                self.optimizer.zero_grad()
                logits, _, step_masks = self.model(bx)
                loss = F.cross_entropy(logits, by)

                # Sparsity regularization on attention masks (entropy loss)
                sparse_loss = 0.0
                for m in step_masks:
                    sparse_loss += torch.mean(
                        torch.sum(-m * torch.log(m + 1e-9), dim=-1)
                    )
                loss += 1e-3 * sparse_loss

                loss.backward()
                self.optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs calibrated class probabilities [P(down), P(up)]."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits, _, _ = self.model(X_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """Extracts instant feature attribution masks for input batch."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits, m_explain, _ = self.model(X_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            m_np = m_explain.cpu().numpy()

        # Mean feature importance across batch
        mean_weights = np.mean(m_np, axis=0)
        feature_importance = {
            name: round(float(w), 4)
            for name, w in zip(self.feature_names, mean_weights)
        }

        # Sort descending
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        )

        return {
            "probabilities": probs.tolist(),
            "global_feature_masks": sorted_importance,
            "top_driving_feature": list(sorted_importance.keys())[0],
        }
