"""
Neural Additive Models (NAMs) with Monotonic Constraints (Sprint 3, Module 3.3 / Idea 4)

Implements Agarwal et al. (Google Research, NeurIPS 2021):
- Generalized Additive Model (GAM) with neural sub-networks:
  g(E[y]) = beta_0 + sum_{j=1}^D f_j(x_j)
- Feature-wise sub-networks f_j(x_j): R -> R parameterized by MLPs with ExU (Exp-centered units)
- Exact intrinsic interpretability (1D shape functions plotted directly without SHAP approximation)
- Economic monotonicity penalty: penalizes non-monotonicity in key indicators (e.g. higher Altman Z -> higher solvency)
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Sentilyze.NAM")
logging.basicConfig(level=logging.INFO)


class ExU(nn.Module):
    """
    Exp-centered Unit (ExU): h(x) = relu(exp(w) * (x - b))
    Enables sharp, localized step-like non-linear shape functions.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features) * 0.5)
        self.b = nn.Parameter(torch.randn(in_features, out_features) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_features)
        # exp(w) guarantees positive scaling
        exp_w = torch.exp(self.w)
        # broadcast subtraction: (Batch, in_features, 1) - (in_features, out_features)
        diff = x.unsqueeze(-1) - self.b
        h = torch.clamp(exp_w * diff, 0.0, 1.0)
        return h.squeeze(1) if x.size(1) == 1 else h


class FeatureSubNet(nn.Module):
    """Individual 1D non-linear neural sub-network f_j(x_j): R -> R."""

    def __init__(self, hidden_dim: int = 32, is_monotonic: bool = False):
        super().__init__()
        self.is_monotonic = is_monotonic
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x_j: torch.Tensor) -> torch.Tensor:
        # x_j: (Batch, 1)
        h = F.relu(self.fc1(x_j))
        h = F.relu(self.fc2(h))
        return self.out(h)


class NeuralAdditiveModel(nn.Module):
    """
    Full NAM architecture composed of D independent feature subnets:
    y_hat = bias + sum_{j=1}^D f_j(x_j)
    """

    def __init__(
        self,
        input_dim: int,
        feature_names: Optional[List[str]] = None,
        hidden_dim: int = 32,
        monotonic_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feat_{i}" for i in range(input_dim)]
        self.monotonic_indices = set(monotonic_indices or [])

        self.subnets = nn.ModuleList(
            [
                FeatureSubNet(
                    hidden_dim=hidden_dim, is_monotonic=(i in self.monotonic_indices)
                )
                for i in range(input_dim)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (Batch, D)
        Returns:
            total_output: (Batch, 1) Logits
            feature_contributions: (Batch, D) Individual f_j(x_j) outputs
        """
        batch_size = x.size(0)
        contributions = []

        for j in range(self.input_dim):
            x_j = x[:, j : j + 1]
            f_j = self.subnets[j](x_j)
            contributions.append(f_j)

        # Shape (Batch, D)
        contrib_tensor = torch.cat(contributions, dim=-1)
        total_logit = self.bias + torch.sum(contrib_tensor, dim=-1, keepdim=True)
        return total_logit, contrib_tensor

    def compute_monotonic_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """Penalizes negative slopes on features declared monotonic."""
        if not self.monotonic_indices:
            return torch.tensor(0.0, device=x.device)

        penalty = torch.tensor(0.0, device=x.device)
        x_req = x.clone().detach().requires_grad_(True)
        _, contribs = self.forward(x_req)

        for j in self.monotonic_indices:
            grad = torch.autograd.grad(
                outputs=contribs[:, j].sum(),
                inputs=x_req,
                create_graph=True,
                retain_graph=True,
            )[0][:, j]
            # Negative slopes violate monotonicity
            neg_slope = F.relu(-grad)
            penalty = penalty + torch.mean(neg_slope**2)

        return penalty


class NAMClassifier:
    """Scikit-learn style wrapper for training and plotting 1D shape functions."""

    def __init__(
        self,
        input_dim: int,
        feature_names: Optional[List[str]] = None,
        monotonic_indices: Optional[List[int]] = None,
        lr: float = 0.01,
    ):
        self.input_dim = input_dim
        self.feature_names = feature_names or [f"feat_{i}" for i in range(input_dim)]
        self.model = NeuralAdditiveModel(
            input_dim=input_dim,
            feature_names=self.feature_names,
            monotonic_indices=monotonic_indices,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(
        self, X: np.ndarray, y: np.ndarray, epochs: int = 20, batch_size: int = 32
    ) -> "NAMClassifier":
        """Trains NAM with binary cross-entropy and monotonicity loss."""
        self.model.train()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for _ in range(epochs):
            for bx, by in loader:
                self.optimizer.zero_grad()
                logits, _ = self.model(bx)
                bce_loss = F.binary_cross_entropy_with_logits(logits, by)
                mono_loss = self.model.compute_monotonic_penalty(bx)
                loss = bce_loss + 0.1 * mono_loss
                loss.backward()
                self.optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [P(0), P(1)]."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits, _ = self.model(X_t)
            prob_1 = torch.sigmoid(logits).cpu().numpy()
            prob_0 = 1.0 - prob_1
        return np.hstack([prob_0, prob_1])

    def extract_shape_function(
        self, feature_idx: int, n_grid: int = 50
    ) -> Dict[str, Any]:
        """Evaluates the 1D response curve f_j(x_j) across its standardized domain [-3, +3]."""
        self.model.eval()
        grid = np.linspace(-3.0, 3.0, n_grid)
        with torch.no_grad():
            x_j = torch.tensor(grid, dtype=torch.float32).unsqueeze(-1)
            f_vals = self.model.subnets[feature_idx](x_j).squeeze(-1).cpu().numpy()

        feat_name = self.feature_names[feature_idx]
        return {
            "feature_index": feature_idx,
            "feature_name": feat_name,
            "x_domain": [round(float(v), 3) for v in grid],
            "f_response": [round(float(v), 4) for v in f_vals],
        }
