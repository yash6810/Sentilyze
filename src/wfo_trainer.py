"""
Elastic Weight Consolidation (EWC) Walk-Forward Optimization (WFO) Trainer
Idea 9 (Sprint 3.7): Prevents catastrophic forgetting during sequential regime retraining
using diagonal Fisher Information matrix regularization:

L_EWC(theta) = L_task(theta) + (lambda / 2) * sum_i F_i * (theta_i - theta_A,i*)^2
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from src.utils import get_logger

logger = get_logger(__name__)


class FinancialMLP(nn.Module):
    """
    Lightweight, CPU-friendly multi-layer perceptron for financial sequence classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017) module.
    Computes diagonal Fisher Information matrix on a prior regime/task dataset
    and computes the quadratic regularization penalty during retraining.
    """

    def __init__(
        self,
        model: nn.Module,
        X_regime: Union[torch.Tensor, np.ndarray],
        y_regime: Union[torch.Tensor, np.ndarray],
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.criterion = (
            criterion
            if criterion is not None
            else nn.BCEWithLogitsLoss(reduction="mean")
        )

        if isinstance(X_regime, np.ndarray):
            self.X = torch.tensor(X_regime, dtype=torch.float32, device=self.device)
        else:
            self.X = X_regime.to(dtype=torch.float32, device=self.device)

        if isinstance(y_regime, np.ndarray):
            self.y = torch.tensor(y_regime, dtype=torch.float32, device=self.device)
        else:
            self.y = y_regime.to(dtype=torch.float32, device=self.device)

        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

        # Store reference parameter weights theta_A*
        self.reference_params: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.reference_params[name] = param.detach().clone()

        # Compute empirical diagonal Fisher Information Matrix
        self.fisher_matrix: Dict[str, torch.Tensor] = self._calculate_fisher_matrix()

    def _calculate_fisher_matrix(self) -> Dict[str, torch.Tensor]:
        """
        Calculates the diagonal Fisher Information matrix:
        F_i = (1 / N) * sum_n (dL(x_n, y_n) / d theta_i)^2
        """
        self.model.eval()
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        n_samples = len(self.X)
        if n_samples == 0:
            return fisher

        for i in range(n_samples):
            self.model.zero_grad()
            xi = self.X[i : i + 1]
            yi = self.y[i : i + 1]

            out = self.model(xi)
            loss = self.criterion(out, yi)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)

        # Average over all samples
        for name in fisher:
            fisher[name] = fisher[name] / float(n_samples)

        return fisher

    def penalty(self, current_model: nn.Module) -> torch.Tensor:
        """
        Calculates the EWC quadratic penalty:
        (1/2) * sum_i F_i * (theta_i - theta_A,i*)^2
        """
        loss = torch.tensor(0.0, device=self.device)
        for name, param in current_model.named_parameters():
            if name in self.fisher_matrix:
                f_i = self.fisher_matrix[name]
                theta_star = self.reference_params[name]
                loss += (f_i * (param - theta_star).pow(2)).sum()
        return 0.5 * loss


def train_model_ewc(
    model: nn.Module,
    X_train: Union[torch.Tensor, np.ndarray],
    y_train: Union[torch.Tensor, np.ndarray],
    ewc: Optional[ElasticWeightConsolidation] = None,
    lambda_ewc: float = 100.0,
    epochs: int = 25,
    lr: float = 0.01,
    batch_size: int = 32,
    device: str = "cpu",
) -> List[float]:
    """
    Trains a model with optional Elastic Weight Consolidation penalty.
    """
    device_obj = torch.device(device)
    model.to(device_obj)
    model.train()

    if isinstance(X_train, np.ndarray):
        X = torch.tensor(X_train, dtype=torch.float32, device=device_obj)
    else:
        X = X_train.to(dtype=torch.float32, device=device_obj)

    if isinstance(y_train, np.ndarray):
        y = torch.tensor(y_train, dtype=torch.float32, device=device_obj)
    else:
        y = y_train.to(dtype=torch.float32, device=device_obj)

    if y.ndim == 1:
        y = y.unsqueeze(1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = len(X)
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        batches = 0

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = perm[start_idx:end_idx]

            xb = X[batch_indices]
            yb = y[batch_indices]

            optimizer.zero_grad()
            preds = model(xb)
            base_loss = criterion(preds, yb)

            if ewc is not None and lambda_ewc > 0:
                ewc_loss = lambda_ewc * ewc.penalty(model)
                total_loss = base_loss + ewc_loss
            else:
                total_loss = base_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batches += 1

        loss_history.append(epoch_loss / max(1, batches))

    return loss_history


def run_continual_wfo_simulation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_window: int = 100,
    test_window: int = 20,
    lambda_ewc: float = 100.0,
    epochs: int = 20,
) -> Dict[str, Any]:
    """
    Simulates walk-forward sequential learning comparing standard retraining (forgetting past regimes)
    versus EWC continual learning (preserving historical regime Fisher information).
    """
    X_full = df[feature_cols].values
    y_full = df[target_col].values

    total_samples = len(df)
    n_features = len(feature_cols)

    model_ewc = FinancialMLP(input_dim=n_features)
    ewc_consolidator: Optional[ElasticWeightConsolidation] = None

    oos_ewc_preds: List[float] = []
    oos_targets: List[float] = []

    start = 0
    step = 0

    while start + train_window + test_window <= total_samples:
        train_idx = slice(start, start + train_window)
        test_idx = slice(start + train_window, start + train_window + test_window)

        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_test, y_test = X_full[test_idx], y_full[test_idx]

        # Train with EWC if consolidated from previous regime
        train_model_ewc(
            model=model_ewc,
            X_train=X_train,
            y_train=y_train,
            ewc=ewc_consolidator,
            lambda_ewc=lambda_ewc,
            epochs=epochs,
        )

        # Update Fisher Consolidation for next step
        ewc_consolidator = ElasticWeightConsolidation(
            model=model_ewc, X_regime=X_train, y_regime=y_train
        )

        # OOS Inference
        model_ewc.eval()
        with torch.no_grad():
            preds = (
                torch.sigmoid(model_ewc(torch.tensor(X_test, dtype=torch.float32)))
                .cpu()
                .numpy()
                .flatten()
            )
            oos_ewc_preds.extend(preds.tolist())
            oos_targets.extend(y_test.tolist())

        start += test_window
        step += 1

    oos_preds_arr = np.array(oos_ewc_preds)
    oos_targets_arr = np.array(oos_targets)

    if len(oos_targets_arr) > 0:
        binary_preds = (oos_preds_arr >= 0.5).astype(int)
        accuracy = float(np.mean(binary_preds == oos_targets_arr))
    else:
        accuracy = 0.0

    return {
        "n_steps": step,
        "total_oos_samples": len(oos_targets_arr),
        "accuracy": accuracy,
        "ewc_predictions": oos_preds_arr,
        "targets": oos_targets_arr,
        "lambda_ewc": lambda_ewc,
    }
