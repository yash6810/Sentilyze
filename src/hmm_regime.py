"""
Paper 15: Gaussian Hidden Markov Model for Market Regime Detection.

Source: Baum et al. (1970) + Hamilton (1989) regime-switching.
Complexity: O(T * K^2) where T=observations, K=states.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional


class GaussianHMMRegimeDetector:
    """
    3-state Gaussian HMM regime classifier: Bull / Normal / Crisis.

    Uses hand-calibrated emission parameters derived from historical
    S&P 500 return distributions. No hmmlearn dependency required.

    The forward algorithm computes filtered state probabilities using
    only past data (no look-ahead bias).
    """

    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.state_names = ["Bull", "Normal", "Crisis"][:n_states]

        # Calibrated from S&P 500 daily returns (2000-2024)
        # State means (daily): Bull=+0.08%, Normal=+0.01%, Crisis=-0.10%
        self.means = np.array([0.0008, 0.0001, -0.0010])[:n_states]
        # State std devs (daily): Bull=0.6%, Normal=1.1%, Crisis=2.5%
        self.stds = np.array([0.006, 0.011, 0.025])[:n_states]

        # Transition matrix: rows=from, cols=to (sticky diagonal)
        self.trans_mat = np.array(
            [
                [0.95, 0.04, 0.01],
                [0.05, 0.90, 0.05],
                [0.02, 0.08, 0.90],
            ]
        )[:n_states, :n_states]
        # Normalize rows
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)

        # Initial state probabilities
        self.state_probs = np.array([0.4, 0.4, 0.2])[:n_states]
        self.state_probs /= self.state_probs.sum()

    def _emission_prob(self, observation: float) -> np.ndarray:
        """Gaussian emission probability for each state."""
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            z = (observation - self.means[k]) / self.stds[k]
            probs[k] = np.exp(-0.5 * z * z) / (self.stds[k] * np.sqrt(2 * np.pi))
        return probs

    def update(self, daily_return: float) -> Dict[str, Any]:
        """
        Forward algorithm step: update filtered state probabilities
        with one new observation.
        """
        # Predict: state_probs @ trans_mat
        predicted = self.state_probs @ self.trans_mat

        # Update: predicted * emission
        emission = self._emission_prob(daily_return)
        updated = predicted * emission

        total = updated.sum()
        if total > 0:
            self.state_probs = updated / total
        else:
            self.state_probs = predicted

        best_state = int(np.argmax(self.state_probs))

        return {
            "regime": self.state_names[best_state],
            "regime_id": best_state,
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.state_names, self.state_probs)
            },
            "is_crisis": bool(best_state == self.n_states - 1),
            "crisis_probability": round(float(self.state_probs[-1]), 4),
        }

    def classify_series(self, returns: np.ndarray) -> pd.DataFrame:
        """Classify an entire return series into regimes."""
        records = []
        for r in returns:
            records.append(self.update(float(r)))
        return pd.DataFrame(records)

    def reset(self):
        """Reset to initial state probabilities."""
        self.state_probs = np.array([0.4, 0.4, 0.2])[: self.n_states]
        self.state_probs /= self.state_probs.sum()
