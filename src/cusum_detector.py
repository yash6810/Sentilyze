"""
Paper 16: CUSUM Sequential Change-Point Detection.

Source: E.S. Page (1954) — "Continuous Inspection Schemes."
Complexity: O(1) per observation (constant time, streaming).
"""

import numpy as np
from typing import Dict, Any, List


class CUSUMDetector:
    """
    Cumulative Sum detector for online mean-shift detection.

    Monitors a numeric stream and raises an alarm when the cumulative
    deviation from the target mean exceeds threshold h.

    Parameters:
        threshold_h: Detection sensitivity (higher = fewer false alarms).
        drift_k: Minimum shift magnitude to detect (slack parameter).
        target_mean: Expected mean of the stream (default 0 for returns).
    """

    def __init__(
        self,
        threshold_h: float = 4.0,
        drift_k: float = 0.5,
        target_mean: float = 0.0,
    ):
        self.threshold_h = threshold_h
        self.drift_k = drift_k
        self.target_mean = target_mean
        self.s_pos = 0.0  # Upper CUSUM
        self.s_neg = 0.0  # Lower CUSUM
        self.n_observations = 0
        self.alarm_history: List[Dict[str, Any]] = []

    def update(self, value: float) -> Dict[str, Any]:
        """Process one observation. Returns alarm status."""
        self.n_observations += 1
        deviation = value - self.target_mean

        self.s_pos = max(0.0, self.s_pos + deviation - self.drift_k)
        self.s_neg = max(0.0, self.s_neg - deviation - self.drift_k)

        alarm_up = self.s_pos > self.threshold_h
        alarm_down = self.s_neg > self.threshold_h
        alarm = alarm_up or alarm_down

        result = {
            "observation": self.n_observations,
            "value": value,
            "s_pos": round(self.s_pos, 6),
            "s_neg": round(self.s_neg, 6),
            "alarm": alarm,
            "direction": "UP" if alarm_up else ("DOWN" if alarm_down else "NONE"),
        }

        if alarm:
            self.alarm_history.append(result)
            self.s_pos = 0.0
            self.s_neg = 0.0

        return result

    def update_batch(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """Process a batch of observations."""
        return [self.update(float(v)) for v in values]

    def reset(self):
        """Reset detector state."""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n_observations = 0
        self.alarm_history = []
