"""
Paper 22: Page-Hinkley Sequential Test for Concept Drift Detection.

Source: Page (1954), Hinkley (1971).
Complexity: O(1) per observation (constant time).
"""

import numpy as np
from typing import Dict, Any, List


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting changes in the mean of a stream.

    Monitors cumulative deviation from running mean. Alarm when
    (PH_t - min(PH)) > threshold.

    Parameters:
        threshold_lambda: Detection threshold (higher = fewer false alarms).
        min_magnitude_delta: Minimum change magnitude to detect.
    """

    def __init__(
        self,
        threshold_lambda: float = 50.0,
        min_magnitude_delta: float = 0.005,
    ):
        self.threshold = threshold_lambda
        self.delta = min_magnitude_delta
        self.n = 0
        self.sum_values = 0.0
        self.ph_sum = 0.0
        self.ph_min = float("inf")
        self.alarm_history: List[Dict[str, Any]] = []

    def update(self, value: float) -> Dict[str, Any]:
        """Process one observation. Returns drift status."""
        self.n += 1
        self.sum_values += value
        running_mean = self.sum_values / self.n

        self.ph_sum += value - running_mean - self.delta
        self.ph_min = min(self.ph_min, self.ph_sum)

        ph_statistic = self.ph_sum - self.ph_min
        drift_detected = ph_statistic > self.threshold

        result = {
            "observation": self.n,
            "value": round(value, 6),
            "running_mean": round(running_mean, 6),
            "ph_statistic": round(ph_statistic, 4),
            "drift_detected": drift_detected,
        }

        if drift_detected:
            self.alarm_history.append(result)
            # Reset after detection
            self.n = 0
            self.sum_values = 0.0
            self.ph_sum = 0.0
            self.ph_min = float("inf")

        return result

    def update_batch(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """Process a batch of observations."""
        return [self.update(float(v)) for v in values]

    def reset(self):
        """Reset detector state."""
        self.n = 0
        self.sum_values = 0.0
        self.ph_sum = 0.0
        self.ph_min = float("inf")
        self.alarm_history = []
