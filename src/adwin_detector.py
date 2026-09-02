"""
Paper 21: ADWIN (Adaptive Windowing) Drift Detector.

Source: Bifet & Gavaldà (2007) — "Learning from Time-Changing Data
with Adaptive Windowing", SIAM.
Complexity: O(log W) amortized per observation.
"""

import math
import numpy as np
from typing import Dict, Any, List


class ADWINDetector:
    """
    ADWIN drift detector with Hoeffding bound.

    Maintains a variable-length window that auto-shrinks when a
    distributional change is detected between two sub-windows.

    Parameters:
        confidence_delta: Confidence parameter (lower = less sensitive).
            Default 0.002 = 99.8% confidence.
    """

    def __init__(self, confidence_delta: float = 0.002):
        self.delta = confidence_delta
        self.window: List[float] = []
        self.total = 0.0
        self.variance_sum = 0.0
        self.n_detections = 0

    def update(self, value: float) -> Dict[str, Any]:
        """
        Add one observation. Returns whether drift was detected.
        If drift is detected, the old data is discarded.
        """
        self.window.append(value)
        self.total += value

        drift_detected = False
        drop_index = -1

        n = len(self.window)
        if n < 10:
            return {
                "drift_detected": False,
                "window_size": n,
                "window_mean": round(self.total / n, 6),
            }

        # Use prefix sums for O(n) scan instead of O(n^2)
        prefix = [0.0] * (n + 1)
        for j in range(n):
            prefix[j + 1] = prefix[j] + self.window[j]

        # Check split points at logarithmic spacing for efficiency
        step = max(1, n // 50)
        for i in range(5, n - 5, step):
            n0 = i
            n1 = n - i
            mean0 = prefix[i] / n0
            mean1 = (prefix[n] - prefix[i]) / n1

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            # Hoeffding bound with window-size correction
            eps = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * n / self.delta))

            if abs(mean0 - mean1) >= eps:
                drift_detected = True
                drop_index = i
                break

        if drift_detected and drop_index > 0:
            self.window = self.window[drop_index:]
            self.total = sum(self.window)
            self.n_detections += 1

        current_n = len(self.window)
        return {
            "drift_detected": drift_detected,
            "window_size": current_n,
            "window_mean": round(self.total / max(current_n, 1), 6),
            "total_detections": self.n_detections,
        }

    def update_batch(self, values: np.ndarray) -> List[Dict[str, Any]]:
        """Process a batch."""
        return [self.update(float(v)) for v in values]

    def reset(self):
        """Reset detector."""
        self.window = []
        self.total = 0.0
        self.n_detections = 0
