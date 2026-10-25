import time
import numpy as np


class DefaultKPASolver:
    """
    Standard Knative Pod Autoscaler (KPA) single-implementation baseline.
    Does not leverage equivalent implementations; routes 100% of traffic to a single
    revision and scales instances based solely on target concurrency up to physical limits.
    """
    def __init__(self, mode='local_only'):
        self.mode = mode

    def solve(self, env):
        start_time = time.perf_counter()
        # Single-implementation KPA has virtually zero optimization overhead (< 0.05ms)
        action = np.zeros(env.num_services, dtype=np.float32)
        solve_duration = (time.perf_counter() - start_time) * 1000.0
        return action, max(0.01, solve_duration)
