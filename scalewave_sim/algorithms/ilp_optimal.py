import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


class ILPOptimalSolver:
    """
    Exact Integer Linear Programming (ILP) solver using SciPy HiGHS backend.
    Provides the theoretical optimal ceiling for the multi-dimensional knapsack problem (Eq. 1).
    """
    def __init__(self):
        pass

    def solve(self, env):
        start_time = time.perf_counter()
        bound = max(1, env.get_demand_bound())
        capacities = env.capacities
        current_used = np.sum(env.replicas[:, None] * env.resource_weights, axis=0)
        avail = np.maximum(0.0, capacities - current_used)
        p_ne = env.throughputs / (np.maximum(1e-5, env.latencies) * np.maximum(1, env.replicas))
        num_services = env.num_services

        # Objective: minimize -sum(p_ne[i] * x[i]) to maximize sum(p_ne[i] * x[i])
        c = -p_ne.astype(np.float64)

        # Constraints: A @ x <= avail  => 0 <= A @ x <= avail
        A = env.resource_weights.T.astype(np.float64)  # shape (num_resources, num_services)
        b_u = avail.astype(np.float64)
        b_l = np.zeros_like(b_u)
        constraints = LinearConstraint(A, b_l, b_u)

        # Bounds: 0 <= x_i <= bound
        bounds = Bounds(0, bound)

        # All variables are integers (integrality=1)
        integrality = np.ones(num_services, dtype=np.int32)

        try:
            res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
            if res.success and res.x is not None:
                sol = np.round(res.x).astype(np.int32)
            else:
                sol = np.zeros(num_services, dtype=np.int32)
        except Exception:
            sol = np.zeros(num_services, dtype=np.int32)

        solve_duration = (time.perf_counter() - start_time) * 1000.0  # ms
        return sol, solve_duration
