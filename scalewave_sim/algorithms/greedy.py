import time
import numpy as np


class GreedyKnapsackSolver:
    """
    Greedy heuristic solver.
    Ranks equivalent services by performance-to-resource-footprint density and allocates greedily.
    """
    def __init__(self):
        pass

    def solve(self, env):
        start_time = time.perf_counter()
        bound = env.get_demand_bound()
        capacities = env.capacities
        current_used = np.sum(env.replicas[:, None] * env.resource_weights, axis=0)
        avail = np.maximum(0.0, capacities - current_used)
        p_ne = env.throughputs / (np.maximum(1e-5, env.latencies) * np.maximum(1, env.replicas))
        num_services = env.num_services

        # Compute normalized multi-resource cost per service
        normalized_costs = np.sum(env.resource_weights / capacities, axis=1)
        densities = p_ne / np.maximum(1e-5, normalized_costs)

        # Sort indices by highest density
        ranked_indices = np.argsort(-densities)

        sol = np.zeros(num_services, dtype=np.int32)
        remaining_avail = avail.copy()

        for idx in ranked_indices:
            r_req = env.resource_weights[idx]
            # Max possible replicas without violating any resource
            with np.errstate(divide='ignore'):
                limits = np.where(r_req > 0, remaining_avail / np.maximum(1e-5, r_req), 999.0)
            max_alloc = int(np.floor(np.min(limits)))
            alloc = min(max_alloc, bound)
            if alloc > 0:
                sol[idx] = alloc
                remaining_avail -= r_req * alloc

        solve_duration = (time.perf_counter() - start_time) * 1000.0  # ms
        return sol, solve_duration
