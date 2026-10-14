import time
import random
import numpy as np


class SimulatedAnnealingSolver:
    """
    Simulated Annealing metaheuristic solver for multi-dimensional knapsack.
    """
    def __init__(self, initial_temp=100.0, cooling_rate=0.92, max_iterations=200):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def solve(self, env):
        start_time = time.perf_counter()
        bound = env.get_demand_bound()
        capacities = env.capacities
        current_used = np.sum(env.replicas[:, None] * env.resource_weights, axis=0)
        avail = np.maximum(0.0, capacities - current_used)
        p_ne = env.throughputs / (np.maximum(1e-5, env.latencies) * np.maximum(1, env.replicas))
        num_services = env.num_services

        max_counts = []
        for i in range(num_services):
            r_req = env.resource_weights[i]
            with np.errstate(divide='ignore'):
                r_limits = np.where(r_req > 0, avail / np.maximum(1e-5, r_req), 999.0)
            max_c = max(1, int(np.floor(np.min(r_limits))))
            max_counts.append(min(max_c, bound))

        def eval_solution(sol):
            val = sum(p_ne[i] * sol[i] for i in range(num_services))
            req = np.sum(np.array(sol)[:, None] * env.resource_weights, axis=0)
            penalty = np.sum(np.maximum(0.0, req - avail))
            return val - (penalty * 100.0)

        current_sol = [random.randint(0, max_counts[i]) for i in range(num_services)]
        current_energy = eval_solution(current_sol)
        best_sol = list(current_sol)
        best_energy = current_energy

        temp = self.initial_temp
        for _ in range(self.max_iterations):
            # Neighbor solution: perturb one service replica count
            idx = random.randint(0, num_services - 1)
            neighbor = list(current_sol)
            delta = random.choice([-1, 1, 2, -2])
            neighbor[idx] = max(0, min(max_counts[idx], neighbor[idx] + delta))

            neighbor_energy = eval_solution(neighbor)
            diff = neighbor_energy - current_energy

            if diff > 0 or random.random() < np.exp(diff / max(1e-5, temp)):
                current_sol = neighbor
                current_energy = neighbor_energy
                if current_energy > best_energy:
                    best_sol = list(current_sol)
                    best_energy = current_energy

            temp *= self.cooling_rate

        solve_duration = (time.perf_counter() - start_time) * 1000.0  # ms
        return np.array(best_sol, dtype=np.int32), solve_duration
