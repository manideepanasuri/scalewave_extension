import random
import numpy as np
import time


class GeneticAlgorithmSolver:
    """
    Paper baseline: Exact Genetic Algorithm from ScaleWave (optimizer.py).
    Solves Eq. 1: max sum(p_ne * x_e) subject to resource capacities and demand bound.
    """
    def __init__(self, population_size=100, generations=100, mutation_rate=0.02, tournament_size=4):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def solve(self, env):
        start_time = time.perf_counter()
        bound = env.get_demand_bound()
        capacities = env.capacities
        current_used = np.sum(env.replicas[:, None] * env.resource_weights, axis=0)
        avail = np.maximum(0.0, capacities - current_used)

        p_ne = env.throughputs / (np.maximum(1e-5, env.latencies) * np.maximum(1, env.replicas))
        num_services = env.num_services

        # Max count per service based on remaining capacity
        max_counts = []
        for i in range(num_services):
            r_req = env.resource_weights[i]
            with np.errstate(divide='ignore'):
                r_limits = np.where(r_req > 0, avail / np.maximum(1e-5, r_req), 999.0)
            max_c = max(1, int(np.floor(np.min(r_limits))))
            max_counts.append(min(max_c, bound))

        def initialize_population():
            return [[random.randint(0, max_counts[i]) for i in range(num_services)]
                    for _ in range(self.population_size)]

        def fitness(solution):
            # Objective: throughput contribution
            tp_contrib = sum(p_ne[i] * solution[i] for i in range(num_services))
            total_req = np.sum(np.array(solution)[:, None] * env.resource_weights, axis=0)
            penalty = np.sum(np.maximum(0.0, total_req - avail))
            return tp_contrib - (penalty * 100.0)

        def select(pop):
            best = random.choice(pop)
            for _ in range(self.tournament_size - 1):
                cont = random.choice(pop)
                if fitness(cont) > fitness(best):
                    best = cont
            return best

        def crossover(parent1, parent2):
            point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2

        def mutate(sol):
            for i in range(len(sol)):
                if random.random() < self.mutation_rate:
                    sol[i] = random.randint(0, max_counts[i])
            return sol

        pop = initialize_population()
        for _ in range(self.generations):
            new_pop = []
            for _ in range(len(pop) // 2):
                p1 = select(pop)
                p2 = select(pop)
                c1, c2 = crossover(p1, p2)
                new_pop.append(mutate(c1))
                new_pop.append(mutate(c2))
            pop = sorted(new_pop + pop, key=fitness, reverse=True)[:self.population_size]

        best_sol = max(pop, key=fitness)
        solve_duration = (time.perf_counter() - start_time) * 1000.0  # ms
        return np.array(best_sol, dtype=np.int32), solve_duration
