import os
import numpy as np
import pandas as pd

from scalewave_sim.env import ScaleWaveEnv
from scalewave_sim.trace_loader import TraceLoader
from scalewave_sim.algorithms.genetic import GeneticAlgorithmSolver
from scalewave_sim.algorithms.ilp_optimal import ILPOptimalSolver
from scalewave_sim.algorithms.greedy import GreedyKnapsackSolver
from scalewave_sim.algorithms.simulated_annealing import SimulatedAnnealingSolver
from scalewave_sim.algorithms.rl_agent import RLPPOAgent


def run_benchmark(concurrency_levels=[18, 35, 70], duration_seconds=3600, dt=5.0, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    loader = TraceLoader()
    # Load all 610 files for the complete 1-hour trace
    arrivals = loader.load_trace(max_files=None, duration_seconds=duration_seconds, dt=dt)
    print(f"[*] Loaded full workload trace: {len(arrivals)} steps, {arrivals.sum()} total requests.")

    model_path = os.path.join("models", "ppo_scalewave.zip")
    algorithms = {
        "ScaleWave (GA)": GeneticAlgorithmSolver(population_size=60, generations=40),
        "Deep RL (PPO)": RLPPOAgent(model_path=model_path),
        "ILP (Optimal Upper Bound)": ILPOptimalSolver(),
        "Greedy Knapsack": GreedyKnapsackSolver(),
        "Simulated Annealing": SimulatedAnnealingSolver(max_iterations=80)
    }

    summary_records = []
    detailed_latencies = {}

    for c in concurrency_levels:
        print(f"\n========================================================")
        print(f"  Running Benchmark for Concurrency c = {c} (Full Dataset)")
        print(f"========================================================")

        for algo_name, solver in algorithms.items():
            print(f"[*] Evaluating {algo_name}...")
            env = ScaleWaveEnv(arrivals=arrivals.copy(), concurrency=c, dt=dt)
            env.reset()

            solve_times = []
            terminated = False

            while not terminated:
                action, solve_ms = solver.solve(env)
                solve_times.append(solve_ms)
                _, _, terminated, _, info = env.step(action)

            served = env.history['requests_served']
            dropped = env.queue
            total_reqs = served + dropped
            all_lats = env.history['latencies'] if len(env.history['latencies']) > 0 else [env.base_latencies.mean()]
            
            avg_lat = float(np.mean(all_lats))
            p50_lat = float(np.percentile(all_lats, 50))
            p95_lat = float(np.percentile(all_lats, 95))
            p99_lat = float(np.percentile(all_lats, 99))
            avg_solve_ms = float(np.mean(solve_times))

            norm_perf = served / max(0.01, avg_lat)
            success_rate = (served / max(1, total_reqs)) * 100.0

            util_matrix = np.array(env.history['resource_utilization'])
            mean_cpu_util = float(np.mean(util_matrix[:, 0])) * 100.0
            mean_gpu_util = float(np.mean(util_matrix[:, 2])) * 100.0
            mean_net_util = float(np.mean(util_matrix[:, 3])) * 100.0

            record = {
                'Concurrency': c,
                'Algorithm': algo_name,
                'Served_Requests': served,
                'Dropped_Requests': dropped,
                'Success_Rate_%': round(success_rate, 2),
                'Avg_Latency_s': round(avg_lat, 3),
                'P95_Latency_s': round(p95_lat, 3),
                'P99_Latency_s': round(p99_lat, 3),
                'Norm_Performance': round(norm_perf, 2),
                'Avg_Solve_Time_ms': round(avg_solve_ms, 3),
                'Mean_CPU_Util_%': round(mean_cpu_util, 2),
                'Mean_GPU_Util_%': round(mean_gpu_util, 2),
                'Mean_Net_Util_%': round(mean_net_util, 2)
            }
            summary_records.append(record)
            detailed_latencies[f"{algo_name}_c{c}"] = all_lats

            print(f"    -> Served: {served}/{total_reqs} ({success_rate:.1f}%), Avg Lat: {avg_lat:.2f}s, P95: {p95_lat:.2f}s, Solve: {avg_solve_ms:.2f}ms")

    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(results_dir, "benchmark_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[✓] Benchmark summary saved to {summary_path}")

    np.savez_compressed(os.path.join(results_dir, "detailed_latencies.npz"), **detailed_latencies)
    return df_summary


if __name__ == "__main__":
    run_benchmark()
