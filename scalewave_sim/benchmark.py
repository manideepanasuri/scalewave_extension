import os
import argparse
import numpy as np
import pandas as pd

from scalewave_sim.env import ScaleWaveEnv
from scalewave_sim.trace_loader import TraceLoader
from scalewave_sim.algorithms.genetic import GeneticAlgorithmSolver
from scalewave_sim.algorithms.rl_agent import RLPPOAgent
from scalewave_sim.algorithms.baselines import DefaultKPASolver


def run_benchmark(dataset="shanghai", concurrency_levels=[18, 35, 70], duration_seconds=3600, dt=5.0, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    loader = TraceLoader()
    arrivals = loader.load_dataset(name=dataset, duration_seconds=duration_seconds, dt=dt)
    print(f"\n========================================================")
    print(f"  DATASET: {dataset.upper()} TRACE")
    print(f"  Steps: {len(arrivals)}, Total Requests: {arrivals.sum()}, Peak: {arrivals.max()} req/5s, Mean: {arrivals.mean():.2f} req/5s")
    print(f"========================================================")

    model_path = os.path.join("models", "ppo_scalewave.zip")
    
    # 3 Single-implementation baselines + 2 Multi-implementation core algorithms (ScaleWave GA vs Deep RL PPO)
    evaluation_suite = [
        # (Display Name, Solver Instance, baseline_mode)
        ("Offload-only (Default KPA)", DefaultKPASolver("offload_only"), "offload_only"),
        ("Local-only (Default KPA)", DefaultKPASolver("local_only"), "local_only"),
        ("Local-offload (Default KPA)", DefaultKPASolver("local_offload"), "local_offload"),
        ("ScaleWave (GA - Paper)", GeneticAlgorithmSolver(population_size=40, generations=20), None),
        ("Deep RL (PPO - Ours)", RLPPOAgent(model_path=model_path), None)
    ]

    summary_records = []
    detailed_latencies = {}

    for c in concurrency_levels:
        print(f"\n--- Concurrency Target c = {c} ({dataset.upper()}) ---")

        for algo_name, solver, b_mode in evaluation_suite:
            print(f"[*] Evaluating {algo_name}...")
            env = ScaleWaveEnv(arrivals=arrivals.copy(), concurrency=c, dt=dt, baseline_mode=b_mode)
            env.reset()

            solve_times = []
            terminated = False

            while not terminated:
                action, solve_ms = solver.solve(env)
                solve_times.append(solve_ms)
                _, _, terminated, _, info = env.step(action)

            served = env.history['requests_served']
            dropped = env.history['requests_dropped'] if env.history['requests_dropped'] > 0 else env.queue
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
                'Dataset': dataset,
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
            detailed_latencies[f"{dataset}_{algo_name}_c{c}"] = all_lats

            print(f"    -> Served: {served}/{total_reqs} ({success_rate:.1f}%), Avg Lat: {avg_lat:.2f}s, P95: {p95_lat:.2f}s, Solve: {avg_solve_ms:.2f}ms")

    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(results_dir, f"benchmark_{dataset}.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[✓] Benchmark summary saved to {summary_path}")

    np.savez_compressed(os.path.join(results_dir, f"detailed_latencies_{dataset}.npz"), **detailed_latencies)
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset Benchmark for ScaleWave and Deep RL")
    parser.add_argument("--dataset", type=str, default="shanghai", choices=["shanghai", "azure", "nasa", "all"],
                        help="Workload dataset to evaluate: shanghai, azure, nasa, or all")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[18, 35, 70],
                        help="Knative concurrency targets c to test")
    args = parser.parse_args()

    if args.dataset == "all":
        all_dfs = []
        for ds in ["shanghai", "azure", "nasa"]:
            df = run_benchmark(dataset=ds, concurrency_levels=args.concurrency)
            all_dfs.append(df)
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv("results/benchmark_all_datasets.csv", index=False)
        master_df.to_csv("results/benchmark_master_all_datasets.csv", index=False)
        
        # Sync Shanghai as default summary
        if os.path.exists("results/benchmark_shanghai.csv"):
            import shutil
            shutil.copy("results/benchmark_shanghai.csv", "results/benchmark_summary.csv")
            shutil.copy("results/detailed_latencies_shanghai.npz", "results/detailed_latencies.npz")
        print("\n[✓] All datasets evaluated and master summary saved to results/benchmark_all_datasets.csv and benchmark_master_all_datasets.csv")
    else:
        run_benchmark(dataset=args.dataset, concurrency_levels=args.concurrency)
