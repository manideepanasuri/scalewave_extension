import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scalewave_sim.trace_loader import TraceLoader


def generate_all_plots(results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "benchmark_summary.csv")
    latencies_path = os.path.join(results_dir, "detailed_latencies.npz")

    if not os.path.exists(summary_path):
        print(f"[!] Summary file {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    algo_order = ["ScaleWave (GA)", "Deep RL (PPO)", "ILP (Optimal Upper Bound)", "Greedy Knapsack", "Simulated Annealing"]
    algo_palette = {
        "ScaleWave (GA)": "#e74c3c",              # Red
        "Deep RL (PPO)": "#2ecc71",               # Green (our proposed approach)
        "ILP (Optimal Upper Bound)": "#3498db",   # Blue
        "Greedy Knapsack": "#f39c12",             # Orange
        "Simulated Annealing": "#9b59b6"          # Purple
    }

    # 1. Workload Trace Arrival Profile
    loader = TraceLoader()
    arrivals = loader.load_trace(max_files=100, duration_seconds=1800, dt=5.0)
    time_minutes = np.arange(len(arrivals)) * (5.0 / 60.0)

    plt.figure(figsize=(11, 4.5))
    plt.plot(time_minutes, arrivals, color="#2c3e50", linewidth=1.5, label="Incoming Requests (per 5s window)")
    plt.fill_between(time_minutes, arrivals, alpha=0.25, color="#34495e")
    plt.axhline(np.mean(arrivals), color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean Arrival Rate ({np.mean(arrivals):.1f} req/window)")
    plt.title("Workload Request Arrival Profile (Shanghai Telecom Trace Replay)", fontsize=13, weight="bold")
    plt.xlabel("Elapsed Time (Minutes)", fontsize=11)
    plt.ylabel("Incoming Requests / 5s", fontsize=11)
    plt.xlim(0, max(time_minutes))
    plt.legend(loc="upper right")
    plt.tight_layout()
    p_trace = os.path.join(results_dir, "workload_trace_profile.png")
    plt.savefig(p_trace, dpi=300)
    plt.close()
    print(f"[✓] Saved {p_trace}")

    # 2. Throughput Comparison across Concurrency Levels
    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=df,
        x="Concurrency",
        y="Served_Requests",
        hue="Algorithm",
        palette=algo_palette,
        hue_order=algo_order
    )
    plt.title("Served Requests across Concurrency Levels (Higher is Better)", fontsize=13, weight='bold')
    plt.ylabel("Total Successful Requests Served", fontsize=11)
    plt.xlabel("Concurrency Target (c)", fontsize=11)
    plt.ylim(4900, 5050)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    p_tp = os.path.join(results_dir, "throughput_comparison.png")
    plt.savefig(p_tp, dpi=300)
    plt.close()
    print(f"[✓] Saved {p_tp}")

    # 3. Normalized Performance (Throughput / Latency)
    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=df,
        x="Concurrency",
        y="Norm_Performance",
        hue="Algorithm",
        palette=algo_palette,
        hue_order=algo_order
    )
    plt.title("Normalized Performance (Throughput / Latency) (Higher is Better)", fontsize=13, weight='bold')
    plt.ylabel("Normalized Performance (req/s)", fontsize=11)
    plt.xlabel("Concurrency Target (c)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    p_np = os.path.join(results_dir, "normalized_performance.png")
    plt.savefig(p_np, dpi=300)
    plt.close()
    print(f"[✓] Saved {p_np}")

    # 4. Tail Latency Comparison (P50, P95, P99) for c=18
    df_c18 = df[df["Concurrency"] == 18]
    tail_metrics = []
    for _, row in df_c18.iterrows():
        tail_metrics.append({"Algorithm": row["Algorithm"], "Percentile": "Median (P50)", "Latency (s)": row["Avg_Latency_s"]})
        tail_metrics.append({"Algorithm": row["Algorithm"], "Percentile": "Tail (P95)", "Latency (s)": row["P95_Latency_s"]})
        tail_metrics.append({"Algorithm": row["Algorithm"], "Percentile": "Worst-case (P99)", "Latency (s)": row["P99_Latency_s"]})
    df_tail = pd.DataFrame(tail_metrics)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=df_tail,
        x="Algorithm",
        y="Latency (s)",
        hue="Percentile",
        palette="Blues_r"
    )
    plt.title("Latency Breakdown (P50, P95, P99) under Bursty Traffic (c = 18)", fontsize=13, weight='bold')
    plt.ylabel("Latency (Seconds - Lower is Better)", fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title="Percentile", frameon=True)
    plt.tight_layout()
    p_tail = os.path.join(results_dir, "tail_latencies_comparison.png")
    plt.savefig(p_tail, dpi=300)
    plt.close()
    print(f"[✓] Saved {p_tail}")

    # 5. Algorithm Decision Latency / Solve Time Comparison (Log Scale)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=df_c18,
        x="Algorithm",
        y="Avg_Solve_Time_ms",
        palette=algo_palette,
        order=algo_order
    )
    plt.yscale("log")
    plt.title("Algorithm Decision Overhead / Solve Latency (Lower is Better)", fontsize=13, weight='bold')
    plt.ylabel("Decision Time (ms, log-scale)", fontsize=11)
    plt.xlabel("Algorithm", fontsize=11)
    plt.xticks(rotation=18, ha='right')

    # Add text annotations with exact values
    for i, p in enumerate(ax.patches):
        val = df_c18.iloc[i]["Avg_Solve_Time_ms"]
        ax.annotate(f"{val:.2f} ms",
                    (p.get_x() + p.get_width() / 2., p.get_height() * 1.3),
                    ha='center', va='bottom', fontsize=10, weight='bold')

    plt.tight_layout()
    p_solve = os.path.join(results_dir, "solve_time_comparison.png")
    plt.savefig(p_solve, dpi=300)
    plt.close()
    print(f"[✓] Saved {p_solve}")

    # 6. Multi-Panel Latency CDF for all Concurrency Levels (c=18, c=35, c=70)
    if os.path.exists(latencies_path):
        data = np.load(latencies_path)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        concurrencies = [18, 35, 70]
        titles = ["Bursty / Sensitive (c = 18)", "Moderate (c = 35)", "High Volume (c = 70)"]

        for idx, c in enumerate(concurrencies):
            ax = axes[idx]
            for algo in algo_order:
                key = f"{algo}_c{c}"
                if key in data:
                    lats = np.sort(data[key])
                    cdf = np.arange(1, len(lats) + 1) / len(lats)
                    ax.plot(lats, cdf, label=algo, linewidth=2.2, color=algo_palette[algo])

            ax.set_title(titles[idx], fontsize=12, weight='bold')
            ax.set_xlabel("End-to-End Latency (seconds)", fontsize=11)
            ax.set_xlim(0, 25)
            ax.grid(True, linestyle="--", alpha=0.6)
            if idx == 0:
                ax.set_ylabel("Cumulative Distribution (CDF)", fontsize=11)

        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.suptitle("Latency CDF Comparison Across All Workload Concurrency Regimes", fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        p_cdf_all = os.path.join(results_dir, "latency_cdf_all.png")
        plt.savefig(p_cdf_all, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Saved {p_cdf_all}")

    # 7. Resource Utilization Breakdown (CPU, GPU, Network)
    res_data = []
    for _, row in df_c18.iterrows():
        res_data.append({"Algorithm": row["Algorithm"], "Resource": "CPU Utilization (%)", "Usage": row["Mean_CPU_Util_%"]})
        res_data.append({"Algorithm": row["Algorithm"], "Resource": "GPU Utilization (%)", "Usage": row["Mean_GPU_Util_%"]})
        res_data.append({"Algorithm": row["Algorithm"], "Resource": "Network Uplink (%)", "Usage": row["Mean_Net_Util_%"]})
    df_res = pd.DataFrame(res_data)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=df_res,
        x="Resource",
        y="Usage",
        hue="Algorithm",
        palette=algo_palette,
        hue_order=algo_order
    )
    plt.title("Cluster Resource Allocation Profile under Bursty Load (c = 18)", fontsize=13, weight='bold')
    plt.ylabel("Mean Utilization (%)", fontsize=11)
    plt.ylim(0, 50)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    p_res = os.path.join(results_dir, "resource_utilization_breakdown.png")
    plt.savefig(p_res, dpi=300)
    plt.close()
    print(f"[✓] Saved {p_res}")


if __name__ == "__main__":
    generate_all_plots()
