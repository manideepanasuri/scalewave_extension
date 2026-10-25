import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_multi_dataset_plots(results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    # Load all 3 dataset summaries
    dfs = []
    shanghai_path = os.path.join(results_dir, "benchmark_summary.csv")
    azure_path = os.path.join(results_dir, "benchmark_azure.csv")
    nasa_path = os.path.join(results_dir, "benchmark_nasa.csv")

    if os.path.exists(shanghai_path):
        df_sh = pd.read_csv(shanghai_path)
        if "Dataset" not in df_sh.columns:
            df_sh.insert(0, "Dataset", "shanghai")
        dfs.append(df_sh)
    
    if os.path.exists(azure_path):
        dfs.append(pd.read_csv(azure_path))
        
    if os.path.exists(nasa_path):
        dfs.append(pd.read_csv(nasa_path))

    if not dfs:
        print("[!] No benchmark result files found.")
        return

    master_df = pd.concat(dfs, ignore_index=True)
    algo_order = [
        "Offload-only (Default KPA)",
        "Local-only (Default KPA)",
        "Local-offload (Default KPA)",
        "ScaleWave (GA - Paper)",
        "Deep RL (PPO - Ours)"
    ]
    master_df = master_df[master_df["Algorithm"].isin(algo_order)].copy()
    master_path = os.path.join(results_dir, "benchmark_master_all_datasets.csv")
    master_df.to_csv(master_path, index=False)
    print(f"[✓] Saved master dataset benchmark to {master_path}")

    sns.set_theme(style="whitegrid", font_scale=1.05)

    algo_palette = {
        "Offload-only (Default KPA)": "#7f8c8d",   # Gray
        "Local-only (Default KPA)": "#c0392b",     # Dark Crimson
        "Local-offload (Default KPA)": "#d35400",  # Dark Orange
        "ScaleWave (GA - Paper)": "#e74c3c",       # Coral Red
        "Deep RL (PPO - Ours)": "#2ecc71"          # Emerald Green
    }

    dataset_labels = {
        "shanghai": "Shanghai Telecom (Cellular)",
        "azure": "Azure Functions (Serverless)",
        "nasa": "NASA Flash Crowd (Surge)"
    }
    master_df["Dataset_Display"] = master_df["Dataset"].map(dataset_labels).fillna(master_df["Dataset"])

    # Figure 1: Average Latency across Datasets (c = 18)
    df_c18 = master_df[master_df["Concurrency"] == 18].copy()
    plt.figure(figsize=(13, 6))
    ax = sns.barplot(
        data=df_c18,
        x="Dataset_Display",
        y="Avg_Latency_s",
        hue="Algorithm",
        palette=algo_palette,
        hue_order=algo_order
    )
    plt.title("End-to-End Latency Across Multi-Domain Benchmarks (Concurrency Target c = 18)", fontsize=13, weight='bold')
    plt.xlabel("Evaluation Workload Dataset", fontsize=11)
    plt.ylabel("Average Request Latency (seconds)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    p_lat = os.path.join(results_dir, "multi_dataset_latency_comparison.png")
    plt.savefig(p_lat, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {p_lat}")

    # Figure 2: Solve Time Overhead (log scale)
    plt.figure(figsize=(10, 5))
    df_solve = df_c18.groupby("Algorithm")["Avg_Solve_Time_ms"].mean().loc[algo_order].reset_index()
    colors = [algo_palette[a] for a in algo_order]
    bars = plt.bar(df_solve["Algorithm"], df_solve["Avg_Solve_Time_ms"], color=colors, edgecolor='black', linewidth=0.8)
    plt.yscale("log")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.ylabel("Decision Overhead per Step (ms, log scale)", fontsize=11)
    plt.title("Optimizer Decision Latency per 5s Control Step (Mean Across All Datasets)", fontsize=13, weight='bold')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h * 1.25, f"{h:.2f} ms", ha='center', va='bottom', fontsize=8.5, weight='bold')
    plt.ylim(0.005, 1000)
    plt.tight_layout()
    p_solve = os.path.join(results_dir, "multi_dataset_solve_overhead.png")
    plt.savefig(p_solve, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {p_solve}")


if __name__ == "__main__":
    generate_multi_dataset_plots()
