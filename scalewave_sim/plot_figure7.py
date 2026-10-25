import os
import io
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def create_figure7_style_plot(data_dict, title_suffix, output_filename, include_deep_rl=True):
    """
    Renders the exact 3-axis Figure 7 style from the ScaleWave paper:
      - 3 horizontal subplots: (a) c=70, (b) c=35, (c) c=18
      - Left Y-axis (Green): Successful Requests (bars)
      - Right Y-axis 1 (Orange): Average Latency (s) (solid line with circular markers)
      - Right Y-axis 2 (Blue, offset outward): Norm. Performance (req/s) (dotted line with square markers)
      - Top unified horizontal legend
    """
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.size': 10,
        'axes.labelsize': 10.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    green_color = '#a6dba0'
    green_edge = '#5aae61'
    orange_color = '#f97316'
    blue_color = '#1f78b4'

    sub_labels = ['(a) c=70', '(b) c=35', '(c) c=18']
    concurrencies = [70, 35, 18]

    for idx, (c, ax1) in enumerate(zip(concurrencies, axes)):
        df_c = data_dict[c]

        x_indices = np.arange(len(df_c))
        deployments = df_c['Deployment'].values
        requests = df_c['Requests'].values
        latencies = df_c['Average_Latency'].values
        perf = df_c['Performance'].values

        # Axis 1: Successful Requests (Left, Green)
        ax1.bar(
            x_indices,
            requests,
            color=green_color,
            edgecolor=green_edge,
            alpha=0.75,
            width=0.55,
            zorder=2
        )
        ax1.set_ylabel('Successful Requests', color='#2ca02c', fontweight='semibold')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(deployments, rotation=18, ha='right')
        ax1.set_xlabel(sub_labels[idx], fontweight='semibold', labelpad=8)
        ax1.set_ylim(0, max(requests) * 1.18 if max(requests) > 0 else 1000)
        ax1.grid(True, linestyle='--', alpha=0.35, zorder=1)

        # Axis 2: Average Latency (Right 1, Orange)
        ax2 = ax1.twinx()
        ax2.plot(
            x_indices,
            latencies,
            color=orange_color,
            linewidth=2.2,
            marker='o',
            markersize=5.5,
            zorder=4
        )
        ax2.set_ylabel('Average Latency (s)', color=orange_color, fontweight='semibold')
        ax2.tick_params(axis='y', labelcolor=orange_color)
        ax2.set_ylim(0, max(latencies) * 1.18 if max(latencies) > 0 else 10)
        ax2.grid(False)

        # Axis 3: Normalized Performance (Right 2, Blue offset)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 45))
        ax3.spines['right'].set_color(blue_color)
        ax3.plot(
            x_indices,
            perf,
            color=blue_color,
            linewidth=2.0,
            linestyle=':',
            marker='s',
            markersize=5.5,
            zorder=5
        )
        ax3.set_ylabel('Norm. Performance (req/s)', color=blue_color, fontweight='semibold')
        ax3.tick_params(axis='y', labelcolor=blue_color)
        ax3.set_ylim(0, max(perf) * 1.18 if max(perf) > 0 else 100)
        ax3.grid(False)

    # Shared Legend centered at the top
    legend_elements = [
        Patch(facecolor=green_color, edgecolor=green_edge, alpha=0.75, label='Successful Requests'),
        Line2D([0], [0], color=orange_color, linewidth=2.2, marker='o', markersize=5.5, label='Average Latency (s)'),
        Line2D([0], [0], color=blue_color, linewidth=2.0, linestyle=':', marker='s', markersize=5.5, label='Normalized Performance (req/s)')
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=10.5
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved {output_filename}")


def generate_all_figure7_plots():
    # 1. Exact Physical Testbed Reproduction (Direct Paper Results from paper_results.zip)
    zip_path = "ScaleWave/evaluations/scripts/simulation/results/paper_results.zip"
    if os.path.exists(zip_path):
        z = zipfile.ZipFile(zip_path)
        paper_data = {}
        for c in [70, 35, 18]:
            rows = []
            for name, tag in [('Offload', 'cloud'), ('Local', 'edge'), ('Local-Offload', 'hybrid'), ('ScaleWave', 'oblique')]:
                fname = f"paper_results/c{c}/requests/output_metrics_{tag}.csv"
                df = pd.read_csv(io.BytesIO(z.read(fname)))
                df_200 = df[df['status_code'] == 200]
                served = len(df_200)
                lat = df_200['latency'].mean()
                perf = served / lat if lat > 0 else 0
                rows.append({'Deployment': name, 'Requests': served, 'Average_Latency': lat, 'Performance': perf})
            paper_data[c] = pd.DataFrame(rows)

        create_figure7_style_plot(
            paper_data,
            "Physical Testbed Reproduction (PerCom 2026 Fig. 7)",
            "results/figure7_paper_physical_reproduction.png",
            include_deep_rl=False
        )

    # 2. Simulation Benchmark with Deep RL Included (Shanghai Telecom)
    summary_path = "results/benchmark_summary.csv"
    if os.path.exists(summary_path):
        df_sh = pd.read_csv(summary_path)
        sim_data = {}
        name_map = {
            'Offload-only (Default KPA)': 'Offload',
            'Local-only (Default KPA)': 'Local',
            'Local-offload (Default KPA)': 'Local-Offload',
            'ScaleWave (GA - Paper)': 'ScaleWave',
            'Deep RL (PPO - Ours)': 'Deep RL'
        }
        for c in [70, 35, 18]:
            sub = df_sh[df_sh['Concurrency'] == c].copy()
            sub['Deployment'] = sub['Algorithm'].map(name_map)
            sub = sub.dropna(subset=['Deployment'])
            sub['Requests'] = sub['Served_Requests']
            sub['Average_Latency'] = sub['Avg_Latency_s']
            sub['Performance'] = sub['Norm_Performance']
            sim_data[c] = sub[['Deployment', 'Requests', 'Average_Latency', 'Performance']].reset_index(drop=True)

        create_figure7_style_plot(
            sim_data,
            "Shanghai Telecom Simulation (ScaleWave vs Deep RL)",
            "results/figure7_shanghai_simulation.png",
            include_deep_rl=True
        )

    # 3. Azure Functions Benchmark
    azure_path = "results/benchmark_azure.csv"
    if os.path.exists(azure_path):
        df_az = pd.read_csv(azure_path)
        azure_data = {}
        for c in [70, 35, 18]:
            sub = df_az[df_az['Concurrency'] == c].copy()
            sub['Deployment'] = sub['Algorithm'].map(name_map)
            sub = sub.dropna(subset=['Deployment'])
            sub['Requests'] = sub['Served_Requests']
            sub['Average_Latency'] = sub['Avg_Latency_s']
            sub['Performance'] = sub['Norm_Performance']
            azure_data[c] = sub[['Deployment', 'Requests', 'Average_Latency', 'Performance']].reset_index(drop=True)

        create_figure7_style_plot(
            azure_data,
            "Azure Functions Serverless Benchmark",
            "results/figure7_azure_functions.png",
            include_deep_rl=True
        )

    # 4. NASA Flash Crowd Benchmark
    nasa_path = "results/benchmark_nasa.csv"
    if os.path.exists(nasa_path):
        df_nasa = pd.read_csv(nasa_path)
        nasa_data = {}
        for c in [70, 35, 18]:
            sub = df_nasa[df_nasa['Concurrency'] == c].copy()
            sub['Deployment'] = sub['Algorithm'].map(name_map)
            sub = sub.dropna(subset=['Deployment'])
            sub['Requests'] = sub['Served_Requests']
            sub['Average_Latency'] = sub['Avg_Latency_s']
            sub['Performance'] = sub['Norm_Performance']
            nasa_data[c] = sub[['Deployment', 'Requests', 'Average_Latency', 'Performance']].reset_index(drop=True)

        create_figure7_style_plot(
            nasa_data,
            "NASA Flash-Crowd Benchmark",
            "results/figure7_nasa_flash_crowd.png",
            include_deep_rl=True
        )


if __name__ == "__main__":
    generate_all_figure7_plots()
