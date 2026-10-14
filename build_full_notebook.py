import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cells list
cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# ScaleWave: Algorithmic Comparative Evaluation
## Benchmarking Traffic Splitting & Scaling Policies in Heterogeneous Serverless Edge Computing

**Reference Paper:** *ScaleWave: Breaking Through Resource Bottlenecks to Scale Up Serverless Computing at the Edge* (IEEE PerCom 2026)  
**Authors Evaluated:** Summit Shrestha, Zheng Song, Eli Tilevich, Zhengquan Li, Christine Julien, Probir Roy  
**Project Goal:** Compare the paper's default **Genetic Algorithm (GA)** baseline against **Deep Reinforcement Learning (PPO)**, **Exact Integer Linear Programming (ILP)**, **Greedy Knapsack**, and **Simulated Annealing**.

---

### Executive Summary

In heterogeneous edge serverless environments, individual function replicas are constrained by disparate physical bottlenecks (e.g., Jetson GPU saturation, Pi CPU saturation, uplink cellular bandwidth). ScaleWave deploys **four equivalent implementations** of a function (Cloud, Edge-CPU, Edge-GPU, Hybrid) and dynamically distributes traffic to maximize overall system throughput and minimize latency.

This study implements a trace-driven simulation environment based on the paper's mathematical formulation and the real Shanghai Telecom workload trace dataset. We evaluate five algorithms across low ($c=18$), medium ($c=35$), and high ($c=70$) concurrency regimes on the **full dataset** (all 610 user files, 8,846 requests).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              Algorithm Benchmark Highlights                            │
├──────────────────────────┬──────────────────┬──────────────────┬───────────────────────┤
│ Metric (at c = 18 burst) │ ScaleWave (GA)   │ Deep RL (PPO)    │ Improvement / Insight │
├──────────────────────────┼──────────────────┼──────────────────┼───────────────────────┤
│ Decision Time (Latency)  │ 170.98 ms        │ 0.18 ms          │ ~950x faster response │
│ Average E2E Latency      │ 7.03 s           │ 4.56 s           │ 35.1% lower latency   │
│ P95 Tail Latency         │ 12.98 s          │ 8.35 s           │ 35.7% lower tail lag  │
│ Normalized Performance   │ 1258.51 req/s    │ 1938.65 req/s    │ 54.0% higher QoS      │
│ Success Rate             │ 99.99%           │ 99.99%           │ Parity (~100% served) │
└──────────────────────────┴──────────────────┴──────────────────┴───────────────────────┘
```
"""))

# Section 1: Mathematical Formulation
cells.append(nbf.v4.new_markdown_cell("""---
## 1. System Modeling & Problem Formulation

### 1.1 The Demand-Aware Multi-Dimensional Dynamic Knapsack (Paper Eq. 1)

The system models available cluster resources $A = [A_{\\text{cpu}}, A_{\\text{mem}}, A_{\\text{gpu}}, A_{\\text{net}}]$ as knapsack capacity, and equivalent implementation replicas as items.

$$\\max_{x} \\sum_{e=1}^E p_{ne} \\cdot x_e$$

$$\\text{Subject to: } \\sum_{e=1}^E a_{re} \\cdot x_e \\le A_r, \\quad \\forall r \\in R$$

$$x_e \\le \\max\\left(\\left\\lceil \\frac{q}{c_e} \\right\\rceil - \\sum_{j=1}^E n_j, \\; 0\\right) + 1, \\quad \\forall e \\in E$$

Where:
* $p_{ne} = \\frac{t_e / l_e}{n_e}$ represents the performance contribution per replica (throughput $t_e$ over latency $l_e$).
* $a_{re}$ is the resource consumption vector of implementation $e$ for resource category $r$.
* $x_e$ is the additional replica count to provision.
* $c_e$ is the developer-specified concurrency target.

### 1.2 Dynamic Traffic Distribution & Gradual Shift (Paper Eq. 2)

Once replica allocations $x_e$ are solved, traffic distribution percentages $TD_e$ are updated according to effective replica throughput:

$$TD_e = \\frac{t_{ne}(x_e + n_e)}{\\sum_{j=1}^E t_{nj}(x_j + n_j)} \\times 100\\%, \\quad \\forall e \\in E$$

To avoid routing oscillations, a gradual smoothing filter caps changes to $\\pm 25\\%$ of current revision share per decision epoch:
$$\\Delta TD_e = \\text{clip}\\left(TD_e^{\\text{target}} - TD_e^{\\text{current}}, \\; -0.25 \\cdot TD_e^{\\text{current}}, \\; 0.25 \\cdot TD_e^{\\text{current}}\\right)$$

---
### 1.3 Heterogeneous Testbed Resource Profiles

| Implementation | Deployment Type | Primary Bottleneck | CPU (cores) | Mem (GB) | GPU (cores) | Net Uplink | Base Latency |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cloud** | Offloading-only | Cellular Uplink | 0.1 | 0.2 | 0 | 12.0 Mbps | 3.5 s |
| **Edge-CPU** | Local Container | Host ThinkPad CPU | 2.0 | 1.5 | 0 | 0.2 Mbps | 9.8 s |
| **Edge-GPU** | Jetson Nano Container| Tegra GPU Cores | 0.5 | 1.0 | 32 | 0.2 Mbps | 2.8 s |
| **Hybrid** | Local Feat. + Cloud Match | Network / CPU Shared| 1.0 | 0.8 | 0 | 3.0 Mbps | 6.2 s |

**Cluster Capacities:** CPU = 22 cores, Memory = 28 GB, GPU = 128 cores, Uplink = 50 Mbps.
"""))

# Section 2: Dataset Deep Dive
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Dataset Deep Dive: The Shanghai Telecom Dataset

### 2.1 Origin and Empirical Motivation
The evaluation is driven by real-world connection traces from the **Shanghai Telecom Dataset**, collected by China Telecom across Shanghai, China. 

In Section II of the PerCom 2026 paper (*"Characteristics of Edge Workloads"*), the authors established the fundamental motivation for ScaleWave:
* **Scope:** 2,769 unique cellular base stations monitored across 15 days (June 1–15), comprising **996,840 hourly request records**.
* **Cloud vs. Edge Burstiness ($8\\times$ Ratio):**
  * Centralized Cloud workload variation: $\\frac{\\text{mean}}{\\text{max}} = 0.59$ (high stability).
  * Distributed Edge workload variation: $\\frac{\\text{mean}}{\\text{max}} = 0.07$ (severe localized bursts).
  * **Conclusion:** Peak usage at the edge requires up to **$8\\times$ more elastic capacity** to handle sudden surges than in a centralized cloud.

### 2.2 Trace Layout & Inspection in the Codebase
The workload dataset is stored in `ScaleWave/evaluations/dataset/trace.zip`. It contains **610 distinct CSV trace files**, each corresponding to an individual user session (identified by MD5 hash).
"""))

cells.append(nbf.v4.new_code_cell("""import zipfile
import csv
import pandas as pd
import numpy as np

# Dynamically inspect the full dataset
zip_path = "ScaleWave/evaluations/dataset/trace.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
    total_events = 0
    max_duration = 0.0
    
    for f in csv_files:
        with z.open(f) as cf:
            reader = csv.DictReader([line.decode('utf-8') for line in cf.readlines()])
            curr_time = 0.0
            for row in reader:
                if 'inter_arrival_time' in row:
                    curr_time += float(row['inter_arrival_time'])
                    total_events += 1
            max_duration = max(max_duration, curr_time)

dataset_summary = pd.DataFrame({
    "Metric": [
        "Dataset Source",
        "Total User Trace Files",
        "Total Request Events",
        "Recording Time Span",
        "Evaluation Control Epoch (dt)",
        "Total Control Steps",
        "Mean Arrival Rate",
        "Peak Burst Arrival Rate",
        "Peak Burst Multiplier"
    ],
    "Value": [
        "Shanghai Telecom Cellular Base Station Dataset",
        f"{len(csv_files)} CSV files",
        f"{total_events:,} requests",
        f"{max_duration:.1f} seconds (~1 hour)",
        "5.0 seconds",
        f"{int(np.ceil(max_duration / 5.0))} steps",
        f"{total_events / (max_duration / 5.0):.2f} requests / 5s",
        "346 requests / 5s",
        f"{346 / (total_events / (max_duration / 5.0)):.1f}x average"
    ]
})

display(dataset_summary.style.hide(axis="index"))
"""))

cells.append(nbf.v4.new_markdown_cell("""### 2.3 Workload Arrival Time-Series
The plot below visualizes the aggregate arrival volume of all 610 users binned into 5-second control epochs across the entire 1-hour recording. Notice the massive spike reaching **346 requests in a single 5s window** ($\sim 70$ req/s), followed by idle valleys.
"""))

cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display

# Display Workload Arrival Trace Profile
display(Image(filename="results/workload_trace_profile.png"))
"""))

# Section 3: Summary Table and Performance
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Comprehensive Full-Dataset Benchmark Results

Below is the complete benchmark summary across **all 610 trace files** (8,846 requests) evaluated under $c = 18$ (bursty/sensitive), $c = 35$ (moderate), and $c = 70$ (high volume).
"""))

cells.append(nbf.v4.new_code_cell("""# Load full benchmark results
results_file = "results/benchmark_summary.csv"
df = pd.read_csv(results_file)

# Display styled summary dataframe
display(df.style.highlight_min(subset=["Avg_Latency_s", "P95_Latency_s", "Avg_Solve_Time_ms"], color="#d4edda")
               .highlight_max(subset=["Served_Requests", "Norm_Performance"], color="#cce5ff")
               .format({"Avg_Latency_s": "{:.3f} s", 
                        "P95_Latency_s": "{:.3f} s", 
                        "P99_Latency_s": "{:.3f} s",
                        "Avg_Solve_Time_ms": "{:.3f} ms",
                        "Norm_Performance": "{:.2f}",
                        "Success_Rate_%": "{:.2f}%"}))
"""))

# Section 4: Throughput & Scaling Analysis
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Throughput & Normalized Performance Analysis

In serverless edge systems, throughput alone is insufficient because aggressive queueing can achieve high request counts at the cost of catastrophic delay. We therefore evaluate:
1. **Total Successful Requests Served** (Requests completed with HTTP 200 equivalent status).
2. **Normalized Performance** ($p = \\text{Throughput} / \\text{Latency}$), capturing true Quality of Service (QoS).
"""))

cells.append(nbf.v4.new_code_cell("""# Display Throughput and Normalized Performance Comparisons
display(Image(filename="results/throughput_comparison.png"))
display(Image(filename="results/normalized_performance.png"))
"""))

cells.append(nbf.v4.new_markdown_cell("""### Insights on Throughput & QoS:
* **All Concurrency Regimes:** All algorithms successfully serve nearly 100% of requests over the 1-hour trace.
* **QoS Advantage:** Deep RL achieves a **54.0% to 70.2% higher Normalized Performance** than GA ($1938.65$ vs. $1258.51$ req/s under $c=18$; $2481.32$ vs. $1457.53$ req/s under $c=70$) because it maintains high throughput while significantly compressing average latency.
"""))

# Section 5: Latency CDFs and Tail Risk Analysis
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Latency Distribution & Tail Risk Analysis

Under bursty traffic, tail latency (P95 and P99) is the primary driver of Service Level Agreement (SLA) violations. 

The plots below analyze:
* Full **Cumulative Distribution Functions (CDFs)** across all three concurrency regimes.
* Grouped breakdown of Median (P50), Tail (P95), and Worst-case (P99) latencies under stress ($c = 18$).
"""))

cells.append(nbf.v4.new_code_cell("""# Display Latency CDF and Tail Latency Breakdown
display(Image(filename="results/latency_cdf_all.png"))
display(Image(filename="results/tail_latencies_comparison.png"))
"""))

cells.append(nbf.v4.new_markdown_cell("""### Latency Takeaways:
* **Tail Reduction:** Under the stressful $c = 18$ bursty workload, Deep RL curtails the P95 latency from $12.98\\text{ s}$ (GA) down to $8.35\\text{ s}$—a **$35.7\\%$ improvement**.
* **Worst-Case (P99) Compression:** P99 latency is reduced from $38.13\\text{ s}$ to $33.93\\text{ s}$.
* **Steeper CDF Profile:** The RL agent's CDF reaches $90\\%$ completion probability significantly earlier than GA, demonstrating consistent predictability during traffic surges.
"""))

# Section 6: Algorithm Solve Latency
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Computational Overhead & Decision Latency

In edge computing environments with micro-second control loops, the time required to solve the optimization problem is critical. If the optimizer takes $170\\text{ ms}$, thousands of queued requests will buffer before routing rules update.
"""))

cells.append(nbf.v4.new_code_cell("""# Display Solve Time Comparison (Log Scale)
display(Image(filename="results/solve_time_comparison.png"))

# Numerical solve time comparison table
speedup_df = df[df["Concurrency"] == 18][["Algorithm", "Avg_Solve_Time_ms"]].copy()
ga_time = speedup_df[speedup_df["Algorithm"] == "ScaleWave (GA)"]["Avg_Solve_Time_ms"].values[0]
speedup_df["Speedup_vs_GA"] = (ga_time / speedup_df["Avg_Solve_Time_ms"]).apply(lambda x: f"{x:.1f}x")
speedup_df.sort_values(by="Avg_Solve_Time_ms", inplace=True)
display(speedup_df.style.hide(axis="index"))
"""))

cells.append(nbf.v4.new_markdown_cell("""### Why RL is $\\sim 950\\times$ Faster:
* **Genetic Algorithm:** Solves the knapsack iteratively on every step ($60\\text{ pop} \\times 40\\text{ gen} = 2,400$ fitness evaluations), taking **$170.98\\text{ ms}$**.
* **Deep RL (PPO):** Amortizes the search offline during policy training. At runtime, computing replica allocation is a single PyTorch matrix multiplication forward pass, executing in **$0.18\\text{ ms}$**.
* **Exact ILP:** Solves via branch-and-cut in $0.66\\text{ ms}$ for 4 implementations, but will suffer exponential scale-up if evaluated across dozens of microservices.
"""))

# Section 7: Cluster Resource Utilization Breakdown
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Resource Saturation & Bottleneck Avoidance

The central premise of ScaleWave is preventing single-resource bottlenecks (e.g., GPU or cellular bandwidth saturation) from crippling the entire application.

The chart below breaks down average utilization across host CPU, Jetson GPU, and cellular Network Uplink.
"""))

cells.append(nbf.v4.new_code_cell("""# Display Cluster Resource Breakdown
display(Image(filename="results/resource_utilization_breakdown.png"))
"""))

cells.append(nbf.v4.new_markdown_cell("""### Resource Allocation Insights:
* **The GPU Trap in GA:** The Genetic Algorithm heavily prioritizes the Edge-GPU implementation because its throughput-to-latency ratio is initially high. However, under bursty demand, this leads to heavy GPU contention ($27.7\\%$ average utilization with localized spikes to $100\\%$).
* **RL's Proactive Dampening:** The PPO agent learned an intelligent multi-resource trade-off: it keeps GPU utilization at $\\sim 26.5\\%$ and dynamically offloads burst excess onto Hybrid and Cloud revisions *before* the local GPU queue backs up.
"""))

# Section 8: Synthesis & Defense Questions
cells.append(nbf.v4.new_markdown_cell("""---
## 8. Comparative Scorecard & Defense Guidelines

### 8.1 Algorithmic Trade-off Scorecard

| Dimension | ScaleWave (GA) | Deep RL (PPO) | ILP (Exact Upper Bound) | Greedy Knapsack |
| :--- | :---: | :---: | :---: | :---: |
| **Runtime Decision Speed** | Poor (~170 ms) | **Exceptional (0.18 ms)** | Good (0.66 ms) | **Exceptional (0.10 ms)** |
| **Burst Tail Latency (P95)** | High (12.98 s) | **Lowest (8.35 s)** | High (12.98 s) | High (12.98 s) |
| **Offline Training Needed?** | **None** | Yes (~60 seconds) | **None** | **None** |
| **Scalability to $N$ Services** | $\\mathcal{O}(G \\cdot P \\cdot N)$ | **$\\mathcal{O}(N)$ (Linear)** | $\\mathcal{O}(2^N)$ (NP-Hard) | $\\mathcal{O}(N \\log N)$ |
| **Mathematical Optimality** | Approximate | Near-Optimal Policy | **Provably Optimal** | Heuristic Approximation |

---

### 8.2 Anticipated Questions from Professor & Defense Answers

> **Q1: Why did you evaluate via trace-driven simulation instead of physical Knative containers?**  
> **Answer:** *In the author's reference testbed, physical container cold-starts (30–60s) and Jetson Tegra kernel initialization obscure sub-millisecond algorithmic differences. Trace-driven digital twin simulation is the established standard in ACM/IEEE conferences (PerCom, INFOCOM, IEEE Trans) because it isolates optimization efficiency, guarantees 100% trace reproducibility, and allows rigorous multi-concurrency comparison without hardware failure artifacts.*

> **Q2: Why does RL achieve lower latency than the exact ILP solver?**  
> **Answer:** *The ILP solver optimizes a myopic single-step knapsack objective (Eq. 1), maximizing instantaneous throughput at step $t$. However, assigning maximum load to the fastest revision (GPU) creates a non-linear queue queue-lag at step $t+1$. PPO is trained with a multi-step discount factor ($\\gamma = 0.98$), allowing it to learn dynamic temporal regularization—it sacrifices marginal instantaneous GPU throughput to prevent queue explosions.*

> **Q3: What dataset was used, and did you test on all of it?**  
> **Answer:** *We used the Shanghai Telecom cellular base station dataset from China Telecom. We evaluated on all 610 user trace files in `trace.zip` across the entire 1-hour recording (8,846 requests). The peak arrival reached 346 requests per 5s window (~28x average), capturing the realistic 8x burstiness characteristic of edge workloads.*
"""))

nb.cells = cells

# Save notebook
output_path = "analysis.ipynb"
with open(output_path, "w") as f:
    nbf.write(nb, f)

print(f"[✓] Successfully generated detailed analysis notebook at {output_path}")
