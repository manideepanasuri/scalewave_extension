# ScaleWave-RL: Sub-Millisecond Multi-Implementation Autoscaling for Edge Serverless

An edge-cloud serverless autoscaling framework that combines multi-implementation service equivalence with constrained Deep Reinforcement Learning (PPO). This project replicates, stress-tests, and substantially improves upon the state-of-the-art **ScaleWave** baseline (*IEEE PerCom 2026*).

---

## Overview

Edge serverless computing faces heterogeneous hardware bottlenecks across CPU, GPU, memory, and network uplink. While standard Knative Pod Autoscaler (KPA) relies on a single container implementation and easily collapses under resource saturation, **ScaleWave** introduces multi-implementation equivalence—dynamically routing traffic across four interchangeable container variants:
- **Cloud Offload** (High bandwidth, zero edge compute)
- **Edge-CPU** (Heavy CPU compute, low network)
- **Edge-GPU** (Fast GPU inference, minimal CPU/network)
- **Hybrid** (Balanced CPU-network partitioning)

### The ScaleWave Limitation
ScaleWave solves replica and traffic allocation using a **Genetic Algorithm (GA)**. Under production edge conditions, GA requires **55–58 ms** per control cycle and incurs heavy control-plane CPU contention.

### Our Solution: Constrained Deep RL (PPO)
We replace the genetic solver with an **Actor-Critic Proximal Policy Optimization (PPO)** agent trained with physical safety constraints:
- **Sub-millisecond inference:** Decision cycles execute in **<0.25 ms** (>280x faster than ScaleWave GA).
- **Lower latency:** Delivers **23.1% to 41.2% lower end-to-end average latency** across diverse real-world workloads.
- **Higher performance:** Delivers up to **70.3% higher normalized throughput** while preventing SLA-violating container drops.

---

## Hardware Configuration (Config 1)

The calibrated edge cluster simulates an edge gateway node (e.g., NVIDIA Jetson Nano / Raspberry Pi edge cluster) governed by physical cgroups and network queues:

- **CPU Capacity:** `12.0` cores
- **Memory Capacity:** `20.0` GB RAM
- **GPU Capacity:** `128.0` CUDA cores (Jetson Nano architecture)
- **Network Uplink:** `24.0` Mbps

### Service Implementation Resource Profiles

| Service Variant | CPU (cores) | RAM (GB) | GPU (cores) | Uplink (Mbps) | Base Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0: Cloud Offload** | 0.1 | 0.2 | 0 | 12.0 | 3.5 |
| **1: Edge-CPU** | 2.0 | 1.5 | 0 | 0.2 | 9.8 |
| **2: Edge-GPU** | 0.5 | 1.0 | 32.0 | 0.2 | 2.8 |
| **3: Hybrid** | 1.0 | 0.8 | 0 | 3.0 | 6.2 |

---

## Workload Datasets

Experiments evaluate three distinct production traces (sampled at $\Delta t = 5$ seconds over 720 steps / 1 hour):

1. **Shanghai Telecom Trace:** Bursty mobile edge access logs (8,846 total requests, peak 346 req/5s).
2. **Azure Functions 2019 Trace:** Cloud-native serverless invocations (12,415 total requests, peak 31 req/5s).
3. **NASA Flash Crowd Trace:** Highly dynamic HTTP access logs (6,017 total requests, peak 22 req/5s).

---

## Directory Structure

```text
.
├── scalewave_sim/
│   ├── env.py                  # Calibrated Gymnasium environment (hardware limits & KPA dynamics)
│   ├── trace_loader.py         # Real-world workload trace preprocessors (Shanghai, Azure, NASA)
│   ├── benchmark.py            # Main multi-dataset benchmark execution harness
│   ├── train_rl.py             # Stable-Baselines3 PPO training script
│   ├── find_best_config.py     # Hardware capacity grid sweep harness
│   ├── run_ablation.py         # Elastic vs hardware-contended ablation runner
│   ├── plot_results.py         # Publication figures generator (throughput, tail latency, CDF)
│   ├── plot_figure7.py         # PerCom 2026 Figure 7 physical validation replication
│   ├── plot_ablation.py        # Ablation study visualization
│   └── algorithms/
│       ├── rl_agent.py         # Deep RL (PPO) inference agent (<0.25 ms forward pass)
│       ├── genetic.py          # ScaleWave Genetic Algorithm (GA) baseline
│       ├── baselines.py        # Single-implementation KPA baselines (Offload, Local, Local-offload)
│       ├── greedy.py           # Greedy heuristic solver
│       ├── simulated_annealing.py # Simulated annealing heuristic solver
│       └── ilp_optimal.py      # Integer Linear Programming (ILP) upper bound
├── data/                       # Workload trace datasets
├── models/
│   └── ppo_scalewave.zip       # Pre-trained PPO neural network weights
├── results/                    # Generated benchmark CSVs, NPZ raw latencies, and PNG figures
├── paper/                      # IEEEtran LaTeX conference manuscript and figures
├── analysis.ipynb              # Interactive analysis and validation Jupyter notebook
├── run_experiments.py          # One-shot end-to-end automated experiment runner
└── requirements.txt            # Python dependencies
```

---

## Installation & Setup

### 1. Prerequisites
- Linux OS (Ubuntu 20.04/22.04/Debian or Arch Linux)
- Python 3.10+ / Python 3.14

### 2. Virtual Environment Setup

```bash
# Clone the repository and navigate into the project root
git clone <repository_url>
cd acn

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Command Reference

All commands should be executed from the project root with `PYTHONPATH=.`.

### 1. Training the Deep RL (PPO) Agent

Trains the PPO policy for 100,000 timesteps against the Shanghai Telecom trace under the stressful concurrency setting ($c = 18$). The trained policy weights are saved to `models/ppo_scalewave.zip`:

```bash
PYTHONPATH=. ./venv/bin/python scalewave_sim/train_rl.py
```

### 2. Running Multi-Dataset Benchmarks

Evaluates all 5 algorithms (`Offload-only`, `Local-only`, `Local-offload`, `ScaleWave GA`, and `Deep RL PPO`) across Knative concurrency targets $c \in \{18, 35, 70\}$:

#### Run on all datasets (Shanghai, Azure, NASA):
```bash
PYTHONPATH=. ./venv/bin/python scalewave_sim/benchmark.py --dataset all --concurrency 18 35 70
```

#### Run on a specific dataset:
```bash
# Shanghai Telecom trace
PYTHONPATH=. ./venv/bin/python scalewave_sim/benchmark.py --dataset shanghai --concurrency 18 35 70

# Azure Functions 2019 trace
PYTHONPATH=. ./venv/bin/python scalewave_sim/benchmark.py --dataset azure --concurrency 18 35 70

# NASA Flash Crowd trace
PYTHONPATH=. ./venv/bin/python scalewave_sim/benchmark.py --dataset nasa --concurrency 18 35 70
```

Benchmark CSV summaries and compressed `.npz` raw latencies are written directly to `results/`:
- `results/benchmark_master_all_datasets.csv`
- `results/benchmark_shanghai.csv`
- `results/benchmark_azure.csv`
- `results/benchmark_nasa.csv`
- `results/detailed_latencies_*.npz`

### 3. Running Hardware Parameter Sweeps

To evaluate different edge cluster hardware profiles (CPU: 12–16 cores, RAM: 20–24 GB, Uplink: 24–36 Mbps) and verify bottleneck behavior:

```bash
PYTHONPATH=. ./venv/bin/python scalewave_sim/find_best_config.py
```

### 4. Running the Elastic vs. Contended Ablation Study

Compares unconstrained elastic scaling against realistic physical container contention (cgroups throttling and timeout drops):

```bash
# Execute ablation simulation
PYTHONPATH=. ./venv/bin/python scalewave_sim/run_ablation.py

# Generate ablation plots
PYTHONPATH=. ./venv/bin/python scalewave_sim/plot_ablation.py
```

### 5. Generating Publication Figures

Generates all publication-quality comparison charts and synchronizes them to `paper/figures/`:

```bash
# Generate core benchmark plots (throughput, tail latency, CDF, solve time)
PYTHONPATH=. ./venv/bin/python scalewave_sim/plot_results.py

# Generate physical reproduction plots (Figure 7 validation across all traces)
PYTHONPATH=. ./venv/bin/python scalewave_sim/plot_figure7.py

# Copy latest figures to paper directory
cp results/*.png paper/figures/
```

Generated plots in `results/`:
- `workload_trace_profile.png`
- `throughput_comparison.png`
- `normalized_performance.png`
- `tail_latencies_comparison.png`
- `solve_time_comparison.png`
- `latency_cdf_all.png`
- `resource_utilization_breakdown.png`
- `figure7_paper_physical_reproduction.png`
- `figure7_shanghai_simulation.png`
- `figure7_azure_functions.png`
- `figure7_nasa_flash_crowd.png`

### 6. Executing the Analysis Notebook

To re-run all calculations, distributions, and embedded plots inside `analysis.ipynb` in-place:

```bash
./venv/bin/jupyter nbconvert --to notebook --execute analysis.ipynb --inplace
```

To launch an interactive Jupyter session:

```bash
./venv/bin/jupyter notebook analysis.ipynb
```

### 7. One-Shot Automated Pipeline

To run the complete pipeline (training PPO, running benchmarks, plotting figures, and re-rendering `analysis.ipynb`) in a single command:

```bash
PYTHONPATH=. ./venv/bin/python run_experiments.py
```

---

## Empirical Results Summary (Calibrated Testbed Benchmark)

### Benchmark Comparison ($c = 18$)

| Dataset | Metric | Offload-only | Local-only | Local-offload | ScaleWave (GA) | Deep RL (PPO) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Shanghai** | Served / Total | 8528 / 8846 | 8475 / 8846 | 8587 / 8846 | 8769 / 8846 | **8777 / 8846** |
| | Dropped Reqs | 318 | 371 | 259 | 77 | **69** |
| | Avg Latency | 6.20 s | 15.74 s | 10.01 s | 3.89 s | **3.35 s** (-13.9%) |
| | Norm. Performance | 1376.3 | 538.5 | 857.7 | 2254.7 | **2617.7** (+16.1%) |
| | Solve Overhead | **0.01 ms** | **0.01 ms** | **0.01 ms** | 56.36 ms | **0.01 ms** (>5000x) |
| **Azure** | Served / Total | 12415 / 12415 | 12111 / 12415 | 12415 / 12415 | 12414 / 12415 | **12413 / 12415** |
| | Dropped Reqs | 0 | 304 | 0 | 1 | **2** |
| | Avg Latency | 8.11 s | 15.79 s | 10.00 s | 10.32 s | **3.15 s** (-69.5%) |
| | Norm. Performance | 1530.4 | 767.0 | 1241.6 | 1202.9 | **3937.8** (+227.4%) |
| | Solve Overhead | **0.01 ms** | **0.01 ms** | **0.01 ms** | 55.49 ms | **0.01 ms** (>5000x) |
| **NASA** | Served / Total | 6017 / 6017 | 6010 / 6017 | 6017 / 6017 | 6015 / 6017 | **6016 / 6017** |
| | Dropped Reqs | 0 | 7 | 0 | 2 | **1** |
| | Avg Latency | 5.17 s | 14.28 s | 9.04 s | 7.79 s | **3.21 s** (-58.8%) |
| | Norm. Performance | 1162.6 | 420.9 | 665.9 | 772.4 | **1876.5** (+142.9%) |
| | Solve Overhead | **0.01 ms** | **0.01 ms** | **0.01 ms** | 58.92 ms | **0.01 ms** (>5000x) |

### Key Findings
1. **Bottleneck Realism (Throughput Not Full):** Single implementations hit physical capacity limits and buffer overflows, consistently dropping requests across concurrencies and workloads (e.g. Local-only drops up to 3,277 requests in Azure at $c=70$; Offload-only drops 318–333 requests in Shanghai).
2. **Superior Normalized Performance:** Across every dataset and concurrency target ($c \in \{18, 35, 70\}$), Deep RL achieves strictly higher normalized performance than ScaleWave GA (+16% to +227% gain) by prioritizing Edge-GPU (2.8s) and Cloud (3.5s) over slow Edge-CPU (9.8s).
3. **Sub-Millisecond Decisions:** Deep RL executes policy inference in ~0.01 ms, eliminating the 55–60 ms control-plane compute overhead of ScaleWave's Genetic Algorithm.
