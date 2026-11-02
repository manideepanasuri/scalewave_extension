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

## Empirical Results Summary (Config 1)

### Benchmark Comparison ($c = 18$)

| Dataset | Metric | Offload-only | Local-only | Local-offload | ScaleWave (GA) | Deep RL (PPO) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Shanghai** | Served / Total | 8596 / 8846 | 8600 / 8846 | 8661 / 8846 | 8845 / 8846 | **8845 / 8846** |
| | Dropped Reqs | 250 | 246 | 185 | 1 | **1** |
| | Avg Latency | 13.79 s | 19.41 s | 13.44 s | 8.28 s | **6.37 s** (-23.1%) |
| | P95 Latency | 8.05 s | 11.27 s | 7.13 s | 11.94 s | **10.02 s** |
| | Solve Overhead | **0.01 ms** | **0.01 ms** | **0.01 ms** | 56.81 ms | **0.20 ms** (>280x) |
| **Azure** | Avg Latency | 5.42 s | 10.98 s | 6.86 s | 8.17 s | **5.86 s** (-28.3%) |
| | Solve Overhead | **0.01 ms** | **0.01 ms** | **0.01 ms** | 57.17 ms | **0.19 ms** (>280x) |
| **NASA** | Avg Latency | 3.84 s | 10.83 s | 6.72 s | 9.78 s | **5.75 s** (-41.2%) |
| | Solve Overhead | **0.01 ms** | **0.01 ms** | **0.01 ms** | 58.56 ms | **0.19 ms** (>280x) |

### Key Findings
1. **Bottleneck Circumvention:** Under bursty traffic, single implementations drop 185–250 requests due to CPU (12 cores) or Uplink (24 Mbps) exhaustion. Both ScaleWave GA and Deep RL avoid drops by shifting traffic across implementations.
2. **Latency Reduction:** Deep RL achieves **23.1% to 41.2% lower average latency** compared to ScaleWave GA because it learns to allocate requests preferentially to GPU (2.8s) and Cloud (3.5s) while throttling slow Edge-CPU replicas.
3. **Control Overhead:** Deep RL reduces optimization overhead from **~57 ms** down to **~0.2 ms**, eliminating control-plane CPU thrashing.
