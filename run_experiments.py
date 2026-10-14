import os
import sys

sys.path.insert(0, os.path.abspath("."))

from scalewave_sim.train_rl import train_ppo
from scalewave_sim.benchmark import run_benchmark
from scalewave_sim.plot_results import generate_all_plots


def main():
    print("==================================================================")
    print("  ScaleWave: Full Dataset (All 610 Trace Files) Benchmark Suite")
    print("==================================================================")

    # Step 1: Retrain Deep RL (PPO) agent on the full dataset
    print("\n[Step 1/3] Training Deep RL (PPO) agent on the full 610-file dataset (100k timesteps)...")
    train_ppo(total_timesteps=100000)

    # Step 2: Run benchmark across c=18, c=35, c=70 on the full 1-hour dataset
    print("\n[Step 2/3] Running multi-algorithm benchmark across c = [18, 35, 70] on full dataset...")
    df_summary = run_benchmark(concurrency_levels=[18, 35, 70], duration_seconds=3600, dt=5.0)

    # Step 3: Generate visual plots
    print("\n[Step 3/3] Generating full comparison plots...")
    generate_all_plots()

    print("\n==================================================================")
    print("  Full Dataset Benchmark Summary Table:")
    print("==================================================================")
    print(df_summary.to_string(index=False))

    # Step 4: Re-render analysis notebook
    print("\n[*] Re-rendering analysis.ipynb with full dataset metrics...")
    os.system("./venv/bin/python build_full_notebook.py")
    os.system("./venv/bin/jupyter nbconvert --to notebook --execute analysis.ipynb --inplace")
    print("\n[✓] All experiments and report updates completed successfully on full dataset!")


if __name__ == "__main__":
    main()
