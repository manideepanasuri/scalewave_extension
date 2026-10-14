import os
from stable_baselines3 import PPO

from scalewave_sim.env import ScaleWaveEnv
from scalewave_sim.trace_loader import TraceLoader


def train_ppo(total_timesteps=100000, save_dir="models"):
    """
    Trains PPO agent on the full Shanghai Telecom trace dataset (all 610 user files).
    """
    os.makedirs(save_dir, exist_ok=True)
    loader = TraceLoader()
    # Load entire dataset: all 610 CSV trace files across full 3600s hour
    arrivals = loader.load_trace(max_files=None, duration_seconds=3600, dt=5.0)
    print(f"[*] Loaded full trace dataset: {len(arrivals)} steps, {arrivals.sum()} total requests.")

    # Train under the most stressful bursty concurrency setting (c=18)
    env = ScaleWaveEnv(arrivals=arrivals, concurrency=18, dt=5.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    )

    print(f"[*] Training Deep RL (PPO) agent for {total_timesteps} steps on CPU...")
    model.learn(total_timesteps=total_timesteps)

    save_path = os.path.join(save_dir, "ppo_scalewave.zip")
    model.save(save_path)
    print(f"[✓] Full-dataset trained model saved to {save_path}")
    return save_path


if __name__ == "__main__":
    train_ppo()
