import time
import os
import numpy as np
from stable_baselines3 import PPO


class RLPPOAgent:
    """
    Deep Reinforcement Learning (PPO) agent for dynamic traffic and replica allocation.
    Infers optimal replica allocation in <1ms forward pass.
    """
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path)

    def set_model(self, model):
        self.model = model

    def solve(self, env):
        start_time = time.perf_counter()
        bound = env.get_demand_bound()

        if self.model is not None:
            obs = env._get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            # Map [0, 1] action to replica count [0, bound]
            sol = np.floor(np.clip(action, 0.0, 1.0) * bound).astype(np.int32)
        else:
            # Fallback uniform if untrained
            sol = np.full(env.num_services, max(1, bound // env.num_services), dtype=np.int32)

        solve_duration = (time.perf_counter() - start_time) * 1000.0  # ms
        return sol, solve_duration
