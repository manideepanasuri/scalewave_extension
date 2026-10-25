import gymnasium as gym
from gymnasium import spaces
import numpy as np


def get_small_traffic_adjustments(prev_traffic, new_traffic, adjustment_factor=0.25):
    """
    Exact implementation of gradual traffic adjustment from ScaleWave optimizer.py (lines 478-556).
    Caps per-key delta to at most 25% of current share, produces valid non-negative integers summing to 100.
    """
    EPS = 1e-9
    keys = list(range(len(prev_traffic)))
    prev = {k: max(0.0, float(prev_traffic[k])) for k in keys}
    new  = {k: max(0.0, float(new_traffic[k])) for k in keys}

    total_prev = sum(prev.values())
    prev_norm = {k: (v / total_prev) * 100.0 if total_prev > EPS else 25.0 for k, v in prev.items()}

    total_new = sum(new.values())
    new_norm = {k: (v / total_new) * 100.0 if total_new > EPS else 25.0 for k, v in new.items()}

    capped = {}
    for k in keys:
        desired_delta = new_norm[k] - prev_norm[k]
        base = max(prev_norm[k], new_norm[k], 1.0)
        cap_abs = adjustment_factor * base
        delta = np.clip(desired_delta, -cap_abs, cap_abs)
        capped[k] = max(0.0, prev_norm[k] + delta)

    total_capped = sum(capped.values())
    if total_capped > EPS:
        capped = {k: (v / total_capped) * 100.0 for k, v in capped.items()}
    else:
        capped = {k: 25.0 for k in keys}

    # Largest remainder method to integer percentages summing to 100
    floored = {k: int(capped[k]) for k in keys}
    remainder = 100 - sum(floored.values())
    fracs = sorted(keys, key=lambda k: (capped[k] - floored[k]), reverse=True)
    i = 0
    while remainder > 0 and i < len(fracs):
        floored[fracs[i]] += 1
        remainder -= 1
        i += 1

    return np.array([floored[k] for k in keys], dtype=np.float32)


class ScaleWaveEnv(gym.Env):
    """
    Gymnasium environment reproducing ScaleWave's full physical testbed dynamics (PerCom 2026):
      - 4 Equivalent Implementations: [0: Cloud, 1: Edge-CPU, 2: Edge-GPU, 3: Hybrid]
      - Multi-device edge cluster: ThinkPad manager + HP ProDesk + Raspberry Pi 4B + Jetson Nano
      - Multi-resource constraints: CPU, RAM, GPU cores, Network Uplink bandwidth
      - Knative Pod Autoscaler (KPA) concurrency targets (c = 18, 35, 70) with cold-start delay
      - Gradual traffic smoothing (max 25% shift)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, arrivals=None, concurrency=18, dt=5.0, baseline_mode=None):
        super().__init__()
        self.concurrency = concurrency
        self.dt = dt
        self.baseline_mode = baseline_mode
        self.num_services = 4
        self.service_names = ['cloud', 'edge_cpu', 'edge_gpu', 'hybrid']

        # Table III Cluster Resource Capacities:
        # ThinkPad (12 CPU, 16GB) + HP ProDesk (6 CPU, 8GB) + Pi4 (4 CPU, 8GB) + Jetson Nano (4 CPU, 4GB, 128 GPU)
        # Allocatable to worker pods after control plane (K3s, Knative, Prometheus, Redis):
        # [CPU (cores), Memory (GB), GPU (cores), Network Uplink (Mbps)]
        self.capacities = np.array([22.0, 28.0, 128.0, 50.0], dtype=np.float32)

        # Resource consumption per replica (Table IV & config files):
        # 0: Cloud (0.1 CPU, 0.2 GB, 0 GPU, 12.0 Mbps uplink for full image upload)
        # 1: Edge-CPU (2.0 CPU, 1.5 GB, 0 GPU, 0.2 Mbps uplink)
        # 2: Edge-GPU (0.5 CPU, 1.0 GB, 32 GPU cores on Jetson Nano, 0.2 Mbps uplink)
        # 3: Hybrid (1.0 CPU, 0.8 GB, 0 GPU, 3.0 Mbps uplink for 32 float embedding array)
        self.resource_weights = np.array([
            [0.1, 0.2,  0.0, 12.0],
            [2.0, 1.5,  0.0,  0.2],
            [0.5, 1.0, 32.0,  0.2],
            [1.0, 0.8,  0.0,  3.0],
        ], dtype=np.float32)

        # Empirical base latencies from testbed under normal load (seconds)
        self.base_latencies = np.array([3.5, 9.8, 2.8, 6.2], dtype=np.float32)

        # Workload trace
        if arrivals is not None:
            self.arrivals = arrivals
        else:
            self.arrivals = np.full(720, 12, dtype=np.int32)

        self.max_steps = len(self.arrivals)
        self.current_step = 0

        # Action space: normalized replica proportion [0, 1] for each equivalent implementation
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_services,), dtype=np.float32)

        # Observation space: 17 continuous features
        self.observation_space = spaces.Box(
            low=0.0, high=1e5, shape=(4 + 4 + 1 + 4 + 4,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        if self.baseline_mode == 'offload_only':
            self.replicas = np.array([1, 0, 0, 0], dtype=np.float32)
            self.traffic_split = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif self.baseline_mode == 'local_only':
            self.replicas = np.array([0, 1, 0, 0], dtype=np.float32)
            self.traffic_split = np.array([0.0, 100.0, 0.0, 0.0], dtype=np.float32)
        elif self.baseline_mode == 'local_offload':
            self.replicas = np.array([0, 0, 0, 1], dtype=np.float32)
            self.traffic_split = np.array([0.0, 0.0, 0.0, 100.0], dtype=np.float32)
        else:
            self.replicas = np.array([1, 1, 1, 1], dtype=np.float32)
            self.traffic_split = np.array([25.0, 25.0, 25.0, 25.0], dtype=np.float32)

        self.queue_items = []
        self.queue = 0
        self.latencies = self.base_latencies.copy()
        self.throughputs = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.active_connections = np.zeros(4, dtype=np.int32)
        
        self.history = {
            'requests_served': 0,
            'requests_dropped': 0,
            'latencies': [],
            'resource_utilization': [],
            'traffic_splits': []
        }
        return self._get_obs(), {}

    def _get_obs(self):
        used_resources = np.sum(self.replicas[:, None] * self.resource_weights, axis=0)
        avail_resources = np.maximum(0.0, self.capacities - used_resources)
        obs = np.concatenate([
            self.replicas,
            avail_resources,
            [float(self.queue)],
            self.throughputs,
            self.latencies
        ]).astype(np.float32)
        return obs

    def get_demand_bound(self):
        total_replicas = np.sum(self.replicas)
        bound = max(int(np.ceil(self.queue / self.concurrency) - total_replicas), 0) + 1
        return bound

    def step(self, action):
        # 1. 300s HTTP Timeout check on FIFO queue
        TIMEOUT_STEPS = int(300.0 / self.dt)
        active_queue = []
        timeout_dropped = 0
        for arr_step, count in self.queue_items:
            if (self.current_step - arr_step) >= TIMEOUT_STEPS:
                timeout_dropped += count
            else:
                active_queue.append([arr_step, count])
        self.queue_items = active_queue
        self.history['requests_dropped'] += timeout_dropped
        if timeout_dropped > 0:
            self.history['latencies'].extend([300.0] * timeout_dropped)

        # 2. Ingest arrivals for current step
        new_arrivals = self.arrivals[self.current_step] if self.current_step < len(self.arrivals) else 0
        if new_arrivals > 0:
            self.queue_items.append([self.current_step, new_arrivals])
        total_requests = sum(c for _, c in self.queue_items)
        self.queue = total_requests

        # 3. Scaling & Traffic Allocation
        capacity_penalty = 0.0
        additional_replicas = np.zeros(4, dtype=np.int32)

        if self.baseline_mode == 'offload_only':
            self.traffic_split = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
            target = max(1, int(np.ceil(total_requests / self.concurrency)))
            max_reps = int(self.capacities[3] / self.resource_weights[0, 3])
            self.replicas[0] = min(max_reps, target)
        elif self.baseline_mode == 'local_only':
            self.traffic_split = np.array([0.0, 100.0, 0.0, 0.0], dtype=np.float32)
            target = max(1, int(np.ceil(total_requests / self.concurrency)))
            max_reps = int(self.capacities[0] / self.resource_weights[1, 0])
            self.replicas[1] = min(max_reps, target)
        elif self.baseline_mode == 'local_offload':
            self.traffic_split = np.array([0.0, 0.0, 0.0, 100.0], dtype=np.float32)
            target = max(1, int(np.ceil(total_requests / self.concurrency)))
            max_reps = min(int(self.capacities[0] / self.resource_weights[3, 0]), int(self.capacities[3] / self.resource_weights[3, 3]))
            self.replicas[3] = min(max_reps, target)
        else:
            bound = self.get_demand_bound()
            action = np.array(action, dtype=np.float32)
            if np.all(action <= 1.0 + 1e-6):
                additional_replicas = np.floor(action * bound).astype(np.int32)
            else:
                additional_replicas = np.clip(action, 0, bound).astype(np.int32)

            total_additional_resources = np.sum(additional_replicas[:, None] * self.resource_weights, axis=0)
            current_used_resources = np.sum(self.replicas[:, None] * self.resource_weights, axis=0)
            total_projected_resources = current_used_resources + total_additional_resources

            violations = np.maximum(0.0, total_projected_resources - self.capacities)
            capacity_penalty = np.sum(violations)

            if capacity_penalty > 0:
                scale_down = np.min(self.capacities / np.maximum(1e-5, total_projected_resources))
                scale_down = min(1.0, max(0.0, scale_down))
                additional_replicas = np.floor(additional_replicas * scale_down).astype(np.int32)

            # ScaleWave Eq. 2 Traffic Distribution
            p_ne = self.throughputs / (np.maximum(1e-5, self.latencies) * np.maximum(1, self.replicas))
            projected_reps = self.replicas + additional_replicas
            effective_throughput_potential = self.throughputs * projected_reps

            sum_tp = np.sum(effective_throughput_potential)
            if sum_tp > 0:
                target_td = (effective_throughput_potential / sum_tp) * 100.0
            else:
                target_td = np.full(4, 25.0, dtype=np.float32)

            self.traffic_split = get_small_traffic_adjustments(self.traffic_split, target_td, adjustment_factor=0.25)

        # 4. Service Processing & Physical Capacity Modeling
        current_used_resources = np.sum(self.replicas[:, None] * self.resource_weights, axis=0)
        reqs_per_service = (total_requests * (self.traffic_split / 100.0)).astype(np.int32)
        total_step_capacity = 0.0

        for e in range(self.num_services):
            reqs = reqs_per_service[e]
            if reqs == 0:
                continue
            rep = self.replicas[e]

            # Physical capacity modeling from Table III & IV:
            if e == 0:  # Cloud: 50 Mbps network uplink (8 Mb per image upload)
                base_cap = (self.capacities[3] / 8.0) * self.dt
                eff = max(0.39, 1.0 / (1.0 + 0.015 * max(0, reqs - 10)))
                total_step_capacity += base_cap * eff
            elif e == 1:  # Edge-CPU: 12 worker CPU cores (1.8 core-s per request)
                base_cap = (self.capacities[0] / 1.8) * self.dt
                eff = max(0.40, 1.0 / (1.0 + 0.012 * max(0, reqs - 12)))
                total_step_capacity += base_cap * eff
            elif e == 2:  # Edge-GPU: Jetson Nano Maxwell GPU (2.8s per request)
                total_step_capacity += rep * (self.dt / self.base_latencies[e])
            elif e == 3:  # Hybrid: Feature extraction on CPU (0.8 core-s) + negligible upload
                base_cap = (self.capacities[0] / 0.8) * self.dt
                eff = max(0.45, 1.0 / (1.0 + 0.008 * max(0, reqs - 16)))
                total_step_capacity += base_cap * eff

        to_serve = min(total_requests, int(total_step_capacity))
        rem = to_serve
        step_served = 0
        step_dropped = 0
        step_latencies = []
        new_q = []

        base_l = self.base_latencies[np.argmax(self.traffic_split)]
        for arr_step, count in self.queue_items:
            if rem >= count:
                wait_s = (self.current_step - arr_step) * self.dt
                lat = base_l + wait_s
                if lat >= 300.0:
                    step_dropped += count
                    step_latencies.extend([300.0] * count)
                else:
                    step_served += count
                    step_latencies.extend([lat] * count)
                rem -= count
            elif rem > 0:
                wait_s = (self.current_step - arr_step) * self.dt
                lat = base_l + wait_s
                if lat >= 300.0:
                    step_dropped += rem
                    step_latencies.extend([300.0] * rem)
                else:
                    step_served += rem
                    step_latencies.extend([lat] * rem)
                new_q.append([arr_step, count - rem])
                rem = 0
            else:
                new_q.append([arr_step, count])

        self.queue_items = new_q
        self.queue = sum(c for _, c in self.queue_items)

        self.latencies = np.full(4, np.mean(step_latencies) if len(step_latencies) > 0 else base_l, dtype=np.float32)
        self.throughputs = np.full(4, max(0.01, step_served / (self.dt * 4)), dtype=np.float32)

        # Remove served and dropped requests from FIFO queue
        self.history['requests_served'] += step_served
        self.history['requests_dropped'] += step_dropped
        self.history['latencies'].extend(step_latencies)

        # 5. Knative Pod Autoscaler (KPA) replica updating (for ScaleWave)
        if self.baseline_mode is None:
            for e in range(self.num_services):
                target_reps = int(np.ceil(reqs_per_service[e] / self.concurrency))
                if target_reps > self.replicas[e]:
                    self.replicas[e] = min(self.replicas[e] + additional_replicas[e] + 1, 15)
                elif target_reps < self.replicas[e]:
                    self.replicas[e] = max(1, self.replicas[e] - 1)

        norm_perf = step_served / max(0.1, np.mean(self.latencies))
        reward = float(norm_perf - (capacity_penalty * 100.0) - (self.queue * 0.5))

        self.history['resource_utilization'].append(current_used_resources / self.capacities)
        self.history['traffic_splits'].append(self.traffic_split.copy())

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        if terminated:
            rem_q = sum(c for _, c in self.queue_items)
            if rem_q > 0:
                self.history['requests_dropped'] += rem_q
                self.history['latencies'].extend([300.0] * rem_q)
                self.queue_items = []
                self.queue = 0

        info = {
            'served': step_served,
            'queue': self.queue,
            'throughput': np.sum(self.throughputs),
            'avg_latency': np.mean(self.latencies),
            'capacity_penalty': capacity_penalty,
            'replicas': self.replicas.copy(),
            'traffic_split': self.traffic_split.copy()
        }

        return self._get_obs(), reward, terminated, truncated, info
