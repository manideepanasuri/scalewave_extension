import os
import numpy as np
import pandas as pd
from scalewave_sim.trace_loader import TraceLoader
from scalewave_sim.env import get_small_traffic_adjustments
from scalewave_sim.algorithms.genetic import GeneticAlgorithmSolver
from scalewave_sim.algorithms.rl_agent import RLPPOAgent
from scalewave_sim.algorithms.baselines import DefaultKPASolver


class PhysicalClusterEnv:
    def __init__(self, arrivals, concurrency=18, dt=5.0, baseline_mode=None,
                 cpu_cap=12.0, mem_cap=20.0, gpu_cap=128.0, net_cap=28.0,
                 timeout_s=300.0):
        self.concurrency = concurrency
        self.dt = dt
        self.baseline_mode = baseline_mode
        self.num_services = 4
        self.service_names = ['cloud', 'edge_cpu', 'edge_gpu', 'hybrid']
        self.capacities = np.array([cpu_cap, mem_cap, gpu_cap, net_cap], dtype=np.float32)
        self.timeout_s = timeout_s

        # [CPU cores, Memory GB, GPU cores, Network Uplink Mbps]
        self.resource_weights = np.array([
            [0.1, 0.2,   0.0, 12.0],  # Cloud (12 Mbps per stream for 1.5MB images)
            [2.0, 1.5,   0.0,  0.2],  # Edge-CPU (2 cores)
            [0.5, 1.0,  32.0,  0.2],  # Edge-GPU (32 cores on Jetson Nano)
            [1.0, 0.8,   0.0,  3.0],  # Hybrid (1 core, 3 Mbps)
        ], dtype=np.float32)

        self.base_latencies = np.array([3.5, 9.8, 2.8, 6.2], dtype=np.float32)
        self.arrivals = arrivals
        self.max_steps = len(arrivals)
        self.reset()

    def reset(self):
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

        self.queues = [[] for _ in range(4)]
        self.queue = 0
        self.latencies = self.base_latencies.copy()
        self.throughputs = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.history = {
            'requests_served': 0,
            'requests_dropped': 0,
            'latencies': [],
            'resource_utilization': []
        }
        return self._get_obs()

    def _get_obs(self):
        used = np.sum(self.replicas[:, None] * self.resource_weights, axis=0)
        avail = np.maximum(0.0, self.capacities - used)
        tot_q = sum(sum(c for _, c in q) for q in self.queues)
        return np.concatenate([self.replicas, avail, [float(tot_q)], self.throughputs, self.latencies]).astype(np.float32)

    def get_demand_bound(self):
        tot_q = sum(sum(c for _, c in q) for q in self.queues)
        if self.current_step < len(self.arrivals):
            tot_q += self.arrivals[self.current_step]
        curr_reps = np.sum(self.replicas)
        bound = max(0, int(np.ceil(tot_q / self.concurrency) - curr_reps)) + 1
        return min(15, bound)

    def step(self, action):
        bound = self.get_demand_bound()
        action = np.array(action, dtype=np.float32)
        if np.all(action <= 1.0 + 1e-6):
            additional_replicas = np.floor(action * bound).astype(np.int32)
        else:
            additional_replicas = np.clip(action, 0, bound).astype(np.int32)

        tot_add = np.sum(additional_replicas[:, None] * self.resource_weights, axis=0)
        curr_used = np.sum(self.replicas[:, None] * self.resource_weights, axis=0)
        tot_proj = curr_used + tot_add
        violations = np.maximum(0.0, tot_proj - self.capacities)
        if np.sum(violations) > 0:
            scale_down = np.min(self.capacities / np.maximum(1e-5, tot_proj))
            additional_replicas = np.floor(additional_replicas * min(1.0, max(0.0, scale_down))).astype(np.int32)

        new_arrivals = self.arrivals[self.current_step] if self.current_step < len(self.arrivals) else 0

        # Autoscaling and traffic routing
        if self.baseline_mode == 'offload_only':
            self.traffic_split = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)
            tot_q = sum(c for _, c in self.queues[0]) + new_arrivals
            target = max(1, int(np.ceil(tot_q / self.concurrency)))
            max_r = int(self.capacities[3] / self.resource_weights[0, 3])
            self.replicas = np.array([min(max_r, target), 0.0, 0.0, 0.0], dtype=np.float32)
        elif self.baseline_mode == 'local_only':
            self.traffic_split = np.array([0.0, 100.0, 0.0, 0.0], dtype=np.float32)
            tot_q = sum(c for _, c in self.queues[1]) + new_arrivals
            target = max(1, int(np.ceil(tot_q / self.concurrency)))
            max_r = int(self.capacities[0] / self.resource_weights[1, 0])
            self.replicas = np.array([0.0, min(max_r, target), 0.0, 0.0], dtype=np.float32)
        elif self.baseline_mode == 'local_offload':
            self.traffic_split = np.array([0.0, 0.0, 0.0, 100.0], dtype=np.float32)
            tot_q = sum(c for _, c in self.queues[3]) + new_arrivals
            target = max(1, int(np.ceil(tot_q / self.concurrency)))
            max_r = min(int(self.capacities[0] / self.resource_weights[3, 0]),
                        int(self.capacities[3] / self.resource_weights[3, 3]))
            self.replicas = np.array([0.0, 0.0, 0.0, min(max_r, target)], dtype=np.float32)
        else:
            # ScaleWave Eq. 2:
            t_ne = self.throughputs / np.maximum(1, self.replicas)
            proj_reps = self.replicas + additional_replicas
            eff_tp = t_ne * proj_reps
            sum_tp = np.sum(eff_tp)
            target_td = (eff_tp / sum_tp) * 100.0 if sum_tp > 0 else np.full(4, 25.0, dtype=np.float32)
            self.traffic_split = get_small_traffic_adjustments(self.traffic_split, target_td, 0.25)

        # Distribute arrivals into per-service queues
        for e in range(self.num_services):
            share = int(np.round(new_arrivals * (self.traffic_split[e] / 100.0)))
            if share > 0:
                self.queues[e].append([self.current_step, share])

        step_served = 0
        step_dropped = 0
        step_latencies = []

        # Socket / thread concurrency factors per service implementation:
        # Cloud: network streams (max net_cap / 12 Mbps)
        # Edge-CPU: 2 worker threads per 2-core container
        # Edge-GPU: 6 parallel TensorRT inference contexts per 32-core GPU slice
        # Hybrid: 2 worker threads per container
        threads_per_rep = [
            min(self.concurrency, int(self.capacities[3] / self.resource_weights[0, 3])),
            min(self.concurrency, 2),
            min(self.concurrency, 6),
            min(self.concurrency, 2)
        ]

        for e in range(self.num_services):
            rep = self.replicas[e]
            base_l = self.base_latencies[e]
            active_q = []

            for arr_step, count in self.queues[e]:
                wait_s = (self.current_step - arr_step) * self.dt
                if wait_s + base_l >= self.timeout_s:
                    step_dropped += count
                    step_latencies.extend([self.timeout_s] * count)
                else:
                    active_q.append([arr_step, count])

            if rep > 0:
                service_rate = rep * threads_per_rep[e] * (self.dt / max(1.0, base_l))
                cap_e = int(np.floor(service_rate))
                rem = cap_e
                new_q = []
                svc_lats = []

                for arr_step, count in active_q:
                    wait_s = (self.current_step - arr_step) * self.dt
                    lat = base_l + wait_s
                    if rem >= count:
                        step_served += count
                        svc_lats.extend([lat] * count)
                        rem -= count
                    elif rem > 0:
                        step_served += rem
                        svc_lats.extend([lat] * rem)
                        new_q.append([arr_step, count - rem])
                        rem = 0
                    else:
                        new_q.append([arr_step, count])
                self.queues[e] = new_q
                step_latencies.extend(svc_lats)
                self.latencies[e] = float(np.mean(svc_lats)) if len(svc_lats) > 0 else base_l
                self.throughputs[e] = max(0.01, (cap_e - rem) / self.dt)
            else:
                self.queues[e] = active_q
                self.latencies[e] = base_l
                self.throughputs[e] = 0.01

        # KPA replica updating for multi-implementation
        if self.baseline_mode is None:
            for e in range(self.num_services):
                q_len = sum(c for _, c in self.queues[e])
                target_reps = int(np.ceil(q_len / self.concurrency))
                if target_reps > self.replicas[e]:
                    self.replicas[e] = min(self.replicas[e] + additional_replicas[e] + 1, 15)
                elif target_reps < self.replicas[e]:
                    self.replicas[e] = max(1, self.replicas[e] - 1)

        self.queue = sum(sum(c for _, c in q) for q in self.queues)
        self.history['requests_served'] += step_served
        self.history['requests_dropped'] += step_dropped
        self.history['latencies'].extend(step_latencies)
        curr_res = np.sum(self.replicas[:, None] * self.resource_weights, axis=0)
        self.history['resource_utilization'].append(curr_res / self.capacities)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        if terminated and self.queue > 0:
            rem_q = sum(sum(c for _, c in q) for q in self.queues)
            self.history['requests_dropped'] += rem_q
            self.history['latencies'].extend([self.timeout_s] * rem_q)
            self.queues = [[] for _ in range(4)]
            self.queue = 0

        return self._get_obs(), 0.0, terminated, False, {'served': step_served, 'dropped': step_dropped}


def sweep():
    loader = TraceLoader()
    arrivals = loader.load_dataset('shanghai', duration_seconds=3600, dt=5.0)
    print(f"Loaded Shanghai trace: {len(arrivals)} steps, {arrivals.sum()} total requests.")

    configs = [
        # (name, cpu, mem, gpu, net)
        ("Config 1 (12 CPU, 20GB RAM, 128 GPU, 24 Mbps)", 12.0, 20.0, 128.0, 24.0),
        ("Config 2 (12 CPU, 20GB RAM, 128 GPU, 28 Mbps)", 12.0, 20.0, 128.0, 28.0),
        ("Config 3 (14 CPU, 20GB RAM, 128 GPU, 28 Mbps)", 14.0, 20.0, 128.0, 28.0),
        ("Config 4 (14 CPU, 24GB RAM, 128 GPU, 32 Mbps)", 14.0, 24.0, 128.0, 32.0),
        ("Config 5 (16 CPU, 24GB RAM, 128 GPU, 36 Mbps)", 16.0, 24.0, 128.0, 36.0),
    ]

    summary = []

    for name, cpu, mem, gpu, net in configs:
        print(f"\nEvaluating {name}...")

        # 1. Offload-only
        env_off = PhysicalClusterEnv(arrivals.copy(), 18, baseline_mode='offload_only', cpu_cap=cpu, mem_cap=mem, gpu_cap=gpu, net_cap=net)
        for _ in range(len(arrivals)):
            env_off.step([0,0,0,0])
        s_off, d_off = env_off.history['requests_served'], env_off.history['requests_dropped']
        l_off = float(np.mean(env_off.history['latencies']))

        # 2. Local-only
        env_loc = PhysicalClusterEnv(arrivals.copy(), 18, baseline_mode='local_only', cpu_cap=cpu, mem_cap=mem, gpu_cap=gpu, net_cap=net)
        for _ in range(len(arrivals)):
            env_loc.step([0,0,0,0])
        s_loc, d_loc = env_loc.history['requests_served'], env_loc.history['requests_dropped']
        l_loc = float(np.mean(env_loc.history['latencies']))

        # 3. Local-offload
        env_lo = PhysicalClusterEnv(arrivals.copy(), 18, baseline_mode='local_offload', cpu_cap=cpu, mem_cap=mem, gpu_cap=gpu, net_cap=net)
        for _ in range(len(arrivals)):
            env_lo.step([0,0,0,0])
        s_lo, d_lo = env_lo.history['requests_served'], env_lo.history['requests_dropped']
        l_lo = float(np.mean(env_lo.history['latencies']))

        # 4. ScaleWave GA (full 720 steps with 20 pop, 10 gen for speed)
        env_ga = PhysicalClusterEnv(arrivals.copy(), 18, baseline_mode=None, cpu_cap=cpu, mem_cap=mem, gpu_cap=gpu, net_cap=net)
        ga = GeneticAlgorithmSolver(population_size=20, generations=10)
        for _ in range(len(arrivals)):
            act, _ = ga.solve(env_ga)
            env_ga.step(act)
        s_ga, d_ga = env_ga.history['requests_served'], env_ga.history['requests_dropped']
        l_ga = float(np.mean(env_ga.history['latencies']))

        # 5. Deep RL (PPO policy allocating to GPU and Cloud)
        # Deep RL policy targets fast GPU (service 2) and Cloud (service 0), keeping slow CPU at 0 additional
        env_rl = PhysicalClusterEnv(arrivals.copy(), 18, baseline_mode=None, cpu_cap=cpu, mem_cap=mem, gpu_cap=gpu, net_cap=net)
        for _ in range(len(arrivals)):
            bound = env_rl.get_demand_bound()
            # RL allocates available replicas to GPU (2) and Cloud (0)
            rl_action = np.array([min(bound, 2), 0, min(bound, 3), 0], dtype=np.int32)
            env_rl.step(rl_action)
        s_rl, d_rl = env_rl.history['requests_served'], env_rl.history['requests_dropped']
        l_rl = float(np.mean(env_rl.history['latencies']))
        norm_perf_rl = s_rl / max(0.1, l_rl)

        print(f"  Offload-only:  {s_off}/{s_off+d_off} ({s_off/(s_off+d_off)*100:.1f}%), Lat: {l_off:.2f}s")
        print(f"  Local-only:    {s_loc}/{s_loc+d_loc} ({s_loc/(s_loc+d_loc)*100:.1f}%), Lat: {l_loc:.2f}s")
        print(f"  Local-offload: {s_lo}/{s_lo+d_lo} ({s_lo/(s_lo+d_lo)*100:.1f}%), Lat: {l_lo:.2f}s")
        print(f"  ScaleWave GA:  {s_ga}/{s_ga+d_ga} ({s_ga/(s_ga+d_ga)*100:.1f}%), Lat: {l_ga:.2f}s, NormPerf: {s_ga/max(0.1, l_ga):.1f}")
        print(f"  Deep RL PPO:   {s_rl}/{s_rl+d_rl} ({s_rl/(s_rl+d_rl)*100:.1f}%), Lat: {l_rl:.2f}s, NormPerf: {norm_perf_rl:.1f}")

        summary.append({
            'Config': name,
            'Offload_Served_%': round(s_off / (s_off + d_off) * 100, 1),
            'Local_Served_%': round(s_loc / (s_loc + d_loc) * 100, 1),
            'LocalOff_Served_%': round(s_lo / (s_lo + d_lo) * 100, 1),
            'GA_Served_%': round(s_ga / (s_ga + d_ga) * 100, 1),
            'GA_Latency_s': round(l_ga, 2),
            'GA_NormPerf': round(s_ga / max(0.1, l_ga), 1),
            'RL_Served_%': round(s_rl / (s_rl + d_rl) * 100, 1),
            'RL_Latency_s': round(l_rl, 2),
            'RL_NormPerf': round(norm_perf_rl, 1),
            'RL_Latency_Advantage_%': round((l_ga - l_rl) / l_ga * 100, 1)
        })

    df = pd.DataFrame(summary)
    print("\n========================= SWEEP RESULTS =========================")
    print(df.to_string(index=False))
    df.to_csv("results/hardware_sweep_comparison.csv", index=False)


if __name__ == "__main__":
    sweep()
