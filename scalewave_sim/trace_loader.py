import os
import zipfile
import csv
import numpy as np


class TraceLoader:
    """
    Unified multi-dataset trace loader supporting standard edge and serverless workloads:
      1. 'shanghai': Shanghai Telecom cellular base station user trace (ScaleWave baseline)
      2. 'azure': Microsoft Azure Functions 2019 production invocation trace (ASPLOS 2020)
      3. 'nasa': NASA Kennedy Space Center HTTP flash-crowd access log (ITA Archive)
    """
    def __init__(self, trace_zip_path=None, data_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if trace_zip_path is None:
            trace_zip_path = os.path.join(base_dir, "ScaleWave", "evaluations", "dataset", "trace.zip")
        self.trace_zip_path = trace_zip_path
        
        if data_dir is None:
            data_dir = os.path.join(base_dir, "data", "processed_traces")
        self.data_dir = data_dir

    def load_dataset(self, name="shanghai", duration_seconds=3600, dt=5.0):
        """
        Loads a pre-processed 1-hour arrival array for the given dataset name.
        """
        name = name.lower().strip()
        npy_path = os.path.join(self.data_dir, f"{name}_arrivals.npy")
        if os.path.exists(npy_path):
            arrivals = np.load(npy_path)
            total_steps = int(np.ceil(duration_seconds / dt))
            if len(arrivals) >= total_steps:
                return arrivals[:total_steps].copy()
            else:
                padded = np.zeros(total_steps, dtype=np.int32)
                padded[:len(arrivals)] = arrivals
                return padded

        if name == "shanghai":
            return self.load_trace(duration_seconds=duration_seconds, dt=dt)
        else:
            raise ValueError(f"Unknown dataset '{name}'. Available: ['shanghai', 'azure', 'nasa']")

    def load_trace(self, max_files=None, duration_seconds=3600, dt=5.0):
        """
        Loads all user trace CSV files from the Shanghai Telecom dataset and bins every arrival event
        into dt-sized intervals over duration_seconds.
        """
        total_steps = int(np.ceil(duration_seconds / dt))
        arrivals = np.zeros(total_steps, dtype=np.int32)

        if not os.path.exists(self.trace_zip_path):
            raise FileNotFoundError(f"Trace zip file not found at {self.trace_zip_path}")

        with zipfile.ZipFile(self.trace_zip_path, 'r') as z:
            file_list = [f for f in z.namelist() if f.endswith('.csv')]
            if max_files is not None:
                file_list = file_list[:max_files]

            for fname in file_list:
                try:
                    with z.open(fname) as csv_file:
                        lines = [line.decode('utf-8') for line in csv_file.readlines()]
                        reader = csv.DictReader(lines)
                        current_time = 0.0
                        for row in reader:
                            if 'inter_arrival_time' in row:
                                try:
                                    iat = float(row['inter_arrival_time'])
                                    current_time += iat
                                    step_idx = int(current_time // dt)
                                    if 0 <= step_idx < total_steps:
                                        arrivals[step_idx] += 1
                                    elif step_idx >= total_steps:
                                        break
                                except ValueError:
                                    continue
                except Exception:
                    continue

        return arrivals
