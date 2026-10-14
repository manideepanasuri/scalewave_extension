import os
import zipfile
import csv
import numpy as np


class TraceLoader:
    """
    Loads and aggregates the entire Shanghai Telecom trace dataset (all 610 user files)
    simulating concurrent edge user request generation over a 1-hour period.
    """
    def __init__(self, trace_zip_path=None):
        if trace_zip_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            trace_zip_path = os.path.join(base_dir, "ScaleWave", "evaluations", "dataset", "trace.zip")
        self.trace_zip_path = trace_zip_path

    def load_trace(self, max_files=None, duration_seconds=3600, dt=5.0):
        """
        Loads all user trace CSV files from the dataset and bins every arrival event
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
