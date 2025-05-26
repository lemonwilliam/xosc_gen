import pandas as pd
import subprocess
import numpy as np
import os


class EsminiSimulator:
    def __init__(self, esmini_bin="./esmini/bin/esmini", dat2csv_script="./esmini/scripts/dat2csv.py"):
        self.esmini_bin = esmini_bin
        self.dat2csv_script = dat2csv_script

    def run(self, xosc_path, record_path, x_offset, y_offset, track_id_mapping=None):
        """
        Run Esmini simulation on the given OpenSCENARIO file, convert the .dat to .csv, 
        and apply coordinate normalization using the given UTM offsets. Optionally remap track IDs.
        
        Args:
            xosc_path (str): Path to OpenSCENARIO file.
            record_path (str): Output path for the simulation .dat file.
            x_offset (float): X UTM offset.
            y_offset (float): Y UTM offset.
            track_id_mapping (dict): Optional mapping {generated_track_id: ground_truth_track_id}.
        """
        os.makedirs(os.path.dirname(record_path), exist_ok=True)

        # Step 1: Run simulation
        subprocess.run([
            self.esmini_bin,
            "--window", "60", "60", "800", "400",
            "--osc", xosc_path,
            "--record", record_path
        ], check=True)

        # Step 2: Convert .dat to .csv using dat2csv.py
        print(f"Running: python {self.dat2csv_script} {record_path}")
        result = subprocess.run(
            ["python3", self.dat2csv_script, record_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("❌ dat2csv.py failed:\n", result.stderr)
            raise RuntimeError("dat2csv.py failed to execute")
        else:
            print("✅ dat2csv.py completed:\n", result.stdout)

        # Step 3: Post-process CSV output
        csv_path = record_path.replace(".dat", ".csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Apply track ID mapping if provided
        track_ids = df[" id"]
        if track_id_mapping:
            df[" id"] = track_ids.map(track_id_mapping)
        else:
            df[" id"] = track_ids  # Default, no remapping

        df_final = pd.DataFrame({
            "trackId": df[" id"],
            "time": df["time"],
            "x": (df[" x"] - x_offset).round(3),
            "y": (df[" y"] - y_offset).round(3),
            "heading": np.degrees(df[" h"]).round(3),
            "velocity": df[" speed"]
        })

        df_final = df_final.sort_values(by=["trackId", "time"])
        df_final.to_csv(csv_path, index=False)
        print(f"✅ Processed and saved to {csv_path}")
