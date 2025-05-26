import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import os
import subprocess
import openai

class Reflection:
    def __init__(self, esmini_dir=None, api_key=None):
        self.esmini_dir = esmini_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided either directly or through the 'OPENAI_API_KEY' environment variable.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def run_esmini_and_convert(self, xosc_path, output_record_path):
        """
        Runs esmini to simulate a scenario and convert the output .dat file to a .csv.
        """
        esmini_bin = os.path.join(self.esmini_dir, "esmini")
        dat2csv_bin = os.path.join(self.esmini_dir, "dat2csv")

        # Step 1: Run simulation
        esmini_command = [
            esmini_bin,
            "--window", "60", "60", "800", "400",
            "--osc", xosc_path,
            "--fixed_timestep", "0.2",
            "--record", output_record_path
        ]
        print("Running esmini simulation...")
        subprocess.run(esmini_command, check=True)

        # Step 2: Convert to CSV
        dat2csv_command = [dat2csv_bin, output_record_path]
        print("Converting .dat to .csv...")
        subprocess.run(dat2csv_command, check=True)

        print("Simulation and conversion complete.")

    def compute_similarity(self, original_csv_path, result_csv_path):
        """
        Merges the original and Esmini trajectories on time,
        then computes position error, heading error, speed error,
        Pearson correlation, and a final similarity score.
        """
        original_df = pd.read_csv(original_csv_path)
        temp_df = pd.read_csv(result_csv_path)
        result_df = temp_df[[" id", "time", " x", " y", " h", " speed"]].copy()
        result_df.columns = ["trackId", "time", "x", "y", "heading", "speed"]
        result_df = result_df.sort_values(by=["trackId", "time"])

        merged = pd.merge(original_df, result_df, on='time', suffixes=('_orig', '_gen'))

        position_diff = merged.apply(
            lambda row: euclidean((row['x_orig'], row['y_orig']), (row['x_gen'], row['y_gen'])), axis=1
        )

        heading_diff = np.abs(merged['heading_orig'] - merged['heading_gen'])
        speed_diff = np.abs(merged['speed_orig'] - merged['speed_gen'])

        avg_position_error = position_diff.mean()
        avg_heading_error = heading_diff.mean()
        avg_speed_error = speed_diff.mean()

        position_corr_x = pearsonr(merged['x_orig'], merged['x_gen'])[0]
        position_corr_y = pearsonr(merged['y_orig'], merged['y_gen'])[0]

        similarity_score = (1 / (1 + avg_position_error + avg_heading_error + avg_speed_error)) * \
                           ((position_corr_x + position_corr_y) / 2)

        return {
            "avg_position_error": avg_position_error,
            "avg_heading_error": avg_heading_error,
            "avg_speed_error": avg_speed_error,
            "position_corr_x": position_corr_x,
            "position_corr_y": position_corr_y,
            "similarity_score": similarity_score
        }

    def suggest_improvements_with_gpt(self, original_csv_path, result_csv_path):
        """
        Uses the OpenAI o3-mini model to analyze the original and generated trajectories,
        and returns suggestions to make the generated trajectory more similar.
        """
        try:
            with open(original_csv_path, 'r') as f:
                original_data = f.read()
            with open(result_csv_path, 'r') as f:
                result_data = f.read()

            messages = [
                {"role": "system", "content": "You are a traffic simulation expert skilled at analyzing and comparing trajectory data."},
                {"role": "user", "content": f"Compare the following original trajectory with the generated one and suggest how the generated trajectory can be improved to match the original more closely.\n\nOriginal Trajectory CSV:\n{original_data}\n\nGenerated Trajectory CSV:\n{result_data}\n\nPlease return your suggestions in natural language."}
            ]

            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=messages
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"An error occurred while generating suggestions: {e}"
