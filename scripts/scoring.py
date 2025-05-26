import pandas as pd
import numpy as np

class Scorer:
    def __init__(self):
        pass

    def is_static(self, traj_df):
        """
        Return True if the agent's velocity is zero for all timesteps.
        """
        return np.allclose(traj_df['velocity'].values, 0.0)

    def compute_fde(self, gt_csv_path, gen_csv_path):
        """
        Compute the Final Displacement Error (FDE) between ground truth and generated trajectories.
        Returns the average FDE across all trackIds.
        """
        gt_df = pd.read_csv(gt_csv_path)
        gen_df = pd.read_csv(gen_csv_path)

        fde_list = []

        for track_id in gt_df['trackId'].unique():
            gt_traj = gt_df[gt_df['trackId'] == track_id].sort_values('time')
            gen_traj = gen_df[gen_df['trackId'] == track_id].sort_values('time')

            if gt_traj.empty or gen_traj.empty or self.is_static(gt_traj):
                continue

            # Final positions
            gt_final = gt_traj.iloc[-1][['x', 'y']].values
            gen_final = gen_traj.iloc[-1][['x', 'y']].values

            # Euclidean distance
            fde = np.linalg.norm(gt_final - gen_final)
            fde_list.append(fde)

        mean_fde = np.mean(fde_list) if fde_list else float('nan')
        return mean_fde
