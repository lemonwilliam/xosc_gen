import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

class Visualization:
    def __init__(self, gt_csv_path, gen_csv_path):
        self.gt_df = pd.read_csv(gt_csv_path)
        self.gen_df = pd.read_csv(gen_csv_path)
        self.dynamic_agents = self._get_dynamic_agents()
        self.current_index = 0

    def _get_dynamic_agents(self):
        track_ids = self.gt_df["trackId"].unique()
        dynamic = []
        for tid in track_ids:
            gt_v = self.gt_df[self.gt_df["trackId"] == tid]["velocity"]
            if not np.allclose(gt_v, 0):
                dynamic.append(tid)
        return dynamic

    def _plot_agent_data(self, tid, axs):
        gt = self.gt_df[self.gt_df["trackId"] == tid]
        gen = self.gen_df[self.gen_df["trackId"] == tid]

        axs[0].cla()
        axs[0].plot(gt["time"], gt["x"], label="Ground Truth", color='blue')
        axs[0].plot(gen["time"], gen["x"], label="Generated", linestyle="--", color='orange')
        axs[0].set_title("X Coordinate")
        axs[0].legend()

        axs[1].cla()
        axs[1].plot(gt["time"], gt["y"], label="Ground Truth", color='blue')
        axs[1].plot(gen["time"], gen["y"], label="Generated", linestyle="--", color='orange')
        axs[1].set_title("Y Coordinate")
        axs[1].legend()

        axs[2].cla()
        axs[2].plot(gt["time"], gt["heading"], label="Ground Truth", color='blue')
        axs[2].plot(gen["time"], gen["heading"], label="Generated", linestyle="--", color='orange')
        axs[2].set_title("Heading")
        axs[2].legend()

        axs[3].cla()
        axs[3].plot(gt["time"], gt["velocity"], label="Ground Truth", color='blue')
        axs[3].plot(gen["time"], gen["velocity"], label="Generated", linestyle="--", color='orange')
        axs[3].set_title("Velocity")
        axs[3].legend()

        for ax in axs:
            ax.set_xlabel("Time")

        fig_title = f"Agent {tid} Comparison"
        return fig_title

    def interactive_view(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()

        ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03])
        slider = Slider(ax_slider, 'Agent Index', 0, len(self.dynamic_agents) - 1,
                        valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            tid = self.dynamic_agents[idx]
            title = self._plot_agent_data(tid, axs)
            fig.suptitle(title)
            fig.canvas.draw_idle()

        update(0)  # initial plot
        slider.on_changed(update)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
