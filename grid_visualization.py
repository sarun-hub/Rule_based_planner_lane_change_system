# This is static animation visualization, cannot do real time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.pygame_config import distance_range, rel_speed_range, grid_resolution
from utils.coverage_utils import CellCoverageModel
from utils.reference_genertor import TrajectoryGenerator
from utils.utils import convert_state3d_to_state2d


class GridVisualizer:
    def __init__(self, update_interval=100):
        self.update_interval = update_interval
        self.cell_coverage_model = CellCoverageModel(
            distance_range, rel_speed_range, grid_resolution
        )
        self.traj_generator = TrajectoryGenerator()
        self.initialize_plot()

    def load_data(self, d, vp, vf):
        """only one time load"""
        self.d = d
        self.vp = vp
        self.vf = vf
        self.states_3d = [(d_, vp_, vf_) for d_, vp_, vf_ in zip(d, vp, vf)]
        self.states_2d = [
            convert_state3d_to_state2d((d_, vp_, vf_))
            for d_, vp_, vf_ in zip(d, vp, vf)
        ]

    def initialize_plot(self):
        self.fig, self.ax = plt.subplots()

        self.x_min = distance_range[0]
        self.x_max = distance_range[1]
        self.y_min = rel_speed_range[0]
        self.y_max = rel_speed_range[1]

        self.distance_grid_size = (self.x_max - self.x_min) / grid_resolution[0]
        self.rel_speed_grid_size = (self.y_max - self.y_min) / grid_resolution[1]

        self.ax.set_title(f"Car 1 Data -> Coverage 0%")
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Relative Velocity (m/s)")
        self.ax.set_xlim(*distance_range)
        self.ax.set_ylim(*rel_speed_range)
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Relative Velocity (m/s)")

        self.rectangles = {}

        # Draw grid
        for i in range(grid_resolution[0]):
            for j in range(grid_resolution[1]):
                # Determine cell position
                distance = distance_range[0] + i * self.distance_grid_size
                rel_speed = rel_speed_range[0] + j * self.rel_speed_grid_size

                # Create grid cell
                rect = plt.Rectangle(
                    (distance, rel_speed),
                    self.distance_grid_size,
                    self.rel_speed_grid_size,
                    edgecolor="black",
                    facecolor="none",
                )

                self.ax.add_patch(rect)
                self.rectangles[(i, j)] = rect

        # Initialize the scatter points for illustrating state
        self.current_state_scatter = self.ax.scatter([], [], color="black",label="current_state")
        self.previous_state_scatter = self.ax.scatter([], [], color="grey", alpha=0.5,label="trace")

        self.current_target_scatter = self.ax.scatter([], [], color="red",label="current target")
        self.ax.legend()

    def update(self, frame):
        state_3d = self.states_3d[frame]
        state_2d = self.states_2d[frame]

        current_cell_status = self.cell_coverage_model.get_current_coverage_status()
        current_target = self.traj_generator.propose(state_3d, current_cell_status)

        i, j = self.cell_coverage_model.state_to_grid(*state_2d)
        rect = self.rectangles[(i, j)]
        rect.set_facecolor("lightblue")
        rect.set_alpha(0.5)

        self.current_state_scatter.set_offsets([*state_2d])
        # For not the first step -> show all previous step
        if frame > 0:
            previous_state = np.array(self.states_2d[:frame])
        else:
            previous_state = np.empty(
                (0, 2)
            )  # Empty array with shape (0,2) - no row, 2 columns

        self.previous_state_scatter.set_offsets(previous_state)

        self.current_target_scatter.set_offsets(
            [convert_state3d_to_state2d(current_target)]
        )

        coverage = self.cell_coverage_model.get_coverage() * 100
        self.ax.set_title(f"Car 1 Data → Coverage {coverage:.1f}%")

        self.cell_coverage_model.update(state_3d)

    def run(self):
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.d),
            interval=self.update_interval,  # ms
            blit=False,
            save_count=50,
        )
        plt.show()


if __name__ == "__main__":
    # only one time
    data = pd.read_csv("output/Car_1_data.csv")

    d = data["longitudinal_distance"].values
    vp = data["longitudinal_ego_velocity"].values
    vf = data["longitudinal_velocity"].values

    visualizer = GridVisualizer(update_interval=100)
    visualizer.load_data(d, vp, vf)
    visualizer.run()
