import numpy as np
from typing import Tuple
from utils.utils import convert_state3d_to_state2d, convert_state2d_to_state3d_pass_affine

# ================== Reference Generator ================


class BaseGenerator:
    pass


class TrajectoryGenerator:
    def __init__(self):
        pass

    def find_closest_empty_cell(self, state_3d, cov_status, tolerance: float = 1e-6):
        empty_cells = cov_status.empty_cells
        distance_range = cov_status.distance_range
        rel_speed_range = cov_status.rel_speed_range
        distance_grid_size = cov_status.distance_grid_size
        rel_speed_grid_size = cov_status.rel_speed_grid_size

        if not empty_cells:
            print("The state-space graph is fully covered.")
            return []

        # find the representative of each cell which is the center of cell
        empty_representatives = []
        for cell in empty_cells:
            i, j = cell
            distance = distance_range[0] + (i + 1 / 2) * distance_grid_size
            rel_speed = rel_speed_range[0] + (j + 1 / 2) * rel_speed_grid_size
            empty_representatives.append((distance, rel_speed))

        current_distance, current_rel_speed = convert_state3d_to_state2d(state_3d)

        # Calculate distances to all empty cells
        distances = [
            np.sqrt(
                (cell[0] - current_distance) ** 2 + (cell[1] - current_rel_speed) ** 2
            )
            for cell in empty_representatives
        ]

        min_distance = np.min(distances)

        closest_cells = [
            cell
            for cell, dist in zip(empty_representatives, distances)
            if abs(dist - min_distance) <= tolerance
        ]

        return closest_cells

    def propose(
        self, state_3d, cov_status, tolerance=1e-6
    ) -> Tuple[float, float, float]:
        closest_uncovered_points = self.find_closest_empty_cell(
            state_3d, cov_status, tolerance
        )
        if closest_uncovered_points != []:
            print(f"Return target: {convert_state2d_to_state3d_pass_affine(closest_uncovered_points[0])}")
            return convert_state2d_to_state3d_pass_affine(closest_uncovered_points[0])
        else:
            print("There is no empty cells left")
            return None
