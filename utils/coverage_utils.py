from typing import Tuple, List
import numpy as np

from dataclasses import dataclass
from utils.utils import convert_state3d_to_state2d


# This file is for putting the coverage assessment model
@dataclass
class CoverageStatus:
    empty_cells: List[Tuple[float, float]]
    distance_range: Tuple[float, float]
    rel_speed_range: Tuple[float, float]
    distance_grid_size: float
    rel_speed_grid_size: float


class CellCoverageModel:
    def __init__(
        self,
        distance_range: Tuple[float, float],
        rel_speed_range: Tuple[float, float],
        grid_resolution: Tuple[float, float],
    ):
        self.distance_range = distance_range
        self.rel_speed_range = rel_speed_range
        self.grid_resolution = grid_resolution

        # Calculate grid cell size
        self.distance_grid_size = (
            distance_range[1] - distance_range[0]
        ) / grid_resolution[0]
        self.rel_speed_grid_size = (
            rel_speed_range[1] - rel_speed_range[0]
        ) / grid_resolution[1]

        # Initialize state-space grid (0 for empty, 1 for filled)
        self.states = []
        self.state_space = np.zeros((self.grid_resolution[0], self.grid_resolution[1]))

    def check_cover(self, state_3d: Tuple[float, float, float]) -> bool:
        # Function that check if the given state is already covered or not
        distance, rel_speed = convert_state3d_to_state2d(state_3d)
        i, j = self.state_to_grid(distance, rel_speed)
        return self.state_space[i, j] == 1
    
    def get_covered_cells(self):
        covered_indices = np.where(self.state_space == 1)

        if covered_indices[0].size == 0:
            return []

        return list(zip(*covered_indices))


    def update(self, state_3d: Tuple[float, float, float]):
        # Function that update the covered cells
        
        distance, rel_speed = convert_state3d_to_state2d(state_3d)
        i, j = self.state_to_grid(distance, rel_speed)
        # Add the history
        self.states.append((distance, rel_speed))
        # Update state-space
        self.state_space[i, j] = 1

    def get_current_coverage_status(self):
        empty_indices = np.where(self.state_space == 0)

        # as (row, col) pairs
        empty_cells = list(zip(empty_indices[0], empty_indices[1]))
        cov = CoverageStatus(
            empty_cells,
            self.distance_range,
            self.rel_speed_range,
            self.distance_grid_size,
            self.rel_speed_grid_size,
        )

        return cov
    
    def get_coverage(self):
        covered_indices = np.where(self.state_space == 1)
        return covered_indices[0].size / (self.grid_resolution[0] * self.grid_resolution[1])


    # ==================== Tools =======================================#
    def state_to_grid(self, distance: float, rel_speed: float):
        """Convert continous state into grid indices"""
        if not (
            self.distance_range[0] <= distance <= self.distance_range[1]
            and self.rel_speed_range[0] <= rel_speed <= self.rel_speed_range[1]
        ):
            # print(f'{(distance,rel_speed)} is not in the area of interest.')
            return 0, 0

        i = int((distance - self.distance_range[0]) / self.distance_grid_size)
        j = int((rel_speed - self.rel_speed_range[0]) / self.rel_speed_grid_size)
        return i, j
