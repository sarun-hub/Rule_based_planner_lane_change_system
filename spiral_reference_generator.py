import numpy as np
from typing import Tuple
from reference_generator import BaseGenerator

# Temporary set
T = 0.1     # will be in vehicle model


# Create a class to generate spiral refernce generator
# fix a_target, and period T (100) -> vary b_target (5-40) [TODO: to be decided]
class SpiralReferenceGenerator(BaseGenerator):
    def __init__(self,initial_state: Tuple[float,float,float], 
                 distance_range: Tuple[float,float],
                 rel_speed_range: Tuple[float,float],
                 N: int = 20,
                 expanding_rate: float = 0.1):
        """
        Requires:
        initial_state: the initial state (don't know what to use)
        distance_range: area of interest distance range
        rel_speed_range: area of interest rel_speed range
        """
        # Initialize parameter
        self.initial_state = initial_state
        self.distance_range = distance_range
        self.rel_speed_range = rel_speed_range
        self.N = N
        self.expanding_rate = expanding_rate
        self.count = 0

        # set the center of the spiral (a_target)
        self.center = self.calculate_center()
        self.create_spiral()

    # for calculating the range of vary b_target (TODO + set default T)
    def set_range(self):
        NotImplementedError

    # calculate center of the spiral from the distance range
    def calculate_center(self):
        min_x, max_x = self.initial_state
        center = (max_x-min_x)/2
        return center
    
    def create_spiral(self):
        max_steps = 100
        # Set period and angular velocity 
        period = 100
        w = 2 * np.pi / period
        # Set range
        b_target_range = (5,40)
        b_target_list = np.linspace(b_target_range[0],b_target_range[1],(b_target_range[1]-b_target_range[0])/self.expanding_rate)

        spiral_trajectory = []
        for i in range(len(b_target_list)):
            target_d = self.center + b_target_list[i] * np.sin(w*i*T)
            target_v_rel = b_target_list[i]*w*np.cos(w*i*T)
            spiral_trajectory.append((target_d,target_v_rel))
        self.spiral_trajectory = spiral_trajectory

    def go_to_next_point(self):
        self.count += 1

    # Note: need to return N = 20 points (same as state-space analysis reference generator)
    def generate_reference(self):
        remaining = len(self.spiral_trajectory) - self.count
        if remaining >= self.N:
            return self.spiral_trajectory[self.count:self.count+self.N]
        else:# if it does not have enough b_target for future, repeat the last value
            return self.spiral_trajectory[self.count:] + [self.spiral_trajectory[-1]] * (self.N - remaining)
        
    
    # TODO: make tester