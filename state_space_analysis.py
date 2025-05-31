import numpy as np
from typing import Tuple, List 
import os
# from MPC_utils import SamplingBasedMPC, OptimizationBasedMPC      # Cannot import here since MPC_utils also import this file
from reference_generator import BaseGenerator       # for base reference_generator
from utils import get_unique_filepath

# ======================================== Target Trajectory Generator ==================================#

# class for state-space analyze reference generator
class TrajectoryGenerator(BaseGenerator):
    def __init__(self, distance_range: Tuple[float,float],
                 rel_speed_range: Tuple[float,float],
                 grid_resolution: Tuple[float,float],
                 num_samples: int = 20,
                 N: int = 20,
                 derivative: bool = False):
        """
        :param
            distance_range: (min_distance, max_distance) in meters
            rel_speed_range: (min_rel_speed, max_rel_speed) in m/s
            grid_resolution: (distance_resolution, speed_resolution) number of grid cells 
        """
        # Set distance range, rel_speed range, grid_resolution, N and calculate grid cell size
        super().__init__(distance_range, rel_speed_range, grid_resolution, N)
        
        self.num_samples = num_samples
        self.derivative = derivative

        # Initialize state-space grid (0 for empty, 1 for filled)
        self.best_trajectory = None
        self.trajectories = None
        
        self.color_state_space = np.zeros((self.grid_resolution[0], self.grid_resolution[1]))
    
    def find_empty_cells(self) -> List[Tuple[float,float]]:
        """
        Find state-space representation of the grid cells without data
        
        :return: list of (distance, state-space) -> center of empty cells
        """
        empty_cells = []
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                if self.state_space[i,j] == 0:
                    distance = self.distance_range[0] + (i+1/2) * self.distance_grid_size
                    rel_speed = self.rel_speed_range[0] + (j+1/2) * self.rel_speed_grid_size
                    empty_cells.append((distance,rel_speed))
        return empty_cells
    
    def find_closest_empty_cells(self, current_distance: float, current_rel_speed: float,
                                 tolerance: float = 1e-6) -> List[Tuple[float,float]]:
        """Find the closest empty cells"""
        empty_cells = self.find_empty_cells()

        if not empty_cells:
            print('The state-space graph is fully covered.')
            return []

        # Calculate distances to all empty cells
        distances = [np.sqrt((cell[0] - current_distance)**2 + 
                             (cell[1] - current_rel_speed)**2) 
                             for cell in empty_cells]

        min_distance = np.min(distances)
        
        closest_cells = [cell for cell, dist in zip(empty_cells, distances)
                         if abs(dist - min_distance) <= tolerance]

        return closest_cells
    
    def generate_trajectory(self, start_distance: float, end_distance:float,
                            start_rel_speed: float, end_rel_speed: float) -> List[Tuple[float,float]]:
        """Generate a smooth trajectory between start (current state) and end states (the closest cell)"""
        trajectory  = []
        
        # Use cubic interpolation for distance to ensure smooth transitions
        # set time step
        t = np.linspace(0,1,self.N)

        # Generate random intermediate control points for variety
        num_control_points = 2 # Number of random control points

        control_distances = np.random.uniform(
            min(start_distance, end_distance),
            max(start_distance, end_distance),
            num_control_points
        )

        control_rel_speeds = np.random.uniform(
            min(start_rel_speed, end_rel_speed),
            max(start_rel_speed, end_rel_speed),
            num_control_points
        )

        # Combine all points for spline fitting
        num_intervals = num_control_points + 1
        intervals = 1/num_intervals
        x = [0] + [ (i+1)*intervals for i in range(num_control_points)] + [1]
        distances  = [start_distance] + list(control_distances) + [end_distance]
        rel_speeds  = [start_rel_speed] + list(control_rel_speeds) + [end_rel_speed]

        # Fit cubic spline for distance
        coeffs = np.polyfit(x, distances, 3)
        # Find coefficent of relative speed (which is derivative of distance)
        rel_speed_coeffs = np.polyder(coeffs) if self.derivative else np.polyfit(x,rel_speeds,3)

        for time in t:
            # Calculate distance at any time step
            distance = np.polyval(coeffs, time)

            # Calculate relative speed at any time step
            rel_speed = np.polyval(rel_speed_coeffs, time)

            trajectory.append((distance, rel_speed))
        
        # print(trajectory[0])
        
        return trajectory
    
    def evaluate_trajectory(self, trajectory: List[Tuple[float, float]]) -> Tuple[int, float]:
        """Evaluate trajectory based on new cells visited and length"""
        new_cells = 0
        total_length = 0
        visited_cells = set()

        for i in range(len(trajectory)):
            distance, rel_speed = trajectory[i]

            # Convert state to grid
            grid_coords = self.state_to_grid(distance, rel_speed)

            # Check if this grid is visited (not yet update to state-space)
            # if visited, add new cell count and mark that it is visited (avoid counting it several times)
            if not self.is_visited(distance, rel_speed) and grid_coords not in visited_cells:
                new_cells += 1
                visited_cells.add(grid_coords)

            # Calculate trajectory length
            if i > 0 :
                prev_distance, prev_rel_speed = trajectory[i-1]
                segment_length = np.sqrt((distance - prev_distance)**2 + 
                                         (rel_speed - prev_rel_speed)**2)
                total_length += segment_length

        return new_cells, total_length
    
    def find_best_trajectory(self, current_distance: float, current_rel_speed: float):
        """Find the best trajectory (from num_samples samples) to the closest cells"""
        # print(current_distance,current_rel_speed)
        closest_cells = self.find_closest_empty_cells(current_distance, current_rel_speed)
        if not closest_cells:
            print('There is no closest cells.')
            return []
        
        trajectories = []
        best_trajectory = []
        max_new_cells = 0
        min_length = float('inf')

        # for each closest cell, generate multiple random trajectories
        for end_cell in closest_cells:
            for _ in range(self.num_samples):
                # Generate random trajectory to this end cell
                trajectory = self.generate_trajectory(
                        current_distance,end_cell[0],
                        current_rel_speed, end_cell[1]
                    )

                trajectories.append(trajectory)
                # Evaluate trajectory
                new_cells, length = self.evaluate_trajectory(trajectory)

                # Update best trajectory
                if new_cells > max_new_cells or (new_cells == max_new_cells and length < min_length):
                    max_new_cells = new_cells
                    min_length = length
                    best_trajectory = trajectory

        self.trajectories = trajectories
        self.best_trajectory = best_trajectory
        self.all_best_trajectories.append(best_trajectory)
        self.all_trajectories.append(trajectories)

        return best_trajectory
    
    # for general calling TODO: refactor the code
    def generate_reference(self,current_distance: float, current_rel_speed: float):
        return self.find_best_trajectory(current_distance, current_rel_speed)

# ===================================== Vehicle Model ===========================================#

def vehicle_model(state: Tuple[float, float, float], 
                 control_input: float,
                 aggressive: float = 0.8,
                 h: float = 1.0,
                 delta_min: float = 5.0,
                 T: float = 0.1,
                 deadtime: float = 0.1,
                 tolerance: float = 1e-5) -> Tuple[float, float, float]:
    """
    Vehicle dynamics model calculating next state based on current state and control input.

    :param
        state: Tuple of (distance, preceding vehicle velocity, following vehicle velocity)
        control_input: Acceleration input for the preceding vehicle
        aggressive: Aggressiveness factor
        h: Time headway (s)
        delta_min: Minimum safe distance (meter)
        T: Time step (s)
        deadtime: System delay (s)

    :return: Tuple of next state (next_distance, next_vp, next_vf)
    """
    d, vp, vf = state

    # Initialize state history if not exists (for the beginning)
    if not hasattr(vehicle_model,'state_history'):
        vehicle_model.state_history = []
        vehicle_model.time_history = []
        vehicle_model.current_time = 0
    
    # Update current time
    vehicle_model.current_time += T

    # Store current state and time
    vehicle_model.state_history.append(state)
    vehicle_model.time_history.append(vehicle_model.current_time)
    
    # Remove old states beyond deadtime
    while (vehicle_model.current_time - vehicle_model.time_history[0]) - deadtime > tolerance:
        vehicle_model.state_history.pop(0)
        vehicle_model.time_history.pop(0)
    
    # Get delayed state for ACC calculation
    if len(vehicle_model.state_history) > 0:
        delayed_d, delayed_vp, delayed_vf = vehicle_model.state_history[0]
        # print(vehicle_model.current_time - vehicle_model.time_history[0])
    else:
        delayed_d, delayed_vp, delayed_vf = state
    
    next_d = d + (vp-vf) * T
    next_vp = vp + control_input * T
    
     # Calculate following vehicle acceleration using delayed states
    acc = (aggressive * delayed_d + delayed_vp - 
           (1 + aggressive * h) * delayed_vf - 
           aggressive * delta_min) / h
    
    # Add safety margin due to deadtime
    safety_factor = 1.2  # Increase safety margin by 20%
    if delayed_d < (delta_min * safety_factor):
        acc = min(acc, -2)  # Stronger deceleration for safety
    
    # Update following vehicle velocity using calculated acceleration
    next_vf = vf + acc * T

    return next_d, next_vp, next_vf

def vehicle_model_without_delay(state: Tuple[float, float, float], 
                 control_input: float,
                 aggressive: float = 0.8,
                 h: float = 1.0,
                 delta_min: float = 5.0,
                 T: float = 0.1) -> Tuple[float, float, float]:
    """
    Vehicle dynamics model calculating next state based on current state and control input.

    :param
        state: Tuple of (distance, preceding vehicle velocity, following vehicle velocity)
        control_input: Acceleration input for the preceding vehicle
        aggressive: Aggressiveness factor
        h: Time headway (s)
        delta_min: Minimum safe distance (meter)
        T: Time step (s)
        deadtime: System delay (s)

    :return: Tuple of next state (next_distance, next_vp, next_vf)
    """
    d, vp, vf = state
    next_d = d + (vp-vf) * T
    next_vp = vp + control_input * T
    next_vf = vf + (aggressive * d + vp - (1+aggressive*h)*vf - aggressive*delta_min)/h * T
    return next_d, next_vp, next_vf


def cost_function(targets: List[Tuple[float, float]],predicted_states: List[Tuple[float, float, float]], input_sequence: List[float]):
    """
    Calculate Cost from the predicted states (state cost) and input sequence (input cost)
    """
    cost = 0
    Q = np.zeros((3,3))
    Q[0,0] = distance_weight
    Q[1,1] = rel_speed_weight
    R = np.zeros((1,1))
    R[0,0] = input_weight
    for state,target in zip(predicted_states,targets):
        d, vp, vf = state
        target_d,target_rel_speed = target
        d_diff = d - target_d
        rel_speed_diff = (vp - vf) - target_rel_speed
        cost = cost + d_diff * Q[0,0] * d_diff + rel_speed_diff * Q[1,1] * rel_speed_diff

    for k in range(len(input_sequence)):
        if k > 0:
            previous_cont = input_sequence[k-1]
            diff_cont = input_sequence[k] - previous_cont
            cost = cost + diff_cont * R[0,0] * diff_cont

    return cost

# Set Global weight
distance_weight = 1
rel_speed_weight = 1
input_weight = 1

def main():
    """In Progress for testing the functions"""
    from MPC_utils import SamplingBasedMPC, OptimizationBasedMPC

    # Initiate state-space and MPC
    distance_range = (5,50)     # Distance range in meters
    rel_speed_range = (-5,5)    # Relative speed range in m/s
    grid_resolution = (10,10)   # Grid resolution
    num_samples = 20
    N = 20

    traj_generator = TrajectoryGenerator(distance_range,rel_speed_range,grid_resolution,num_samples,N,derivative = False)

    # Initialize mpc
    sampling_mpc = SamplingBasedMPC(vehicle_model,cost_function, N, num_samples, traj_generator)
    optimize_mpc = OptimizationBasedMPC(vehicle_model,cost_function,N,traj_generator)

    # Set objective function weight
    optimize_mpc.distance_weight = distance_weight
    optimize_mpc.rel_speed_weight = rel_speed_weight
    optimize_mpc.input_weight = input_weight

    # Define initial state
    initial_state = (12, 3, 0)  # Distance, preceding vehicle speed, following vehicle speed

    state = initial_state
    max_steps = 100

    # ================ SET MODE =================
    mode = 'optimize'

    optimal_input_sequence = np.zeros(N)
    mpc = sampling_mpc if mode == 'sampling' else optimize_mpc
    # Generate and evaluate trajectories
    for step in range(max_steps):
        # TODO: Can be put in mpc.solve (TBD)
        traj_generator.marked_visited(state[0],state[1]-state[2])
        if mode == 'optimize':
            optimal_input_sequence = mpc.solve(state,optimal_input_sequence)
        elif mode == 'sampling':
            optimal_input_sequence = mpc.solve(state)
        else:
            raise NotImplementedError(f'Mode {mode} is not supported.')
        optimal_input = optimal_input_sequence[0]

        # print(f'Optimal acceleration is {optimal_input} with cost {optimal_cost}.')
        predicted_state = mpc.predict_states(state,optimal_input_sequence)
        predicted_state = [(state_[0],state_[1]-state_[2]) for state_ in predicted_state]

        traj_generator.add_predicted_state(predicted_state)
        state = vehicle_model(state,optimal_input)


    # Visualize state-space and trajectory
    if mode == 'optimize':
        save_path = get_unique_filepath('OptimizeBasedMPC','state_space_animation_opt','.gif')
    elif mode == 'sampling':
        save_path = get_unique_filepath('SamplingBasedMPC_delayed','state_space_animation','.gif')
    
    save_path = ''
    traj_generator.plot_state_space(max_steps,show = True, save_path=save_path)

if __name__ == '__main__':
    main()

