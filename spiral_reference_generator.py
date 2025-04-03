import numpy as np
from typing import Tuple, List
from reference_generator import BaseGenerator
from utils import get_unique_filepath

# Temporary set
T = 0.1     # will be in vehicle model


# Create a class to generate spiral refernce generator
# fix a_target, and period T (100) -> vary b_target (5-40) [TODO: to be decided]
class SpiralReferenceGenerator(BaseGenerator):
    def __init__(self,initial_state: Tuple[float,float,float], 
                 distance_range: Tuple[float,float],
                 rel_speed_range: Tuple[float,float],
                 grid_resolution: Tuple[float,float],
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
        super().__init__(distance_range, rel_speed_range, grid_resolution, N)
        self.expanding_rate = expanding_rate
        self.count = 0

        # set the center of the spiral (a_target)
        self.center = self.calculate_center()
        self.create_spiral()

    # for calculating the range of vary b_target (TODO + set default T)
    def set_range(self):
        pass

    # calculate center of the spiral from the distance range
    def calculate_center(self):
        min_x, max_x = self.distance_range
        center = (max_x-min_x)/2
        return center
    
    def create_spiral(self):
        max_steps = 100
        # Set period and angular velocity 
        period = 100
        w = 2 * np.pi / period
        # Set range
        b_target_range = (5,40)
        b_target_list = np.linspace(b_target_range[0],b_target_range[1],int((b_target_range[1]-b_target_range[0])/self.expanding_rate))

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

# TODO: make tester
def main():
    from MPC_utils import SamplingBasedMPC, OptimizationBasedMPC

    # Initialize state-space and MPC
    distance_range = (5,50)         # Distance range in meters
    rel_speed_range = (-5,5)        # Relative speed range in m/s
    grid_resolution = (10, 10)      # Grid Resolution
    expanding_rate = 1              # Expanding rate of the spiral
    N = 20

    # Define initial state
    initial_state = (12, 3, 0)  # Distance, preceding vehicle speed, following vehicle speed

    spiral_generator = SpiralReferenceGenerator(initial_state,distance_range,rel_speed_range,grid_resolution,N,expanding_rate)

    # Initialize MPC
    num_samples = 20
    sampling_mpc = SamplingBasedMPC(vehicle_model, cost_function,N,num_samples,spiral_generator)
    optimize_mpc = OptimizationBasedMPC(vehicle_model,cost_function,N,spiral_generator)
    
    state = initial_state
    max_steps = 100

    # ================= SET MODE ======================= #
    mode = 'sampling'

    optimal_input_sequence = np.zeros(N)
    mpc = sampling_mpc if mode == 'sampling' else optimize_mpc

    # Generate and evaluate trajectories
    for step in range(max_steps):
        spiral_generator.marked_visited(state[0], state[1]-state[2])
        if mode == 'optimize':
            optimal_input_sequence = mpc.solve(state,optimal_input_sequence)
        elif mode == 'sampling':
            optimal_input_sequence = mpc.solve(state)
        else :
            raise NotImplementedError(f'Mode {mode} is not supported.')
        optimal_input = optimal_input_sequence[0]
    
        predicted_state = mpc.predict_states(state,optimal_input_sequence)
        predicted_state = [(state_[0],state_[1]-state_[2]) for state_ in predicted_state]

        spiral_generator.add_predicted_state(predicted_state)
        state = vehicle_model(state,optimal_input)
    
    # Visualize state-space and trajectory
    if mode == 'opitmize':
        save_path = get_unique_filepath('Spiral_OptimizedBasedMPC','state_space_animation_opt','.gif')
    elif mode == 'sampling':
        save_path = get_unique_filepath('Spiral_SamplingBasedMPC','state_space_animation','.gif')
    
    save_path = ''
    spiral_generator.plot_state_space(max_steps,show=True,save_path=save_path)

if __name__ == '__main__':
    main()