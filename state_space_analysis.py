import numpy as np
from typing import Tuple, List, Callable
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class TrajectoryGenerator:
    def __init__(self, distance_range: Tuple[float,float],
                 rel_speed_range: Tuple[float,float],
                 grid_resolution: Tuple[float,float],
                 num_samples: int = 20,
                 N: int = 20):
        """
        :param
            distance_range: (min_distance, max_distance) in meters
            rel_speed_range: (min_rel_speed, max_rel_speed) in m/s
            grid_resolution: (distance_resolution, speed_resolution) number of grid cells 
        """
        self.distance_range = distance_range
        self.rel_speed_range = rel_speed_range
        self.grid_resolution = grid_resolution
        self.N = N
        self.num_samples = num_samples

        # Calculate size of one grid cell
        self.distance_grid_size = (distance_range[1] - distance_range[0])/grid_resolution[0]
        self.rel_speed_grid_size = (rel_speed_range[1] - rel_speed_range[0])/grid_resolution[1]

        # Initialize state-space grid (0 for empty, 1 for filled)
        self.state_space = np.zeros((self.grid_resolution[0], self.grid_resolution[1]))
        self.best_trajectory = None
        self.trajectories = None

    def state_to_grid(self, distance: float, rel_speed: float):
        """Convert continous state into grid indices"""
        if not (self.distance_range[0] <= distance <= self.distance_range[1] and
                self.rel_speed_range[0] <= rel_speed <= self.rel_speed_range[1]):
            # print(f'{(distance,rel_speed)} is not in the area of interest.')
            return 0,0
        
        i = int((distance - self.distance_range[0])/self.distance_grid_size)
        j = int((rel_speed - self.rel_speed_range[0])/self.rel_speed_grid_size)
        return i,j

    def marked_visited(self, distance: float, rel_speed: float):
        """Mark a grid cell as visited from the state (distance, rel_speed)"""
        i,j = self.state_to_grid(distance, rel_speed)
        self.state_space[i,j] = 1

    def is_visited(self, distance: float, rel_speed: float):
        """Check if the state has been visited (grid where this state belongs to is visited or not)"""
        i,j = self.state_to_grid(distance, rel_speed)
        return self.state_space[i,j] == 1
    
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
        rel_speed_coeffs = np.polyder(coeffs)

        for time in t:
            # Calculate distance at any time step
            distance = np.polyval(coeffs, time)

            # Calculate relative speed at any time step
            rel_speed = np.polyval(rel_speed_coeffs, time)

            trajectory.append((distance, rel_speed))
        
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

        return best_trajectory
    
    def plot_stat_space(self):
        """Visualize a state-space and trajectories."""
        fig, ax = plt.subplots(figsize=(10,6))

        # Draw grid
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                # Determine cell position
                distance = self.distance_range[0] + i * self.distance_grid_size
                rel_speed = self.rel_speed_range[0] + j * self.rel_speed_grid_size


                # Fill color to visited cells
                if self.state_space[i,j] == 1:
                    color_rect = Rectangle((distance,rel_speed),self.distance_grid_size,self.rel_speed_grid_size,color = 'red', alpha = 0.5)
                    ax.add_patch(color_rect)
                else : 
                # Draw grid line for empty cell
                    rect = Rectangle((distance,rel_speed),self.distance_grid_size,self.rel_speed_grid_size)
                    ax.add_patch(rect)

        # Plot random trajectories with low opacity
        for trajectory in self.trajectories:
            x, y = zip(*trajectory)
            ax.plot(x, y, color='blue', alpha=0.5, linestyle='--', linewidth=1)

        # Plot best trajectory with high opacity
        if self.best_trajectory:
            x, y = zip(*self.best_trajectory)
            ax.plot(x, y, color='green', alpha=1.0, linewidth=2, label='Best Trajectory')
            ax.scatter(x, y, color='green', alpha=1.0)

        # Set labels and title
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Relative Speed (m/s)")
        ax.set_title("State-Space and Trajectories")
        ax.set_xlim(self.distance_range)
        ax.set_ylim(self.rel_speed_range)
        ax.legend()
        plt.show()

class SamplingBasedMPC():
    def __init__(self, model: Callable[[Tuple[float,float,float],float, float, float, float, float], Tuple[float,float,float]],
                 cost_function: Callable[[Tuple[float, float],List[Tuple[float, float, float]]],List[float]],
                 N: int,
                 num_samples: int,
                 state_space: TrajectoryGenerator):
        """
        :param
            model: vehicle model (including preceding and following vehicles [ACC])
            cost_fucntion: cost function that is used to evaluate the sample
            N: prediction horizon
            num_samples: number of samples that will be generated
        """
        self.model = model
        self.cost_function = cost_function
        self.N = N
        self.num_samples = num_samples
        self.state_space = state_space

    def generate_random_inputs(self):
        """
        Generate random input sequeences for N step 
        """
        return [np.random.uniform(low = -3, high = 3, size = self.N) for _ in range(self.num_samples)]

    def predict_states(self,initial_state: Tuple[float, float, float],
                         input_sequence: List[float])-> List[Tuple[float, float, float]]:
        """
        Generate states (for N steps) from input sequences (acceleration)

        :param
            initial_state: initial state [current state] (distance, preceding speed, following speed)
            input_sequence: list of input (acceleration)
        :return: List of state (predicted states for N steps)
        """
        states = [initial_state]
        state = initial_state
        for u in input_sequence:
            state = self.model(state, u)
            states.append(state)
        return states
    
    def compute_costs(self, initial_state: Tuple[float, float, float],
                      input_sequences: List[List[float]]) -> List[float]:
        """
        Compute costs of all random input sequences

        :param
            initial_state: initial state [current state] (distance, preceding speed, following speed)
            input_sequences: list of all random input sequences (acceleration) 
        :return: List of cost (for all random inputs of num_samples)
        """
        costs = []
        constraint_distance = (self.state_space.distance_range[0],self.state_space.distance_range[1])
        for input_sequence in input_sequences:
            predicted_states = self.predict_states(initial_state, input_sequence)
            cost = self.cost_function(self.state_space.find_best_trajectory(initial_state[0],(initial_state[1]-initial_state[2])),predicted_states, input_sequence)
            if not all(constraint_distance[0]<=state[0]<=constraint_distance[0] for state in predicted_states):
                cost = cost + 1e6

            costs.append(cost)
        return costs
    
    def select_optimal_input_sequence(self, input_sequences: List[List[float]],
                                      costs: List[float]) -> Tuple[List[float], float]:
        """
        Select the minimum cost input sequence

        :param
            input_sequences: list of all random input sequences
            costs: list of cost of all random input sequences
        :return: input sequence with the least cost and that cost
        """
        min_cost_index = np.argmin(costs)
        return input_sequences[min_cost_index], costs[min_cost_index]

    def solve(self, initial_state: Tuple[float, float, float]) -> List[float]:
        """
        Aggrregate all functions to run at once
        :param initial_state: initial state [current state] (distance, preceding speed, following speed)
        :return: optimal input sequence
        """
        input_sequences = self.generate_random_inputs()
        costs = self.compute_costs(initial_state, input_sequences)
        optimal_input_sequence, _ = self.select_optimal_input_sequence(input_sequences, costs)
        return optimal_input_sequence
    

def vehicle_model(state: Tuple[float, float, float], 
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
    Q[0,0] = 0.5
    Q[1,1] = 1
    R = np.zeros((1,1))
    R[0,0] = 1
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


def main():
    """In Progress for testing the functions"""
    # Initiate state-space and MPC
    distance_range = (5,50)     # Distance range in meters
    rel_speed_range = (-5,5)    # Relative speed range in m/s
    grid_resolution = (10,10)   # Grid resolution
    num_samples = 20
    N = 20

    state_space = TrajectoryGenerator(distance_range,rel_speed_range,grid_resolution,num_samples,N)
    mpc = SamplingBasedMPC(vehicle_model,cost_function, N, num_samples, state_space)

    # Define initial state
    initial_state = (12, 3, 0)  # Distance, preceding vehicle speed, following vehicle speed

    state = initial_state
    max_steps = 10
    # Generate and evaluate trajectories
    for step in range(max_steps):
        
        input_sequences = mpc.generate_random_inputs()
        costs = mpc.compute_costs(state, input_sequences)
        optimal_input_sequence, optimal_cost = mpc.select_optimal_input_sequence(input_sequences,costs)
        optimal_input = optimal_input_sequence[0]
        print(f'Optimal acceleration is {optimal_input} with cost {optimal_cost}.')
        state = vehicle_model(state,optimal_input)


    # Visualize state-space and trajectory
    state_space.plot_stat_space()

if __name__ == '__main__':
    main()

