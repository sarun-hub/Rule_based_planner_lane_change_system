import numpy as np
from typing import Tuple, List 
import os
# from MPC_utils import SamplingBasedMPC, OptimizationBasedMPC      # Cannot import here since MPC_utils also import this file
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

# ======================================== Target Trajectory Generator ==================================#

class TrajectoryGenerator:
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
        self.distance_range = distance_range
        self.rel_speed_range = rel_speed_range
        self.grid_resolution = grid_resolution
        self.N = N
        self.num_samples = num_samples
        self.derivative = derivative

        # Calculate size of one grid cell
        self.distance_grid_size = (distance_range[1] - distance_range[0])/grid_resolution[0]
        self.rel_speed_grid_size = (rel_speed_range[1] - rel_speed_range[0])/grid_resolution[1]

        # Initialize state-space grid (0 for empty, 1 for filled)
        self.state_space = np.zeros((self.grid_resolution[0], self.grid_resolution[1]))
        self.best_trajectory = None
        self.trajectories = None
        self.states = []
        self.all_trajectories = []
        self.all_best_trajectories = []
        self.color_state_space = np.zeros((self.grid_resolution[0], self.grid_resolution[1]))
        self.predicted_states = []

    def add_predicted_state(self,predicted_state):
        self.predicted_states.append(predicted_state)

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
        self.states.append((distance,rel_speed))
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
    
    def plot_stat_space(self,max_steps,show = True,save_path = None):
        """Visualize a state-space and trajectories."""
        fig, ax = plt.subplots(figsize=(10,6))

        rectangles = {}
        # Draw grid
        for i in range(self.grid_resolution[0]):
            for j in range(self.grid_resolution[1]):
                # Determine cell position
                distance = self.distance_range[0] + i * self.distance_grid_size
                rel_speed = self.rel_speed_range[0] + j * self.rel_speed_grid_size

                # Create grid cell
                rect = Rectangle((distance, rel_speed), self.distance_grid_size, self.rel_speed_grid_size,
                                edgecolor='black', facecolor='none')
                
                ax.add_patch(rect)
                rectangles[(i, j)] = rect

        # Initialize trajectory line
        random_trajectories = [ax.plot([], [], color='blue', alpha=0.3, linestyle='--', linewidth=0.5)[0] for _ in range(self.num_samples)]
        best_trajectory_line, = ax.plot([], [], color='green', linewidth=2, label='Best Target Trajectory')
        predicted_state_line, = ax.plot([], [], color='purple', linewidth=2, label='Predicted State')

        # Initialize scatter points for trajectory
        # scatter_points = ax.scatter([], [], color='green', alpha=1.0)

        current_state_scatter = ax.scatter([],[], color = 'red')
        previous_state_scatter = ax.scatter([],[], color = 'grey', alpha = 0.5)

        # Set labels and title
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Relative Speed (m/s)")
        ax.set_title("State-Space and Trajectories")
        # ax.set_xlim(self.distance_range)
        # ax.set_ylim(self.rel_speed_range)
        ax.legend()
        
        def init():
            """Initialize animation."""
            # Reset grid colors
            for rect in rectangles.values():
                rect.set_facecolor('none')
                rect.set_alpha(1.0)

            # Reset scatter points and lines
            for traj_line in random_trajectories:
                traj_line.set_data([], [])
            best_trajectory_line.set_data([], [])
            predicted_state_line.set_data([], [])
            current_state_scatter.set_offsets(np.empty((0, 2)))
            previous_state_scatter.set_offsets(np.empty((0, 2)))

            return random_trajectories + [best_trajectory_line, current_state_scatter, previous_state_scatter] + list(rectangles.values())

        def update(step):
            """Update animation at each step"""
            current_state = self.states[step]
            # Update visited cells
            
            i, j = self.state_to_grid(current_state[0], current_state[1])
            rect = rectangles[(i,j)]
            rect.set_facecolor('lightblue')
            rect.set_alpha(0.5)
            
            
            # Update scatter for current state
            
            current_state_scatter.set_offsets([current_state[:2]])  # Only use (distance, rel_speed)
            
            if step > 0:
                previous_state = np.array([(state[0], state[1]) for state in self.states[:step]])
            else:
                previous_state = np.empty((0, 2))  # Empty array with shape (0, 2)

            previous_state_scatter.set_offsets(previous_state)

            # Update random trajectories
            trajectories = self.all_trajectories[step]
            for traj_line, trajectory in zip(random_trajectories, trajectories):
                x, y = zip(*[(state[0], state[1]) for state in trajectory])
                traj_line.set_data(x, y)

            # Update best trajectory
            best_trajectory = self.all_best_trajectories[step]
            x_opt, y_opt = zip(*[(state[0], state[1]) for state in best_trajectory])
            best_trajectory_line.set_data(x_opt, y_opt)
            # scatter_points.set_offsets(list(zip(x_opt, y_opt)))

            predicted_state = self.predicted_states[step]
            predicted_x, predicted_y = zip(*[(state[0],state[1]) for state in predicted_state])
            predicted_state_line.set_data(predicted_x, predicted_y)

            margin_x = 5
            margin_y = 0.5
            x_min = self.distance_range[0] if current_state[0] > self.distance_range[0] else current_state[0] - margin_x
            y_min = self.rel_speed_range[0] if current_state[1] > self.rel_speed_range[0] else current_state[1] - margin_y
            x_max = self.distance_range[1] if current_state[0] < self.distance_range[1] else current_state[0] + margin_x
            y_max = self.rel_speed_range[1] if current_state[1] < self.rel_speed_range[1] else current_state[1] + margin_y

            ax.set_xlim((x_min,x_max))
            ax.set_ylim((y_min,y_max))

            # return random_trajectories + [best_trajectory_line, scatter_points, current_state_scatter] + list(rectangles.values())
            return random_trajectories + [best_trajectory_line,  current_state_scatter, previous_state_scatter] + list(rectangles.values())
        
        # Animate
        anim = FuncAnimation(fig, update, frames=max_steps, init_func=init , interval=500, blit=False, repeat = True, )

        # Save the animation
        if save_path:
            anim.save(save_path, fps=2, writer='pillow')
            print(f"Animation saved to {save_path}")

        if show :
            plt.show()

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

# ========================================== Helper Function =========================================#

def get_unique_filepath(base_dir: str, base_filename: str, extension: str) -> str:
    """
    Generate a unique file path by appending a number if the file already exists.

    :param base_dir: Directory where the file will be saved.
    :param base_filename: Base name of the file (without number or extension).
    :param extension: File extension (e.g., '.gif').
    :return: Unique file path.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # Create the directory if it doesn't exist

    # Construct initial file path
    counter = 1
    filepath = os.path.join(base_dir, f"{base_filename}_{counter}{extension}")
    
    # Increment counter until a unique file path is found
    while os.path.exists(filepath):
        counter += 1
        filepath = os.path.join(base_dir, f"{base_filename}_{counter}{extension}")
    
    return filepath

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

    state_space = TrajectoryGenerator(distance_range,rel_speed_range,grid_resolution,num_samples,N,derivative = False)

    # Initialize mpc
    sampling_mpc = SamplingBasedMPC(vehicle_model,cost_function, N, num_samples, state_space)
    optimize_mpc = OptimizationBasedMPC(vehicle_model,cost_function,N,state_space)

    # Set objective function weight
    optimize_mpc.distance_weight = distance_weight
    optimize_mpc.rel_speed_weight = rel_speed_weight
    optimize_mpc.input_weight = input_weight

    # Define initial state
    initial_state = (12, 3, 0)  # Distance, preceding vehicle speed, following vehicle speed

    state = initial_state
    max_steps = 100

    # ================ SET MODE =================
    mode = 'sampling'

    optimal_input_sequence = np.zeros(N)
    mpc = sampling_mpc if mode == 'sampling' else optimize_mpc
    # Generate and evaluate trajectories
    for step in range(max_steps):
        state_space.marked_visited(state[0],state[1]-state[2])
        if mode == 'optimize':
            optimal_input_sequence = mpc.solve(state,optimal_input_sequence)
        elif mode == 'sampling':
            optimal_input_sequence = mpc.solve(state)
        optimal_input = optimal_input_sequence[0]

        # print(f'Optimal acceleration is {optimal_input} with cost {optimal_cost}.')
        predicted_state = mpc.predict_states(state,optimal_input_sequence)
        predicted_state = [(state_[0],state_[1]-state_[2]) for state_ in predicted_state]

        state_space.add_predicted_state(predicted_state)
        state = vehicle_model(state,optimal_input)


    # Visualize state-space and trajectory
    if mode == 'optimize':
        save_path = get_unique_filepath('OptimizeBasedMPC','state_space_animation_opt','.gif')
    elif mode == 'sampling':
        save_path = get_unique_filepath('SamplingBasedMPC_delayed','state_space_animation','.gif')
    
    state_space.plot_stat_space(max_steps,show = False, save_path=save_path)

if __name__ == '__main__':
    main()

