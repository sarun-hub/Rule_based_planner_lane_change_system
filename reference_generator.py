import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
from typing import Tuple
from matplotlib.animation import FuncAnimation

# ================================== FOR ADDING WARNING ============================
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format
# ==================================================================================

class BaseGenerator():
    """Common interface for trajectory/reference generators."""
    def __init__(self, distance_range: Tuple[float,float],
                 rel_speed_range: Tuple[float,float],
                 grid_resolution: Tuple[float,float],
                 N : int = 20):
        self.distance_range = distance_range
        self.rel_speed_range = rel_speed_range
        self.grid_resolution = grid_resolution
        self.N = N

        # Calculate grid cell size
        self.distance_grid_size = (distance_range[1] - distance_range[0])/grid_resolution[0]
        self.rel_speed_grid_size = (rel_speed_range[1] - rel_speed_range[0])/grid_resolution[1]

        # Initialize state-space grid (0 for empty, 1 for filled)
        self.state_space = np.zeros((self.grid_resolution[0], self.grid_resolution[1]))
        
        self.predicted_states = []

        # For collecting states, trajectories, best trajectory history
        self.states = []
        self.all_trajectories = []                  # all random trajectories
        self.all_best_trajectories = []             # all optimal trajectories


    # ==================== Tools =======================================#
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
        # Add the history
        self.states.append((distance,rel_speed))
        # Update state-space
        self.state_space[i,j] = 1

    def is_visited(self, distance: float, rel_speed: float):
        """Check if the state has been visited (grid where this state belongs to is visited or not)"""
        i,j = self.state_to_grid(distance, rel_speed)
        return self.state_space[i,j] == 1
    
    def add_predicted_state(self,predicted_state):
        self.predicted_states.append(predicted_state)
    # ==================================================================#

    def generate_reference(self):
        raise NotImplementedError("Subclasses must implement generate()")
    
    # ==================== Plot Illustration (Animation) ===============================#

    def plot_state_space(self,max_steps,show = True, save_path = None):
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
                rect = Rectangle((distance,rel_speed),self.distance_grid_size,self.rel_speed_grid_size,
                                 edgecolor='black', facecolor='none')
                
                ax.add_patch(rect)
                rectangles[(i,j)] = rect
        
        # Initialize for beginning of the animation
        # Initialize trajectory line
        if self.check_instance(self,"TrajectoryGenerator"):
            random_trajectories = [ax.plot([],[], color = 'blue', alpha = 0.3, linestyle='--', linewidth =0.5)[0] for _ in range(self.num_samples)]
            label_best_traj = 'Best Target Trajectory'
        elif self.check_instance(self,"SpiralReferenceGenerator"):
            random_trajectories = []
            label_best_traj = 'Target Trajectory'
        else :
            random_trajectories = []
            label_best_traj = 'Target Trajectory'
            warnings.warn(f'Generator {type(self).__name__} is not yet supported.\n \
                          There might be something wrong with the illustration, Recheck again')
        
        # Initialize best trajectory line and predicted state line
        best_trajectory_line, = ax.plot([], [], color='green', linewidth=2, label=label_best_traj)
        predicted_state_line, = ax.plot([], [], color='purple', linewidth=2, label='Predicted State')

        # Initialize the scatter points for illustrating state
        current_state_scatter = ax.scatter([],[],color = 'red')
        previous_state_scatter = ax.scatter([],[],color = 'grey', alpha = 0.5)

        # Set labels and title
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Relative speed (m/s)')
        ax.set_title('State-Spacea and Trajectories')
        # ax.set_xlim(self.distance_range)
        # ax.set_ylim(self.rel_speed_range)
        ax.legend()
    
        def init():
            """Intialize animation"""
            # Reset grid colors (for loop start)
            for rect in rectangles.values():
                rect.set_facecolor('none')
                rect.set_alpha(1.0)
            
            # Reset scatter points and line (for loop start)
            for traj_line in random_trajectories:
                traj_line.set_data([],[])
            best_trajectory_line.set_data([],[])
            predicted_state_line.set_data([],[])
            current_state_scatter.set_offsets(np.empty((0,2)))
            previous_state_scatter.set_offsets(np.empty((0,2)))

            # No need to return prediced_state_line since blit = False (it explicitly update)
            return random_trajectories + [best_trajectory_line, current_state_scatter, previous_state_scatter] + list(rectangles.values())

        def update(step):
            """Update animation at each step"""
            current_state = self.states[step]
            
            # Update visited cells (fill color - lightblue)
            i, j = self.state_to_grid(current_state[0],current_state[1])
            rect = rectangles[(i,j)]
            rect.set_facecolor('lightblue')
            rect.set_alpha(0.5)

            # Update scatter for current state
            current_state_scatter.set_offsets([current_state[:2]]) # Only use (distance, rel_speed) - Actually there are only two parameters already.

            # For not the first step -> show all previous step
            if step > 0 :
                previous_state = np.array([(state[0],state[1]) for state in self.states[:step]])
            else :
                previous_state = np.empty((0,2))    # Empty array with shape (0,2) - no row, 2 columns

            previous_state_scatter.set_offsets(previous_state)

            # Update random trajectories (if any) -> exists in TrajectoryGenerator but not in SprialReferenceGenerator
            trajectories = self.all_trajectories[step]
            for traj_line, trajectory in zip(random_trajectories, trajectories):
                x, y = zip(*[(state[0],state[1]) for state in trajectory])
                traj_line.set_data(x,y)
            
            # Update the best trajectory (in both TrajectoryGenerator and SpiralReferenceGenerator)
            best_trajectory = self.all_best_trajectories[step]
            x_opt, y_opt = zip(*[(state[0],state[1]) for state in best_trajectory])
            best_trajectory_line.set_data(x_opt,y_opt)

            # Update predicted state
            predicted_state = self.predicted_states[step]
            predicted_x, predicted_y = zip(*[(state[0],state[1]) for state in predicted_state])
            predicted_state_line.set_data(predicted_x, predicted_y)

            # Allow out of area of interest illustration
            margin_x = 5
            margin_y = 0.5
            x_min = self.distance_range[0] if current_state[0] > self.distance_range[0] else current_state[0] - margin_x
            y_min = self.rel_speed_range[0] if current_state[1] > self.rel_speed_range[0] else current_state[1] - margin_y
            x_max = self.distance_range[1] if current_state[0] < self.distance_range[1] else current_state[0] + margin_x
            y_max = self.rel_speed_range[1] if current_state[1] < self.rel_speed_range[1] else current_state[1] + margin_y

            ax.set_xlim((x_min,x_max))
            ax.set_ylim((y_min,y_max))

            return random_trajectories + [best_trajectory_line,  current_state_scatter, previous_state_scatter] + list(rectangles.values())

        # Animate
        anim = FuncAnimation(fig, update, frames=max_steps, init_func = init, interval = 500, blit = False, repeat = True)

        # Save the animation
        if save_path:
            anim.save(save_path, fps=2, writer='pillow')
            print(f'Animation saved to {save_path}')
        
        if show :
            plt.show()

    # For dealing with the instance type check
    def check_instance(self, obj, obj_type:str):
        return type(obj).__name__ == obj_type