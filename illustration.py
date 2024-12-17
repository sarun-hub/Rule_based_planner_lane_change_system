import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class CarDataVisualizer:
    def __init__(self, number_of_surrounding_cars, update_interval=100, mode='3d'):
        """
        Initializes the visualizer.
        :param number_of_surrounding_cars: Number of surrounding cars to plot.
        :param update_interval: Interval for updating the plot in milliseconds.
        :param mode: Visualization mode, either '2d' or '3d'.
        """
        self.number_of_surrounding_cars = number_of_surrounding_cars
        self.update_interval = update_interval
        self.mode = mode.lower()
        self.fig = None
        self.grid_size = 10

        # Initialize the figure and subplots
        if self.mode == '3d':
            self.axes_3d = []
            self._initialize_3dplot()
        elif self.mode == '2d':
            self.axes_2d = []
            self._initialize_2dplot()
        else:
            raise ValueError("Invalid mode. Choose either '2d' or '3d'.")

    def grid_length(self, x_min=5, x_max=50, y_min=-2, y_max=2):
        """Calculate the size of each grid cell."""
        return (x_max - x_min) / self.grid_size, (y_max - y_min) / self.grid_size
    
    def is_grid_covered(self, i, j, points, x_min=5, x_max=50, y_min=-2, y_max=2):
        """Check if grid cell (i, j) is covered by points on the ellipse."""
        grid_x_len, grid_y_len = self.grid_length(x_min, x_max, y_min, y_max)
        for x, y in points:
            if (x_min + i * grid_x_len <= x <= x_min + (i + 1) * grid_x_len and
                y_min + j * grid_y_len <= y <= y_min + (j + 1) * grid_y_len):
                return True
        return False

    def _initialize_3dplot(self):
        """Initializes the 3D subplots."""
        self.fig, axes = plt.subplots(1, self.number_of_surrounding_cars, figsize=(12, 6))
        self.axes_3d = []
        for i, ax in enumerate(axes):
            ax_3d = self.fig.add_subplot(1, self.number_of_surrounding_cars, i + 1, projection='3d')
            self.axes_3d.append(ax_3d)
            ax_3d.set_title(f"Car {i + 1} Data")
            ax_3d.set_xlabel("Longitudinal Distance (m)")
            ax_3d.set_ylabel("Lateral Distance (m)")
            ax_3d.set_zlabel("Longitudinal Relative Velocity (m/s)")

    def _initialize_2dplot(self):
        """Initializes the 2D subplots."""
        self.fig, axes = plt.subplots(1, self.number_of_surrounding_cars, figsize=(12, 6))
        self.axes_2d = list(axes)  # Store axes as a list for consistency
        # Set max/min data
        x_max = 20      
        x_min = -5
        y_max = 4
        y_min = -4
        grid_x_len, grid_y_len = (x_max - x_min) / self.grid_size, (y_max - y_min) / self.grid_size

        for i, ax in enumerate(self.axes_2d):
            ax.set_title(f"Car {i + 1} Data -> Coverage 0%")
            ax.set_xlabel("Longitudinal Distance (m)")
            ax.set_ylabel("Longitudinal Relative Velocity (m/s)")

            # Draw grid cells
            for grid_i in range(self.grid_size):
                for grid_j in range(self.grid_size):
                    rect = plt.Rectangle(
                        (x_min + grid_i * grid_x_len, y_min + grid_j * grid_y_len),
                        grid_x_len, grid_y_len,
                        edgecolor='black', facecolor='none'
                    )
                    ax.add_patch(rect)
            # ax.grid()

    def load_dataframe(self, filename):
        """Loads a CSV file into a Pandas DataFrame with error handling."""
        if not os.path.exists(filename):
            print(f"Error: {filename} does not exist.")
            return pd.DataFrame()
        if os.stat(filename).st_size == 0:
            print(f"Warning: {filename} is empty.")
            return pd.DataFrame(columns=["longitudinal_distance", "lateral_distance", "longitudinal_relative_velocity"])

        try:
            df = pd.read_csv(filename)
            return df
        except pd.errors.EmptyDataError:
            print(f"Error: {filename} contains no data.")
            return pd.DataFrame(columns=["longitudinal_distance", "lateral_distance", "longitudinal_relative_velocity"])
        
    def update_3d(self, frame):
        """Updates the 3D scatter plots."""
        for i in range(self.number_of_surrounding_cars):
            # Load the updated data
            file = f'Car_{i+1}_data_mod.csv'
            data = self.load_dataframe(file)
            lon = data['longitudinal_distance'].to_list()
            lat = data['lateral_distance'].to_list()
            speed = data['longitudinal_relative_velocity'].to_list()

            # Clear the previous plot
            self.axes_3d[i].cla()

            # Redraw the updated plot
            self.axes_3d[i].scatter(lon, lat, speed, c='r', marker='o')
            self.axes_3d[i].set_title(f"Car {i + 1} Data")
            self.axes_3d[i].set_xlabel("Longitudinal Distance (m)")
            self.axes_3d[i].set_ylabel("Lateral Distance (m)")
            self.axes_3d[i].set_zlabel("Longitudinal Relative Velocity (m/s)")

    def update_2d(self, frame):
        """Updates the 2D scatter plots."""
        # Set max/min data
        x_max = 20      
        x_min = -5
        y_max = 4
        y_min = -4
        grid_x_len, grid_y_len = (x_max - x_min) / self.grid_size, (y_max - y_min) / self.grid_size
        for i in range(self.number_of_surrounding_cars):
            # Load the updated data
            file = f'Car_{i+1}_data_mod.csv'
            data = self.load_dataframe(file)
            lon = data['longitudinal_distance'].to_list()
            speed = data['longitudinal_relative_velocity'].to_list()

            # Clear the previous plot
            self.axes_2d[i].cla()

            covered_grid = set()
            points = [(lat_, speed_) for lat_, speed_ in zip(lon, speed)]
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    if self.is_grid_covered(j, k, points, x_min, x_max, y_min, y_max):
                        covered_grid.add((j, k))

            # Draw grid cells
            for grid_i in range(self.grid_size):
                for grid_j in range(self.grid_size):
                    rect = plt.Rectangle(
                        (x_min + grid_i * grid_x_len, y_min + grid_j * grid_y_len),
                        grid_x_len, grid_y_len,
                        edgecolor='black', facecolor='none'
                    )
                    self.axes_2d[i].add_patch(rect)

            # Highlight the best-covered grid cells
            for (grid_i, grid_j) in covered_grid:
                rect = plt.Rectangle(
                    (x_min + grid_i * grid_x_len, y_min + grid_j * grid_y_len),
                    grid_x_len, grid_y_len,
                    edgecolor='black', facecolor='lightblue', alpha=0.5
                )
                self.axes_2d[i].add_patch(rect)

            # Redraw the updated plot
            self.axes_2d[i].scatter(lon, speed, c='red')
            self.axes_2d[i].set_title(f"Car {i + 1} Data -> Coverage {len(covered_grid)/self.grid_size**2 * 100 :.2f}%")
            self.axes_2d[i].set_xlabel("Longitudinal Distance (m)")
            self.axes_2d[i].set_ylabel("Longitudinal Relative Velocity (m/s)")
            # self.axes_2d[i].grid()


    def run(self):
        """Starts the visualization."""
        if self.mode == '3d':
            ani = FuncAnimation(self.fig, self.update_3d, interval=self.update_interval, save_count=50)
        elif self.mode == '2d':
            ani = FuncAnimation(self.fig, self.update_2d, interval=self.update_interval, save_count=50)
        plt.show()

if __name__ == "__main__":
    # Change the mode to '2d' or '3d' to test both options
    visualizer = CarDataVisualizer(number_of_surrounding_cars=2, update_interval=100, mode='2d')
    visualizer.run()
