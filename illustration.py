import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CarDataVisualizer:
    def __init__(self, number_of_surrounding_cars, update_interval=100):
        """
        Initializes the visualizer.
        :param number_of_surrounding_cars: Number of surrounding cars to plot.
        :param update_interval: Interval for updating the plot in milliseconds.
        """
        self.number_of_surrounding_cars = number_of_surrounding_cars
        self.update_interval = update_interval
        self.fig = None
        self.axes_3d = []

        # Initialize the figure and subplots
        self._initialize_plot()

    def _initialize_plot(self):
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

    def load_dataframe(self, filename):
        """Loads a CSV file into a Pandas DataFrame."""
        return pd.read_csv(filename)

    def update(self, frame):
        """Updates the 3D scatter plots."""
        for i in range(self.number_of_surrounding_cars):
            # Load the updated data
            data = self.load_dataframe(f'Car_{i+1}_data.csv')
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

    def run(self):
        """Starts the visualization."""
        ani = FuncAnimation(self.fig, self.update, interval=self.update_interval)
        plt.show()


if __name__ == "__main__":
    visualizer = CarDataVisualizer(number_of_surrounding_cars=2, update_interval=10)
    visualizer.run()
