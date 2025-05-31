import pygame
import numpy as np
import time
import random
import math
import pandas as pd
from typing import Tuple, List
from MPC_utils import SamplingBasedMPC, OptimizationBasedMPC
from state_space_analysis import TrajectoryGenerator, get_unique_filepath

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 1800, 600
LANE_WIDTH = 70.0
NUM_LANES = 2
FPS = 60
NUM_VEHICLES = 3    # number of vehicles including ego vehicles

# Colors 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)

# Vehicle properties
CAR_WIDTH, CAR_HEIGHT = 91, 35

# Simulation parameters
MAX_SPEED = 5
LANE_CHANGING_SPEED = 5
LANE_CHANGE_BUFFER = 200    # Minimum gap for lane change

# Set Global weight
distance_weight = 1
rel_speed_weight = 1
input_weight = 1

# Set Global range (for state-space analysis)
distance_range = (5,50)     # Distance range in meters
rel_speed_range = (-5,5)    # Relative speed range in m/s
grid_resolution = (10,10)   # Grid resolution
num_samples = 20
N = 20

# Set Global deadtime
global_deadtime = 0.3
global_tolerance = 1e-5

# Initialize optimal input sequence
optimal_input_sequence = np.zeros(N)

# Define vehicle model for surrounding vehicle 1
def vehicle_model(state: Tuple[float, float, float], 
                 control_input: float,
                 aggressive: float = 0.8,
                 h: float = 1.0,
                 delta_min: float = 5.0,
                 T: float = 0.1,
                 deadtime: float = global_deadtime,
                 tolerance: float = global_tolerance) -> Tuple[float, float, float]:
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

# Define cost func
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

class DataCollection:
    def __init__(self):
        self.lon_distance = []
        self.lat_distance = []
        self.lon_relative_velocity = []
        self.lon_ego_velocity = []
        self.lon_velocity = []

    def update_value(self,lon_distance, lat_distance, 
                     lon_relative_velocity, lon_ego_velocity,
                     lon_velocity):
        self.lon_distance.append(lon_distance)
        self.lat_distance.append(lat_distance)
        self.lon_relative_velocity.append(lon_relative_velocity)
        self.lon_ego_velocity.append(lon_ego_velocity)
        self.lon_velocity.append(lon_velocity)

    def update_DataFrame(self):
        dict = {'longitudinal_distance': self.lon_distance, 'lateral_distance': self.lat_distance, 'longitudinal_relative_velocity': self.lon_relative_velocity,
                'longitudinal_ego_velocity': self.lon_ego_velocity,'longitudinal_velocity':self.lon_velocity} 
        df = pd.DataFrame(dict)
        return df
    
class Vehicle:
    def __init__(self, x, y, speed, color, ego = False):
        self.x = x
        self.y = y
        self.speed = speed              # Speed in pygame
        self.original_speed = speed 
        self.scaled_speed = 0           # Actual speed 
        self.color = color
        self.original_color = color
        self.previous_x = x             # Location of previous frame
        self.previous_time = time.time()# Time of previous frame 
        self.current_time = None        # Time of current frame
        self.time_interval = None       # Time interval between frame
        self.target_y = y               # Target lane during lane change
        self.data_collections = DataCollection()
        self.ego = ego                  # Check if it's ego vehicle
        self.ego_speed = 0
        self.acceleration = 0

    def accelerate(self):
        self.speed += self.acceleration * (self.time_interval)

    def move(self):
        # print(self.time_interval) if self.ego else None
        self.x += (self.speed - self.ego_speed)*(self.time_interval)
        # self.x += (self.speed - self.ego_speed)
        # self.x += (self.speed )*(self.time_interval)
    
    def draw(self, screen, number):
        # Draw the car 
        car = pygame.draw.rect(screen, self.color, (self.x, self.y, CAR_WIDTH, CAR_HEIGHT))
        font = pygame.font.Font(None,24)
        text = font.render(f'{number}' if self.ego == False else 'Ego', True, BLACK if self.ego else WHITE)
        text_loc = text.get_rect(center=car.center) 
        screen.blit(text, text_loc)

def draw_dashed_line(screen, color, start_pos, end_pos, dash_length = 10, offset = 0):
    """
    Draw a dashed line on the screen.
    :param screen: Pygame screen object.
    :param color: Line color.
    :param start_pos: Starting position (x, y).
    :param end_pos: Ending position (x, y).
    :param dash_length: Length of each dash.
    :param offset: Offset to move the dashes.
    """
    x1, y1 = start_pos
    x2, y2 = end_pos
    total_length = math.hypot(x2 - x1, y2 - y1)
    dashes = int(total_length // dash_length)

    # Adjust offset to loop within the dash length
    offset = offset % dash_length
    for i in range(dashes):
        # Calculate the start and end positions of each dash, considering the offset
        start = (
            x1 + (x2 - x1) * ((i * dash_length + offset) / total_length),
            y1 + (y2 - y1) * ((i * dash_length + offset) / total_length),
        )
        end = (
            x1 + (x2 - x1) * (((i + 0.5) * dash_length + offset) / total_length),
            y1 + (y2 - y1) * (((i + 0.5) * dash_length + offset) / total_length),
        )
        pygame.draw.line(screen, color, start, end, 2)

def calculate_x_distance(vehicle1, vehicle2):
    """
    Calculate the longitudinal (X-axis) distance between two vehicles.
    :param vehicle1: First vehicle (object).
    :param vehicle2: Second vehicle (object).
    :return: Longitudinal distance between the two vehicles.
    """
    
    return (vehicle2.x - vehicle1.x) - CAR_WIDTH

def calculate_y_distance(vehicle1, vehicle2):
    """
    Calculate the lateral (Y-axis) distance between two vehilces.
    :param vehicle1: First vehicle (object).
    :param vehicle2: Second vehicle (object).
    :return: Lateral distance between the two vehicles.
    """

    return vehicle2.y - vehicle1.y

class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("Lane Changing Simulation")
        self.clock = pygame.time.Clock()
        self.vehicles = self.initialize_vehicles(randomize = False)
        self.initialize_sur1_controller()
        self.offset = 0

    def initialize_vehicles(self, randomize = True):
        vehicles = []
        if randomize:
            for i in range(NUM_VEHICLES): # Randomly generate vehicles
                x = random.randint(0, WIDTH)
                lane = random.randint(0, NUM_LANES - 1)
                y = lane * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
                speed = random.randint(5, MAX_SPEED)
                color = GREEN if i == 0 else BLUE # First vehicle is the "ego" vehicle (green)
                vehicles.append(Vehicle(x, y, speed, color))
        else :
            # Ego Vehicle
            x_ego = 500
            lane_ego = 0
            y_ego = lane_ego * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            speed_ego = 100         # 10 
            vehicles.append(Vehicle(x_ego,y_ego,speed_ego,GREEN,ego=True))

            x_sur1 = 900
            lane_sur1 = 0
            y_sur1 = lane_sur1 * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            speed_sur1 = 80          # 8
            vehicles.append(Vehicle(x_sur1,y_sur1,speed_sur1,BLUE,ego=False))

            x_sur2 = 800
            lane_sur2 = 1
            y_sur2 = lane_sur2 * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            speed_sur2 = 60             # 6
            vehicles.append(Vehicle(x_sur2,y_sur2,speed_sur2,BLUE,ego=False))

        return vehicles
    
    def check_lane_change(self, ego):
        current_lane = ego.y // LANE_WIDTH
        for direction in [-1, 1]:       # Check both left (-1) and right (+1) lanes
            target_lane = current_lane + direction
            if 0 <= target_lane < NUM_LANES:
                target_y = target_lane * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
                # Check if the lane is safe
                if all(not (v.y == target_y and abs(calculate_x_distance(ego, v)) + (ego.speed - v.speed) * ego.time_interval < LANE_CHANGE_BUFFER) for v in self.vehicles):
                    return target_y , ego.speed # Return the new lane if safe
        return ego.y , ego.speed # Stay in the current lane if no safe option
        
    def rule_based_lane_change(self):
        for vehicle in self.vehicles:
            if vehicle.ego == True:
                ego = vehicle

        if any(v.y == ego.y and 0 < calculate_x_distance(ego, v) < LANE_CHANGE_BUFFER and not v.ego for v in self.vehicles):
            print("LESS THAN LANE CHANGE BUFFER")
            print(f"distance is {[calculate_x_distance(ego, v) for v in self.vehicles if v.y == ego.y and not v.ego]}")
            
            ego.target_y, ego.speed = self.check_lane_change(ego)

            # Check if lane change is possible
            safe_lane_change = ego.target_y != ego.y and self.check_lane_change(ego)
            print(f'Try Changing lane: {ego.target_y != ego.y}')
            if safe_lane_change:
                ego.target_y, ego.speed = self.check_lane_change(ego)  # Initiate lane change
            else:
                # No safe lane: Fallback to ACC
                print("No safe lane change possible, running ACC")
                self.ACC(ego)
        else:
            # Ego vehicle is already in target lane or no need for lane change
            if ego.speed >= ego.original_speed:
                print("ACC is WORKING")
                self.ACC(ego)
            else:
                print(f'{ego.speed}<{ego.original_speed}')
                print("Returning to OG Speed")
                self.return_to_og_speed(ego)

    def return_to_og_speed(self,ego):
        returning_acceleration = 40         # 2 m/s (40 pixels/s)
        ego.speed += returning_acceleration * ego.time_interval

    def initialize_sur1_controller(self):
        self.state_space = TrajectoryGenerator(distance_range, rel_speed_range, grid_resolution, num_samples, N, derivative= False)
        # ======================== SET MODE =========================== #
        self.mode = 'sampling'

        # initialize mpc
        sampling_mpc = SamplingBasedMPC(vehicle_model,cost_function, N, num_samples, self.state_space)
        optimize_mpc = OptimizationBasedMPC(vehicle_model,cost_function,N,self.state_space)

        self.mpc = sampling_mpc if self.mode == 'sampling' else optimize_mpc

        for vehicle in self.vehicles:
            if vehicle.ego :
                ego = vehicle

        for vehicle in self.vehicles:
            if vehicle.x >= ego.x and vehicle.y == ego.y and vehicle.ego == False:
                sur1 = vehicle
                break
        self.sur1 = sur1

    def surrounding1_controller(self):
        for vehicle in self.vehicles:
            if vehicle.ego :
                ego = vehicle
        
        # for vehicle in self.vehicles:
        #     if vehicle.x >= ego.x and vehicle.y == ego.y and vehicle.ego == False:
        #         sur1 = vehicle
        #         break
        sur1 = self.sur1

        # Mark visited grid and initialize state (with unit meter)
        distance = calculate_x_distance(ego,sur1)/20
        relative_speed = (sur1.scaled_speed - ego.scaled_speed)/20
        state = (distance,sur1.scaled_speed/20,ego.scaled_speed/20)
        self.state_space.marked_visited(distance, relative_speed)
        
        # solve for optimal input sequence
        if self.mode == 'optimize':
            optimal_input_sequence = self.mpc.solve(state,optimal_input_sequence)
        elif self.mode == 'sampling':
            optimal_input_sequence = self.mpc.solve(state)

        optimal_input = optimal_input_sequence[0]
        
        # update in state space analysis
        predicted_state = self.mpc.predict_states(state,optimal_input_sequence)
        predicted_state = [(state_[0],state_[1]-state_[2]) for state_ in predicted_state]

        self.state_space.add_predicted_state(predicted_state)

        # Set acceleration of surrounding car 1 (with unit pixel)
        sur1.acceleration = optimal_input*20
        print(f'sur1 acceleration is {sur1.acceleration}')


    def ACC(self, ego):
        aggressive = 0.8
        h = 1                   # Headway time
        delta_min = 5          # Minimum safe distance   (5 meter -> 100 pixels)
        deadtime = global_deadtime
        tolerance = global_tolerance

        # TODO: Need to adjust to Dynamic!!
        # for vehicle in self.vehicles:
        #     if vehicle.x >= ego.x and vehicle.y == ego.y and vehicle.ego == False:
        #         sur1 = vehicle
        #         break
        sur1 = self.sur1

        # Store historical control action
        if not hasattr(ego,'control_history'):
            ego.control_history = []

        current_time = time.time()

        # Add current state to history
        ego.control_history.append({
            'time': current_time,
            'distance': calculate_x_distance(ego,sur1),
            'preceding_speed': sur1.speed,
            'speed': ego.speed,
            'acceleration': ego.acceleration
        })

        # Remove old control actions beyond deadtime
        ego.control_history = [
            control for control in ego.control_history 
            if (current_time - control['time']) - deadtime <= tolerance
        ]

        # Get the delayed state (from deadtime ago)
        delayed_state = ego.control_history[0] if ego.control_history else None

        # Use delayed state if available, otherwise use current state
        # control_speed = delayed_state['speed'] if delayed_state else ego.speed
        delayed_d = delayed_state['distance'] if delayed_state else calculate_x_distance(ego,sur1)
        delayed_vp = delayed_state['preceding_speed'] if delayed_state else sur1.speed
        delayed_vf = delayed_state['speed'] if delayed_state else ego.speed

        for v in self.vehicles:
            if v.y == ego.y and v.x - ego.x > 0:
                delta = calculate_x_distance(ego, v)
                # desired_speed = max(0, (v.speed - aggressive * (h * ego.speed + delta_min*20 - delta)))
                # ego.acceleration = (desired_speed - ego.speed) / h
                # desired_speed = max(0, (v.speed - aggressive * (h * control_speed + delta_min*20 - delta)))
                # ego.acceleration = (desired_speed - control_speed) / h
                acc = (aggressive * delayed_d + delayed_vp - 
                        (1 + aggressive * h) * delayed_vf - 
                        aggressive * delta_min) / h
                ego.acceleration = acc
          
    def update(self):
        for vehicle in self.vehicles:
            if vehicle.ego == True:
                ego = vehicle
        # Rule-based lane changing
        self.rule_based_lane_change()

        # MPC for surrounding vehicle 1
        self.surrounding1_controller()

        # Update vehicle positions
        for vehicle in self.vehicles:
            current_time = time.time()
            vehicle.current_time = current_time
            distance_traveled = vehicle.x - vehicle.previous_x
            vehicle.scaled_speed = distance_traveled / (vehicle.current_time - vehicle.previous_time) + vehicle.ego_speed
            vehicle.time_interval = vehicle.current_time - vehicle.previous_time
            print(f'tell {vehicle.time_interval}') if vehicle.ego else None
            if vehicle.y != vehicle.target_y:
                vehicle.y += math.copysign(LANE_CHANGING_SPEED, vehicle.target_y - vehicle.y)
            vehicle.previous_x = vehicle.x 
            vehicle.previous_time = vehicle.current_time
            
            vehicle.accelerate()
            vehicle.move()
            

        # Collision detection and handling
        for vehicle1 in self.vehicles:
            rect1 = pygame.Rect(vehicle1.x, vehicle1.y, CAR_WIDTH, CAR_HEIGHT)
            collision_detected = False  # Track if this vehicle is in a collision
            for vehicle2 in self.vehicles:
                if vehicle1 != vehicle2:  # Avoid self-collision
                    rect2 = pygame.Rect(vehicle2.x, vehicle2.y, CAR_WIDTH, CAR_HEIGHT)
                    if rect1.colliderect(rect2):
                        # Collision detected: Change color and equalize speed
                        vehicle1.color = RED
                        vehicle2.color = RED
                        slower_speed = min(vehicle1.speed, vehicle2.speed)
                        vehicle1.speed = slower_speed
                        vehicle2.speed = slower_speed
                        collision_detected = True
            if not collision_detected:
                # No collisions for this vehicle: Reset color to original
                vehicle1.color = vehicle1.original_color
                vehicle2.color = vehicle2.original_color

        for i,vehicle in enumerate(self.vehicles[1:]):
            distance_x = calculate_x_distance(ego,vehicle)/20
            distance_y = calculate_y_distance(ego,vehicle)/20
            relative_speed = (vehicle.scaled_speed - ego.scaled_speed)/20
            vehicle.ego_speed = ego.speed
            vehicle.data_collections.update_value(distance_x,distance_y,relative_speed,ego.scaled_speed/20,vehicle.scaled_speed/20)

        ego.ego_speed = ego.speed
            


    def save_df(self):
        for i,vehicle in enumerate(self.vehicles[1:]):
            df = vehicle.data_collections.update_DataFrame()
            df.to_csv(f'Car_{i+1}_data_mod_delayed.csv',index=False)

    def show_verbose(self):
        ego = self.vehicles[0]
        distances_x = []
        distances_y = []
        for i,vehicle in enumerate(self.vehicles[1:]):
            distance_x = calculate_x_distance(ego, vehicle)
            distance_y = calculate_y_distance(ego, vehicle)
            distances_x.append(distance_x)
            distances_y.append(distance_y)
        
        font = pygame.font.Font(None, 24)
        distance_text = ['Distance from Ego car (in x-axis)']
        line_spacing = 10  # Spacing between lines
        for i,distance in enumerate(distances_x):
            distance_text.append(f'Car {i+1}: {distance:.2f} pixels -> {distance/20:.2f} meters')
        x,y = 100, LANE_WIDTH * NUM_LANES + 50
        for i, line in enumerate(distance_text):
            line_render = font.render(line, True, WHITE)
            self.screen.blit(line_render,(x,y+i*font.get_height()+line_spacing))

        distance_text = ['Distance from Ego car (in y-axis)']
        line_spacing = 10  # Spacing between lines
        for i,distance in enumerate(distances_y):
            distance_text.append(f'Car {i+1}: {distance:.2f} pixels -> {distance/20:.2f} meters')
        x,y = 500, LANE_WIDTH * NUM_LANES + 50
        for i, line in enumerate(distance_text):
            line_render = font.render(line, True, WHITE)
            self.screen.blit(line_render,(x,y+i*font.get_height()+line_spacing))

        speed_text = ['Speed (in x-axis)']
        for i,vehicle in enumerate(self.vehicles):
            if i == 0: 
                speed_text.append(f'Ego car: {vehicle.scaled_speed:.2f} pixels/s -> {vehicle.scaled_speed/20:.2f} m/s') 
            else:
                speed_text.append(f'Car {i}: {vehicle.scaled_speed:.2f} pixels/s -> {vehicle.scaled_speed/20:.2f} m/s') 
        
        x,y = 900, LANE_WIDTH * NUM_LANES + 50
        for i, line in enumerate(speed_text):
            line_render = font.render(line, True, WHITE)
            self.screen.blit(line_render,(x,y+i*font.get_height()+line_spacing))

    
    def draw(self):
        ego = self.vehicles[0]
        self.screen.fill(GREY)
        # Draw lanes
        pygame.draw.line(self.screen, WHITE, (0, 0), (WIDTH, 0), 2)
        pygame.draw.line(self.screen, WHITE, (0,NUM_LANES* LANE_WIDTH), (WIDTH, NUM_LANES* LANE_WIDTH), 2)
        self.offset -= ego.speed * ego.time_interval
        for i in range(1,NUM_LANES):
            draw_dashed_line(self.screen, WHITE, (0, i * LANE_WIDTH), (WIDTH, i * LANE_WIDTH), dash_length=20, offset=self.offset)
        # Draw vehicles
        for num,vehicle in enumerate(self.vehicles):
            vehicle.draw(self.screen,num)

        for vehicle in self.vehicles[1:]:
            distance = calculate_x_distance(ego, vehicle)
            font = pygame.font.Font(None, 24)
            distance_text = font.render(f"{distance:.2f}", True, BLACK)
            self.screen.blit(distance_text, (vehicle.x + CAR_WIDTH, vehicle.y))
        
        self.show_verbose()

        pygame.display.flip()

    def reset(self):
        "Reset simulation to initial state."
        self.vehicles = self.initialize_vehicles(randomize=False)
        print('Restart Simulation!')

    def run(self):
        running = True
        paused = True  # for pausing
        pause_start_time = time.time()  # Record pause start time
        total_pause_duration = 0
        self.update()
        self.draw()
        reset = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in {pygame.K_ESCAPE, pygame.K_q}:  # Check if ESC or Q is pressed
                        running = False
                    if event.key in {pygame.K_p, pygame.K_SPACE, pygame.K_RETURN}:
                        paused = not paused
                        if paused:
                            pause_start_time = time.time()  # Record when the pause started
                        else:
                            # Accumulate total paused duration
                            total_pause_duration += time.time() - pause_start_time
                            if reset :
                                for vehicle in self.vehicles:
                                    vehicle.previous_time = time.time() - total_pause_duration  # Reset timing for each vehicle
                                reset = False
                            pause_start_time = 0
                            # Adjust all vehicle's `previous_time` for accurate intervals
                            for vehicle in self.vehicles:
                                vehicle.previous_time += total_pause_duration
                            total_pause_duration = 0  # Reset for next pause
                    if event.key == pygame.K_r:
                        reset = not reset
                        self.reset()
                        self.update()
                        self.draw()

            if not paused:  # Only update and draw when not paused
                self.update()
                self.draw()
                self.save_df()

            if paused:
            
                font = pygame.font.Font(None, 48)
                text = font.render("Paused", True, RED)
                self.screen.blit(text, (WIDTH // 2 - 50, HEIGHT // 2))
                pygame.display.flip()

            self.clock.tick(FPS)
        print('Saving files!')
        save_path = get_unique_filepath('SamplingBasedMPC_pygame','state_space_animation','.gif')
        self.state_space.plot_stat_space(100,show = False, save_path=save_path)    
        pygame.quit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()

