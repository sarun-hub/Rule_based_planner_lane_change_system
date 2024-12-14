import pygame
import time
import random
import math
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1800, 600
LANE_WIDTH = 70.0
# NUM_LANES = HEIGHT // LANE_WIDTH
NUM_LANES = 2
FPS = 60
NUM_VEHICLES = 3

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GREY = (128,128,128)

# Vehicle properties
CAR_WIDTH, CAR_HEIGHT = 91, 35

# Simulation parameters
MAX_SPEED = 5
LANE_CHANGING_SPEED = 5
LANE_CHANGE_BUFFER = 200  # Minimum gap for lane change

class DataCollection:
    def __init__(self):
        self.lon_distance = []
        self.lat_distance = []
        self.lon_relative_velocity = []
        self.lon_ego_velocity = []
        self.lon_velocity = []

    def update_value(self,lon_distance,lat_distance,lon_relative_velocity,lon_ego_velocity,lon_velocity):
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
    def __init__(self, x, y, speed, color):
        self.x = x
        self.y = y
        self.speed = speed
        self.original_speed = speed
        self.scaled_speed = 0
        self.color = color
        self.original_color = color    # Save original color 
        self.previous_x = x         # Set previous x
        self.previous_time = time.time()         # Set previous time
        self.target_y = y  # Target lane during lane changes
        self.data_collections = DataCollection()

    def move(self):
        self.x += self.speed

    def draw(self, screen, number):
        if self.x > 0 :
            # Draw the car normally within the screen (not yet have visualization for backward loop: TODO)
            car = pygame.draw.rect(screen, self.color, (self.x, self.y, CAR_WIDTH, CAR_HEIGHT))
        else:
            car_previous = pygame.draw.rect(screen, self.color, (self.x+WIDTH, self.y, CAR_WIDTH , CAR_HEIGHT))
            car = pygame.draw.rect(screen, self.color, (0, self.y, CAR_WIDTH +self.x, CAR_HEIGHT))
        font = pygame.font.Font(None,24)
        text = font.render(f'{number}' if number != 0 else 'Ego',True, BLACK if number ==0 else WHITE)
        text_car = text.get_rect(center=car.center)
        screen.blit(text,text_car)

    def is_ahead_of(self,other_vehicle):
        if self.x > other_vehicle.x:
            return True
        elif self.x < other_vehicle.x:
            return False
        else:
            return None


def draw_dashed_line(screen, color, start_pos, end_pos, dash_length=10):
    """
    Draw a dashed line on the screen.
    :param screen: Pygame screen object.
    :param color: Line color.
    :param start_pos: Starting position (x, y).
    :param end_pos: Ending position (x, y).
    :param dash_length: Length of each dash.
    """
    x1, y1 = start_pos
    x2, y2 = end_pos
    total_length = math.hypot(x2 - x1, y2 - y1)
    dashes = int(total_length // dash_length)
    for i in range(dashes):
        start = (
            x1 + (x2 - x1) * (i / dashes),
            y1 + (y2 - y1) * (i / dashes),
        )
        end = (
            x1 + (x2 - x1) * ((i + 0.5) / dashes),
            y1 + (y2 - y1) * ((i + 0.5) / dashes),
        )
        pygame.draw.line(screen, color, start, end, 2)

def calculate_distance(vehicle1, vehicle2):
    """
    Calculate the Euclidean distance between two vehicles.
    :param vehicle1: First vehicle (object).
    :param vehicle2: Second vehicle (object).
    :return: Distance between the two vehicles.
    """
    dx = vehicle2.x - vehicle1.x
    dy = vehicle2.y - vehicle1.y
    return math.sqrt(dx ** 2 + dy ** 2)

def calculate_x_distance(vehicle1, vehicle2):
    """
    Calculate the longitudinal (X-axis) distance between two vehicles.
    :param vehicle1: First vehicle (object).
    :param vehicle2: Second vehicle (object).
    :return: Longitudinal distance between the two vehicles.
    """
    
    return min(abs(vehicle2.x - vehicle1.x),abs(abs(vehicle2.x - vehicle1.x)-WIDTH)) - CAR_WIDTH

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
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Lane Changing Simulation")
        self.clock = pygame.time.Clock()
        self.vehicles = self.initialize_vehicles(randomize=False)
        self.fig, self.axes = plt.subplots(1,2,figsize=(8, 3))  # Create a Matplotlib figure

    def initialize_vehicles(self,randomize = True):
        vehicles = []
        if randomize:
            for i in range(NUM_VEHICLES):  # Randomly generate vehicles
                x = random.randint(0, WIDTH)
                lane = random.randint(0, NUM_LANES - 1)
                y = lane * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
                speed = random.randint(5, MAX_SPEED)
                # speed = 1
                color = GREEN if i == 0 else BLUE  # First vehicle is the "ego" vehicle (green)
                vehicles.append(Vehicle(x, y, speed, color))

        else:
            # Ego Vehicle
            x_ego = 100
            lane_ego = 0
            y_ego = lane_ego * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            speed_ego = 10
            vehicles.append(Vehicle(x_ego,y_ego,speed_ego,GREEN))

            x_sur1 = 500
            lane_sur1 = 0
            y_sur1 = lane_sur1 * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            speed_sur1 = 8
            vehicles.append(Vehicle(x_sur1,y_sur1,speed_sur1,BLUE))

            x_sur2 = 400
            lane_sur2 = 1
            y_sur2 = lane_sur2 * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            speed_sur2 = 6
            vehicles.append(Vehicle(x_sur2,y_sur2,speed_sur2,BLUE))


            # x_sur3 = 0
            # lane_sur3 = 1
            # y_sur3 = lane_sur3 * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
            # speed_sur3 = 6
            # vehicles.append(Vehicle(x_sur3,y_sur3,speed_sur3,BLUE))


        return vehicles

    def check_lane_change(self, ego):
        current_lane = ego.y // LANE_WIDTH
        for direction in [-1, 1]:  # Check both left (-1) and right (+1) lanes
            target_lane = current_lane + direction
            if 0 <= target_lane < NUM_LANES:
                target_y = target_lane * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
                # Check if the lane is safe
                if all(not (v.y == target_y and abs(v.x - ego.x) + ego.speed - v.speed < LANE_CHANGE_BUFFER) for v in self.vehicles):
                    return target_y , ego.speed # Return the new lane if safe
        return ego.y , 0 # Stay in the current lane if no safe option

    def update(self):
        ego = self.vehicles[0]
        # Rule-based lane changing
        if any(v.y == ego.y and 0 < v.x - ego.x < LANE_CHANGE_BUFFER for v in self.vehicles):
            ego.target_y, ego.speed = self.check_lane_change(ego)
        if ego.speed == 0 :
            for v in self.vehicles:
                if v.y == ego.y and 0 < v.x - ego.x < LANE_CHANGE_BUFFER:
                    ego.speed = v.speed
        else:
            ego.speed = ego.original_speed

        # Check speed 
        for vehicle in self.vehicles:
            current_time = time.time()
            # Calculate the distance traveled, preserving the direction (TODO: looped around from left to right)
            if vehicle.x < 0 and vehicle.previous_x < WIDTH - CAR_WIDTH and vehicle.previous_x>0 :
                # Vehicle has looped around from right to left
                distance_traveled = (vehicle.x) - (vehicle.previous_x - WIDTH)
            else:
                # Normal case, no wrapping
                distance_traveled = vehicle.x - vehicle.previous_x

            # Calculate scaled_speed
            vehicle.scaled_speed = distance_traveled / (current_time - vehicle.previous_time) if current_time - vehicle.previous_time > 0 else 0
            if vehicle.scaled_speed/20 < 1 and vehicle.scaled_speed >0:
                print(f'x is {vehicle.x}, prev_x is {vehicle.previous_x} at problem')
            vehicle.previous_x = vehicle.x 
            vehicle.previous_time = current_time 

        # Update vehicle positions
        for vehicle in self.vehicles:
            if vehicle.y != vehicle.target_y:
                vehicle.y += math.copysign(LANE_CHANGING_SPEED, vehicle.target_y - vehicle.y)
            vehicle.move()
            if vehicle.x > WIDTH - CAR_WIDTH:
                vehicle.x = -CAR_WIDTH  # Loop from right to left
            elif vehicle.x < -CAR_WIDTH:
                vehicle.x = WIDTH - CAR_WIDTH  # Loop from left to right

        # Collision detection and handling
        for i, vehicle1 in enumerate(self.vehicles):
            rect1 = pygame.Rect(vehicle1.x, vehicle1.y, CAR_WIDTH, CAR_HEIGHT)
            for j, vehicle2 in enumerate(self.vehicles):
                if i != j:  # Avoid self-collision
                    rect2 = pygame.Rect(vehicle2.x, vehicle2.y, CAR_WIDTH, CAR_HEIGHT)
                    if rect1.colliderect(rect2):
                        # Collision detected: Change color and equalize speed
                        vehicle1.color = RED
                        vehicle2.color = RED
                        # Both vehicles take the slower speed
                        slower_speed = min(vehicle1.speed, vehicle2.speed)
                        vehicle1.speed = slower_speed
                        vehicle2.speed = slower_speed
                    else:
                        # No collision: Reset color to original
                        vehicle1.color = vehicle1.original_color
                        vehicle2.color = vehicle2.original_color

        ego = self.vehicles[0]

        for i,vehicle in enumerate(self.vehicles[1:]):
            distance_x = calculate_x_distance(ego,vehicle)/20
            distance_y = calculate_y_distance(ego,vehicle)/20
            relative_speed = (vehicle.scaled_speed - ego.scaled_speed)/20
            vehicle.data_collections.update_value(distance_x,distance_y,relative_speed,ego.scaled_speed/20,vehicle.scaled_speed/20)
            # self.plotters[i].add_data(distance_x, distance_y, relative_speed)
            
    def save_df(self):
        for i,vehicle in enumerate(self.vehicles[1:]):
            df = vehicle.data_collections.update_DataFrame()
            df.to_csv(f'Car_{i+1}_data.csv',index=False)


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
            distance_text.append(f'Car {i+1}: {distance} pixels -> {distance/20:.2f} meters')
        x,y = 100, LANE_WIDTH * NUM_LANES + 50
        for i, line in enumerate(distance_text):
            line_render = font.render(line, True, WHITE)
            self.screen.blit(line_render,(x,y+i*font.get_height()+line_spacing))

        distance_text = ['Distance from Ego car (in y-axis)']
        line_spacing = 10  # Spacing between lines
        for i,distance in enumerate(distances_y):
            distance_text.append(f'Car {i+1}: {distance} pixels -> {distance/20:.2f} meters')
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
        for i in range(1,NUM_LANES):
            draw_dashed_line(self.screen, WHITE, (0, i * LANE_WIDTH), (WIDTH, i * LANE_WIDTH), dash_length=20)
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
        plot = True
        self.update()
        self.draw() 
        while running:
            frame_start_time = time.time()  # Time at the start of the frame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:  # Check if ESC or Q is pressed
                        running = False
                    if event.key == pygame.K_p or event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        paused = not paused
                    if event.key == pygame.K_r:
                        self.reset()
                        self.update()
                        self.draw()
                    if event.key == pygame.K_a:
                        plot = not plot

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
        pygame.quit()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
