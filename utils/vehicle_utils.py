import pygame
from utils.pygame_config import *
import pandas as pd

# Note: In pygame, the pixel is * 20 of actual data


class DataCollection:
    def __init__(self):
        self.lon_distance = []
        self.lat_distance = []
        self.lon_relative_velocity = []
        self.lon_ego_velocity = []
        self.lon_velocity = []

    def update_value(
        self,
        lon_distance,
        lat_distance,
        lon_relative_velocity,
        lon_ego_velocity,
        lon_velocity,
    ):
        self.lon_distance.append(lon_distance)
        self.lat_distance.append(lat_distance)
        self.lon_relative_velocity.append(lon_relative_velocity)
        self.lon_ego_velocity.append(lon_ego_velocity)
        self.lon_velocity.append(lon_velocity)

    def update_DataFrame(self):
        data = {
            "longitudinal_distance": self.lon_distance,
            "lateral_distance": self.lat_distance,
            "longitudinal_relative_velocity": self.lon_relative_velocity,
            "longitudinal_ego_velocity": self.lon_ego_velocity,
            "longitudinal_velocity": self.lon_velocity,
        }
        df = pd.DataFrame(data)
        return df


class Vehicle:
    def __init__(self, x, y, speed, color, ego=False):
        self.x = x
        self.y = y
        self.speed = speed  # Speed in pygame (pixel/s)
        self.color = color
        self.original_color = color
        self.time_interval = 0  # Time interval between frame
        self.target_y = y  # Target lane during lane change
        self.data_collections = DataCollection()
        self.ego = ego  # Check if it's ego vehicle
        self.ego_speed = 0
        self.acceleration = 0

    def accelerate(self):
        self.speed += self.acceleration * (self.time_interval)

    # move compare to ego car (since the screen is moving with ego_speed)
    def move(self):
        self.x += (self.speed - self.ego_speed) * (self.time_interval)

    def draw(self, screen, number):
        # Draw the car
        car = pygame.draw.rect(
            screen, self.color, (self.x, self.y, CAR_WIDTH, CAR_HEIGHT)
        )
        font = pygame.font.Font(None, 24)
        text = font.render(
            f"{number}" if self.ego == False else "Ego",
            True,
            BLACK if self.ego else WHITE,
        )
        text_loc = text.get_rect(center=car.center)
        screen.blit(text, text_loc)


# =============== Controller ========================
def load_acc_config(aggressive=0.8, h=1, delta_min=5):
    return aggressive, h, delta_min


def ACC_controller(ego: Vehicle, front_car: Vehicle):
    aggressive, h, delta_min = load_acc_config()
    dist = ((front_car.x - ego.x) - CAR_WIDTH) / PIXEL_PER_METER
    vp = front_car.speed / PIXEL_PER_METER
    vf = ego.speed / PIXEL_PER_METER

    if front_car is None:
        ego.acceleration = 0.0  # no acceleration when no front car
    else:
        acc = (
            aggressive * dist + vp - (1 + aggressive * h) * vf - aggressive * delta_min
        ) / h
        ego.acceleration = acc * PIXEL_PER_METER
