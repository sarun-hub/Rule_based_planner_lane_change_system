import pygame, time
import random
from utils.pygame_config import *
from utils.pygame_utils import (
    draw_dashed_line,
    calculate_x_distance,
    calculate_y_distance,
    calculate_y_pos_from_lane_num,
)
from utils.vehicle_utils import Vehicle, ACC_controller

# ======================== pygame simulator ==================

# Initialize pygame
pygame.init()

# Simulation Parameters
MAX_SPEED = 5
# Unused function (TODO(20/12/2025): later feature)
LANE_CHANGING_SPEED = 5
LANE_CHANGE_BUFFER = 200  # Minimum gap for lane change


class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("ACC Simulation")
        self.clock = pygame.time.Clock()
        self.vehicles = self.initialize_vehicles(randomize=False)
        self.sur1 = self.initialize_sur1()  # find sur1
        self.offset = 0

    def initialize_vehicles(self, randomize=True):
        vehicles = []
        # Randomly generate vehicles speed and location
        if randomize:
            for i in range(NUM_VEHICLES):
                x = random.randint(0, WIDTH)
                lane = random.randint(0, NUM_LANES - 1)
                y = lane * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
                speed = random.randint(
                    5, MAX_SPEED
                )  # Max speed is 5 (so currently fix to be 5)
                color = (
                    GREEN if i == 0 else BLUE
                )  # First car is always ego (green), else are blue
                vehicles.append(Vehicle(x, y, speed, color))
        else:
            # Ego car
            x_ego = 500
            lane_ego = 0
            y_ego = calculate_y_pos_from_lane_num(lane_ego)
            speed_ego = 100  # 5 m/s
            ego = Vehicle(x_ego, y_ego, speed_ego, GREEN, ego=True)
            vehicles.append(ego)

            # =================== Sur 1 =================
            x_sur1 = 900
            lane_sur1 = 0
            y_sur1 = calculate_y_pos_from_lane_num(lane_sur1)
            speed_sur1 = 80  # 4 m/s
            vehicles.append(Vehicle(x_sur1, y_sur1, speed_sur1, BLUE, ego=False))

            # Other Surs (TODO: 21/12/2025)
        return vehicles

    def initialize_sur1(self):
        sur1 = None
        if len(self.vehicles) < 2:
            return None

        for vehicle in self.vehicles:
            if vehicle.ego:
                ego = vehicle

        # TODO: 21/12/2025 - Now it chooses the first car on the same lane that is not ego to be sur1,
        # Need to make it check the only the first front car
        for vehicle in self.vehicles:
            if (
                vehicle.x >= ego.x
                and ego.y - LANE_WIDTH <= vehicle.y <= ego.y + LANE_WIDTH
                and vehicle.ego == False
            ):
                sur1 = vehicle
                break
        # if not found, return None
        return sur1

    def update_vehicle(self):
        for vehicle in self.vehicles:
            current_time = time.time()
            vehicle.current_time = current_time
            distance_traveled = vehicle.x - vehicle.previous_x
            interval = current_time - vehicle.previous_time
            if interval <= 1e-6:
                interval = 1e-6
            vehicle.time_interval = interval
            vehicle.scaled_speed = distance_traveled / (vehicle.time_interval) + vehicle.ego_speed

            vehicle.previous_x = vehicle.x
            vehicle.previous_time = vehicle.current_time

            # TODO 21/12/2025 - doesn't consider lane change

            vehicle.accelerate()
            vehicle.move()

    def check_collision(self):
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
        # # 1. Reset all vehicles first (every frame)
        # for vehicle in self.vehicles:
        #     vehicle.color = vehicle.original_color

        # # 2. Check pairwise collisions
        # for i, vehicle1 in enumerate(self.vehicles):
        #     rect1 = pygame.Rect(vehicle1.x, vehicle1.y, CAR_WIDTH, CAR_HEIGHT)

        #     for vehicle2 in self.vehicles[i + 1 :]:  # avoid double-check
        #         rect2 = pygame.Rect(vehicle2.x, vehicle2.y, CAR_WIDTH, CAR_HEIGHT)

        #         if rect1.colliderect(rect2):
        #             # Collision detected
        #             vehicle1.color = RED
        #             vehicle2.color = RED

        #             slower_speed = min(vehicle1.speed, vehicle2.speed)
        #             vehicle1.speed = slower_speed
        #             vehicle2.speed = slower_speed

    def collect_data(self, ego):
        for _, vehicle in enumerate(self.vehicles[1:]):
            distance_x = calculate_x_distance(ego, vehicle) / PIXEL_PER_METER
            distance_y = calculate_y_distance(ego, vehicle) / PIXEL_PER_METER
            relative_speed = (vehicle.scaled_speed - ego.scaled_speed) / PIXEL_PER_METER
            vehicle.ego_speed = ego.speed
            vehicle.data_collections.update_value(
                distance_x,
                distance_y,
                relative_speed,
                ego.scaled_speed / PIXEL_PER_METER,
                vehicle.scaled_speed / PIXEL_PER_METER,
            )

    def update(self):
        for vehicle in self.vehicles:
            if vehicle.ego == True:
                ego = vehicle

        sur1 = self.sur1

        # get current info of ego, and front car to calcualte acceleration
        if sur1 is not None:
            ACC_controller(ego, sur1)

        # update vehicle position and speed
        self.update_vehicle()

        # check the vehicle collision
        self.check_collision()

        # collect data in each car data
        self.collect_data(ego)

        ego.ego_speed = ego.speed

    def save_df(self):
        for i, vehicle in enumerate(self.vehicles[1:]):
            df = vehicle.data_collections.update_DataFrame()
            df.to_csv(f"output/Car_{i+1}_data.csv", index=False)

    def show_verbose(self):
        ego = self.vehicles[0]
        distances_x = []
        distances_y = []
        for i, vehicle in enumerate(self.vehicles[1:]):
            distance_x = calculate_x_distance(ego, vehicle)
            distance_y = calculate_y_distance(ego, vehicle)
            distances_x.append(distance_x)
            distances_y.append(distance_y)

        font = pygame.font.Font(None, 24)
        line_spacing = 10

        def write_verbose(text_list, x, y):
            for i, line in enumerate(text_list):
                line_render = font.render(line, True, WHITE)
                self.screen.blit(
                    line_render, (x, y + i * font.get_height() + line_spacing)
                )

        # For x distance
        distance_x_text = ["Distance from Ego car (in x-axis)"]
        for i, distance in enumerate(distances_x):
            distance_x_text.append(
                f"Car {i+1}: {distance:.2f} pixels -> {distance/PIXEL_PER_METER:.2f} meters"
            )

        # For y distance
        distance_y_text = ["Distance from Ego car (in y-axis)"]
        for i, distance in enumerate(distances_y):
            distance_y_text.append(
                f"Car {i+1}: {distance:.2f} pixels -> {distance/PIXEL_PER_METER:.2f} meters"
            )

        # For speed text
        speed_text = ["Speed (in x-axis)"]
        for i, vehicle in enumerate(self.vehicles):
            if i == 0:
                speed_text.append(
                    f"Ego car: {vehicle.scaled_speed:.2f} pixels/s -> {vehicle.scaled_speed/PIXEL_PER_METER:.2f} m/s"
                )
            else:
                speed_text.append(
                    f"Car {i}: {vehicle.scaled_speed:.2f} pixels/s -> {vehicle.scaled_speed/PIXEL_PER_METER:.2f} m/s"
                )

        write_verbose(distance_x_text, 100, LANE_WIDTH * NUM_LANES + 50)
        write_verbose(distance_y_text, 500, LANE_WIDTH * NUM_LANES + 50)
        write_verbose(speed_text, 900, LANE_WIDTH * NUM_LANES + 50)

    def draw(self):
        ego = self.vehicles[0]
        self.screen.fill(GREY)

        # Draw lanes
        pygame.draw.line(self.screen, WHITE, (0, 0), (WIDTH, 0), 2)
        pygame.draw.line(
            self.screen,
            WHITE,
            (0, NUM_LANES * LANE_WIDTH),
            (WIDTH, NUM_LANES * LANE_WIDTH),
            2,
        )

        # Move screen backward with ego speed to lock the ego car on the screen
        self.offset -= ego.speed * ego.time_interval

        for i in range(1, NUM_LANES):
            draw_dashed_line(
                self.screen,
                WHITE,
                (0, i * LANE_WIDTH),
                (WIDTH, i * LANE_WIDTH),
                dash_length=20,
                offset=self.offset,
            )

        # Draw vehicles
        for num, vehicle in enumerate(self.vehicles):
            vehicle.draw(self.screen, num)

        for vehicle in self.vehicles[1:]:
            distance = calculate_x_distance(ego, vehicle)
            font = pygame.font.Font(None, 24)
            distance_text = font.render(f"{distance:.2f}", True, BLACK)
            self.screen.blit(distance_text, (vehicle.x + CAR_WIDTH, vehicle.y))

        # show verbose
        self.show_verbose()

        pygame.display.flip()

    def reset(self):
        "Reset simulation to initial state."
        self.vehicles = self.initialize_vehicles(randomize=False)
        print("Restart Simulation!")

    def run(self):
        running = True
        paused = True  # for pausing
        pause_start_time = time.time()  # Record pause start time
        total_pause_duration = 0
        # Initialize =====
        self.update()
        self.draw()
        # ================
        reset = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in {
                        pygame.K_ESCAPE,
                        pygame.K_q,
                    }:  # Check if ESC or Q is pressed
                        running = False
                    if event.key in {pygame.K_p, pygame.K_SPACE, pygame.K_RETURN}:
                        paused = not paused
                        if paused:
                            pause_start_time = (
                                time.time()
                            )  # Record when the pause started
                        else:
                            # Accumulate total paused duration
                            total_pause_duration += time.time() - pause_start_time
                            if reset:
                                for vehicle in self.vehicles:
                                    vehicle.previous_time = (
                                        time.time() - total_pause_duration
                                    )  # Reset timing for each vehicle
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
        # print('Saving files!')
        # save_path = get_unique_filepath('SamplingBasedMPC_pygame','state_space_animation','.gif')
        # self.state_space.plot_stat_space(100,show = False, save_path=save_path)
        pygame.quit()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()
