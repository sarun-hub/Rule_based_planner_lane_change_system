import math, pygame
from utils.pygame_config import *

# =========================== FUNCTIONS ======================

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


def calculate_y_pos_from_lane_num(lane_num):
    """
    Convert the lane num into the y position in pygame
    the order is from top to bottom starts from 0 to NUM_LANES
    if more than NUM_LANES, it will return to the top.
    """
    if lane_num >= NUM_LANES :
        return calculate_y_pos_from_lane_num(lane_num % NUM_LANES)
    return lane_num * LANE_WIDTH + LANE_WIDTH // 2 - CAR_HEIGHT // 2
