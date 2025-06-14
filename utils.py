import os
import numpy as np
from typing import List, Tuple
from MPC_config import Q_weight, R_weight

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

# ======================================== Vehicle Model =============================================#

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

def vehicle_model_without_delay(state: Tuple[float, float, float], 
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



def new_vehicle_model(state: Tuple[float,float,float],
                      control_input: float,
                      aggressive: float = 0.8,
                      h: float = 1.0,
                      T: float = 1.0):
    """
    Vehicle dynamics model calculating next state based on current state and control input
    Without considering delta min (but will include in constraint) (add following velocity constraint manually)

    :param
        state: Tuple of (distance, preceding vehicle velocity, following vehicle velocity)
        control_input: Acceleration input for the preceding vehicle
        aggressive: Aggressiveness factor
        h: Time headway (s)
        T: Time step (s)
    :return: Tuple for next state (next_distance, next_vp, next_vf)
    """
    min_following_acc = -3
    max_following_acc = 3

    d, vp, vf = state
    next_d = d + (vp-vf) * T
    next_vp = vp + control_input * T 
    delta_min = 5.0

    following_acc = max(min_following_acc, min(max_following_acc, (aggressive * d + vp - (1+aggressive*h) * vf - aggressive * delta_min)/h))
    next_vf = vf + following_acc * T

    
    return next_d, next_vp, next_vf

# ======================================== cost function =============================================#

# Cost function considering one target (for heading goal): 
def cost_function1(
        target: Tuple[float,float],
        predicted_states: List[Tuple[float, float, float]],
        input_sequence: List[float]):
    """
    Calculate Cost from the predicted states (state cost) and input sequence (input cost)
    """
    cost = 0
    Q = Q_weight
    R = R_weight

    target_d, target_rel_speed = target

    # find cost for state difference
    for state in predicted_states:
        d, vp, vf = state
        d_diff = d - target_d
        rel_speed_diff = (vp - vf) - target_rel_speed
        cost = cost + d_diff * Q[0,0] * d_diff + rel_speed_diff * Q[1,1] * rel_speed_diff

    # find cost for input  
    for k in range(len(input_sequence)):
        if k > 0:
            previous_cont = input_sequence[k-1]
            diff_cont = input_sequence[k] - previous_cont
            cost = cost + diff_cont * R[0,0] * diff_cont
    
    return cost

# Cost function considering N steps of target (for tracking)
def cost_function(
        targets: List[Tuple[float, float]],
        predicted_states: List[Tuple[float, float, float]], 
        input_sequence: List[float]):
    """
    Calculate Cost from the predicted states (state cost) and input sequence (input cost)
    """
    cost = 0
    Q = Q_weight
    R = R_weight
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