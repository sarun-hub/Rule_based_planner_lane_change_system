import numpy as np
from typing import List, Tuple
from utils.utils import (
    get_discretize_matrix,
    compute_next_state,
    compute_affine_shift,
    convert_state3d_to_state2d,
)
from utils.vehicle_utils import load_acc_config

# ============ MPC config ====================================
# (might move to other files)

Q_weight = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 0]))

R_weight = np.array(([1],))

distance_constraint = (5, 50)

# ============ Vehicle Model & Cost Function ================


# Discretized vehicle_model
def discretized_vehicle_model(
    state: Tuple[float, float, float],
    control_input: float,
    T: float = 0.1,
):
    aggressive, h, delta_min = load_acc_config()
    # setup system
    A_u = np.array(
        [[0, 1, -1], [0, 0, 0], [aggressive / h, 1 / h, -(1 + aggressive * h) / h]],
        dtype=float,
    )
    B_u = np.array([[0], [1], [0]])
    C_u = np.array([[1, 0, 0], [0, 1, -1]], dtype=float)
    D_u = np.array([[0], [0]], dtype=float)
    offset = np.array([[0], [0], [-aggressive / h * delta_min]])

    # use augmented matrix to find discretized matrix
    A_aug = np.block([[A_u, offset], [np.zeros((1, 4))]])
    B_aug = np.vstack((B_u, np.zeros((1, 1))))
    C_aug = np.hstack((C_u, np.zeros((2, 1))))
    D_aug = np.zeros((2, 1))

    A_d_aug, B_d_aug, C_d_aug, _ = get_discretize_matrix(A_aug, B_aug, C_aug, D_aug)

    # Use A 3x3 from augmented system
    A_d = A_d_aug[:3, :3]
    B_d = B_d_aug[:3, :]
    C_d = C_d_aug[:, :3]
    offset_d = A_d_aug[:3, 3:4]  # the last column (only 3 rows) is offset

    # predict the next state from control input
    next_state = compute_next_state(state, control_input, A_d, B_d, offset_d)
    return next_state


# Cost function for one target
def cost_function(
    target: Tuple[float, float, float],
    predicted_states: List[Tuple[float, float, float]],
    input_sequence: List[float],
):

    cost = 0
    Q = Q_weight
    R = R_weight

    target_d, target_rel_speed = convert_state3d_to_state2d(target)

    # find cost for state difference ()
    for state in predicted_states:
        d, vp, vf = state
        d_diff = d - target_d
        rel_speed_diff = (vp - vf) - target_rel_speed
        cost = (
            cost + d_diff * Q[0, 0] * d_diff + rel_speed_diff * Q[1, 1] * rel_speed_diff
        )

    # find cost for input
    for k in range(len(input_sequence)):
        if k > 0:
            previous_cont = input_sequence[k - 1]
            diff_cont = input_sequence[k] - previous_cont
            cost = cost + diff_cont * R[0, 0] * diff_cont

    return cost


# =================== MPC system ============================


class SamplingBasedMPC:
    def __init__(self, model, cost_function, N: int, num_samples: int):
        self.model = model
        self.cost_function = cost_function
        self.N = N
        self.num_samples = num_samples
        self.max_acc = 3
        self.min_acc = -3

    def generate_random_inputs(self):
        """
        Generate random input sequeences for N step
        """
        return [
            np.random.uniform(low=self.min_acc, high=self.max_acc, size=self.N)
            for _ in range(self.num_samples)
        ]

    def predict_states(self, initial_state, input_sequence):
        """
        Generate states (for N steps) from input sequences (acceleration)

        :param
            initial_state: initial state [current state] (distance, preceding speed, following speed)
            input_sequence: list of input (acceleration)
        :return: List of state (predicted states for N steps)
        """
        states = [initial_state]
        state = initial_state
        for u in input_sequence:
            state = self.model(state, u)
            states.append(state)
        return states

    def compute_costs(self, initial_state, input_sequences, target):
        """
        Compute costs of all random input sequences

        :param
            initial_state: initial state [current state] (distance, preceding speed, following speed)
            input_sequences: list of all random input sequences (acceleration)
        :return: List of cost (for all random inputs of num_samples)
        """
        costs = []
        for input_sequence in input_sequences:
            predicted_states = self.predict_states(initial_state, input_sequence)
            cost = self.cost_function(target, predicted_states, input_sequence)
            if not np.all(
                distance_constraint[0] <= state[0] <= distance_constraint[1]
                for state in predicted_states
            ):
                cost = cost + 1e6
            costs.append(cost)
        
        return costs

    def select_optimal_input_sequence(self, input_sequences, costs):
        """
        Select the minimum cost input sequence

        :param
            input_sequences: list of all random input sequences
            costs: list of cost of all random input sequences
        :return: input sequence with the least cost and that cost
        """
        min_cost_index = np.argmin(costs)
        return input_sequences[min_cost_index], costs[min_cost_index]

    def solve(self, initial_state, target):
        """
        Aggrregate all functions to run at once
        :param initial_state: initial state [current state] (distance, preceding speed, following speed)
        :param target_state: target state [target state] (distance, preceding speed, following speed)
        :return: optimal input sequence
        """
        input_sequences = self.generate_random_inputs()
        costs = self.compute_costs(initial_state, input_sequences, target)
        optimal_input_sequence, _ = self.select_optimal_input_sequence(
            input_sequences, costs
        )
        return optimal_input_sequence


class OptimizedBasedMPC:
    pass
