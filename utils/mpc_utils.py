from casadi import *
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

Q_weight = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1]))

R_weight = np.array(([1],))

distance_constraint = (5, 50)

# ============ Vehicle Model & Cost Function ================


# Discretized vehicle_model
def discretized_vehicle_model(
    state: Tuple[float, float, float],
    control_input: float,
    dt: float = 0.1,
    return_tuple: bool = True,
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

    A_d_aug, B_d_aug, C_d_aug, _ = get_discretize_matrix(A_aug, B_aug, C_aug, D_aug, dt)

    # Use A 3x3 from augmented system
    A_d = A_d_aug[:3, :3]
    B_d = B_d_aug[:3, :]
    C_d = C_d_aug[:, :3]
    offset_d = A_d_aug[:3, 3:4]  # the last column (only 3 rows) is offset

    # predict the next state from control input
    next_state = compute_next_state(
        state, control_input, A_d, B_d, offset_d, return_tuple
    )

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

    # use 3d simulation
    target_d, target_vp, target_vf = target
    print(target_d)
    print(target_vp)
    print(target_vf)

    # find cost for state difference ()
    for state in predicted_states:
        d, vp, vf = state
        d_diff = d - target_d
        vp_diff = vp - target_vp
        vf_diff = vf - target_vf

        # TODO: 24/12/2025 ⭐⭐⭐ now quick fix due to x_p in conversion target array inside tuple problem
        diff_state = np.array([d_diff, vp_diff, vf_diff])  # 3x1
        cost += diff_state.T @ Q @ diff_state  # 1x1

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


class OptimizationBasedMPC:
    def __init__(self, model, cost_function, N: int):
        # self.model = self._intialize_model(model)
        self.model = model
        self.cost_function = cost_function
        self.N = N
        self.distance_weight = 1
        self.rel_speed_weight = 1
        self.input_weight = 1

        self.aggressive, self.h, self.delta_min = load_acc_config()
        # TODO: 23/12/2025 ⭐⭐⭐
        # currently set to be 0.1, but it should refer to current dt (from FPS)
        self.T = 0.1

        # Initialize CasADi variables
        self.d = SX.sym("d")
        self.vp = SX.sym("vp")
        self.vf = SX.sym("vf")
        self.ap = SX.sym("ap")

        # State and control variables
        self.states = vertcat(self.d, self.vp, self.vf)
        self.n_states = self.states.numel()
        self.controls = vertcat(self.ap)
        self.n_controls = self.controls.numel()

        # Prediction and parameter variables
        self.U = SX.sym("U", self.n_controls, self.N)
        self.P = SX.sym("P", self.n_states + self.n_states)
        self.X = SX.sym("X", self.n_states, N + 1)

        self._build_dynamics()

    def _initialize_model(self, model):
        """Convert matrix model into SX model"""

    def _build_dynamics(self):
        """Define the vehicle dynamic"""
        # discretized model
        next_state = self.model(
            (self.d, self.vp, self.vf), self.ap, self.T, return_tuple=False
        )
        rhs = vertcat(DM(next_state))

        self.f = Function("f", [self.states, self.controls], [rhs])

    def _build_solver(
        self,
        current_state: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ):
        """Build the optimization problem"""
        self.X[:, 0] = self.P[: self.n_states]

        # Populate predicted using dynamics
        for k in range(self.N):
            st = self.X[:, k]
            cont = self.U[:, k]
            f_value = self.f(st, cont)
            # Since the f function give the next_state
            self.X[:, k + 1] = f_value

        obj = self.set_objective_function(current_state, target)

        # Constraint
        g = []
        for k in range(self.N + 1):
            g = vertcat(g, self.X[0, k])
            g = vertcat(g, self.X[1, k])
            g = vertcat(g, self.X[2, k])

            # a_f (following acceleration) limit
            dynamic_constraint = (
                self.aggressive * self.X[0, k]
                + self.X[1, k]
                - (1 + self.aggressive * self.h) * self.X[2, k]
                - self.aggressive * self.delta_min
            ) / (self.h + 1e-6)
            g = vertcat(g, dynamic_constraint)

        opt_variables = reshape(self.U, (self.n_controls * self.N, 1))

        # Set up the problem
        nlp_prob = {"f": obj, "x": opt_variables, "g": g, "p": self.P}

        opts = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        self.solver = nlpsol("solver", "ipopt", nlp_prob, opts)

    def set_objective_function(
        self,
        current_state: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ):

        # Objective function
        obj = 0

        Q = SX.zeros(self.n_states, self.n_states)  # Weight matrix of states diff

        Q[0, 0] = self.distance_weight
        Q[0, 1] = 0
        Q[1, 1] = self.rel_speed_weight
        Q[1, 0] = 0
        R = SX.zeros(self.n_controls, self.n_controls)  # Weight matrix of control diff
        R[0, 0] = self.input_weight

        # Objective function for states diff
        for k in range(self.N):

            target_d, target_vp, target_vf = target
            d_st = self.X[0, k]
            vp_st = self.X[1, k]
            vf_st = self.X[2, k]
            d_diff = d_st - target_d
            vp_diff = vp_st - target_vp
            vf_diff = vf_st - target_vf

            diff_state = np.array([[d_diff], [vp_diff], [vf_diff]])  # 3x1

            obj += diff_state.T @ Q @ diff_state  # 1x1

        # Objective function for input diff
        for k in range(self.N):
            if k > 0:
                cont = self.U[:, k]
                previous_cont = self.U[:, k - 1]
                diff_cont = cont - previous_cont
                obj += diff_cont.T @ R @ diff_cont

        return obj

    def predict_states(
        self, current_state: Tuple[float, float, float], input_sequence: List[float]
    ) -> List[Tuple[float, float, float]]:
        """
        Generate states (for N steps) from input sequences (acceleration)

        :param
            initial_state: initial state [current state] (distance, preceding speed, following speed)
            input_sequence: list of input (acceleration)
        :return: List of state (predicted states for N steps)
        """
        states = [current_state]
        state = current_state
        for u in input_sequence:
            state = self.model(state, u)
            states.append(state)
        return states

    def solve(
        self,
        current_state: Tuple[float, float, float],
        current_u: float,
        target: Tuple[float, float, float],
    ) -> List[float]:
        """Solve the MPC optimization problem."""

        self._build_solver(current_state, target)
        arg = {}

        # preceding acceleration
        arg["lbx"] = -2  # -4
        arg["ubx"] = 2  # 4

        # Set upper and lower bounds for distance and speed separately
        g_lb = []
        g_ub = []

        for _ in range(self.N + 1):
            # Distance bounds
            g_lb.append(5)  # Lower bound for distance
            g_ub.append(120)  # 120 #60 # Upper bound for distance

            # Speed bounds
            g_lb.append(0)  # Lower bound for preceding speed
            g_ub.append(float("inf"))  # 43.7  # Upper bound for preceding speed

            g_lb.append(0)  # -inf # Lower bound for following speed
            g_ub.append(43.8)  # 43.8   # Upper bound for following speed

            g_lb.append(
                -2
            )  # -2 # Lower bound for dynamic constraint (following acceleration)
            g_ub.append(
                2
            )  # 2 # Upper bound for dynamic constraint (following acceleration)

        arg["lbg"] = g_lb
        arg["ubg"] = g_ub

        arg["p"] = vertcat(*current_state, *target)
        arg["x0"] = DM(reshape(current_u, (self.n_controls, 1)))

        sol = self.solver(
            x0=arg["x0"],
            lbx=arg["lbx"],
            ubx=arg["ubx"],
            lbg=arg["lbg"],
            ubg=arg["ubg"],
            p=arg["p"],
        )
        u = reshape(sol["x"].T, self.n_controls, self.N)
        return u.full().flatten().tolist()
