from casadi import *
import numpy as np
from typing import List, Tuple
from utils.utils import (
    get_discretize_matrix,
    compute_next_state,
    compute_affine_shift,
    compute_orthogonal_projection_matrix
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
        state, control_input, A_d, B_d, offset=0, return_tuple=return_tuple
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
        self.model = model  # Set up Vehicle Model
        self.cost_function = cost_function  # for cost function (TODO: 09/01/2026 -> use it, now manually calcualte cost)
        self.N = N  # Number of Steps (prediction horizon)

        # Setup Weights
        self.distance_weight = 1.0
        self.preceding_speed_weight = 1.0
        self.following_speed_weight = 1.0
        self.input_weight = 1.0

        self.aggressive, self.h, self.delta_min = load_acc_config()
        # TODO: 23/12/2025 ⭐⭐⭐
        # currently set to be 0.1, but it should refer to current dt (from FPS)
        self.T = 0.1

        # dimensions
        self.n_states = 3
        self.n_controls = 1

        # CasADi symbols
        self.x = SX.sym("x", self.n_states)  # state (d, vp, vf)
        self.u = SX.sym("u", self.n_controls)  # control
        self.U = SX.sym(
            "U", self.n_controls, N
        )  # control sequence (prediction horizon)
        self.P = SX.sym("P", 2 * self.n_states)  # [x0, x_target]

        self._build_dynamics()
        self._build_solver()

    def _build_dynamics(self):
        next_state = self.model(
            (self.x[0], self.x[1], self.x[2]), self.u, self.T, return_tuple=False
        )

        # right hand size
        rhs = vertcat(next_state)

        self.f = Function("f", [self.x, self.u], [rhs])

    def _build_solver(self):
        Q = SX.zeros(self.n_states, self.n_states)  # Weight matrix of states diff

        Q[0, 0] = self.distance_weight
        Q[1, 1] = self.preceding_speed_weight
        Q[2, 2] = self.following_speed_weight

        R = SX.zeros(self.n_controls, self.n_controls)  # Weight matrix of control diff
        R[0, 0] = self.input_weight

        g = []
        # set objective function
        obj = 0

        # initial state
        xk = self.P[: self.n_states]
        # target state
        x_target = self.P[self.n_states :]

        # iterate through prediction horizon
        for k in range(self.N):

            # Cost from state
            diff_state = xk - x_target
            obj += diff_state.T @ Q @ diff_state

            # Cost from control (for control input smoothness)
            uk = self.U[:, k]
            # calculate obj from uk
            obj += uk.T @ R @ uk
            #  calculate obj from uk_diff
            # if k > 0:
            #     diff_control = self.U[:, k] - self.U[:, k - 1]
            #     obj += diff_control.T @ R @ diff_control

            # get next state
            xk = self.f(xk, uk)

            # state constraints
            g.append(xk[0])  # distance
            g.append(xk[1])  # vp
            g.append(xk[2])  # vf

            # ACC dynamic constraint (for ACC acceleration)
            dyn = (
                self.aggressive * xk[0]
                + xk[1]
                - (1 + self.aggressive * self.h) * xk[2]
                - self.aggressive * self.delta_min
            )  # scaled h

            g.append(dyn)

        g = vertcat(*g)

        opt_vars = reshape(self.U, self.n_controls * self.N, 1)

        nlp_prob = {
            "f": obj,  # objective function (cost)
            "x": opt_vars,  # control input param
            "g": g,  # constraint
            "p": self.P,  # initial state and target state
        }

        opts = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        self.solver = nlpsol("solver", "ipopt", nlp_prob, opts)

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
        # bound for control
        lbx = [-2.0] * self.N
        ubx = [2.0] * self.N

        # constraints for state (4 per steps - d, vp, vf, dyn)
        lbg = []
        ubg = []
        for _ in range(self.N):
            lbg += [5.0, 0.0, 0.0, -2.0 * self.h]  # scaled h
            ubg += [120.0, inf, 43.8, 2.0 * self.h]

        p = vertcat(*current_state, *target)
        sol = self.solver(
            x0=[0.0] * self.N,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=p,
        )

        u_opt = sol["x"].full().flatten()
        return u_opt.tolist()


class OptimizationBasedMPC_reachable_constraint:
    def __init__(self, model, A_d, offset_d, N: int):
        self.model = model  # Set up Vehicle Model
        self.A_d = A_d
        self.N = N  # Number of Steps (prediction horizon)

        # Setup Weights
        self.distance_weight = 1.0
        self.preceding_speed_weight = 1.0
        self.following_speed_weight = 1.0
        self.input_weight = 1.0

        self.aggressive, self.h, self.delta_min = load_acc_config()
        # TODO: 23/12/2025 ⭐⭐⭐
        # currently set to be 0.1, but it should refer to current dt (from FPS)
        self.T = 0.1

        # Set up affine shift
        self.x_p = compute_affine_shift(A_d, offset_d)

        # dimensions
        self.n_states = 3
        self.n_controls = 1

        # CasADi symbols
        self.x = SX.sym("x", self.n_states)  # state (d, vp, vf)
        self.u = SX.sym("u", self.n_controls)  # control
        self.U = SX.sym(
            "U", self.n_controls, N
        )  # control sequence (prediction horizon)
        self.P = SX.sym("P", 2 * self.n_states)  # [x0, x_target]

        self._last_u = [0.0] * self.N  # initial guess

        self._build_dynamics()
        self._build_solver()

    def _build_dynamics(self):
        next_state = self.model(
            (self.x[0], self.x[1], self.x[2]), self.u, self.T, return_tuple=False
        )

        # right hand size
        rhs = vertcat(next_state)

        self.f = Function("f", [self.x, self.u], [rhs])

    def _build_solver(self):
        Q = SX.zeros(self.n_states, self.n_states)  # Weight matrix of states diff

        Q[0, 0] = self.distance_weight
        Q[1, 1] = self.preceding_speed_weight
        Q[2, 2] = self.following_speed_weight

        R = SX.zeros(self.n_controls, self.n_controls)  # Weight matrix of control diff
        R[0, 0] = self.input_weight

        g = []
        # set objective function
        obj = 0

        # initial state
        x0 = self.P[: self.n_states]
        # iterate state
        xk = self.P[: self.n_states]
        # target state
        x_target = self.P[self.n_states :]

        # Hard-coded perpendicular projection
        project_perpendicular = compute_orthogonal_projection_matrix()

        # since rank of project_perpendicular is 1, only 1 row is important.
        # this part will find which row (1x3) is necessary.
        row_norms = np.linalg.norm(project_perpendicular, axis=1)
        independent_row_idx = np.argmax(row_norms)

        # convert to symbolic variable
        project_perpendicular_SX = SX(
            project_perpendicular[independent_row_idx : independent_row_idx + 1, :]
        )

        A_pow = SX.eye(self.n_states)  # will store A^k, starts as I (A^0)

        # iterate through prediction horizon
        for k in range(self.N):
            # Cost from state difference
            diff_state = xk - x_target
            obj += diff_state.T @ Q @ diff_state

            # Cost from control (for control input smoothness)
            uk = self.U[:, k]
            # calculate obj from uk
            obj += uk.T @ R @ uk

            # get next state
            xk = self.f(xk, uk)

            # get constraint for reachable set for step >= 3 (index >= 2)
            # update A_pow
            A_pow = A_pow @ SX(self.A_d)  # A_pow = A^(k+1)
            if k >= 2:
                deviation = xk - A_pow @ x0
                # residual will be 1x1 (from 1x3 @ 3x1)
                residual = project_perpendicular_SX @ deviation
                g.append(residual)

            # state constraints
            g.append(xk[0])  # distance
            g.append(xk[1])  # vp
            g.append(xk[2])  # vf

        # add terminal state instead of hard constraint
        terminal_state_diff = xk - x_target
        Q_terminal = 100 * Q  # Set the terminal state more than weight of og state
        obj += terminal_state_diff.T @ Q_terminal @ terminal_state_diff

        g = vertcat(*g)

        opt_vars = reshape(self.U, self.n_controls * self.N, 1)

        print("Problem generated!")
        nlp_prob = {
            "f": obj,  # objective function (cost)
            "x": opt_vars,  # control input param
            "g": g,  # constraint
            "p": self.P,  # initial state and target state
        }

        opts = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        self.solver = nlpsol("solver", "ipopt", nlp_prob, opts)

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
            state = self.model(state, u, self.T)
            states.append(state)
        return states

    def solve(
        self,
        current_state: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> List[float]:
        # bound for control
        lbx = [-inf] * self.N
        ubx = [inf] * self.N

        # constraints for state (4 per steps - d, vp, vf, dyn)
        lbg = []
        ubg = []
        for k in range(self.N):
            if k >= 2:
                # reachable set equality: 3 rows, **but only 1 is independent**
                lbg += [0.0]
                ubg += [0.0]

            lbg += [0.0 - self.x_p[0], 0.0 - self.x_p[1], 0.0 - self.x_p[2]]
            ubg += [120 - self.x_p[0], inf - self.x_p[1], 43.8 - self.x_p[2]]

        p = vertcat(*current_state, *target)
        sol = self.solver(
            x0=self._last_u,  # will update it every step
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=p,
        )

        u_opt = sol["x"].full().flatten().tolist()

        # change the last control input
        # Shift: drop first control, repeat last at the end
        self._last_u = u_opt[1:] + [u_opt[-1]]

        return u_opt