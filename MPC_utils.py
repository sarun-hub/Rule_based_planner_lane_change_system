from casadi import *
from typing import Tuple, List, Callable, Dict
import inspect
from state_space_analysis import TrajectoryGenerator


class SamplingBasedMPC():
    def __init__(self, model: Callable[[Tuple[float,float,float],float, float, float, float, float], Tuple[float,float,float]],
                 cost_function: Callable[[Tuple[float, float],List[Tuple[float, float, float]]],List[float]],
                 N: int,
                 num_samples: int,
                 state_space: TrajectoryGenerator):
        """
        :param
            model: vehicle model (including preceding and following vehicles [ACC])
            cost_fucntion: cost function that is used to evaluate the sample
            N: prediction horizon
            num_samples: number of samples that will be generated
        """
        self.model = model
        self.cost_function = cost_function
        self.N = N
        self.num_samples = num_samples
        self.state_space = state_space

    def generate_random_inputs(self):
        """
        Generate random input sequeences for N step 
        """
        return [np.random.uniform(low = -3, high = 3, size = self.N) for _ in range(self.num_samples)]

    def predict_states(self,initial_state: Tuple[float, float, float],
                         input_sequence: List[float])-> List[Tuple[float, float, float]]:
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
    
    def compute_costs(self, initial_state: Tuple[float, float, float],
                      input_sequences: List[List[float]]) -> List[float]:
        """
        Compute costs of all random input sequences

        :param
            initial_state: initial state [current state] (distance, preceding speed, following speed)
            input_sequences: list of all random input sequences (acceleration) 
        :return: List of cost (for all random inputs of num_samples)
        """
        costs = []
        constraint_distance = (self.state_space.distance_range[0],self.state_space.distance_range[1])
        target_list =self.state_space.find_best_trajectory(initial_state[0],(initial_state[1]-initial_state[2]))
        for input_sequence in input_sequences:
            predicted_states = self.predict_states(initial_state, input_sequence)
            cost = self.cost_function(target_list,predicted_states, input_sequence)
            if not all(constraint_distance[0]<=state[0]<=constraint_distance[0] for state in predicted_states):
                cost = cost + 1e6

            costs.append(cost)
        return costs
    
    def select_optimal_input_sequence(self, input_sequences: List[List[float]],
                                      costs: List[float]) -> Tuple[List[float], float]:
        """
        Select the minimum cost input sequence

        :param
            input_sequences: list of all random input sequences
            costs: list of cost of all random input sequences
        :return: input sequence with the least cost and that cost
        """
        min_cost_index = np.argmin(costs)
        return input_sequences[min_cost_index], costs[min_cost_index]

    def solve(self, initial_state: Tuple[float, float, float]) -> List[float]:
        """
        Aggrregate all functions to run at once
        :param initial_state: initial state [current state] (distance, preceding speed, following speed)
        :return: optimal input sequence
        """
        input_sequences = self.generate_random_inputs()
        costs = self.compute_costs(initial_state, input_sequences)
        optimal_input_sequence, _ = self.select_optimal_input_sequence(input_sequences, costs)
        return optimal_input_sequence
    

class OptimizationBasedMPC():
    def __init__(self, model: Callable[[Tuple[float,float,float],float, float, float, float, float], Tuple[float,float,float]],
                 cost_function: Callable[[Tuple[float, float],List[Tuple[float, float, float]]],List[float]],
                 N: int,
                 state_space: TrajectoryGenerator):
        self.model = model
        self.cost_function = cost_function
        self.N = N
        self.state_space = state_space

        self.distance_weight = 1
        self.rel_speed_weight = 1
        self.input_weight = 1

        model_signature = inspect.signature(model)
        self.aggressive = model_signature.parameters['aggressive'].default
        self.h = model_signature.parameters['h'].default
        self.delta_min = model_signature.parameters['delta_min'].default
        self.T = model_signature.parameters['T'].default

        # Initialize CasADi variables
        self.d = SX.sym('d')
        self.vp = SX.sym('vp')
        self.vf = SX.sym('vf')
        self.ap = SX.sym('ap')

        # State and control variables
        self.states = vertcat(self.d, self.vp, self.vf)
        self.n_states = self.states.numel()
        self.controls = vertcat(self.ap)
        self.n_controls = self.controls.numel()

        # Prediction and parameter variables
        self.U = SX.sym('U', self.n_controls, self.N)
        self.P = SX.sym('P', self.n_states + self.n_states)
        self.X = SX.sym('X', self.n_states, N+1)

        self._build_dynamics()

    def _build_dynamics(self):
        """Define the vehicle dynamic"""
        rhs = vertcat(self.vp - self.vf,
                      self.ap,
                      (self.aggressive*self.d+self.vp-(1+self.aggressive*self.h)*self.vf-self.aggressive*self.delta_min)/(self.h+1e-6))

        self.f = Function('f',[self.states,self.controls],[rhs])

    def _build_solver(self,current_state: Tuple[float, float, float]):
        """Build the optimization problem"""
        self.X[:,0] = self.P[:self.n_states]

        # Populate predicted using dynamics
        for k in range(self.N):
            st = self.X[:,k]
            cont = self.U[:,k]
            f_value = self.f(st,cont)
            self.X[:,k+1] = st + self.T * f_value

        obj = self.set_objective_function(current_state)
        
        # Constraint
        g = []
        for k in range(self.N+1):
            g = vertcat(g,self.X[0,k])
            g = vertcat(g,self.X[1,k])
            g = vertcat(g,self.X[2,k])

            dynamic_constraint = (self.aggressive * self.X[0, k] + self.X[1, k] - (1 + self.aggressive * self.h) * self.X[2, k] - self.aggressive * self.delta_min) / (self.h+1e-6)
            g = vertcat(g, dynamic_constraint)
        
        opt_variables = reshape(self.U,(self.n_controls*self.N,1))

        # Set up the problem
        nlp_prob = {'f': obj, 'x': opt_variables, 'g': g, 'p': self.P}

        opts = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6
        }

        self.solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

    def set_objective_function(self, current_state: Tuple[float,float,float]) :
        target_list =self.state_space.find_best_trajectory(current_state[0],(current_state[1]-current_state[2]))

        # Objective function
        obj = 0

        Q = SX.zeros(self.n_states, self.n_states)      # Weight matrix of states diff

        Q[0,0] = self.distance_weight
        Q[0,1] = 0
        Q[1,1] = self.rel_speed_weight
        Q[1,0] = 0
        R = SX.zeros(self.n_controls,self.n_controls)             # Weight matrix of control diff
        R[0,0] = self.input_weight

        # Objective function for states diff
        for k in range(self.N):
            target_d,target_rel_speed = target_list[k]
            d_st = self.X[0,k]
            v_rel_st = self.X[1,k] - self.X[2,k]
            d_diff = d_st - target_d
            v_rel_diff = v_rel_st - target_rel_speed

            obj += d_diff * Q[0,0] * d_diff + v_rel_diff * Q[1,1] * v_rel_diff + d_diff * Q[0,1] * v_rel_diff

        # Objective function for input diff
        for k in range(self.N):
            if k > 0:
                cont = self.U[:,k]
                previous_cont = self.U[:,k-1] 
                diff_cont = cont - previous_cont
                obj += diff_cont.T @ R @ diff_cont
        
        return obj
    
    def predict_states(self,current_state: Tuple[float, float, float],
                         input_sequence: List[float])-> List[Tuple[float, float, float]]:
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

    def solve(self, current_state: Tuple[float, float, float], current_u) -> List[float]:
        """Solve the MPC optimization problem."""

        self._build_solver(current_state)
        arg = {}
        

        # preceding acceleration
        arg["lbx"] = -2   #-4
        arg["ubx"] = 2    # 4

        # Set upper and lower bounds for distance and speed separately
        g_lb = []
        g_ub = []

        for _ in range(self.N+1):
            # Distance bounds
            g_lb.append(5)  # Lower bound for distance
            g_ub.append(120) #120 #60 # Upper bound for distance
        
            # Speed bounds
            g_lb.append(0)  # Lower bound for preceding speed
            g_ub.append(float('inf')) #43.7  # Upper bound for preceding speed

            g_lb.append(0)  #-inf # Lower bound for following speed
            g_ub.append(43.8) #43.8   # Upper bound for following speed

            g_lb.append(-2) # -2 # Lower bound for dynamic constraint (following acceleration)
            g_ub.append(2)  # 2 # Upper bound for dynamic constraint (following acceleration)

        arg["lbg"] = g_lb
        arg["ubg"] = g_ub

        target = [10,0,0]
        arg["p"] = vertcat(*current_state, *target)
        arg["x0"] = DM(reshape(current_u, (self.n_controls * self.N, 1)))

        sol = self.solver(x0=arg["x0"], lbx=arg["lbx"], ubx=arg["ubx"], lbg=arg["lbg"], ubg=arg["ubg"], p=arg["p"])
        u = reshape(sol['x'].T, self.n_controls, self.N)
        return u.full().flatten().tolist()    