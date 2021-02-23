# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime as dt
from scipy import linalg

try:
    import sympy  # Optional package - only for generating abstract matrices
except ImportError:
    sympy = None

from ochre.FileIO import save_to_csv


class ModelException(Exception):
    pass


class RCModel:
    """
    Discrete Time State Space RC Model

    Generates an model based on RC parameters provided as a dictionary. The naming convention is as follows:
     - Resistors: "R_{node1}_{node2}" (order of nodes doesn't matter)
     - Capacitors: "C_{node}"

    From the RC parameter dictionary, the model generates collects all internal and external system nodes.
    Nodes are internal if there is a capacitance associated with it; otherwise it is external.
    The initialization process is as follows:
     - Load RC parameter dictionary
     - Create state names from internal nodes: "T_{node_internal}"
     - Create input names from internal and external nodes: "T_{node_external}" and "H_{node_internal}"
     - Create A and B continuous-time matrices from RC values
     - Discretize A and B matrices using datetime.timedelta parameter 'time_res' (required)
     - Creates initial state vector using paramter 'initial_states' (required)
     - Creates default input vector using paramter 'default_inputs' (optional, default sets all inputs to 0)
    """
    name = 'Generic RC'

    def __init__(self, time_res, rc_params=None, ext_node_names=None, **kwargs):
        self.time_res = time_res
        self.high_res = self.time_res < dt.timedelta(minutes=5)

        # Load RC parameters
        if rc_params is None:
            rc_params = self.load_rc_data(**kwargs)
        if not rc_params:
            raise ModelException('No RC Parameters found for {} Model'.format(self.name))

        # Create A and B matrices from RC parameters and get state and input names
        A_c, B_c, state_names, input_names = self.create_matrices(rc_params, ext_node_names)
        self.A_c = A_c
        self.B_c = B_c
        # Load A and B abstract matrices
        # A_c, B_c, state_names, input_names = self.create_matrices(rc_params, print_abstract=True)

        # Create default input vector
        self.input_names = input_names
        self.default_inputs = self.load_default_inputs(**kwargs)
        self.inputs = self.default_inputs.copy()

        # remove unused inputs
        self.remove_unused_inputs(**kwargs)

        # Create initial state vector
        self.state_names = state_names
        self.states = self.load_initial_state(**kwargs)
        self.state_capacitances = np.array([rc_params['C_' + state[2:]] for state in self.state_names])

        # Convert matrices to discrete time
        A, B = self.to_discrete(self.A_c, self.B_c, time_res)
        self.A = A
        self.B = B

        if kwargs.get('save_matrices', False):
            # convert A and B matrices to a data frame, save as csv files
            save_a, save_b = self.to_discrete(self.A_c, self.B_c, kwargs.get('save_matrices_time_res', self.time_res))
            df_a = pd.DataFrame(save_a, index=self.state_names, columns=self.state_names)
            df_b = pd.DataFrame(save_b, index=self.state_names, columns=self.input_names)
            save_to_csv(df_a, '{}_{}_matrixA.csv'.format(kwargs['house_name'], self.name), **kwargs)
            save_to_csv(df_b, '{}_{}_matrixB.csv'.format(kwargs['house_name'], self.name), **kwargs)

    def load_rc_data(self, rc_filename=None, name_col='Name', val_col='Value', **kwargs):
        if rc_filename is None:
            raise ModelException('Missing filename with RC parameters for {}'.format(self.name))
        # Load file
        df = pd.read_csv(rc_filename, index_col=name_col)

        # Convert to dict of {Parameter Name: Parameter Value}
        return df[val_col].to_dict()

    def create_matrices(self, rc_params, ext_node_names_check=None, return_abstract=False):
        # uses RC parameter names to get list of internal/external nodes
        # C names should be 'C_{node}'; R names should be 'R_{node1}_{node2}'
        if sympy is None:
            return_abstract = False

        # parse RC names
        all_cap = {'_'.join(name.split('_')[1:]).upper(): val for name, val in rc_params.items() if name[0] == 'C'}
        all_res = {'_'.join(name.split('_')[1:]).upper(): val for name, val in rc_params.items() if name[0] == 'R'}

        # get all internal and external nodes (internal nodes have a C)
        internal_nodes = list(all_cap.keys())
        res_nodes = [node for name in all_res.keys() for node in name.split('_')]
        all_nodes = sorted(set(res_nodes), key=res_nodes.index)
        external_nodes = [node for node in all_nodes if node not in internal_nodes]

        bad = [node for node in internal_nodes if node not in all_nodes]
        if bad:
            raise ModelException(
                'Some nodes have capacitors but no connected resistors for {}: {}'.format(self.name, bad))
        if ext_node_names_check is not None:
            bad = [node for node in external_nodes if node not in ext_node_names_check]
            if bad:
                raise ModelException('Undefined external nodes for {}: {}'.format(self.name, bad))

        # Define states and inputs
        state_names = ['T_' + node for node in internal_nodes]
        input_names = ['T_' + node for node in external_nodes] + ['H_' + node for node in internal_nodes]
        n = len(state_names)
        m = len(input_names)

        # Create A, B matrices
        A = np.zeros((n, n))
        b_diag = [1 / all_cap[node] for node in internal_nodes]
        B = np.concatenate((np.zeros((n, m - n)), np.diag(b_diag)), axis=1)

        # Create A and B abstract matrices
        if return_abstract:
            cap_abstract = {name: sympy.Symbol('C_' + name) for name in all_cap.keys()}
            res_abstract = {name: sympy.Symbol('R_' + name) for name in all_res.keys()}
            A_abstract = sympy.zeros(n, n)
            b_diag = [1 / c for c in cap_abstract.values()]
            B_abstract = np.concatenate((sympy.zeros(n, m - n), np.diag(b_diag)), axis=1)
        else:
            A_abstract = None
            B_abstract = None

        def add_matrix_values(node1, node2, r_val, res_name):
            # add 1/RC term to A and B matrices (R is between node1 and node2)
            if node1 in internal_nodes and node2 in internal_nodes:
                # both are internal nodes - only update A
                i1 = internal_nodes.index(node1)
                c1 = all_cap[node1]
                i2 = internal_nodes.index(node2)
                c2 = all_cap[node2]
                A[i1, i1] -= 1 / c1 / r_val
                A[i2, i2] -= 1 / c2 / r_val
                A[i1, i2] += 1 / c1 / r_val
                A[i2, i1] += 1 / c2 / r_val
                if return_abstract:
                    r = res_abstract[res_name]
                    c1 = cap_abstract[node1]
                    c2 = cap_abstract[node2]
                    A_abstract[i1, i1] -= 1 / c1 / r
                    A_abstract[i2, i2] -= 1 / c2 / r
                    A_abstract[i1, i2] += 1 / c1 / r
                    A_abstract[i2, i1] += 1 / c2 / r
            else:
                if node1 in internal_nodes:
                    # node2 is external, update A and B
                    i_ext = external_nodes.index(node2)
                    i_int = internal_nodes.index(node1)
                    c = all_cap[node1]
                elif node2 in internal_nodes:
                    # node1 is external, update A and B
                    i_ext = external_nodes.index(node1)
                    i_int = internal_nodes.index(node2)
                    c = all_cap[node2]
                else:
                    # neither is internal, raise an error
                    raise ModelException('Cannot parse resistor {}, no internal nodes defined'.format(res_name))
                A[i_int, i_int] -= 1 / c / r_val
                B[i_int, i_ext] += 1 / c / r_val
                if return_abstract:
                    r = res_abstract[res_name]
                    c = cap_abstract[node1] if node1 in internal_nodes else cap_abstract[node2]
                    A_abstract[i_int, i_int] -= 1 / c / r
                    B_abstract[i_int, i_ext] += 1 / c / r

        # Iterate through resistances to build A, B matrices
        for res_name, res in all_res.items():
            n1, n2 = tuple(res_name.split('_'))
            if n1 not in all_nodes:
                raise ModelException('Error parsing resistor {}. {} not in {}.'.format(res_name, n1, self.name))
            if n2 not in all_nodes:
                raise ModelException('Error parsing resistor {}. {} not in {}.'.format(res_name, n2, self.name))

            add_matrix_values(n1, n2, res, res_name)

        if return_abstract:
            return A_abstract, B_abstract
        else:
            return A, B, state_names, input_names

    @staticmethod
    def to_discrete(A, B, time_res):
        # 2 options for discretization, see https://en.wikipedia.org/wiki/Discretization
        n, m = B.shape

        # first option
        A_d = linalg.expm(A * time_res.total_seconds())
        B_d = np.dot(np.dot(linalg.inv(A), A_d - np.eye(n)), B)

        # second option, without inverse
        # M = np.block([[A, B], [np.zeros((m, n + m))]])
        # M_exp = linalg.expm(M * time_res.total_seconds())
        # A_d = M_exp[:n, :n]
        # B_d = M_exp[:n, n:]
        return A_d, B_d

    def load_initial_state(self, initial_states=None, **kwargs):
        # can take initial states as a dict, list, or number
        # if initial_states is a number, all states are equal to that number
        if isinstance(initial_states, dict) and all([state in initial_states.keys() for state in self.state_names]):
            initial_states = [initial_states[state] for state in self.state_names]
        elif isinstance(initial_states, (int, float)):
            initial_states = [initial_states] * len(self.state_names)
        elif isinstance(initial_states, (list, np.ndarray)) and len(initial_states) == len(self.state_names):
            pass
        else:
            raise ModelException('Initial state cannot be loaded from: {}'.format(initial_states))
        return np.array(initial_states, dtype=float)

    def load_default_inputs(self, default_inputs=None, **kwargs):
        # can take default inputs as a dict or list or None
        # if default_inputs is a dict, inputs not included in the dict are set to 0.
        if isinstance(default_inputs, dict) and all([u in self.input_names for u in default_inputs.keys()]):
            u0 = [default_inputs.get(u, 0) for u in self.input_names]
        elif isinstance(default_inputs, list) and len(default_inputs) == len(self.input_names):
            u0 = default_inputs
        elif default_inputs is None:
            u0 = [0] * len(self.input_names)
        else:
            raise ModelException('Default inputs cannot be loaded from: {}'.format(default_inputs))
        return np.array(u0, dtype=float)

    def remove_unused_inputs(self, unused_inputs=None, **kwargs):
        if unused_inputs is None:
            return

        keep_input_idx = [i for i, name in enumerate(self.input_names) if name not in unused_inputs]
        self.input_names = [name for name in self.input_names if name not in unused_inputs]
        self.default_inputs = self.default_inputs[keep_input_idx]
        self.inputs = self.inputs[keep_input_idx]
        self.B_c = self.B_c[:, keep_input_idx]

    @staticmethod
    def par(*args):
        return 1 / sum([1 / a for a in args])

    def update_inputs(self, inputs):
        if inputs is None or len(inputs) == 0:
            return

        # For speed, if all inputs are provided, do not check input names
        if len(inputs) == len(self.inputs):
            if isinstance(inputs, dict):
                inputs = list(inputs.values())
            self.inputs = np.array(inputs, dtype=float)
        else:
            for input_name, new_val in inputs.items():
                if input_name not in self.input_names:
                    raise ModelException('Input {} not in {} Model'.format(input_name, self.name))
                idx = self.input_names.index(input_name)
                self.inputs[idx] = new_val

    def update(self, inputs=None, reset_inputs=True, return_states=False):
        if reset_inputs:
            self.inputs = self.default_inputs.copy()
        self.update_inputs(inputs)

        new_states = self.A.dot(self.states) + self.B.dot(self.inputs)
        if return_states:
            return new_states
        else:
            self.states = new_states

    def get_states(self):
        # convert states to dictionary
        return {name: val for name, val in zip(self.state_names, self.states)}

    def get_inputs(self):
        # convert inputs to dictionary
        return {name: val for name, val in zip(self.input_names, self.inputs)}

    def get_state_or_input(self, name):
        # get input or state
        if name in self.state_names:
            return self.states[self.state_names.index(name)]
        elif name in self.input_names:
            return self.inputs[self.input_names.index(name)]
        else:
            raise ModelException('{} not in {} states or inputs.'.format(name, self.name))

    def solve_for_input(self, x_idx, u_idx, x_desired):
        # if 1 state is fixed, solve for 1 input that controls state to desired setpoint
        # Accepts state/input indices or state/inputs names
        if isinstance(x_idx, str) and x_idx in self.state_names:
            x_idx = self.state_names.index(x_idx)
        if isinstance(u_idx, str) and u_idx in self.input_names:
            u_idx = self.input_names.index(u_idx)

        return self.solve_for_inputs(x_idx, [u_idx], x_desired)

    def solve_for_inputs(self, x_idx, u_idxs, x_desired, u_ratios=None):
        # solve for n inputs that controls state to desired setpoint
        # assumes 1 state is fixed at setpoint, and ratio of n inputs are known
        # Returns input with a ratio of 1 (usually the sum of the inputs)
        # Note: inputs remain the same as they were (not set to defaults)
        if u_ratios is None:
            # if ratios not given, assume all are constant and sum to 1
            u_ratios = np.ones(len(u_idxs)) / len(u_idxs)
        u_ratios = np.array(u_ratios)
        u_idxs = np.array(u_idxs)

        a_i = self.A[x_idx, :]
        b_i = self.B[x_idx, :]
        u_factor = self.B[x_idx, u_idxs].dot(u_ratios)

        u_desired = (x_desired - a_i.dot(self.states) - b_i.dot(self.inputs)) / u_factor
        return u_desired
