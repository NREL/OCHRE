# -*- coding: utf-8 -*-
"""
@author: mblonsky
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
from itertools import combinations

from . import StateSpaceModel, ModelException


def transform_floating_node(float_node, all_resistors):
    # use star-mesh transform to remove floating node, see https://en.wikipedia.org/wiki/Star-mesh_transform
    adj_resistors = {nodes: val for nodes, val in all_resistors.items() if float_node in nodes}
    adj_resistors = {
        node: val for nodes, val in adj_resistors.items() for node in nodes if node != float_node
    }

    new_resistors = {nodes: val for nodes, val in all_resistors.items() if float_node not in nodes}
    zeros = [node for node, r in adj_resistors.items() if r == 0]
    if any(zeros):
        # one resistor value is 0, replace float node with connected node
        assert len(zeros) == 1  # only one zero allowed
        node1 = zeros[0]
        for node2, r in adj_resistors.items():
            if node1 == node2:
                pass
            elif (node1, node2) in new_resistors:
                new_resistors[(node1, node2)] = RCModel.par(new_resistors[(node1, node2)], r)
            elif (node2, node1) in new_resistors:
                new_resistors[(node2, node1)] = RCModel.par(new_resistors[(node2, node1)], r)
            else:
                new_resistors[(node1, node2)] = r
    else:
        # use star-mesh transform

        r_parallel = RCModel.par(*adj_resistors.values())
        for node1, node2 in combinations(adj_resistors.keys(), 2):
            r1 = adj_resistors[node1]
            r2 = adj_resistors[node2]
            r_new = r1 * r2 / r_parallel
            if (node1, node2) in new_resistors:
                new_resistors[(node1, node2)] = RCModel.par(new_resistors[(node1, node2)], r_new)
            elif (node2, node1) in new_resistors:
                new_resistors[(node2, node1)] = RCModel.par(new_resistors[(node2, node1)], r_new)
            else:
                new_resistors[(node1, node2)] = r_new

    return new_resistors


class RCModel(StateSpaceModel):
    """
    Discrete Time State Space RC Model

    Generates a model based on RC parameters provided as a dictionary. The naming convention is as follows:
     - Resistors: "R_{node1}_{node2}" (order of nodes doesn't matter)
     - Capacitors: "C_{node}"

    From the RC parameter dictionary and a list of external node names, the model generates all internal and external
    system nodes. Internal nodes with an associated capacitance are included as states. Internal nodes without a
    capacitance are removed using the star-mesh transform.

    The initialization process is as follows:
     - Load RC parameter dictionary
     - Create state names from internal nodes: "T_{node_internal}"
     - Create input names from internal and external nodes: "T_{node_external}" and "H_{node_internal}"
     - Create A and B continuous-time matrices from RC values
     - Discretize A and B matrices using datetime.timedelta parameter 'time_res' (required)
     - Creates initial state vector using parameter 'initial_states' (required)

    In general, units are not explicit. However, it is recommended to use the following units:
     - Resistors: K / W
     - Capacitors: J / K
     - Temperature: degrees C
     - Heat: W
    """

    name = "Generic RC"

    def __init__(
        self,
        capacitances,
        resistances,
        external_nodes,
        energy_flow_states=None,
        unused_inputs=None,
        **kwargs,
    ):
        # saves parameters for faster solving, see self.solve_for_multi_inputs
        self.solver_params = None

        # Parse RC parameters
        self.capacitances = np.array(list(capacitances.values()))

        # Get node names from RC parameters
        internal_nodes = list(capacitances.keys())
        self.n_internal_nodes = len(internal_nodes)
        self.n_external_nodes = len(external_nodes)
        all_nodes = set([node for name in resistances.keys() for node in name])

        # remove floating nodes using star-mesh transform
        floating_nodes = [node for node in all_nodes if node not in internal_nodes + external_nodes]
        for node in floating_nodes:
            resistances = transform_floating_node(node, resistances)

        # check for zero or negative RC parameters
        bad_params = [c for c, val in capacitances.items() if val <= 0]
        bad_params += [r for r, val in resistances.items() if val <= 0]
        if bad_params:
            raise ModelException(
                f"RC parameters for {self.name} Model must be positive: {bad_params}"
            )

        # Define state and input names
        state_names = ["T_" + node for node in internal_nodes]
        input_names = ["T_" + node for node in external_nodes]
        input_names += ["H_" + node for node in internal_nodes]

        # Create A and B matrices from RC parameters and get state and input names
        A_c, B_c = self.create_rc_matrices(
            capacitances, resistances, internal_nodes, external_nodes
        )

        # add energy flow states
        # TODO: don't allow energy flow states with model reduction
        if energy_flow_states is None:
            energy_flow_states = []
        self.n_energy_flow_states = len(energy_flow_states)
        A_c, B_c = self.add_energy_flow_states(
            energy_flow_states, A_c, B_c, internal_nodes, external_nodes, resistances
        )
        state_names.extend(
            [f"H_{node_from}_{node_to}" for (node_from, node_to) in energy_flow_states]
        )

        # remove unused inputs
        if unused_inputs is not None:
            good_input_idx = [
                i for (i, name) in enumerate(input_names) if name not in unused_inputs
            ]
            B_c = B_c[:, good_input_idx]
            input_names = [name for name in input_names if name not in unused_inputs]

        # initialize states based on matrices
        states = self.initialize_state(state_names, input_names, A_c, B_c, **kwargs)

        # create state space model
        super().__init__(states, input_names, matrices=(A_c, B_c), **kwargs)

        self.high_res = self.time_res < dt.timedelta(minutes=5)

        # remove summation of energy flow states in discrete time matrices
        for i in range(self.n_internal_nodes, self.n_internal_nodes + self.n_energy_flow_states):
            assert self.A[i, i] == 1
            self.A[i, i] = 0

        if kwargs.get("save_matrices", False) and self.output_path is not None:
            # convert A and B matrices to a data frame
            # Note: set save_matrices_time_res to dt.timedelta(0) to save continuous time matrices
            A, B = self.to_discrete(time_res=kwargs.get("save_matrices_time_res"))
            df_a = pd.DataFrame(A, index=self.state_names, columns=self.state_names)
            df_b = pd.DataFrame(B, index=self.state_names, columns=self.input_names)
            df_c = pd.DataFrame(self.C, index=self.output_names, columns=self.state_names)

            # save as csv files
            file_name_format = os.path.join(self.output_path, f"{self.name}_{self.main_sim_name}")
            df_a.to_csv(file_name_format + "_matrixA.csv", index=True)
            df_b.to_csv(file_name_format + "_matrixB.csv", index=True)
            df_c.to_csv(file_name_format + "_matrixC.csv", index=True)

    # @staticmethod
    # def create_abstract_matrices(all_cap, all_res, internal_nodes, external_nodes):
    #     # Create A and B abstract matrices (not working anymore)
    #     import sympy
    #     all_cap = {name: sympy.Symbol("C_" + name) for name in all_cap.keys()}
    #     all_res = {name: sympy.Symbol("R_" + "_".join(name)) for name in all_res.keys()}
    #     return RCModel.create_rc_matrices(all_cap, all_res, internal_nodes, external_nodes)

    @staticmethod
    def create_rc_matrices(all_cap, all_res, internal_nodes, external_nodes):
        # uses RC parameter names to get list of internal/external nodes
        # C names should be 'C_{node}'; R names should be 'R_{node1}_{node2}'
        n = len(internal_nodes)
        m = len(external_nodes)

        # Create A, B matrix templates
        A = np.zeros((n, n))
        b_diag = [1 / all_cap[node] for node in internal_nodes]
        B = np.concatenate((np.zeros((n, m)), np.diag(b_diag)), axis=1)

        # Iterate through resistances to build A, B matrices
        for (node1, node2), r_val in all_res.items():
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
                    raise ModelException(
                        f"Cannot parse resistor R_{node1}_{node2}, no internal nodes defined"
                    )
                A[i_int, i_int] -= 1 / c / r_val
                B[i_int, i_ext] += 1 / c / r_val

        return A, B

    @staticmethod
    def add_energy_flow_states(
        energy_flow_states, A_c, B_c, internal_nodes, external_nodes, resistances
    ):
        # add states for energy flows through specific resistors
        # extend A and B matrices
        if not energy_flow_states:
            return (
                A_c,
                B_c,
            )

        n_new = len(energy_flow_states)
        A_c = np.pad(A_c, ((0, n_new), (0, n_new)))
        B_c = np.pad(B_c, ((0, n_new), (0, 0)))

        # fill in energy flow equations:
        # x = Q (kWh), x_dot = H (energy flow through R) = (T_2 - T_1) / R
        row = len(internal_nodes)
        for node_from, node_to in energy_flow_states:
            # get resistance value
            if (node_from, node_to) in resistances:
                res = resistances[(node_from, node_to)]
            else:
                res = resistances[(node_to, node_from)]

            # get node indices and update A and B matrices
            if node_from in internal_nodes:
                i1 = internal_nodes.index(node_from)
                A_c[row, i1] = 1 / res
            else:
                i1 = external_nodes.index(node_from)
                B_c[row, i1] = 1 / res
            i2 = internal_nodes.index(node_to)
            A_c[row, i2] = -1 / res
            row += 1

        return A_c, B_c

    @staticmethod
    def initialize_state(state_names, input_names, A_c, B_c, **kwargs):
        # Optional function to set initial states based on A and B matrices
        return state_names

    @staticmethod
    def par(*args):
        if any([a == 0 for a in args]):
            return 0
        else:
            return 1 / sum([1 / a for a in args])

    def solve_for_input(self, y_idx, u_idx, x_desired, solve_as_output=None):
        # if 1 state or output is fixed, solve for 1 input that controls state to desired setpoint
        # Accepts input/state/output indices or input/state/output names
        if isinstance(y_idx, str) and y_idx in self.state_names:
            y_idx = self.state_names.index(y_idx)
            solve_as_output = False
        elif isinstance(y_idx, str) and y_idx in self.output_names:
            y_idx = self.output_names.index(y_idx)
            solve_as_output = True
        if isinstance(u_idx, str) and u_idx in self.input_names:
            u_idx = self.input_names.index(u_idx)

        if solve_as_output is None:
            raise ModelException("Must specify if y_idx is a state or an output.")
        return self.solve_for_inputs(y_idx, [u_idx], x_desired, solve_as_output=solve_as_output)

    def solve_for_inputs(
        self, y_idx, u_idxs, y_desired, u_ratios=None, solve_as_output=True, use_inputs_init=True
    ):
        # solve for n inputs that controls 1 state or output to desired setpoint
        # assumes 1 state or output is fixed at setpoint, and ratio of n inputs are known
        # Returns input with a ratio of 1 (usually the sum of the inputs)
        # Note: inputs remain the same as they were (not set to defaults)
        if u_ratios is None:
            # if ratios not given, assume all are constant and sum to 1
            u_ratios = np.ones(len(u_idxs)) / len(u_idxs)
        u_ratios = np.array(u_ratios)
        u_idxs = np.array(u_idxs)
        inputs = self.inputs_init if use_inputs_init else self.inputs

        if solve_as_output:
            # solves: y_desired = c_i * (A * x + B * (u + u')) + d_i * (u + u')
            # u' = u_desired * [dict(zip(u_idxs, u_ratios)).get(idx, 0) for idx in range(len(u))]
            c_i = self.C[y_idx, :]
            d_i = self.D[y_idx, :]
            u_factor = (d_i + c_i.dot(self.B))[u_idxs].dot(u_ratios)
            y_current = c_i.dot(self.A.dot(self.states) + self.B.dot(inputs)) + d_i.dot(inputs)
            u_desired = 1 / u_factor * (y_desired - y_current)
        else:
            # solves: y_desired = a_i * x + b_i * (u + u')
            # u' = u_desired * [dict(zip(u_idxs, u_ratios)).get(idx, 0) for idx in range(len(u))]
            a_i = self.A[y_idx, :]
            b_i = self.B[y_idx, :]
            u_factor = self.B[y_idx, u_idxs].dot(u_ratios)
            u_desired = 1 / u_factor * (y_desired - a_i.dot(self.states) - b_i.dot(inputs))

        return u_desired

    def setup_multi_input_solver(self, y_names, u_info, solve_as_output=True):
        # sets up a method to solve for multiple inputs that control multiple outputs (or states) to desired values
        # y_names is a list of output names (or state names if solve_as_output is False)
        # u_info is a list (with the same length as y_names) of:
        #  - inputs names (assumes these inputs can vary to acheive setpoints)
        #  - dictionaries of the form {input_name: input_ratio} (input ratios are maintained to acheive setpoints)
        # Returns an inverse B matrix for solving and a transformation matrix for converting back to u
        if solve_as_output:
            y_idxs = np.array([self.output_names.index(name) for name in y_names])
        else:
            y_idxs = np.array([self.state_names.index(name) for name in y_names])

        # Create vector of input ratios using u_data
        input_ratios = {}
        for y_idx, u_data in zip(y_idxs, u_info):
            if isinstance(u_data, str):
                u_idx = self.input_names.index(u_data)
                input_ratios[y_idx] = np.zeros(self.nu)
                input_ratios[y_idx][u_idx] = 1
            if isinstance(u_data, dict):
                input_ratios[y_idx] = np.array(
                    [u_data.get(u_name, 0) for u_name in self.input_names]
                )
        input_ratios = pd.DataFrame(input_ratios)

        if solve_as_output:
            m_i = self.C[y_idxs, :].dot(self.B) + self.D[y_idxs, :]
        else:
            m_i = self.B[y_idxs, :]
        m_i_inv = np.linalg.inv(m_i.dot(input_ratios))

        # save solver parameters
        self.solver_params = (m_i_inv, input_ratios, solve_as_output)

    def solve_for_multi_inputs(self, y_values, use_inputs_init=True):
        # solves for multiple inputs that control multiple states to desired setpoints
        # takes parameters from setup_multi_input_solver
        # Returns change in input vector u
        # Note: inputs remain the same as they were (not set to defaults)
        y_values = np.array(y_values)
        m_i_inv, input_ratios, solve_as_output = self.solver_params
        y_idxs = input_ratios.columns
        inputs = self.inputs_init if use_inputs_init else self.inputs

        if solve_as_output:
            # solves: y_values = c_i * (A * x + B * (u + u')) + d_i * (u + u')
            # u' = u_desired * [dict(zip(u_idxs, u_ratios)).get(idx, 0) for idx in range(len(u))]
            # m_i = c_i * B + d_i
            c_i = self.C[y_idxs, :]
            d_i = self.D[y_idxs, :]
            y_current = c_i.dot(self.A.dot(self.states) + self.B.dot(inputs)) + d_i.dot(inputs)
            u_desired = m_i_inv.dot(y_values - y_current)
        else:
            # solves: y_values = a_i * x + b_i * (u + u')
            # u' = u_desired * [dict(zip(u_idxs, u_ratios)).get(idx, 0) for idx in range(len(u))]
            # m_i = b_i
            a_i = self.A[y_idxs, :]
            b_i = self.B[y_idxs, :]
            u_desired = m_i_inv.dot(y_values - a_i.dot(self.states) - b_i.dot(inputs))

        u_final = input_ratios.dot(u_desired)
        return u_final


class OneNodeRCModel(RCModel):
    """
    1R-1C Discrete Time Model

    Generates a 1-node RC model with the following parameters:
     - 1 State: "T_INT"
     - 2 Inputs: "T_EXT" and "H_INT"
     - 1 Output: "T_INT"
    """

    name = "One Node RC"
    int_name = "INT"
    ext_name = "EXT"

    def __init__(self, resistance, capacitance, **kwargs):
        self.resistance = resistance
        self.capacitance = capacitance
        capacitances = {self.int_name: self.capacitance}
        resistances = {(self.int_name, self.ext_name): self.resistance}

        super().__init__(capacitances, resistances, [self.ext_name], **kwargs)
