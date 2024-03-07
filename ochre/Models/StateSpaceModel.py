# -*- coding: utf-8 -*-
"""
@author: mblonsky
"""

import datetime as dt
import numpy as np
import pandas as pd
from scipy import linalg

from ochre.Simulator import Simulator


class ModelException(Exception):
    pass


class StateSpaceModel(Simulator):
    """
    Discrete Time State Space Model
    (Requires Python 3.6 or later)

    A state space model based on a set of states and inputs:
    x_new = A*x + B*u
    y = C*x + D*u

    where:
     - x = state vector, length-nx
     - u = input vector, length-nu
     - y = output vector, length-ny

    Initialization requires:
     - states: list or dictionary of states: {state_name: initial_value}, length-nx. If a list, all default state values
      are 0.
     - inputs: list or dictionary of inputs: {input_name: default_value}, length-nu. If a list, all default input values
      are 0.
     - outputs (optional): list of strings corresponding to state names that are observable. By default, all states are
      observable.
     - matrices: tuple of length 2, 3, or 4 corresponding to continuous-time matrices (A, B, [C, [D]])
     - time_res (optional): time resolution for discretization, as a datetime.timedelta object
     - schedule (optional): pandas DataFrame or csv file to load with time-series information for model inputs. If
     DataFrame contains a DatetimeIndex (or a column called 'Time'), time_res will be inferred from the index.

    """
    name = 'Generic State Space'

    def __init__(self, states, inputs, outputs=None, matrices=None, **kwargs):
        super().__init__(**kwargs)

        # Define states
        self.nx = len(states)
        if isinstance(states, dict):
            self.states = np.array(list(states.values()), dtype=float)
        else:
            self.states = np.zeros(self.nx, dtype=float)
        self.state_names = list(states)
        self.next_states = self.states  # for saving states of next time step

        # Define inputs
        self.nu = len(inputs)
        if isinstance(inputs, dict):
            self.inputs = np.array(list(inputs.values()), dtype=float)
        else:
            self.inputs = np.zeros(self.nu, dtype=float)
        self.input_names = list(inputs)
        self.use_schedule_for_inputs = all([col in self.input_names for col in self.schedule.columns])
        self.inputs_init = self.inputs  # for saving values from update_inputs step

        # Define outputs
        if outputs is None:
            self.ny = self.nx
            self.output_names = self.state_names.copy()
        else:
            self.ny = len(outputs)
            self.output_names = outputs
        self.outputs = np.zeros(self.ny, dtype=float)
        self.next_outputs = self.outputs  # for saving outputs of next time step

        # Define continuous-time matrices
        self.A_c, self.B_c, self.C, self.D = self.create_matrices(matrices)

        # Update output values
        self.outputs = self.C.dot(self.states) + self.D.dot(self.inputs)

        # Reduce model order (i.e. number of states)
        self.reduced = False
        self.transformation_matrix = None
        if 'reduced_states' in kwargs or 'reduced_min_accuracy' in kwargs:
            self.reduce_model(update_discrete=False, **kwargs)

        # Create A, B discrete matrices
        self.A, self.B = self.to_discrete()

    def create_matrices(self, matrices):
        a = np.array(matrices[0], ndmin=2, dtype=float)
        b = np.array(matrices[1], ndmin=2, dtype=float)

        if len(matrices) > 2:
            c = np.array(matrices[2], ndmin=2, dtype=float)
        else:
            # if C isn't defined, output names must match state names
            bad_outputs = [output for output in self.output_names if output not in self.state_names]
            if bad_outputs:
                raise ModelException(f'Outputs must match state names if C matrix is not defined.'
                                     f' Invalid outputs: {bad_outputs}')

            output_idx = [list(self.state_names).index(output_name) for output_name in self.output_names]
            c = np.eye(self.nx, dtype=float)[output_idx, :]

        if len(matrices) > 3:
            d = np.array(matrices[3], ndmin=2, dtype=float)
        else:
            # if D isn't defined, set to 0
            d = np.zeros((self.ny, self.nu))

        # if outputs match state names, check that output equation is correct
        for i, name in enumerate(self.output_names):
            if name not in self.state_names:
                continue
            j = self.state_names.index(name)
            check = np.zeros(self.nx)
            check[j] = 1
            if not (c[i, :] == check).all() or not (d[i, :] == np.zeros(self.nu)).all():
                raise ModelException(f'Output equation for {name} does not equal state with same name.')

        return a, b, c, d

    def get_input_weights(self):
        return np.ones(self.nu)

    def get_output_weights(self):
        return np.ones(self.ny)

    def reduce_model(self, reduced_states=None, reduced_min_accuracy=None, input_weights=None, output_weights=None, 
                     update_discrete=True, **kwargs):
        # reduce number of states using balanced truncation model reduction algorithm
        # see Gugercin 2000, section 2.1.1, https://ieeexplore.ieee.org/abstract/document/914153
        a, b, c = self.A_c, self.B_c, self.C
        x = self.states

        # update B with input weights
        if input_weights is None:
            input_weights = self.get_input_weights()
        if isinstance(input_weights, dict):
            input_weights = np.array([input_weights.get(input_name, 1) for input_name in self.input_names])
        b *= input_weights

        # update C with output weights
        if output_weights is None:
            output_weights = self.get_output_weights()
        if isinstance(output_weights, dict):
            output_weights = np.array([output_weights.get(output_name, 1) for output_name in self.output_names])
        c = (c.T * output_weights).T

        # Create lyapunov matrices
        p = linalg.solve_continuous_lyapunov(a, -b.dot(b.T))
        q = linalg.solve_continuous_lyapunov(a.T, -c.T.dot(c))

        # Get eigenvalues
        sigma = linalg.eigvals(p.dot(q)) ** 0.5

        # Solve for U and L
        u = linalg.cholesky(p).T
        l = linalg.cholesky(q, lower=True)

        # SVD of U*L
        z, s, yh = linalg.svd(u.T.dot(l))
        y = yh.T

        # Solve for state transformation matrix
        t = np.diag(s ** 0.5).dot(z.T).dot(linalg.inv(u))
        # t_check = np.diag(s ** -0.5).dot(Y.T).dot(L.T)
        # print(t - t_check)

        # TODO: Can you scale each new state to maintain units?
        #  - If old states have same units, then new state can be weighted averages (rows of T sum to 1)
        #  - Would it help or hurt the matrix inversion?
        # t /= t.sum(axis=1)

        # Transform SS matrices and states
        t_inv = linalg.inv(t)
        a_t = t.dot(a).dot(t_inv)
        b_t = t.dot(b)
        c_t = c.dot(t_inv)
        x_t = t.dot(x)

        # Check transformed (balanced) system
        # Pt = linalg.solve_continuous_lyapunov(At, -Bt.dot(Bt.T))
        # Qt = linalg.solve_continuous_lyapunov(At.T, -Ct.T.dot(Ct))

        # update B and C with input/output weights, back to original model
        b_t /= input_weights
        c_t = (c_t.T / output_weights).T

        # Determine number of reduced states
        if reduced_states is None:
            # variation = s.cumsum() / s.sum()
            max_error = s[::-1].cumsum()[::-1]
            available_states = np.where(max_error < reduced_min_accuracy)[0]
            if len(available_states):
                reduced_states = available_states[0]
            else:
                self.warn(f'Cannot achieve minimum accuracy for {self.name} Model ({reduced_min_accuracy}). '
                      f'Creating 1 state model with accuracy {max_error[-1]}')
                reduced_states = 1

        # save transformation matrix as DataFrame with named index and columns
        new_state_names = [f'x{i + 1}' for i in range(reduced_states)]
        self.transformation_matrix = pd.DataFrame(t[:reduced_states, :],
                                                  index=new_state_names, columns=self.state_names)

        # update states and state names - default state names are ['x1', 'x2', ...]
        self.state_names = new_state_names
        self.states = x_t[:reduced_states]
        self.nx = len(self.states)

        # update matrices
        self.A_c = a_t[:reduced_states, :reduced_states]
        self.B_c = b_t[:reduced_states, :]
        self.C = c_t[:, :reduced_states]
        if update_discrete:
            self.A, self.B = self.to_discrete()

        # update output values
        # self.outputs = self.C.dot(self.states) + self.D.dot(self.inputs)
        self.reduced = True

    def to_discrete(self, time_res=None):
        # 2 options for discretization, see https://en.wikipedia.org/wiki/Discretization
        if time_res is None:
            time_res = self.time_res
        if time_res == dt.timedelta(0):
            return self.A_c, self.B_c

        n, m = self.B_c.shape
        # first option - if A is invertible
        try:
            A_d = linalg.expm(self.A_c * time_res.total_seconds())
            B_d = np.dot(np.dot(linalg.inv(self.A_c), A_d - np.eye(n)), self.B_c)
        except linalg.LinAlgError:
            # second option, without inverse
            M_block = np.block([[self.A_c, self.B_c], [np.zeros((m, n + m))]])
            M_exp = linalg.expm(M_block * time_res.total_seconds())
            A_d = M_exp[:n, :n]
            B_d = M_exp[:n, n:]

        return A_d, B_d

    def update_inputs(self, schedule_inputs=None):
        super().update_inputs(schedule_inputs)

        if isinstance(schedule_inputs, (list, np.ndarray)) and len(schedule_inputs) == self.nu:
            # For speed, if all inputs are provided as a list, do not check input names
            self.inputs_init[:] = schedule_inputs
        elif self.use_schedule_for_inputs:
            # update inputs from dictionary (keys can be input_name or index)
            # TODO: need faster method for large models, maybe require schedule to be same order as input_names?
            for input_name, new_val in self.current_schedule.items():
                if isinstance(input_name, int):
                    input_idx = input_name
                elif input_name in self.input_names:
                    input_idx = self.input_names.index(input_name)
                else:
                    raise ModelException(f'Unknown input name {input_name} for {self.name}')
                self.inputs_init[input_idx] = new_val

    def update_model(self, control_signal=None):
        # Calculates the model states and outputs, but does NOT overwrite the existing states.
        # Note: control_signal inputs replace the existing input values, they they do not add
        # """
        # Updates the model states and outputs. By default, it will set input values to 0 or get their value from the next row
        # of the model schedule if it exists.
        # :param inputs: (optional) dictionary of {input_name: input_value} pairs. If provided, these will override
        # the defaults from the model schedule.
        # :param reset_inputs: (optional) boolean. If False, inputs will keep their previous value and not be set to 0 or
        # updated by the model schedule. If True (default), the model schedule will advance 1 time step.
        # :param update_states: (optional) boolean. If False, model states and outputs will not get updated, but new
        # outputs are still returned. This may be useful for running different inputs at at same state.
        # :return: numpy.ndarray of updated model outputs
        # """

        self.inputs = self.inputs_init.copy()
        if control_signal is None:
            pass
        elif isinstance(control_signal, (list, np.ndarray)) and len(control_signal) == self.nu:
            # For speed, if all inputs are provided as a list, do not check input names
            self.inputs[:] = control_signal
        else:
            # update inputs from control signal dictionary (keys can be input_name or index)
            for input_name, new_val in control_signal.items():
                input_idx = self.input_names.index(input_name) if not isinstance(input_name, int) else input_name
                self.inputs[input_idx] = new_val

        # Calculate new states and outputs
        self.next_states = self.A.dot(self.states) + self.B.dot(self.inputs)
        self.next_outputs = self.C.dot(self.next_states) + self.D.dot(self.inputs)

        super().update_model(control_signal)

    def get_inputs(self):
        # return dictionary of inputs
        return dict(zip(self.input_names, self.inputs))

    def get_states(self):
        # return dictionary of states
        return dict(zip(self.state_names, self.states))

    def get_outputs(self):
        # return dictionary of outputs
        return dict(zip(self.output_names, self.outputs))

    def generate_results(self):
        results = super().generate_results()

        # Note: only includes inputs and states if save_results is True
        if self.save_results:
            if self.verbosity >= 5:
                results.update(self.get_inputs())
            if self.verbosity >= 1:
                results.update(self.get_outputs())
            if self.verbosity >= 9:
                results.update(self.get_states())

        return results

    def update_results(self):
        current_results = super().update_results()

        # Update next time step states and outputs
        self.states = self.next_states
        self.outputs = self.next_outputs

        return current_results

    def get_value(self, name):
        # return value of input, output, or state name
        if name in self.input_names:
            return self.inputs[self.input_names.index(name)]
        elif name in self.output_names:
            return self.outputs[self.output_names.index(name)]
        elif name in self.state_names:
            return self.states[self.state_names.index(name)]
        else:
            raise ModelException(f'Unknown variable {name}, not in {self.name} model.')
