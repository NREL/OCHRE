import pandas as pd
import datetime as dt
import numpy as np

from ochre import Simulator
from ochre.utils import load_csv


class Equipment(Simulator):
    name = 'Generic Equipment'
    end_use = 'Other'
    is_electric = True
    is_gas = False
    modes = ['On', 'Off']  # On and Off assumed as default modes
    zone_name = 'Indoor'

    def __init__(self, zone_name=None, envelope_model=None, ext_time_res=None, save_ebm_results=False, **kwargs):
        """
        Base class for all equipment in a dwelling.
        All equipment must have:
         - A set of modes (default is ['On', 'Off'])
         - Fuel variables (by default, is_electric=True, is_gas=False)
         - A control algorithm to determine the mode (update_internal_control)
         - A method to determine the power and heat outputs (calculate_power_and_heat)
        Optional features for equipment include:
         - A control algorithm to use for external control (update_external_control)
         - A ZIP model for voltage-dependent real and reactive power
         - A parameters file to get loaded as self.parameters
        Equipment can use data from:
         - The dwelling schedule (or from a player file)
         - Any other information from the dwelling (passed through house_args)
        """
        if zone_name is not None:
            self.zone_name = zone_name

        # If envelope model exists, save the zone
        if envelope_model is not None:
            self.zone = envelope_model.zones.get(self.zone_name)
        else:
            self.zone = None

        super().__init__(**kwargs)

        # General parameters
        self.parameters = self.initialize_parameters(**kwargs)
        self.results_name = self.end_use if self.end_use not in ['Lighting', 'Other'] else self.name
        self.save_ebm_results = save_ebm_results

        # Power parameters
        self.electric_kw = 0  # in kW
        self.reactive_kvar = 0  # in kVAR
        self.gas_therms_per_hour = 0  # in therms/hour
        if 'Zp' in kwargs:
            self.zip_data = (
                np.array([kwargs['Zp'], kwargs['Ip'], kwargs['Pp']]),  # real ZIP parameters
                np.array([kwargs['Zq'], kwargs['Iq'], kwargs['Pq']]),  # reactive ZIP parameters
                np.tan(np.arccos(kwargs['pf']))  # power factor multiplier (+ for capacitative, - for inductive)
            )
        else:
            self.zip_data = None

        # Sensible and latent heat parameters
        self.sensible_gain = 0  # in W
        self.latent_gain = 0  # in W

        # Mode and controller parameters (assuming a duty cycle)
        self.mode = 'Off'
        self.time_in_mode = dt.timedelta(minutes=0)
        # self.tot_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
        self.mode_cycles = {mode: 0 for mode in self.modes}

        # Minimum On/Off Times
        on_time = kwargs.get(self.end_use + ' Minimum On Time', 0)
        off_time = kwargs.get(self.end_use + ' Minimum Off Time', 0)
        self.min_time_in_mode = {mode: dt.timedelta(minutes=on_time) for mode in self.modes}
        self.min_time_in_mode['Off'] = dt.timedelta(minutes=off_time)

        self.ext_time_res = ext_time_res
        self.ext_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
        self.duty_cycle_by_mode = {mode: 0 for mode in self.modes}  # fraction of time per mode, should sum to 1
        self.duty_cycle_by_mode['Off'] = 1

    def initialize_parameters(self, parameter_file=None, name_col='Name', value_col='Value', **kwargs):
        if parameter_file is None:
            return {}

        # assumes a parameters file with columns for name and value
        df = load_csv(parameter_file, sub_folder=self.end_use, index_col=name_col)
        if value_col is None:
            return df
        else:
            parameters = df[value_col].to_dict()

            # update parameters from kwargs (overrides the parameters file values)
            parameters.update({key: val for key, val in kwargs.items() if key in parameters})
            return parameters

    def update_duty_cycles(self, *duty_cycles):
        duty_cycles = list(duty_cycles)
        if len(duty_cycles) == len(self.modes) - 1:
            duty_cycles.append(1 - sum(duty_cycles))
        if len(duty_cycles) != len(self.modes):
            raise Exception('Error parsing duty cycles. Expected a list of length equal or 1 less than ' +
                                     'the number of modes ({}): {}'.format(len(self.modes), duty_cycles))

        self.duty_cycle_by_mode = dict(zip(self.modes, duty_cycles))

    def calculate_mode_priority(self, *duty_cycles):
        """
        Calculates the mode priority based on duty cycles from external controller. Always prioritizes current mode
        first. Other modes are prioritized based on the order of Equipment.modes. Excludes modes that have already
        "used up" their time in the external control cycle.
        :param duty_cycles: iterable of duty cycles from external controller, as decimals. Order should follow the order
        of Equipment.modes. Length of list must be equal to or 1 less than the number of modes. If length is 1 less, the
        final mode duty cycle is equal to 1 - sum(duty_cycles).
        :return: list of mode names in order of priority
        """
        if self.ext_time_res is None:
            raise Exception('External control time resolution is not defined for {}.'.format(self.name))
        if duty_cycles:
            self.update_duty_cycles(*duty_cycles)

        if (self.current_time - self.start_time) % self.ext_time_res == 0 or \
                sum(self.ext_mode_counters.values(), dt.timedelta(0)) >= self.ext_time_res:
            # reset mode counters
            self.ext_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}

        modes_with_time = [mode for mode in self.modes
                           if self.ext_mode_counters[mode] / self.ext_time_res < self.duty_cycle_by_mode[mode]]

        # move previous mode to top of priority list
        if self.mode in modes_with_time and modes_with_time[0] != self.mode:
            modes_with_time.pop(modes_with_time.index(self.mode))
            modes_with_time = [self.mode] + modes_with_time

        if not len(modes_with_time):
            self.warn('No available modes, keeping the current mode. '
                      'Duty cycles: {}; Time per mode: {}'.format(duty_cycles, self.ext_mode_counters))
            modes_with_time.append(self.mode)

        return modes_with_time

    def update_external_control(self, control_signal):
        # Overwrite if external control might exist
        raise Exception('Must define external control algorithm for {}'.format(self.name))

    def update_internal_control(self):
        # Returns the equipment mode; can return None if the mode doesn't change
        # Overwrite if internal control exists
        raise NotImplementedError()

    def calculate_power_and_heat(self):
        raise NotImplementedError()

    def add_gains_to_zone(self):
        if self.zone is None:
            return
        
        self.zone.internal_sens_gain += self.sensible_gain
        self.zone.internal_latent_gain += self.latent_gain

    def run_zip(self, v, v0=1):
        if v == 0:
            self.electric_kw = 0

        if self.electric_kw == 0:
            self.reactive_kvar = 0
            return
        if not self.zip_data:
            return
        
        if v == v0:
            pf_mult = self.zip_data[2]
            self.reactive_kvar = self.electric_kw * pf_mult
        else:
            zip_p, zip_q, pf_mult = self.zip_data
            v_quadratic = np.array([(v / v0) ** 2, v / v0, 1])

            self.reactive_kvar = self.electric_kw * pf_mult * zip_p.dot(v_quadratic)
            self.electric_kw = self.electric_kw * zip_q.dot(v_quadratic)

    def update_model(self, control_signal=None):
        # run equipment controller to determine mode
        if control_signal:
            mode = self.update_external_control(control_signal)
        else:
            mode = self.update_internal_control()

        if mode is not None and self.time_in_mode < self.min_time_in_mode[self.mode]:
            # Don't change mode if minimum on/off time isn't met
            mode = self.mode

        # Get voltage, if disconnected then set mode to off
        voltage = self.current_schedule.get('Voltage (-)', 1)
        if voltage == 0:
            mode = 'Off'

        if mode is None or mode == self.mode:
            self.time_in_mode += self.time_res
        else:
            if mode not in self.modes:
                raise Exception(
                    "Can't set {} mode to {}. Valid modes are: {}".format(self.name, mode, self.modes))
            self.mode = mode
            self.time_in_mode = self.time_res
            self.mode_cycles[self.mode] += 1

        if control_signal:
            self.ext_mode_counters[self.mode] += self.time_res

        # calculate electric and gas power and heat gains
        heat_data = self.calculate_power_and_heat()

        # Run update for subsimulators (e.g., water tank, battery thermal model)
        super().update_model(heat_data)

        # Add heat gains to zone
        self.add_gains_to_zone()
        
        # Update electric real/reactive power with ZIP model
        self.run_zip(voltage)

    def make_equivalent_battery_model(self):
        # returns a dictionary of equivalent battery model parameters
        # model definition:
        #  - Continuous time model: E_dot = eta*P - eta_d*P_d - P_b
        #  - Constraints:           E_min <= E <= E_max, 0 <= P <= P_max, E(t=0)=E_0
        #  - Discharge Constraints: 0 <= P_d <= P_d,max, P * P_d = 0 (i.e. no simultaneous charge/discharge) 
        # where:
        #  - E = energy state (kWh)
        #  - E_min, E_max = energy state constraints (kWh)
        #  - E_0 = initial energy state (kWh)
        #  - P = total power, consuming/charging (kW)
        #  - P_max = maximum power, consuming/charging (kW)
        #  - eta = efficiency, consuming/charging (-)
        #  - P_b = baseline power, a disturbance that may be stochastic, time-varying, and dependent on E (kW)
        #  - P_d = power, generating/discharging (kW) (only for devices that can generate power)
        #  - P_d,max = maximum power, generating/discharging (kW) (only for devices that can generate power)
        #  - eta_d = efficiency, generating/discharging (-) (only for devices that can generate power)
        return {
            f'{self.results_name} EBM Energy (kWh)': None,
            f'{self.results_name} EBM Min Energy (kWh)': 0,
            f'{self.results_name} EBM Max Energy (kWh)': None,
            f'{self.results_name} EBM Max Power (kW)': None,
            f'{self.results_name} EBM Efficiency (-)': 1,
            # f'{self.results_name} EBM Baseline Power (kW)': 0,
            # f'{self.results_name} EBM Max Discharge Power (kW)': 0,
            # f'{self.results_name} EBM Discharge Efficiency (-)': 1,
        }

    def generate_results(self):
        results = super().generate_results()

        # Note: end use power is included in Dwelling.generate_results
        # Note: individual equipment powers are included in ScheduledLoad.generate_results
        if self.main_simulator:
            if self.is_electric:
                results[f'{self.results_name} Electric Power (kW)'] = self.electric_kw
                results[f'{self.results_name} Reactive Power (kVAR)'] = self.reactive_kvar
            if self.is_gas:
                results[f'{self.results_name} Gas Power (therms/hour)'] = self.gas_therms_per_hour

        if self.verbosity >= 6:
            results[f'{self.results_name} Mode'] = self.mode

        return results

    def reset_time(self, start_time=None, mode=None, **kwargs):
        # TODO: option to remove equipment mode, set initial state
        super().reset_time(start_time=start_time, **kwargs)

        if mode is not None:
            self.mode = mode

        self.time_in_mode = dt.timedelta(minutes=0)
        self.mode_cycles = {mode: 0 for mode in self.modes}
        self.ext_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
        # self.tot_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
