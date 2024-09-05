import pandas as pd
import datetime as dt
import numpy as np

from ochre import Simulator
from ochre.utils import OCHREException, load_csv


class Equipment(Simulator):
    name = 'Generic Equipment'
    end_use = 'Other'
    is_electric = True
    is_gas = False
    zone_name = 'Indoor'

    def __init__(self, zone_name=None, envelope_model=None, save_ebm_results=False, **kwargs):
        """
        Base class for all equipment in a dwelling. All equipment must have:
         - Fuel variables (by default, is_electric=True, is_gas=False)
         - A control algorithm to turn on (run_internal_control)
         - A method to determine the power and heat outputs (calculate_power_and_heat)
        
        Optional features for equipment include:
         - Control parameters provided by an external control (parse_control_signal)
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

        # Mode and controller parameters
        self.on = 0  # fraction of time on (0-1)
        self.on_new = 0  # fraction of time on (0-1)
        self.time_on = dt.timedelta(minutes=0)  # time continuously on
        self.time_off = dt.timedelta(minutes=0)  # time continuously off
        self.cycles = 0

        # Minimum On/Off Times
        self.min_on_time = kwargs.get(self.end_use + ' Minimum On Time', 0)
        self.min_off_time = kwargs.get(self.end_use + ' Minimum Off Time', 0)

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

    def parse_control_signal(self, control_signal):
        # Overwrite if external control might exist
        raise OCHREException('Must define external control algorithm for {}'.format(self.name))

    def run_internal_control(self):
        # Returns the equipment mode (0 or 1); can return None if the mode doesn't change
        raise NotImplementedError()

    def calculate_power_and_heat(self):
        # Sets equipment power and thermal gains to zone
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

    def update_mode_times(self):
        # updates mode times
        if self.on_new:
            self.time_on += self.time_res * self.on_new
            if not self.on:
                self.time_off = dt.timedelta(minutes=0)
                # increase number of cycles if equipment was off and turns on
                self.cycles += 1
        else:
            self.time_off += self.time_res
            if self.on:
                self.time_on = dt.timedelta(minutes=0)


    def update_model(self, control_signal=None):
        # update equipment based on control signal
        if control_signal:
            self.parse_control_signal(control_signal)
            
        # run equipment controller to determine on/off mode
        self.on_new = self.run_internal_control()

        # Keep existing on fraction if not defined or if minimum time limit isn't reached
        if self.on_new is None:
            self.on_new = self.on
        elif not self.on_new and self.time_on < self.min_on_time:
            self.on_new = self.on
        elif self.on_new and self.time_off < self.min_off_time:
            self.on_new = self.on

        # Get voltage, if disconnected then set to off
        voltage = self.current_schedule.get("Voltage (-)", 1)
        if voltage == 0:
            self.on_new = 0

        # calculate electric and gas power and heat gains
        heat_data = self.calculate_power_and_heat()

        # Add heat gains to zone
        self.add_gains_to_zone()
        
        # Run update for subsimulators (e.g., water tank, battery thermal model)
        super().update_model(heat_data)

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

    def update_results(self):
        self.update_mode_times()
        self.on = self.on_new

        return super().update_results()
    
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

        if self.save_ebm_results:
            results.update(self.make_equivalent_battery_model())

        return results

    def reset_time(self, start_time=None, mode=None, **kwargs):
        # TODO: option to remove equipment mode, set initial state
        super().reset_time(start_time=start_time, **kwargs)

        if mode is not None:
            self.mode = mode

        self.time_on = dt.timedelta(minutes=0)
        # self.tot_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
