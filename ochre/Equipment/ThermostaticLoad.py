import numpy as np
import datetime as dt

from ochre.utils import OCHREException
from ochre.utils.units import convert, kwh_to_therms
from ochre.Equipment import Equipment
from ochre.Models import OneNodeWaterModel, TwoNodeWaterModel, StratifiedWaterModel, IdealWaterModel


class ThermostaticLoad(Equipment):
    optional_inputs = [
        "Water Heating Setpoint (C)",
        "Water Heating Deadband (C)",
        "Water Heating Max Power (kW)",
        "Zone Temperature (C)",  # Needed for Water tank model
    ]  
    
    def __init__(self, thermal_model=None, use_ideal_mode=None, prevent_overshoot=True, **kwargs):
        """
        Equipment that controls a StateSpaceModel using thermostatic control
        methods. Equipment thermal capacity and power may be controlled
        through three modes:

          - Thermostatic mode: A thermostat control with a deadband is used to
            turn the equipment on and off. Capacity and power are zero or at
            their maximum values.

          - Ideal mode: Capacity is calculated at each time step to perfectly
            maintain the desired setpoint. Power is determined by the fraction
            of time that the equipment is on in various modes.

          - Thermostatic mode without overshoot: First, the thermostatic mode
            is used. If the temperature exceeds the deadband, then the ideal
            mode is used to achieve at temperature exactly at the edge of the
            deadband.

        """

        super().__init__(**kwargs)

        self.thermal_model = thermal_model
        self.sub_simulators.append(self.thermal_model)

        # By default, use ideal mode if time resolution >= 15 minutes
        if use_ideal_mode is None:
            use_ideal_mode = self.time_res >= dt.timedelta(minutes=15)
        self.use_ideal_mode = use_ideal_mode

        # By default, prevent overshoot in tstat mode
        self.prevent_overshoot = prevent_overshoot

        # Control parameters
        # note: bottom of deadband is (setpoint_temp - deadband_temp)
        self.temp_setpoint = kwargs['Setpoint Temperature (C)']
        self.temp_setpoint_ext = None
        self.setpoint_ramp_rate = kwargs.get('Max Setpoint Ramp Rate (C/min)')  # max setpoint ramp rate, in C/min
        # TODO: convert to deadband min and max temps
        self.temp_deadband = kwargs.get('Deadband Temperature (C)', 5.56)  # deadband range, in delta degC, i.e. Kelvin
        self.max_power = kwargs.get('Max Power (kW)')
        self.force_off = False

        # Thermal model parameters
        self.capacity = 0  # heat output from main element, in W
        self.delivered_heat = 0  # total heat delivered to the model, in W

    def parse_control_signal(self, control_signal):
        # Options for external control signals:
        # - Load Fraction: 1 (no effect) or 0 (forces WH off)
        # - Setpoint: Updates setpoint temperature from the default (in C)
        #   - Note: Setpoint will only reset back to default value when {'Setpoint': None} is passed.
        # - Deadband: Updates deadband temperature (in C)
        #   - Note: Deadband will only be reset if it is in the schedule
        # - Max Power: Updates maximum allowed power (in kW)
        #   - Note: Max Power will only be reset if it is in the schedule
        #   - Note: Will not work for HPWH in HP mode

        ext_setpoint = control_signal.get("Setpoint")
        if ext_setpoint is not None:
            if f"{self.end_use} Setpoint (C)" in self.current_schedule:
                self.current_schedule[f"{self.end_use} Setpoint (C)"] = ext_setpoint
            else:
                # Note that this overrides the ramp rate
                self.temp_setpoint = ext_setpoint

        ext_db = control_signal.get("Deadband")
        if ext_db is not None:
            if f"{self.end_use} Deadband (C)" in self.current_schedule:
                self.current_schedule[f"{self.end_use} Deadband (C)"] = ext_db
            else:
                self.temp_deadband = ext_db

        max_power = control_signal.get("Max Power")
        if max_power is not None:
            if f"{self.end_use} Max Power (kW)" in self.current_schedule:
                self.current_schedule[f"{self.end_use} Max Power (kW)"] = max_power
            else:
                self.max_power = max_power

        # If load fraction = 0, force off (max power = 0)
        load_fraction = control_signal.get("Load Fraction", 1)
        if load_fraction == 0:
            self.current_schedule[f"{self.end_use} Max Power (kW)"] = 0
        elif load_fraction != 1:
            raise OCHREException(f"{self.name} can't handle non-integer load fractions")

    def update_setpoint(self):
        # get setpoint from schedule
        if "Water Heating Setpoint (C)" in self.current_schedule:
            t_set_new = self.current_schedule["Water Heating Setpoint (C)"]
        else:
            t_set_new = self.temp_setpoint
        
        # update setpoint with ramp rate
        if self.setpoint_ramp_rate and self.temp_setpoint != t_set_new:
            delta_t = self.setpoint_ramp_rate * self.time_res.total_seconds() / 60  # in C
            self.temp_setpoint = min(max(t_set_new, self.temp_setpoint - delta_t),
                                     self.temp_setpoint + delta_t,
            )
        else:
            self.temp_setpoint = t_set_new
        
        # get other controls from schedule - deadband and max power
        if "Water Heating Deadband (C)" in self.current_schedule:
            self.temp_deadband = self.current_schedule["Water Heating Deadband (C)"]
        if "Water Heating Max Power (kW)" in self.current_schedule:
            self.max_power = self.current_schedule["Water Heating Max Power (kW)"]

    def solve_ideal_capacity(self):
        # calculate ideal capacity based on achieving lower node setpoint temperature
        # Run model with heater off, updates next_states
        self.thermal_model.update_model()
        off_states = self.thermal_model.next_states

        # calculate heat needed to reach setpoint - only use nodes at and above lower node
        set_states = np.ones(len(off_states)) * self.temp_setpoint
        h_desired = np.dot(set_states[:self.t_lower_idx + 1] - off_states[:self.t_lower_idx + 1],  # in W
                           self.thermal_model.capacitances[:self.t_lower_idx + 1]) / self.time_res.total_seconds()

        # return ideal capacity, maintain min/max bounds
        return min(max(h_desired, 0), self.capacity_rated)

    def run_thermostat_control(self):
        # use thermostat with deadband control
        if self.thermal_model.n_nodes <= 2:
            t_lower = self.thermal_model.states[self.t_lower_idx]
        else:
            # take average of lower node and node above
            t_lower = (self.thermal_model.states[self.t_lower_idx] + self.thermal_model.states[self.t_lower_idx - 1]) / 2

        if t_lower < self.temp_setpoint - self.temp_deadband:
            return 'On'
        if t_lower > self.temp_setpoint:
            return 'Off'

    def update_internal_control(self):
        self.update_setpoint()

        if self.use_ideal_mode:
            if self.thermal_model.n_nodes == 1:
                # FUTURE: remove if not being used
                # calculate ideal capacity based on tank model - more accurate than self.solve_ideal_capacity

                # Solve for desired heat delivered, subtracting external gains
                h_desired = self.thermal_model.solve_for_input(self.thermal_model.t_1_idx, self.thermal_model.h_1_idx, self.temp_setpoint,
                                                       solve_as_output=False)

                # Only allow heating, convert to duty cycle
                self.capacity = min(max(h_desired, 0), self.capacity_rated)
            else:
                self.capacity = self.solve_ideal_capacity()

            return "On" if self.capacity > 0 else "Off"
        else:
            return self.run_thermostat_control()

    def generate_results(self):
        results = super().generate_results()

        # Note: using end use, not equipment name, for all results
        if self.verbosity >= 3:
            results[f'{self.end_use} Delivered (W)'] = self.delivered_heat
        if self.verbosity >= 6:
            cop = self.delivered_heat / (self.electric_kw * 1000) if self.electric_kw > 0 else 0
            results[f'{self.end_use} COP (-)'] = cop
            results[f'{self.end_use} Total Sensible Heat Gain (W)'] = self.sensible_gain
            results[f'{self.end_use} Deadband Upper Limit (C)'] = self.temp_setpoint
            results[f'{self.end_use} Deadband Lower Limit (C)'] = self.temp_setpoint - self.temp_deadband

        if self.save_ebm_results:
            results.update(self.make_equivalent_battery_model())

        return results
