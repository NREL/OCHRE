import datetime as dt

from ochre.utils import OCHREException
from ochre.Equipment import Equipment


class ThermostaticLoad(Equipment):
    setpoint_deadband_position = 0.5  # setpoint at midpoint of deadband
    is_heater = True
    heat_mult = 1  # 1=heating, -1=cooling
    
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

        # Model parameters
        self.t_control_idx = None  # state index for thermostat control
        self.h_control_idx = None  # input index for thermostat control

        # By default, use ideal mode if time resolution >= 15 minutes
        if use_ideal_mode is None:
            use_ideal_mode = self.time_res >= dt.timedelta(minutes=15)
        self.use_ideal_mode = use_ideal_mode

        # By default, prevent overshoot in tstat mode
        self.prevent_overshoot = prevent_overshoot

        # Setpoint and deadband parameters
        self.temp_setpoint = kwargs['Setpoint Temperature (C)']
        self.temp_setpoint_old = self.temp_setpoint
        self.setpoint_ramp_rate = kwargs.get('Max Setpoint Ramp Rate (C/min)')  # max setpoint ramp rate, in C/min
        self.temp_deadband_range = kwargs.get('Deadband Temperature (C)', 5.56)  # deadband range, in delta degC, i.e. Kelvin
        self.temp_deadband_on = None
        self.temp_deadband_off = None
        self.set_deadband_limits()

        # Other control parameters
        self.max_power = kwargs.get('Max Power (kW)')
        self.force_off = False

        # Thermal model parameters
        self.capacity = 0  # heat output from main element, in W
        self.delivered_heat = 0  # total heat delivered to the model, in W

    def set_deadband_limits(self):
        self.temp_deadband_off = self.temp_setpoint + (1 - self.setpoint_deadband_position) * self.temp_deadband_range * self.heat_mult
        self.temp_deadband_on = self.temp_setpoint - self.setpoint_deadband_position * self.temp_deadband_range * self.heat_mult

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
                self.temp_setpoint = ext_setpoint

        ext_db = control_signal.get("Deadband")
        if ext_db is not None:
            if f"{self.end_use} Deadband (C)" in self.current_schedule:
                self.current_schedule[f"{self.end_use} Deadband (C)"] = ext_db
            else:
                self.temp_deadband_range = ext_db

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
        update_deadband_temps = False

        # get setpoint from schedule
        if f"{self.end_use} Setpoint (C)" in self.current_schedule:
            self.temp_setpoint = self.current_schedule[f"{self.end_use} Setpoint (C)"]
    
        # constrain setpoint based on max ramp rate
        # TODO: create temp_setpoint_old and update in update_results.
        # Could get run multiple times per time step in update_model
        if self.setpoint_ramp_rate and self.temp_setpoint != self.temp_setpoint_old:
            delta_t = self.setpoint_ramp_rate * self.time_res.total_seconds() / 60  # in C
            self.temp_setpoint = min(
                max(self.temp_setpoint, self.temp_setpoint_old - delta_t),
                self.temp_setpoint_old + delta_t,
            )
        
        # get other controls from schedule - deadband and max power
        if f"{self.end_use} Deadband (C)" in self.current_schedule:
            self.temp_deadband_range = self.current_schedule[f"{self.end_use} Deadband (C)"]
        if f"{self.end_use} Max Power (kW)" in self.current_schedule:
            self.max_power = self.current_schedule[f"{self.end_use} Max Power (kW)"]

    def solve_ideal_capacity(self):
        # Solve thermal model for input heat injection to achieve setpoint
        capacity = self.thermal_model.solve_for_input(
            self.t_control_idx,
            self.h_control_idx,
            self.temp_setpoint,
            solve_as_output=False,
        )
        return capacity

    def update_capacity(self):
        return self.solve_ideal_capacity()

    def run_thermostat_control(self):
        # use thermostat with deadband control
        t_control = self.thermal_model.states[self.t_control_idx]

        if (t_control - self.temp_deadband_on) * self.heat_mult < 0:
            return "On"
        elif (t_control - self.temp_deadband_off) * self.heat_mult > 0:
            return "Off"
        else:
            # maintains existing mode
            return None

    def run_internal_control(self):
        # Update setpoint from schedule
        self.update_setpoint()

        if self.use_ideal_mode:
            # run ideal capacity calculation here
            # FUTURE: capacity update is done twice per loop, could but updated to improve speed
            self.capacity = self.update_capacity()
            return "On" if self.capacity > 0 else "Off"
        else:
            # Run thermostat controller and set speed
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
            results[f'{self.end_use} Deadband Lower Limit (C)'] = self.temp_setpoint - self.temp_deadband_range

        if self.save_ebm_results:
            results.update(self.make_equivalent_battery_model())

        return results
