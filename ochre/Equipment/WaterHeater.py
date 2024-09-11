# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 13:24:32 2018

@author: kmckenna, mblonsky
"""
import numpy as np
import datetime as dt

from ochre.utils import OCHREException
from ochre.utils.units import convert, kwh_to_therms
from ochre.Equipment import ThermostaticLoad
from ochre.Models import OneNodeWaterModel, TwoNodeWaterModel, StratifiedWaterModel, IdealWaterModel


class WaterHeater(ThermostaticLoad):
    name = 'Water Heater'
    end_use = 'Water Heating'
    default_capacity = 4500  # in W
    default_deadband = 5.56  # in C
    setpoint_deadband_position = 1  # setpoint at top of the deadband range
    optional_inputs = [
        "Water Heating Setpoint (C)",
        "Water Heating Deadband (C)",
        "Water Heating Max Power (kW)",
        "Zone Temperature (C)",  # Needed for Water tank model
    ]  
    
    def __init__(self, model_class=None, **kwargs):
        # Create water tank model
        if model_class is None:
            nodes = kwargs.get('water_nodes', 2)
            if nodes == 1:
                model_class = OneNodeWaterModel
            elif nodes == 2:
                model_class = TwoNodeWaterModel
            else:
                model_class = StratifiedWaterModel

        water_tank_args = {
            'main_sim_name': kwargs.get('name', self.name),
            **kwargs,
            'name': None,
            **kwargs.get('Water Tank', {}),
        }
        thermal_model = model_class(**water_tank_args)

        super().__init__(thermal_model=thermal_model, **kwargs)

        # Set control node to the bottom of the tank (for gas WH or 1-node model)
        lower_node = str(self.thermal_model.n_nodes)
        self.t_control_idx = self.thermal_model.state_names.index("T_WH" + lower_node)
        self.h_control_idx = (
            self.thermal_model.input_names.index("H_WH" + lower_node) - self.thermal_model.h_1_idx
        )

        # Control parameters
        self.temp_max = kwargs.get('Max Tank Temperature (C)', convert(140, 'degF', 'degC'))

    def update_inputs(self, schedule_inputs=None):
        # Add zone temperature to schedule inputs for water tank
        if not self.main_simulator:
            schedule_inputs['Zone Temperature (C)'] = schedule_inputs[f'{self.zone_name} Temperature (C)']
    
        super().update_inputs(schedule_inputs)

    def solve_ideal_capacity_by_mode(self, setpoint=None, mode=None, lowest_node=None):
        if setpoint is None:
            setpoint = self.temp_setpoint
        if lowest_node is None:
            lowest_node = self.t_control_idx

        # get next tank states based on mode
        if mode is None:
            # default is off - no heat added from water heater
            self.thermal_model.update_model()
        else:
            raise NotImplementedError()
        next_states = self.thermal_model.next_states

        # get ideal setpoint states
        set_states = np.ones(len(next_states)) * setpoint

        # get thermal energy to achieve setpoint states
        delta_t = set_states[: lowest_node + 1] - next_states[: lowest_node + 1]
        capacitances = self.thermal_model.capacitances[: lowest_node + 1]
        q_desired = np.dot(delta_t, capacitances)  # in J

        # return thermal capacity to achieve setpoint states, in W
        h_desired = q_desired / self.time_res.total_seconds()
        h_desired = max(h_desired, 0)
        return h_desired

    def solve_ideal_capacity(self, setpoint=None):
        if self.thermal_model.n_nodes == 1:
            # calculate ideal capacity using tank model directly
            # more accurate than code below
            return super().solve_ideal_capacity(setpoint)

        # calculate heat needed to reach setpoint 
        #  - only use nodes at and above control node
        return self.solve_ideal_capacity_by_mode(setpoint)

    def solve_deadband_mode(self, t_control=None, on_limit=None, off_limit=None):
        if t_control is None and self.thermal_model.n_nodes > 2:
            # take average of lower node and node above
            t_control = 1 / 2 * (
                self.thermal_model.states[self.t_control_idx]
                + self.thermal_model.states[self.t_control_idx - 1]
            )

        return super().solve_deadband_mode(t_control, on_limit, off_limit)

    def finish_sub_update(self, sub):
        # add heat losses from model to sensible gains
        self.sensible_gain += sub.h_loss

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

        return results

    def make_equivalent_battery_model(self):
        # returns a dictionary of equivalent battery model parameters
        total_cap = convert(sum(self.thermal_model.capacitances), 'J', 'kWh')  # in kWh/K
        ref_temp = 0  # temperature at Energy=0, in C
        if self.thermal_model.n_nodes <= 2:
            tank_temp = self.thermal_model.states[self.t_lower_idx]
        else:
            # take average of lower node and node above
            tank_temp = (self.thermal_model.states[self.t_lower_idx] + self.thermal_model.states[self.t_lower_idx - 1]) / 2
        baseline_power = (self.thermal_model.h_loss + self.thermal_model.h_delivered) / 1000  # from conduction losses and water draw
        return {
            f'{self.end_use} EBM Energy (kWh)': total_cap * (tank_temp - ref_temp),
            f'{self.end_use} EBM Min Energy (kWh)': total_cap * (self.temp_setpoint - self.temp_deadband - ref_temp),
            f'{self.end_use} EBM Max Energy (kWh)': total_cap * (self.temp_setpoint - ref_temp),
            f'{self.end_use} EBM Max Power (kW)': self.capacity_rated / self.efficiency / 1000,
            f'{self.end_use} EBM Efficiency (-)': self.efficiency,
            f'{self.end_use} EBM Baseline Power (kW)': baseline_power,
        }


class ElectricResistanceWaterHeater(WaterHeater):
    name = 'Electric Resistance Water Heater'
    modes = ['Upper On', 'Lower On', 'Off']

    def __init__(self, model_class=None, **kwargs):
        super().__init__(model_class, **kwargs)

        # Get tank nodes for upper and lower heat injections
        upper_node = "3" if self.thermal_model.n_nodes >= 12 else "1"
        self.t_upper_idx = self.thermal_model.state_names.index("T_WH" + upper_node)
        self.h_upper_idx = (
            self.thermal_model.input_names.index("H_WH" + upper_node) - self.thermal_model.h_1_idx
        )

        lower_node = "10" if self.thermal_model.n_nodes >= 12 else str(self.thermal_model.n_nodes)
        self.t_lower_idx = self.thermal_model.state_names.index("T_WH" + lower_node)
        self.h_lower_idx = (
            self.thermal_model.input_names.index("H_WH" + lower_node) - self.thermal_model.h_1_idx
        )

        # Mode and control parameters - for upper and lower elements
        # TODO: add these anywhere regular parameters are found
        self.capacity_upper = 0  # heat output from upper element, in W
        self.upper_on_frac = 0  # fraction of time on (0-1)
        self.upper_on_frac_new = 0  # fraction of time on (0-1)
        self.capacity_lower = 0  # heat output from lower element, in W
        self.lower_on_frac = 0  # fraction of time on (0-1)
        self.lower_on_frac_new = 0  # fraction of time on (0-1)
        self.upper_cycles = 0
        self.lower_cycles = 0

    def run_ideal_control(self, setpoint=None):
        if self.thermal_model.n_nodes == 1:
            return super().run_ideal_control(setpoint)

        # calculate heat for full tank to reach setpoint
        h_total = self.solve_ideal_capacity_by_mode(setpoint, lowest_node=self.t_lower_idx)

        # calculate heat for upper portion of tank to reach setpoint
        h_upper = self.solve_ideal_capacity_by_mode(setpoint, lowest_node=self.t_upper_idx)

        # constraint and save capacities
        if h_total > self.capacity_rated:
            # can't reach setpoint, prioritize upper element
            self.capacity = self.capacity_rated
            self.capacity_upper = min(h_upper, self.capacity_rated)
            self.capacity_lower = self.capacity - self.capacity_upper
        else:
            # can reach setpoint, prioritize lower element
            self.capacity = h_total
            self.capacity_upper = 0
            self.capacity_lower = h_total

        # save upper/lower on fractions
        self.upper_on_frac_new = self.capacity_upper / self.capacity_rated
        self.lower_on_frac_new = self.capacity_lower / self.capacity_rated

        # return total on fraction
        return self.capacity / self.capacity_rated

    def solve_deadband_mode(self, t_control=None, on_limit=None, off_limit=None):
        if t_control is None: 
            if self.thermal_model.n_nodes <= 2:
                # use lower index, not t_control_idx
                t_control = self.thermal_model.states[self.t_lower_idx]
            else:
                # take average of lower node and node above
                t_control = 1 / 2 * (
                    self.thermal_model.states[self.t_lower_idx]
                    + self.thermal_model.states[self.t_lower_idx - 1]
                )

        return super().solve_deadband_mode(t_control, on_limit, off_limit)

    def run_thermostat_control(self):
        if self.thermal_model.n_nodes == 1:
            return super().run_thermostat_control()

        # check if upper and lower tank temperatures are within deadband
        self.lower_on_frac_new = self.solve_deadband_mode()
        t_upper = self.thermal_model.states[self.t_upper_idx]
        self.upper_on_frac_new = self.solve_deadband_mode(t_upper)

        # update on fractions from last time step
        if self.upper_on_frac_new is None:
            self.upper_on_frac_new = self.upper_on_frac
        if self.lower_on_frac_new is None:
            self.lower_on_frac_new = self.lower_on_frac

        # prioritize upper element if both should be on
        if self.upper_on_frac_new and self.lower_on_frac_new:
            self.lower_on_frac_new = 0

        # Set capacities from on fraction
        on_frac_new = self.upper_on_frac_new + self.lower_on_frac_new
        self.capacity = on_frac_new * self.capacity_rated
        self.capacity_upper = self.upper_on_frac_new * self.capacity_rated
        self.capacity_lower = self.lower_on_frac_new * self.capacity_rated

        # return main on fraction
        return on_frac_new

    def limit_overshoot(self):
        if self.thermal_model.n_nodes == 1:
            return super().limit_overshoot()

        # check if upper or lower deadband limits are hit
        lower_on_frac_new = self.solve_deadband_mode()
        t_upper = self.thermal_model.states[self.t_upper_idx]
        upper_on_frac_new = self.solve_deadband_mode(t_upper)

        if lower_on_frac_new is None and upper_on_frac_new is None:
            # no overshoot, no change in mode
            return
        elif lower_on_frac_new == 1 and upper_on_frac_new == 0:
            # determine upper on fraction at deadband off temp
            self.on_frac_new = self.run_ideal_control(setpoint=self.temp_deadband_off)
            upper_on_tmp = self.upper_on_frac_new

            # determine lower on fraction at deadband on temp
            self.on_frac_new = self.run_ideal_control(setpoint=self.temp_deadband_on)
            
            # reset upper on fraction and capacity
            # may overestimate lower tank temp
            self.upper_on_frac_new = upper_on_tmp
            self.capacity_upper = self.upper_on_frac_new * self.capacity_rated
            self.on_frac_new = self.upper_on_frac_new + self.lower_on_frac_new
            self.capacity = self.on_frac_new * self.capacity_rated
            if self.capacity > self.capacity_rated:
                # reduce lower on fraction
                self.on_frac_new = 1
                self.capacity = self.capacity_rated
                self.capacity_lower = self.capacity - self.capacity_upper
                self.lower_on_frac_new = 1 - self.upper_on_frac_new

        elif lower_on_frac_new == 1:
            # reached lower deadband limit, turn lower on at end of time step
            self.on_frac_new = self.run_ideal_control(setpoint=self.temp_deadband_on)
            self.on_at_end = True

        elif upper_on_frac_new == 0:
            # reached upper deadband limit, turn off at end of time step
            self.on_frac_new = self.run_ideal_control(setpoint=self.temp_deadband_off)
            self.on_at_end = False
        else:
            # unexpected, print warning and do nothing
            self.warn(f"Unexpected on fractions for upper ({upper_on_frac_new}) "
                      f"and lower ({lower_on_frac_new}) elements")

    def add_heat_from_mode(self, mode, heats_to_tank=None, pct_time_on=1):
        if heats_to_tank is None:
            heats_to_tank = np.zeros(self.thermal_model.n_nodes, dtype=float)

        if mode == "Upper On":
            heats_to_tank[self.h_upper_idx] += self.capacity_rated * pct_time_on
        elif mode in ["On", "Lower On"]:
            # Works for 'On' or 'Lower On', treated the same
            heats_to_tank[self.h_lower_idx] += self.capacity_rated * pct_time_on

        return heats_to_tank

    def get_heat_to_model(self):
        if self.thermal_model.n_nodes == 1:
            return super().get_heat_to_model()
        
        return {self.h_upper_idx: self.capacity_upper,
                self.h_lower_idx: self.capacity_lower}


class HeatPumpWaterHeater(ElectricResistanceWaterHeater):
    name = 'Heat Pump Water Heater'
    modes = ['Heat Pump On', 'Lower On', 'Upper On', 'Off']
    optional_inputs = WaterHeater.optional_inputs + ['Zone Wet Bulb Temperature (C)']
    
    def __init__(self, hp_only_mode=False, water_nodes=12, **kwargs):
        super().__init__(water_nodes=water_nodes, **kwargs)

        # Control parameters
        self.hp_only_mode = hp_only_mode
        self.er_only_mode = False  # True when ambient temp is very hot or cold, forces HP off
        hp_on_time = kwargs.get('HPWH Minimum On Time (min)', 10)
        hp_off_time = kwargs.get('HPWH Minimum Off Time (min)', 0)
        self.min_time_in_mode['Heat Pump On'] = dt.timedelta(minutes=hp_on_time)
        self.min_time_in_mode['Off'] = dt.timedelta(minutes=hp_off_time)

        self.deadband_temp = kwargs.get('Deadband Temperature (C)', 8.17)  # different default than ERWH

        # Nominal COP based on simulation of the UEF test procedure at varying COPs
        self.cop_nominal = kwargs['HPWH COP (-)']
        self.hp_cop = self.cop_nominal
        if self.cop_nominal < 2:
            self.warn("Low Nominal COP:", self.cop_nominal)

        # Heat pump capacity and power parameters - hardcoded for now
        if 'HPWH Capacity (W)' in kwargs:
            self.hp_capacity_nominal = kwargs['HPWH Capacity (W)']  # max heating capacity, in W
        else:
            hp_power_nominal = kwargs.get('HPWH Power (W)', 500)  # in W
            self.hp_capacity_nominal = hp_power_nominal * self.hp_cop  # in W
        self.hp_capacity = self.hp_capacity_nominal  # in W
        self.parasitic_power = kwargs.get('HPWH Parasitics (W)', 1)  # Standby power in W
        self.fan_power = kwargs.get('HPWH Fan Power (W)', 35)  # in W

        # Dynamic capacity coefficients
        # curve format: [1, t_in_wet, t_in_wet ** 2, t_lower, t_lower ** 2, t_lower * t_in_wet]
        self.hp_capacity_coeff = np.array([0.563, 0.0437, 0.000039, 0.0055, -0.000148, -0.000145])
        self.cop_coeff = np.array([1.1332, 0.063, -0.0000979, -0.00972, -0.0000214, -0.000686])

        # Sensible and latent heat parameters
        self.shr_nominal = kwargs.get('HPWH SHR (-)', 0.88)  # unitless
        lost_heat_default = 0.75 if self.zone_name == 'Indoor' else 1  # for sensible heat gain
        self.lost_heat_fraction = 1 - kwargs.get('HPWH Interaction Factor (-)', lost_heat_default)
        self.wall_heat_fraction = kwargs.get('HPWH Wall Interaction Factor (-)', 0.5)
        if self.wall_heat_fraction and self.zone:
            walls = [s for s in self.zone.surfaces if s.boundary_name == 'Interior Wall']
            if not walls:
                raise OCHREException(f'Interior wall surface not found, required for {self.name} model.')
            self.wall_surface = walls[0]
        else:
            self.wall_surface = None
            # if self.wall_heat_fraction:
            #     zone_name = self.zone_name if self.zone_name is not None else 'External'
            #     self.warn(f'Removing HPWH wall heat fraction because zone is {zone_name}')
            #     self.wall_heat_fraction = 0

        # nodes used for HP delivered heat, also used for t_lower for biquadratic equations
        if self.thermal_model.n_nodes == 1:
            self.hp_nodes = np.array([1])
        elif self.thermal_model.n_nodes == 2:
            self.hp_nodes = np.array([0, 1])
        elif self.thermal_model.n_nodes == 12:
            self.hp_nodes = np.array([0, 0, 0, 0, 0, 5, 10, 15, 20, 25, 30, 5]) / 110
        else:
            raise OCHREException('{} model not defined for tank with {} nodes'.format(self.name, self.thermal_model.n_nodes))

    def update_inputs(self, schedule_inputs=None):
        # Add wet and dry bulb temperatures to schedule
        if not self.main_simulator:
            schedule_inputs['Zone Temperature (C)'] = schedule_inputs[f'{self.zone_name} Temperature (C)']
            schedule_inputs['Zone Wet Bulb Temperature (C)'] = schedule_inputs[f'{self.zone_name} Wet Bulb Temperature (C)']

        super().update_inputs(schedule_inputs)

    def run_ideal_control(self):
        # calculate ideal capacity based on future thermostat control
        if self.er_only_mode:
            super().run_ideal_control()
            self.duty_cycle_by_mode['Heat Pump On'] = 0
            return

        # Run model with heater off
        self.thermal_model.update_model()
        off_states = self.thermal_model.next_states.copy()
        # off_mode = self.run_thermostat_control(use_future_states=True)

        # Run model with HP on 100% (uses capacity from last time step)
        self.thermal_model.update_model(self.add_heat_from_mode('Heat Pump On'))
        hp_states = self.thermal_model.next_states.copy()
        hp_mode = self.run_thermostat_control(use_future_states=True)

        # aim 1/4 of deadband below setpoint to reduce temps at top of tank.
        set_states = np.ones(len(off_states)) * (self.temp_setpoint - self.deadband_temp / 4)

        if not self.hp_only_mode and hp_mode == 'Upper On':
            # determine ER duty cycle to achieve setpoint temp
            h_upper = np.dot(set_states[:self.t_upper_idx + 1] - hp_states[:self.t_upper_idx + 1],  # in W
                             self.thermal_model.capacitances[:self.t_upper_idx + 1]) / self.time_res.total_seconds()
            d_upper = min(max(h_upper / self.capacity_rated, 0), 1)

            # force HP on for the rest of the time
            d_hp = 1 - d_upper
        else:
            d_upper = 0

            # determine HP duty cycle to achieve setpoint temp
            # FUTURE: check against lab data
            h_hp = np.dot(set_states[:self.t_lower_idx + 1] - off_states[:self.t_lower_idx + 1],  # in W
                          self.thermal_model.capacitances[:self.t_lower_idx + 1]) / self.time_res.total_seconds()
            # using HP capacity from previous time step
            d_hp = min(max(h_hp / self.hp_capacity, 0), 1)

        self.duty_cycle_by_mode = {
            'Heat Pump On': d_hp,
            'Upper On': d_upper,
            'Lower On': 0,
            'Off': 1 - d_upper - d_hp,
        }

    def run_thermostat_control(self, use_future_states=False):
        # TODO: Need HPWH control logic validation
        if self.er_only_mode:
            if self.mode == 'Heat Pump On':
                self.mode = 'Off'
            return super().run_thermostat_control()

        model_temps = self.thermal_model.states if not use_future_states else self.thermal_model.next_states
        t_upper = model_temps[self.t_upper_idx]
        t_lower = model_temps[self.t_lower_idx]
        t_control = (3 / 4) * t_upper + (1 / 4) * t_lower

        if not self.hp_only_mode:
            if t_upper < self.temp_setpoint - 13 or (self.mode == 'Upper On' and t_upper < self.temp_setpoint):
                return 'Upper On'
            elif self.mode in ['Upper On', 'Lower On'] and t_lower < self.temp_setpoint - 15:
                return 'Lower On'

        if self.mode in ['Upper On', 'Lower On'] or t_control < self.temp_setpoint - self.deadband_temp:
            return 'Heat Pump On'
        elif t_control >= self.temp_setpoint:
            return 'Off'
        elif t_upper >= self.temp_setpoint + 1:  # TODO: Could mess with this a little
            return 'Off'

    def run_internal_control(self):
        # operate as ERWH when ambient temperatures are out of bounds
        t_amb = self.current_schedule['Zone Temperature (C)']
        if t_amb < 7.222 or t_amb > 43.333:
            self.er_only_mode = True
        else:
            self.er_only_mode = False

        return super().run_internal_control()

    def add_heat_from_mode(self, mode, heats_to_tank=None, duty_cycle=1):
        heats_to_tank = super().add_heat_from_mode(mode, heats_to_tank, duty_cycle)
        if mode == 'Heat Pump On':
            capacity_hp = self.hp_capacity * duty_cycle  # max heat from HP, in W
            heats_to_tank += self.hp_nodes * capacity_hp
        return heats_to_tank

    def update_cop_and_capacity(self, t_wet):
        t_lower = np.dot(self.hp_nodes, self.thermal_model.states)  # use node connected to condenser
        vector = np.array([1, t_wet, t_wet ** 2, t_lower, t_lower ** 2, t_lower * t_wet])
        self.hp_capacity = self.hp_capacity_nominal * np.dot(self.hp_capacity_coeff, vector)
        self.hp_cop = self.cop_nominal * np.dot(self.cop_coeff, vector)

    def calculate_power_and_heat(self):
        t_dry = self.current_schedule['Zone Temperature (C)']
        t_wet = self.current_schedule['Zone Wet Bulb Temperature (C)']

        # calculate dynamic capacity and COP
        self.update_cop_and_capacity(t_wet)

        # get delivered heat, note power and sensible/latent gains get overwritten
        heats_to_model = super().calculate_power_and_heat()

        # get HP and ER delivered heat and power
        if self.use_ideal_mode:
            d_hp = self.duty_cycle_by_mode['Heat Pump On']
            d_er = self.duty_cycle_by_mode['Upper On'] + self.duty_cycle_by_mode['Lower On']
        else:
            d_hp = 1 if 'Heat Pump' in self.mode else 0
            d_er = 1 if self.mode in ['Upper On', 'Lower On'] else 0

        delivered_hp = self.hp_capacity * d_hp  # in W
        power_hp = delivered_hp / self.hp_cop  # in W
        power_hp_other = self.fan_power * d_hp + self.parasitic_power * (1 - d_hp)  # in W
        delivered_er = self.capacity_rated * d_er
        power_er = delivered_er / self.efficiency  # in W

        # update shr based on humidity
        shr = self.shr_nominal if (t_dry - t_wet) > 0.1 else 1

        # update power and heat gains
        # note: heat gains from tank losses are added in parse_sub_update
        self.electric_kw = (power_hp + power_er + power_hp_other) / 1000
        self.sensible_gain = (power_hp - delivered_hp) * shr + power_hp_other + (power_er - delivered_er)
        self.sensible_gain *= 1 - self.lost_heat_fraction
        self.latent_gain = (1 - self.lost_heat_fraction) * (power_hp - delivered_hp) * (1 - shr)

        return heats_to_model

    def add_gains_to_zone(self):
        if self.wall_surface is not None:
            # split gains to zone and interior walls
            self.zone.internal_sens_gain += self.sensible_gain * (1 - self.wall_heat_fraction)
            self.wall_surface.internal_gain += self.sensible_gain * self.wall_heat_fraction
            self.zone.internal_latent_gain += self.latent_gain
        else:
            super().add_gains_to_zone()


    def finish_sub_update(self, sub):
        # add heat losses from model to sensible gains
        h_loss = sub.h_loss * (1 - self.lost_heat_fraction)
        self.sensible_gain += h_loss

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 6:
            if self.use_ideal_mode:
                hp_on_frac = self.duty_cycle_by_mode['Heat Pump On']
            else:
                hp_on_frac = 1 if 'Heat Pump' in self.mode else 0
            results[f'{self.end_use} Heat Pump Max Capacity (W)'] = self.hp_capacity
            results[f'{self.end_use} Heat Pump On Fraction (-)'] = hp_on_frac
            results[f'{self.end_use} Heat Pump COP (-)'] = self.hp_cop
        return results


class GasWaterHeater(WaterHeater):
    name = 'Gas Water Heater'
    is_gas = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs['Energy Factor (-)'] < 0.7:
            self.is_electric = False  # can stay on in islanded mode
            self.skin_loss_frac = 0.64
        elif kwargs['Energy Factor (-)'] < 0.8:
            self.skin_loss_frac = 0.91
        else:
            self.skin_loss_frac = 0.96

    def finish_sub_update(self, sub):
        # add heat losses from model to sensible gains
        # note: no sensible gains from heater (all is vented), tank losses reduced by skin loss frac
        self.sensible_gain = sub.h_loss * self.skin_loss_frac


class TanklessWaterHeater(WaterHeater):
    name = 'Tankless Water Heater'
    default_capacity = 20000  # in W

    def __init__(self, **kwargs):
        kwargs.update({'use_ideal_mode': True,
                       'model_class': IdealWaterModel})
        super().__init__(**kwargs)

        # update initial state to top of deadband (for 1-node model)
        self.thermal_model.states[self.t_control_idx] = self.temp_setpoint

    def update_setpoint(self):
        super().update_setpoint()

        # set state to setpoint temperature
        self.thermal_model.states[self.t_control_idx] = self.temp_setpoint

    def solve_ideal_capacity(self, setpoint=None):
        capacity = super().solve_ideal_capacity(setpoint)

        if capacity > self.capacity_rated:
            # cannot meet setpoint temperature. Update outlet temp for unmet loads
            t_set = self.temp_setpoint
            t_mains = self.thermal_model.current_schedule['Mains Temperature (C)']
            t_outlet = t_mains + (t_set - t_mains) * (self.capacity_rated / self.capacity)
            self.thermal_model.states[self.thermal_model.t_1_idx] = t_outlet
            self.thermal_model.update_water_draw()

        return capacity

    def update_results(self):
        current_results = super().update_results()

        # Reset tank model to setpoint temperature
        self.thermal_model.states[self.thermal_model.t_1_idx] = self.temp_setpoint

        return current_results


class GasTanklessWaterHeater(TanklessWaterHeater):
    name = 'Gas Tankless Water Heater'
    is_gas = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # get parasitic power
        self.parasitic_power = kwargs['Parasitic Power (W)'] / 1000  # in kW

    def calculate_power_and_heat(self):
        heats_to_model = super().calculate_power_and_heat()

        # add constant parasitic power
        self.electric_kw = self.parasitic_power

        return heats_to_model