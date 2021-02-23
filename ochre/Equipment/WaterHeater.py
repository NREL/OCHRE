# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt

from ochre.Equipment import Equipment, EquipmentException
from ochre.Models import OneNodeWaterModel, TwoNodeWaterModel, StratifiedWaterModel, IdealWaterModel
from ochre.Models.Envelope import ZONES as ENVELOPE_NODES
from ochre import Units

# Water Constants
water_density = 1000  # kg/m^3
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_conductivity = 0.6406  # W/m-K


class WaterHeater(Equipment):
    name = 'Water Heater'
    end_use = 'Water Heating'

    def __init__(self, use_ideal_capacity=None, model_class=None, **kwargs):
        super().__init__(**kwargs)

        # By default, use ideal capacity if time resolution > 5 minutes
        if use_ideal_capacity is None:
            use_ideal_capacity = self.time_res >= dt.timedelta(minutes=5)
        self.use_ideal_capacity = use_ideal_capacity

        # Create water tank model
        if model_class is None:
            nodes = kwargs.get('water_nodes', 2)
            if nodes == 1:
                model_class = OneNodeWaterModel
            elif nodes == 2:
                model_class = TwoNodeWaterModel
            if nodes > 2:
                model_class = StratifiedWaterModel
        self.model = model_class(**kwargs)

        upper_node = '3' if self.model.n_nodes >= 12 else '1'
        self.t_upper_idx = self.model.state_names.index('T_WH' + upper_node)
        self.h_upper_idx = self.model.input_names.index('H_WH' + upper_node) - 1  # accounting for T_AMB input

        lower_node = '10' if self.model.n_nodes >= 12 else str(self.model.n_nodes)
        self.t_lower_idx = self.model.state_names.index('T_WH' + lower_node)
        self.h_lower_idx = self.model.input_names.index('H_WH' + lower_node) - 1  # accounting for T_AMB input

        # Capacity and efficiency parameters
        power_rated = kwargs['rated input power (W)']  # maximum input power, in W
        self.efficiency = kwargs.get('eta_c', 1)
        self.capacity_rated = power_rated * self.efficiency  # maximum heat delivered, in W
        self.delivered_heat = 0  # heat delivered to the tank, in W

        if kwargs.get('water heater location') == 'living':
            self.zone = 'LIV'
        elif kwargs.get('water heater location') in ['basement', 'crawlspace']:
            self.zone = 'FND'
        elif kwargs.get('water heater location') == 'garage':
            self.zone = 'GAR'

        # Control parameters
        self.upper_threshold_temp = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        deadband_temp = kwargs.get('deadband (delta C)', 5.56)
        self.lower_threshold_temp = self.upper_threshold_temp - deadband_temp
        self.max_temp = Units.F2C(kwargs.get('max tank temperature (F)', 140))

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - Load Fraction: 1 (no effect) or 0 (forces WH off)
        # - Setpoint: Updates setpoint temperature from the dwelling schedule (in C)
        #   - Note: Setpoint will not reset back to original value
        # - Deadband: Updates deadband temperature (in C)
        #   - Note: Deadband will not reset back to original value
        # - Duty Cycle: Forces WH on for fraction of external time step (as fraction [0,1])
        #   - If 0 < Duty Cycle < 1, the equipment will cycle once every 2 external time steps
        #   - For HPWH: Can supply HP and ER duty cycles
        #   - Note: does not use clock on/off time

        # If load fraction = 0, force off
        load_fraction = ext_control_args.get('Load Fraction', 1)
        if load_fraction == 0:
            return 'Off'
        elif load_fraction != 1:
            raise Exception("{} can't handle non-integer load fractions".format(self.name))

        ext_setpoint = ext_control_args.get('Setpoint')
        if ext_setpoint is not None:
            if ext_setpoint > self.max_temp:
                self.warn('Setpoint cannot exceed {}C. Setting setpoint to maximum value.'.format(self.max_temp))
                ext_setpoint = self.max_temp

            # keep deadband the same
            self.lower_threshold_temp += ext_setpoint - self.upper_threshold_temp
            self.upper_threshold_temp = ext_setpoint

        ext_db = ext_control_args.get('Deadband')
        if ext_db is not None:
            self.lower_threshold_temp = self.upper_threshold_temp - ext_db

        # Force off if temperature exceeds maximum, and print warning
        t_tank = self.model.states[self.t_upper_idx]
        if t_tank > self.max_temp:
            self.warn('Temperature over maximum temperature ({}C), forcing off'.format(self.max_temp))
            return 'Off'

        if 'Duty Cycle' in ext_control_args:
            # Parse duty cycles into list for each mode
            duty_cycles = ext_control_args.get('Duty Cycle')
            if isinstance(duty_cycles, (int, float)):
                duty_cycles = [duty_cycles]
            if not isinstance(duty_cycles, list) or not (0 <= sum(duty_cycles) <= 1):
                raise EquipmentException('Error parsing {} duty cycle control: {}'.format(self.name, duty_cycles))

            return self.run_duty_cycle_control(schedule, duty_cycles)
        else:
            return self.update_internal_control(schedule)

    def run_duty_cycle_control(self, schedule, duty_cycles):
        if self.use_ideal_capacity:
            # Set capacity directly from duty cycle
            self.update_duty_cycles(*duty_cycles)
            return [mode for mode, duty_cycle in self.duty_cycle_by_mode.items() if duty_cycle > 0][0]

        else:
            # Use internal mode if available, otherwise use mode with highest priority
            mode_priority = self.calculate_mode_priority(*duty_cycles)
            internal_mode = self.update_internal_control(schedule)
            if internal_mode is None:
                internal_mode = self.mode
            if internal_mode in mode_priority:
                return internal_mode
            else:
                return mode_priority[0]  # take highest priority mode (usually current mode)

    def solve_ideal_capacity(self, schedule):
        # calculate ideal capacity based on achieving lower node setpoint temperature
        # Run model with heater off, updates next_states
        self.model.update(schedule=schedule, check_bounds=False)
        off_states = self.model.next_states

        # calculate heat needed to reach setpoint - only use nodes at and above lower node
        set_states = np.ones(len(off_states)) * self.upper_threshold_temp
        h_desired = np.dot(set_states[:self.t_lower_idx + 1] - off_states[:self.t_lower_idx + 1],  # in W
                           self.model.state_capacitances[:self.t_lower_idx + 1]) / self.time_res.total_seconds()

        # Convert to duty cycle, maintain min/max bounds
        duty_cycle = min(max(h_desired / self.capacity_rated, 0), 1)
        self.duty_cycle_by_mode = {'On': duty_cycle, 'Off': 1 - duty_cycle}

    def run_thermostat_control(self, schedule):
        # use thermostat with deadband control
        t_tank = self.model.states[self.t_lower_idx]  # use lower node for gas WH, not for ERWH
        if t_tank < self.lower_threshold_temp:
            return 'On'
        if t_tank > self.upper_threshold_temp:
            return 'Off'

    def update_internal_control(self, schedule):
        if self.use_ideal_capacity:
            if self.model.n_nodes == 1:
                # FUTURE: remove if not being used
                # calculate ideal capacity based on tank model - more accurate than self.solve_ideal_capacity
                # Get water draw inputs and ambient temp
                heats_to_tank = self.model.update_water_draw(schedule)
                water_inputs = np.insert(heats_to_tank, self.model.t_amb_idx, schedule['Indoor'])

                # Solve for desired heat delivered, subtracting external gains
                self.model.update_inputs(water_inputs)
                h_desired = self.model.solve_for_input(self.model.t_1_idx, self.model.h_1_idx,
                                                       self.upper_threshold_temp)

                # Only allow heating, convert to duty cycle
                h_desired = min(max(h_desired, 0), self.capacity_rated)
                duty_cycle = h_desired / self.capacity_rated
                self.duty_cycle_by_mode = {mode: 0 for mode in self.modes}
                self.duty_cycle_by_mode[self.modes[0]] = duty_cycle
                self.duty_cycle_by_mode['Off'] = 1 - duty_cycle
            else:
                self.solve_ideal_capacity(schedule)

            return [mode for mode, duty_cycle in self.duty_cycle_by_mode.items() if duty_cycle > 0][0]
        else:
            return self.run_thermostat_control(schedule)

    def add_heat_to_tank(self, mode, duty_cycle=1):
        heats_to_tank = np.zeros(self.model.n_nodes, dtype=float)
        if 'Upper' in mode:
            heats_to_tank[self.h_upper_idx] = self.capacity_rated * duty_cycle
        elif 'On' in mode:
            # Works for 'On' or 'Lower On', treated the same
            heats_to_tank[self.h_lower_idx] = self.capacity_rated * duty_cycle

        return heats_to_tank

    def calculate_power_and_heat(self, schedule):
        # get heat injections from water heater
        heats_to_tank = np.zeros(self.model.n_nodes, dtype=float)
        if self.use_ideal_capacity:
            for mode, duty_cycle in self.duty_cycle_by_mode.items():
                heats_to_tank += self.add_heat_to_tank(mode, duty_cycle)
        else:
            heats_to_tank += self.add_heat_to_tank(self.mode)

        self.delivered_heat = heats_to_tank.sum()
        power = self.delivered_heat / self.efficiency  # in W

        if self.is_gas:
            # note: no sensible gains from heater (all is vented)
            self.gas_therms_per_hour = Units.kWh2therms(power / 1000)  # W to therms/hour
            self.sensible_gain = 0
        else:
            self.electric_kw = power / 1000
            self.sensible_gain = power - self.delivered_heat  # in W

        self.latent_gain = 0

        # run model update to get heat loss. Assumes water heater is in main indoor node, assumes no latent gains
        heat_loss = self.model.update(heats_to_tank, schedule)
        self.sensible_gain += heat_loss

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            t_avg = np.dot(self.model.vol_fractions, self.model.states)  # weighted average temperature
            return {self.name: {'T Average': t_avg,
                                'T Upper Node': self.model.states[self.t_upper_idx],
                                'T Lower Node': self.model.states[self.t_lower_idx],
                                'T Upper Limit': self.upper_threshold_temp,
                                'T Lower Limit': self.lower_threshold_temp,
                                'Is On': 'On' in self.mode,
                                'Tank Volume': self.model.volume}}
        else:
            if verbosity >= 3:
                # Note: using end use, not equipment name, for all results
                results.update({self.end_use + ' Delivered (kW)': self.delivered_heat / 1000})
            if verbosity >= 6:
                cop = self.delivered_heat / (self.electric_kw * 1000) if self.electric_kw > 0 else 0
                results.update({self.end_use + ' COP (-)': cop,
                                self.end_use + ' Sensible Heat to Air (kW)': self.sensible_gain / 1000})
            results.update(self.model.generate_results(verbosity, to_ext))
        return results

    def update_model(self, schedule):
        super().update_model(schedule)

        # water model update is already done. Set states to next_states
        self.model.states = self.model.next_states


class ElectricResistanceWaterHeater(WaterHeater):
    name = 'Electric Resistance Water Heater'
    modes = ['Upper On', 'Lower On', 'Off']

    def run_duty_cycle_control(self, schedule, duty_cycles):
        if len(duty_cycles) == len(self.modes) - 2:
            d_er_total = duty_cycles[-1]
            if self.use_ideal_capacity:
                # determine optimal allocation of upper/lower elements
                self.solve_ideal_capacity(schedule)

                # keep upper duty cycle as is, update lower based on external control
                d_upper = self.duty_cycle_by_mode['Upper On']
                d_lower = d_er_total - d_upper
                self.duty_cycle_by_mode['Lower On'] = d_lower
                self.duty_cycle_by_mode['Off'] = 1 - d_er_total
            else:
                # copy duty cycle for Upper On and Lower On, and calculate Off duty cycle
                duty_cycles.append(d_er_total)
                duty_cycles.append(1 - sum(duty_cycles[:-1]))

        mode = super().run_duty_cycle_control(schedule, duty_cycles)

        if not self.use_ideal_capacity:
            # If duty cycle forces WH on, may need to swap to lower element
            t_upper = self.model.states[self.t_upper_idx]
            if mode == 'Upper On' and t_upper > self.upper_threshold_temp:
                mode = 'Lower On'

            # If mode is ER, add time to both mode_counters
            if mode == 'Upper On':
                self.ext_mode_counters['Lower On'] += self.time_res
            if mode == 'Lower On':
                self.ext_mode_counters['Upper On'] += self.time_res

        return mode

    def solve_ideal_capacity(self, schedule):
        # calculate ideal capacity based on upper and lower node setpoint temperatures
        # Run model with heater off
        self.model.update(schedule=schedule, check_bounds=False)
        off_states = self.model.next_states

        # calculate heat needed to reach setpoint - only use nodes at and above upper/lower nodes
        set_states = np.ones(len(off_states)) * self.upper_threshold_temp
        h_total = np.dot(set_states[:self.t_lower_idx + 1] - off_states[:self.t_lower_idx + 1],  # in W
                         self.model.state_capacitances[:self.t_lower_idx + 1]) / self.time_res.total_seconds()
        h_upper = np.dot(set_states[:self.t_upper_idx + 1] - off_states[:self.t_upper_idx + 1],  # in W
                         self.model.state_capacitances[:self.t_upper_idx + 1]) / self.time_res.total_seconds()
        h_lower = h_total - h_upper

        # Convert to duty cycle, maintain min/max bounds, upper gets priority
        d_upper = min(max(h_upper / self.capacity_rated, 0), 1)
        d_lower = min(max(h_lower / self.capacity_rated, 0), 1 - d_upper)
        self.duty_cycle_by_mode = {'Upper On': d_upper, 'Lower On': d_lower, 'Off': 1 - d_upper - d_lower}

    def run_thermostat_control(self, schedule):
        # use thermostat with deadband control, upper element gets priority over lower element
        t_upper = self.model.states[self.t_upper_idx]
        if self.model.n_nodes <= 2:
            t_lower = self.model.states[self.t_lower_idx]
        else:
            # take average of lower node and node above
            t_lower = (self.model.states[self.t_lower_idx] + self.model.states[self.t_lower_idx - 1]) / 2

        if t_upper < self.lower_threshold_temp or (self.mode == 'Upper On' and t_upper < self.upper_threshold_temp):
            return 'Upper On'
        if t_lower < self.lower_threshold_temp:
            return 'Lower On'
        if self.mode == 'Upper On' and t_upper > self.upper_threshold_temp:
            return 'Off'
        if t_lower > self.upper_threshold_temp:
            return 'Off'


class HeatPumpWaterHeater(ElectricResistanceWaterHeater):
    name = 'Heat Pump Water Heater'
    modes = ['Heat Pump On', 'Lower On', 'Upper On', 'Off']

    def __init__(self, hp_only_mode=False, water_nodes=12, **kwargs):
        super().__init__(water_nodes=water_nodes, **kwargs)

        # Control parameters
        self.t_upper_prev = self.model.states[self.t_upper_idx]
        self.upper_threshold_temp_prev = self.upper_threshold_temp
        self.hp_only_mode = hp_only_mode
        self.er_only_mode = False  # True when ambient temp is very hot or cold, forces HP off

        # Nominal COP based on simulation of the UEF test procedure at varying COPs
        self.cop_nominal = kwargs.get('HPWH COP', 1.174536058 * (0.60522 + kwargs['EF']) / 1.2101)
        self.hp_cop = self.cop_nominal

        # Heat pump capacity - hardcoded for now
        self.hp_power_nominal = kwargs.get('HPWH Power', 979)  # in W
        self.hp_power = self.hp_power_nominal

        # Dynamic capacity coefficients
        # curve format: [1, t_in_wet, t_in_wet ** 2, t_lower, t_lower ** 2, t_lower * t_in_wet]
        self.hp_power_coeff = np.array([0.563, 0.0437, 0.000039, 0.0055, -0.000148, -0.000145])
        self.cop_coeff = np.array([1.1332, 0.063, -0.0000979, -0.00972, -0.0000214, -0.000686])

        # Other HP coefficients
        self.shr_nominal = kwargs.get('HPWH SHR', 0.98)  # unitless
        self.parasitic_power = kwargs.get('HPWH Parasitics (W)', 3.0)  # Standby power in W
        self.fan_power = kwargs.get('HPWH Fan Power (W)', 0.0462 * 181)  # in W

        # nodes used for HP delivered heat, also used for t_lower for biquadratic equations
        if self.model.n_nodes == 1:
            self.hp_nodes = np.array([1])
        elif self.model.n_nodes == 2:
            self.hp_nodes = np.array([0, 1])
        elif self.model.n_nodes == 12:
            self.hp_nodes = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1]) / 12
        else:
            raise Exception('{} model not defined for tank with {} nodes'.format(self.name, self.model.n_nodes))

    def update_external_control(self, schedule, ext_control_args):
        if any([dc in ext_control_args for dc in ['HP Duty Cycle', 'ER Duty Cycle']]):
            # Add HP duty cycle to ERWH control
            duty_cycles = [ext_control_args.get('HP Duty Cycle', 0),
                           ext_control_args.get('ER Duty Cycle', 0) if not self.hp_only_mode else 0]
            ext_control_args['Duty Cycle'] = duty_cycles

        return super().update_external_control(schedule, ext_control_args)

    def solve_ideal_capacity(self, schedule):
        # calculate ideal capacity based on future thermostat control
        if self.er_only_mode:
            return super().solve_ideal_capacity(schedule)

        # Run model with heater off and with HP on 100%
        self.model.update(schedule=schedule, check_bounds=False)
        off_states = self.model.next_states.copy()
        # off_mode = self.run_thermostat_control(schedule, use_future_states=True)

        all_hp_heats = self.add_heat_to_tank('Heat Pump On')
        self.model.update(all_hp_heats, schedule=schedule, check_bounds=False)
        hp_states = self.model.next_states
        hp_mode = self.run_thermostat_control(schedule, use_future_states=True)

        # aim 1/4 of deadband below setpoint to reduce temps at top of tank.
        set_states = np.ones(len(off_states)) * (self.upper_threshold_temp - 3.89 / 4)

        if not self.hp_only_mode and hp_mode == 'Upper On':
            # determine ER duty cycle to achieve setpoint temp
            h_upper = np.dot(set_states[:self.t_upper_idx + 1] - hp_states[:self.t_upper_idx + 1],  # in W
                             self.model.state_capacitances[:self.t_upper_idx + 1]) / self.time_res.total_seconds()
            d_upper = min(max(h_upper / self.capacity_rated, 0), 1)

            # force HP on for the rest of the time
            d_hp = 1 - d_upper
        else:
            d_upper = 0

            # determine HP duty cycle to achieve setpoint temp
            # FUTURE: check against lab data
            h_hp = np.dot(set_states[:self.t_lower_idx + 1] - off_states[:self.t_lower_idx + 1],  # in W
                          self.model.state_capacitances[:self.t_lower_idx + 1]) / self.time_res.total_seconds()
            # using HP power and COP from previous time step
            d_hp = min(max(h_hp / (self.hp_power * self.hp_cop), 0), 1)

        self.duty_cycle_by_mode = {'Heat Pump On': d_hp, 'Upper On': d_upper, 'Lower On': 0, 'Off': 1 - d_upper - d_hp}

    def run_thermostat_control(self, schedule, use_future_states=False):
        # TODO: Need HPWH control logic validation
        if self.er_only_mode:
            return super().solve_ideal_capacity(schedule)

        model_temps = self.model.states if not use_future_states else self.model.next_states
        t_upper = model_temps[self.t_upper_idx]
        t_lower = model_temps[self.t_lower_idx]
        t_control = (3 / 4) * t_upper + (1 / 4) * t_lower

        if not self.hp_only_mode and (t_upper < self.upper_threshold_temp - 18.5 or
                                      (self.mode == 'Upper On' and t_upper < self.upper_threshold_temp)):
            return 'Upper On'
        elif t_control < self.upper_threshold_temp - 3.89:
            return 'Heat Pump On'
        elif self.mode == 'Upper On' and t_upper >= self.upper_threshold_temp:
            return 'Off'
        elif t_control >= self.upper_threshold_temp:
            return 'Off'

    def update_internal_control(self, schedule):
        # operate as ERWH when ambient temperatures are out of bounds
        ambient_node = ENVELOPE_NODES[self.zone] if self.zone is not None else 'ambient_dry_bulb'
        t_amb = schedule[ambient_node]
        if t_amb < 7.222 or t_amb > 43.333:
            self.er_only_mode = True
        else:
            self.er_only_mode = False

        return super().update_internal_control(schedule)

    def add_heat_to_tank(self, mode, duty_cycle=1):
        if 'Heat Pump' in mode:
            capacity_hp = self.hp_power * self.hp_cop * duty_cycle  # max heat from HP, in W
            heats_to_tank = self.hp_nodes * capacity_hp
            return heats_to_tank
        else:
            return super().add_heat_to_tank(mode, duty_cycle=duty_cycle)

    def update_cop_and_power(self, schedule):
        # TODO: update if HPWH not in Indoor zone
        t_in_wet = schedule['Indoor Wet Bulb']
        t_lower = np.dot(self.hp_nodes, self.model.states)  # use noded connected to condenser
        vector = np.array([1, t_in_wet, t_in_wet ** 2, t_lower, t_lower ** 2, t_lower * t_in_wet])
        self.hp_power = self.hp_power_nominal * np.dot(self.hp_power_coeff, vector)
        self.hp_cop = self.cop_nominal * np.dot(self.cop_coeff, vector)

    def calculate_power_and_heat(self, schedule):
        # calculate dynamic capacity and COP
        self.update_cop_and_power(schedule)

        # get delivered heat and update model, note power and sensible/latent gains get overwritten
        super().calculate_power_and_heat(schedule)

        if 'Heat Pump' not in self.mode:
            # Heat pump is off, add parasitic power
            self.electric_kw += self.parasitic_power / 1000
            return

        # get HP and ER delivered heat and power
        if self.use_ideal_capacity:
            d_hp = self.duty_cycle_by_mode['Heat Pump On']
            d_er = self.duty_cycle_by_mode['Upper On'] + self.duty_cycle_by_mode['Lower On']
        else:
            d_hp = 1 if 'Heat Pump' in self.mode else 0
            d_er = 1 if self.mode in ['Upper On', 'Lower On'] else 0

        power_hp = self.hp_power * d_hp  # in W
        power_hp_other = self.fan_power * d_hp + self.parasitic_power * (1 - d_hp)  # in W
        delivered_hp = self.hp_cop * power_hp
        delivered_er = d_er * self.capacity_rated
        power_er = delivered_er / self.efficiency  # in W

        # update shr based on humidity
        w = schedule['Indoor Humidity Ratio']
        shr = self.shr_nominal if w > 0.0001 else 1

        # update power, heat gains from heat pump
        self.electric_kw = (power_hp + power_er + power_hp_other) / 1000
        self.sensible_gain = ((power_hp - delivered_hp) * shr + power_hp_other +
                              (power_er - delivered_er))
        self.latent_gain = (power_hp - delivered_hp) * (1 - shr)
        self.sensible_gain += self.model.h_loss

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if not to_ext:
            if verbosity >= 6:
                if self.use_ideal_capacity:
                    hp_on_frac = self.duty_cycle_by_mode['Heat Pump On']
                else:
                    hp_on_frac = 1 if 'Heat Pump' in self.mode else 0
                results.update({self.end_use + ' Heat Pump Max Power (kW)': self.hp_power / 1000,
                                self.end_use + ' Heat Pump On Fraction (-)': hp_on_frac,
                                self.end_use + ' Heat Pump COP (-)': self.hp_cop})
        return results


class GasWaterHeater(WaterHeater):
    name = 'Gas Water Heater'
    is_gas = True
    is_electric = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if kwargs['EF'] < 0.7:
            self.skin_loss_frac = 0.64
        elif kwargs['EF'] < 0.8:
            self.skin_loss_frac = 0.91
        else:
            self.skin_loss_frac = 0.96

    def calculate_power_and_heat(self, schedule):
        super().calculate_power_and_heat(schedule)

        # note: no sensible gains from heater (all is vented), tank losses reduced by skin loss frac
        self.sensible_gain = self.model.h_loss * self.skin_loss_frac


class TanklessWaterHeater(WaterHeater):
    name = 'Tankless Water Heater'

    def __init__(self, **kwargs):
        kwargs.update({'use_ideal_capacity': True,
                       'model_class': IdealWaterModel})
        super().__init__(**kwargs)

        # Control parameters - reduce efficiency by tankless derate percentage
        tankless_derate = 1 - 0.08
        self.efficiency *= tankless_derate
        self.capacity_rated *= tankless_derate  # in W
        self.heat_from_draw = 0

        # update initial state to top of deadband (for 1-node model)
        self.model.states[self.t_upper_idx] = self.upper_threshold_temp

    def update_internal_control(self, schedule):
        self.heat_from_draw = -self.model.update_water_draw(schedule)[0]
        self.heat_from_draw = max(self.heat_from_draw, 0)

        return 'On' if self.heat_from_draw > 0 else 'Off'

    def calculate_power_and_heat(self, schedule):
        if self.heat_from_draw > self.capacity_rated:
            # cannot meet setpoint temperature. Update outlet temp for 1 time step
            t_set = self.upper_threshold_temp
            t_mains = schedule.get('mains_temperature')
            t_outlet = t_mains + (t_set - t_mains) * (self.capacity_rated / self.heat_from_draw)
            self.model.states[self.model.t_1_idx] = t_outlet
            self.model.update_water_draw(schedule)

            # Reset tank model and update delivered heat
            self.model.states[self.model.t_1_idx] = t_set
            self.delivered_heat = self.capacity_rated
        else:
            self.delivered_heat = self.heat_from_draw

        self.electric_kw = self.delivered_heat / self.efficiency / 1000
        # for now, no extra heat gains for tankless water heater
        # self.sensible_gain = self.delivered_heat * (1 / self.efficiency - 1)
        self.sensible_gain = 0


class GasTanklessWaterHeater(TanklessWaterHeater):
    name = 'Gas Tankless Water Heater'
    is_electric = True  # parasitic power is electric
    is_gas = True

    def calculate_power_and_heat(self, schedule):
        super().calculate_power_and_heat(schedule)

        # gas power in therms/hour
        power_kw = self.delivered_heat / self.efficiency / 1000
        self.gas_therms_per_hour = Units.kWh2therms(power_kw)

        if self.mode == 'On':
            self.electric_kw = 65 / 1000  # hardcoded parasitic electric power
        else:
            self.electric_kw = 5 / 1000  # hardcoded electric power
