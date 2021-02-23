# -*- coding: utf-8 -*-
import datetime as dt
import numpy as np
import scipy.interpolate
import psychrolib

from ochre.Equipment import Equipment, EquipmentException
from ochre import Psychrometrics_HVAC, Units

PsyCalc = Psychrometrics_HVAC.Psychrometrics()
psychrolib.SetUnitSystem(psychrolib.SI)

SPEED_TYPES = {
    1: 'Single',
    2: 'Double',
    4: 'Variable',
}

cp_air = 1.005  # kJ/kg/K
rho_air = 1.2041  # kg/m^3


class HVAC(Equipment):
    """
    Base HVAC Equipment Class. Options for static and ideal capacity. `hvac_type` must be specified in child classes.

    The ideal capacity algorithm uses the envelope model to determine the exact HVAC
     capacity to maintain the setpoint temperature. It does not account for heat gains from other equipment in the
     same time step.
    """
    name = 'Generic HVAC'
    hvac_type = None  # Options are 'heat' and 'cool'
    n_speeds = 1

    def __init__(self, envelope_model=None, use_ideal_capacity=None, **kwargs):
        super().__init__(**kwargs)

        if self.hvac_type == 'heat':
            self.is_heater = True
            self.hvac_mult = 1
        elif self.hvac_type == 'cool':
            self.is_heater = False
            self.hvac_mult = -1
        else:
            raise EquipmentException('HVAC type for {} Equipment must be "heat" or "cool".'.format(self.name))

        # use ideal or static/dynamic capacity depending on time resolution and number of speeds
        # 4 speeds are used for variable speed equipment, which must use ideal capacity
        if use_ideal_capacity is None:
            use_ideal_capacity = self.time_res >= dt.timedelta(minutes=5) or self.n_speeds >= 4
        self.use_ideal_capacity = use_ideal_capacity

        # Capacity parameters
        self.speed_idx = 0  # speed index, varies for multi-speed equipment
        self.rated_capacity_list = kwargs[self.hvac_type + 'ing capacity (W)']  # rated capacities in W, by speed
        self.capacity = self.rated_capacity_list[self.speed_idx]
        self.capacity_max = max(self.rated_capacity_list)  # controllable for ideal equipment, in W
        self.capacity_min = kwargs.get(self.hvac_type + 'ing minimum capacity (W)', 0)  # for ideal equipment, in W
        self.space_fraction = kwargs.get(self.hvac_type + 'ing conditioned space fraction', 1.0)
        self.delivered_heat = 0  # in W

        # Efficiency and loss parameters
        self.rated_eir_list = kwargs[self.hvac_type + 'ing EIR']  # Energy Input Ratios, by speed
        self.eir = self.rated_eir_list[self.speed_idx]
        self.eir_max = max(self.rated_eir_list)
        self.duct_dse = kwargs[self.hvac_type + 'ing duct dse']  # Duct distribution system efficiency
        self.duct_location = kwargs.get('duct location', 'living')
        if self.duct_location == 'living':
            self.duct_location = None
        elif self.duct_location in ['crawlspace', 'basement']:
            self.duct_location = 'FND'
        self.basement_heat_frac = kwargs.get('basement HVAC airflow ratio')

        # Air flow parameters
        temp_setpoint = kwargs['initial_schedule']['{}ing_setpoint'.format(self.hvac_type)]
        shr_rated = kwargs.get(self.hvac_type + 'ing SHR', 1)
        if isinstance(self, DynamicHVAC):
            # calculate flow rates based on capacity and supply air temperature
            if self.is_heater:
                delta_t = Units.F2C(105) - temp_setpoint
            else:
                # interpolate to get cooling supply temp based on rated SHR, must be between 54-58 F
                cool_supply_temp = np.clip(54 + (58 - 54) * (shr_rated - 0.8) / (0.85 - 0.80), 54, 58)
                delta_t = temp_setpoint - Units.F2C(cool_supply_temp)
            self.flow_rates = [cap / 1000 / rho_air / cp_air / delta_t for cap in self.rated_capacity_list]  # in m^3/s
        else:
            # Use nominal flow rates
            flow_rates = kwargs[self.hvac_type + 'ing airflow rate (cfm)']
            self.flow_rates = [Units.cfm2m3_s(rate) for rate in flow_rates]  # Air flow rates, in m^3/s
        self.max_flow_rate = max(self.flow_rates)

        # Fan power parameters
        self.rated_fan_power_list = [Units.m3_s2cfm(flow_rate) * kwargs[self.hvac_type + 'ing fan power (W/cfm)']
                                     for flow_rate in self.flow_rates]  # power output from fan, in W, by speed
        self.fan_power = 0  # in W
        self.fan_power_max = max(self.rated_fan_power_list)
        self.fan_power_ratio = self.fan_power_max / (self.capacity_max * self.eir_max)  # For ideal equipment
        self.coil_input_db = temp_setpoint  # Dry bulb temperature after increase from fan power
        self.coil_input_wb = temp_setpoint  # Wet bulb temperature after increase from fan power

        # check length of rated lists
        for rated_list in [self.rated_capacity_list, self.rated_eir_list, self.rated_fan_power_list]:
            if len(rated_list) != self.n_speeds:
                raise EquipmentException('Number of speeds ({}) does not match length of rated list ({})'
                                         .format(self.n_speeds, len(rated_list)))

        # Humidity and SHR parameters, cooling only
        self.shr_rated = shr_rated
        if not self.is_heater:
            rated_dry_bulb = Units.F2C(80)  # in degrees C
            rated_wet_bulb = Units.F2C(67)  # in degrees C
            rated_pressure = 101.3  # in kPa
            # TODO: should Ao vary with speed?
            self.Ao = PsyCalc.CoilAoFactor_SI(rated_dry_bulb, rated_wet_bulb, rated_pressure,  # Coil Ao factor
                                              self.capacity_max / 1000, self.max_flow_rate, self.shr_rated)
            # shr_check = PsyCalc.CalculateSHR_SI(rated_dry_bulb, rated_wet_bulb, rated_pressure,
            #                                     self.capacity_max / 1000, self.max_flow_rate, self.Ao)

        else:
            self.Ao = None
        self.shr = self.shr_rated

        # Thermostat Control Parameters
        self.temp_setpoint = temp_setpoint
        self.temp_deadband = kwargs.get(self.hvac_type + 'ing deadband temperature (C)', 1)
        self.ext_ignore_thermostat = kwargs.get('ext_ignore_thermostat', False)
        self.setpoint_ramp_rate = kwargs.get('setpoint_ramp_rate', 0.1)  # max setpoint ramp rate, in C/min
        self.temp_indoor_prev = self.temp_setpoint
        self.duty_cycle_capacity = None  # Option to set capacity from duty cycle

        # Minimum On/Off Times
        on_time = kwargs.get(self.hvac_type + 'ing Minimum On Time', 0)
        off_time = kwargs.get(self.hvac_type + 'ing Minimum Off Time', 0)
        self.min_time_in_mode = {mode: dt.timedelta(minutes=on_time) for mode in self.modes}
        self.min_time_in_mode['Off'] = dt.timedelta(minutes=off_time)

        # Building envelope parameters - required for calculating ideal capacity
        self.envelope_model = envelope_model

        # Results options
        self.show_eir_shr = kwargs.get('show_eir_shr', False)

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - Load Fraction: 1 (no effect) or 0 (forces HVAC off)
        # - Setpoint: Updates heating (cooling) setpoint temperature from the dwelling schedule (in C)
        #   - Note: Setpoint must be provided every timestep or it will revert back to the dwelling schedule
        # - Deadband: Updates heating (cooling) deadband temperature (in C)
        #   - Note: Deadband will not reset back to original value
        # - Duty Cycle: Forces HVAC on for fraction of external time step (as fraction [0,1])
        #   - If 0 < Duty Cycle < 1, the equipment will cycle once every 2 external time steps
        #   - For ASHP: Can supply HP and ER duty cycles
        #   - Note: does not use clock on/off time

        # If load fraction = 0, force off
        load_fraction = ext_control_args.get('Load Fraction', 1)
        if load_fraction == 0:
            return 'Off'
        elif load_fraction != 1:
            raise EquipmentException("{} can't handle non-integer load fractions".format(self.name))

        ext_setpoint = ext_control_args.get('Setpoint')
        if ext_setpoint is not None:
            schedule = schedule.copy()
            schedule['{}ing_setpoint'.format(self.hvac_type)] = ext_setpoint

        ext_db = ext_control_args.get('Deadband')
        if ext_db is not None:
            self.temp_deadband = ext_db

        if any(['Duty Cycle' in key for key in ext_control_args]):
            return self.run_duty_cycle_control(schedule, ext_control_args)
        else:
            return self.update_internal_control(schedule)

    def run_duty_cycle_control(self, schedule, ext_control_args):
        duty_cycles = ext_control_args.get('Duty Cycle', 0)

        # Parse duty cycles
        if isinstance(duty_cycles, (int, float)):
            duty_cycles = [duty_cycles]
        assert 0 <= sum(duty_cycles) <= 1

        if self.use_ideal_capacity:
            # Set capacity to constant value based on duty cycle
            self.duty_cycle_capacity = duty_cycles[0] * self.capacity_max
            if self.duty_cycle_capacity < self.capacity_min:
                self.duty_cycle_capacity = 0

            return 'On' if self.duty_cycle_capacity > 0 else 'Off'

        else:
            # Set mode based on duty cycle from external controller
            mode_priority = self.calculate_mode_priority(*duty_cycles)
            thermostat_mode = self.run_thermostat_control(schedule)
            thermostat_mode = thermostat_mode if thermostat_mode is not None else self.mode

            if thermostat_mode in mode_priority and not self.ext_ignore_thermostat:
                return thermostat_mode
            else:
                return mode_priority[0]  # take highest priority mode (usually current mode)

    def update_internal_control(self, schedule):
        if self.use_ideal_capacity:
            self.duty_cycle_capacity = None

            # Update setpoint from schedule file
            self.update_setpoint(schedule)

            # run ideal capacity calculation here, just to determine mode
            # FUTURE: capacity update is done twice per loop, could but updated to improve speed
            capacity = self.update_capacity(schedule)

            return 'On' if capacity > 0 else 'Off'

        else:
            # Run thermostat controller
            new_mode = self.run_thermostat_control(schedule)

            # Override mode switch with minimum on/off times
            new_mode = self.update_clock_on_off_time(new_mode)

            return new_mode

    def update_setpoint(self, schedule):
        # updates setpoint with ramp rate constraints
        t_set = schedule['{}ing_setpoint'.format(self.hvac_type)]
        if self.setpoint_ramp_rate is not None:
            delta_t = self.setpoint_ramp_rate * self.time_res.total_seconds() / 60  # in C
            self.temp_setpoint = np.clip(t_set, self.temp_setpoint - delta_t, self.temp_setpoint + delta_t)
        else:
            self.temp_setpoint = t_set

    def run_thermostat_control(self, schedule):
        temp_indoor = schedule['Indoor']

        # Update setpoint from schedule file
        self.update_setpoint(schedule)

        # On and off limits depend on heating vs. cooling
        temp_turn_on = self.temp_setpoint - self.hvac_mult * self.temp_deadband / 2
        temp_turn_off = self.temp_setpoint + self.hvac_mult * self.temp_deadband / 2

        # Determine mode
        if self.hvac_mult * (temp_indoor - temp_turn_on) < 0:
            return 'On'
        if self.hvac_mult * (temp_indoor - temp_turn_off) > 0:
            return 'Off'

    def update_clock_on_off_time(self, new_mode):
        # This logic ensures that the HVAC abides by minimum on and off time operating requirements
        prev_mode = self.mode
        if new_mode is not None and self.time_in_mode < self.min_time_in_mode[prev_mode]:
            # Force mode to remain as is
            new_mode = prev_mode

        return new_mode

    def solve_ideal_capacity(self):
        # Update capacity using ideal algorithm - maintains setpoint exactly
        x_desired = self.temp_setpoint

        # Solve for desired H_LIV, accounting for heat to other zones from ducts and finished basement fraction
        # Note: all envelope inputs are updated already
        zone = self.envelope_model.indoor_zone
        zone_idxs = [zone.h_idx]
        zone_ratios = [self.duct_dse]
        if self.duct_location is not None:
            duct_idx = self.envelope_model.zones[self.duct_location].h_idx
            zone_idxs.append(duct_idx)
            zone_ratios.append(1 - self.duct_dse)
        if self.basement_heat_frac:
            fnd_idx = self.envelope_model.zones['FND'].h_idx
            if fnd_idx in zone_idxs:
                # basement has ducts and heat pct, update ratios using max
                fnd_ratio = max(1 - self.duct_dse, self.basement_heat_frac)
                zone_ratios = [1 - fnd_ratio, fnd_ratio]
            else:
                # split non-duct heat to indoor and foundation zones
                zone_idxs.append(fnd_idx)
                zone_ratios.append(self.basement_heat_frac * self.duct_dse)
                zone_ratios[0] = self.duct_dse * (1 - self.basement_heat_frac)

        h_desired = self.envelope_model.solve_for_inputs(zone.t_idx, zone_idxs, x_desired, u_ratios=zone_ratios)  # in W

        # Account for fan power and SHR - slightly different for heating/cooling
        # assumes SHR and EIR from previous time step
        if self.is_heater:
            return h_desired / (self.shr + self.eir * self.fan_power_ratio)
        else:
            return -h_desired / (self.shr - self.eir * self.fan_power_ratio)

    def update_capacity(self, schedule):
        if self.use_ideal_capacity:
            if self.duty_cycle_capacity is not None:
                return self.duty_cycle_capacity

            # Solve for capacity to meet setpoint
            h_hvac = self.solve_ideal_capacity()

            # force min <= capacity <= max, or off. If ideal capacity is out of bounds, setpoint won't be met
            if h_hvac < self.capacity_min:
                # If less than minimum capacity (or capacity is negative), force off
                return 0
            else:
                return min(h_hvac, self.capacity_max)

        else:
            # set to rated value when on, set to 0 when off
            return self.rated_capacity_list[self.speed_idx]

    def update_shr(self, schedule):
        self.coil_input_db = schedule['Indoor']
        self.coil_input_wb = schedule['Indoor Wet Bulb']

        # increase coil input temps based on fan power
        est_fan_power = self.update_fan_power()  # assumes fan power from previous time step for ideal capacity
        speed = self.speed_idx if 'On' in self.mode else 0
        if self.use_ideal_capacity and isinstance(speed, float):
            low_speed = int(speed // 1)
            speed_frac = speed % 1
            airflow_rate = speed_frac * self.flow_rates[low_speed] + (1 - speed_frac) * self.flow_rates[low_speed + 1]
            capacity = (speed_frac * self.rated_capacity_list[low_speed] +
                        (1 - speed_frac) * self.rated_capacity_list[low_speed + 1])
        else:
            airflow_rate = self.flow_rates[speed]
            capacity = self.rated_capacity_list[speed]
        self.coil_input_db += est_fan_power / 1000 / airflow_rate / rho_air / cp_air if airflow_rate > 0 else 0

        if self.is_heater:
            # no need to run psychrometric equations
            return self.shr_rated

        w_in = schedule['Indoor Humidity Ratio']
        pres_ext = schedule['ambient_pressure']
        pres_int = pres_ext
        rh_in = min(max(psychrolib.GetRelHumFromHumRatio(self.coil_input_db, w_in, pres_int * 1000), 0), 1)
        if est_fan_power > 0:
            # calculate new wet bulb temperature
            self.coil_input_wb = psychrolib.GetTWetBulbFromHumRatio(self.coil_input_db, w_in, pres_int * 1000)
            w_new = psychrolib.GetHumRatioFromRelHum(self.coil_input_db, rh_in, pres_int * 1000)
            assert abs(w_new - w_in) < 1e-5

        if rh_in == 0:
            return 1
        else:
            return PsyCalc.CalculateSHR_SI(self.coil_input_db, self.coil_input_wb, pres_int, capacity / 1000,
                                           airflow_rate, self.Ao)

    def update_eir(self, schedule):
        # set to rated values when on, set to 0 when off
        return self.rated_eir_list[self.speed_idx]

    def update_fan_power(self):
        if self.use_ideal_capacity:
            # Update fan power as proportional to capacity
            return self.capacity * self.eir * self.fan_power_ratio
        else:
            # Fan power set by speed (if only 1 speed, it is set to that speed)
            return self.rated_fan_power_list[self.speed_idx]

    def calculate_power_and_heat(self, schedule):
        # Calculate delivered heat to envelope model
        if 'On' in self.mode:
            self.shr = self.update_shr(schedule)
            self.capacity = self.update_capacity(schedule)
            self.eir = self.update_eir(schedule)
            self.fan_power = self.update_fan_power()
        else:
            # if 'Off', set capacity and fan power to 0
            self.capacity = 0
            self.fan_power = 0
            if self.show_eir_shr:
                self.shr = self.update_shr(schedule)
                self.eir = self.update_eir(schedule)

        heat_gain = self.hvac_mult * self.capacity  # Heat gain in W, positive=heat, negative=cool

        # Delivered heat: heat gain subject to SHR, both heat gain and fan power subject to DSE
        self.delivered_heat = (heat_gain * self.shr + self.fan_power) * self.duct_dse
        self.sensible_gain = {self.zone: self.delivered_heat}  # using dict version, may have gains in 2+ zones
        self.latent_gain = heat_gain * (1 - self.shr) * self.duct_dse  # 0 for heating, negative for cooling

        # update sensible and latent gains for duct losses and basement fraction
        if self.duct_location is not None:
            self.sensible_gain[self.duct_location] = (heat_gain * self.shr + self.fan_power) * (1 - self.duct_dse)
            # self.latent_gain[self.duct_location] = heat_gain * (1 - self.shr) * (1 - self.duct_dse)
        if self.basement_heat_frac:
            # reduce indoor sensible/latent heat, add to foundation
            sensible_to_fnd = self.sensible_gain[self.zone] * self.basement_heat_frac
            self.sensible_gain[self.zone] -= sensible_to_fnd
            # latent_to_fnd = self.latent_gain[self.zone] * self.basement_heat_frac
            # self.latent_gain[self.zone] -= latent_to_fnd
            if self.duct_location == 'FND':
                self.sensible_gain['FND'] += sensible_to_fnd
                # self.latent_gain['FND'] += latent_to_fnd
            else:
                self.sensible_gain['FND'] = sensible_to_fnd
                # self.latent_gain['FND'] = latent_to_fnd

        # Total power: includes fan power when on
        power_kw = abs(heat_gain) / 1000 * self.eir
        if self.is_gas:
            self.gas_therms_per_hour = Units.kWh2therms(power_kw)
            if self.is_electric:
                self.electric_kw = self.fan_power / 1000
        elif self.is_electric:
            self.electric_kw = power_kw + self.fan_power / 1000

        # reduce delivered heat (only for results) and power output based on space fraction
        # Note: sensible/latent gains to envelope are not updated
        self.delivered_heat *= self.space_fraction
        self.electric_kw *= self.space_fraction
        self.gas_therms_per_hour *= self.space_fraction

        # update previous indoor temperature
        self.temp_indoor_prev = schedule['Indoor']

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        on = 'On' in self.mode
        if to_ext:
            return {self.end_use: {
                # 'T Indoor': temp_indoor,
                'T Setpoint': self.temp_setpoint,
                'T Upper Limit': self.temp_setpoint + self.temp_deadband / 2,
                'T Lower Limit': self.temp_setpoint - self.temp_deadband / 2,
                'Power': self.electric_kw,
                'COP': 1 / self.eir if on else 0,  # Note: using biquadratic for dynamic HVAC equipment
                'Capacity': Units.W2Ton(self.capacity),  # Note: using biquadratic for dynamic HVAC equipment
                'Mode': self.hvac_mult * int(on),
            }}
        else:
            if verbosity >= 3:
                # Note: using end use, not equipment name, for all results
                results.update({
                    self.end_use + ' Delivered (kW)': abs(self.delivered_heat) / 1000,
                })
            if verbosity >= 6:
                results.update({
                    self.end_use + ' Main Power (kW)': self.capacity * self.eir * self.space_fraction / 1000,
                    self.end_use + ' Fan Power (kW)': self.fan_power * self.space_fraction / 1000,
                    self.end_use + ' Latent Gains (kW)': self.latent_gain * self.space_fraction / 1000,
                    self.end_use + ' COP (-)': 1 / self.eir if on or self.show_eir_shr else 0,
                    self.end_use + ' SHR (-)': self.shr if on or self.show_eir_shr else 0,
                    self.end_use + ' Capacity (tons)': Units.W2Ton(self.capacity),
                    self.end_use + ' Max Capacity (kW)': self.capacity_max / 1000,
                })
        return results


class Heater(HVAC):
    hvac_type = 'heat'
    end_use = 'HVAC Heating'
    name = 'Generic Heater'


class Cooler(HVAC):
    hvac_type = 'cool'
    end_use = 'HVAC Cooling'
    name = 'Generic Cooler'


class ElectricFurnace(Heater):
    name = 'Electric Furnace'


class ElectricBoiler(Heater):
    name = 'Electric Boiler'


class ElectricBaseboard(Heater):
    name = 'Electric Baseboard'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # force duct dse to 1
        self.duct_dse = 1


class GasFurnace(Heater):
    name = 'Gas Furnace'
    is_gas = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # update fan power to include parasitic power, defaults to 79W for gas furnaces
        parasitic_power = kwargs.get(self.hvac_type + 'ing parasitic power (W)', 79)  # in W
        self.rated_fan_power_list = [fan_power + parasitic_power for fan_power in self.rated_fan_power_list]
        self.fan_power_max = max(self.rated_fan_power_list)
        self.fan_power_ratio = self.fan_power_max / (self.capacity_max * self.eir_max)


class GasBoiler(Heater):
    name = 'Gas Boiler'
    is_electric = False  # no fan power
    is_gas = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Boiler specific inputs
        self.condensing = self.eir_max < 1 / 0.9  # Condensing if efficiency (AFUE) > 90%

        if self.condensing:
            self.outlet_temp = 65.56  # outlet_water_temp [C] (150 F)
            self.efficiency_coeff = np.array([1.058343061, -0.052650153, -0.0087272,
                                              -0.001742217, 0.00000333715, 0.000513723], dtype=float)
        else:
            self.outlet_temp = 82.22  # self.outlet_water_temp [C] (180F)
            self.efficiency_coeff = np.array([1.111720116, 0.078614078, -0.400425756, 0, -0.000156783, 0.009384599,
                                              0.234257955, 0.00000132927, -0.004446701, -0.0000122498], dtype=float)

    def update_eir(self, schedule):
        # update EIR based on part load ratio, input/output temperatures
        plr = self.capacity / max(self.rated_capacity_list)  # part-load-ratio
        t_in = schedule['Indoor']
        t_out = self.outlet_temp
        if self.condensing:
            eff_var = np.array([1, plr, plr ** 2, t_in, t_in ** 2, plr * t_in], dtype=float)
            eff_curve_output = np.dot(eff_var, self.efficiency_coeff)
        else:
            eff_var = np.array([1, plr, plr ** 2, t_out, t_out ** 2, plr * t_out, plr ** 3, t_out ** 3,
                                plr ** 2 * t_out, plr * t_out ** 2], dtype=float)
            eff_curve_output = np.dot(eff_var, self.efficiency_coeff)
        return self.eir_max / eff_curve_output


class DynamicHVAC(HVAC):
    """
    HVAC Equipment Class using dynamic capacity algorithm. This uses a biquadratic model to update the EIR and capacity
    at each time step. Equipment is defined by the speed_type:
     - Single: Single-speed equipment with on/off modes
     - Double: Two-speed equipment with a low and high setting
     - Variable: Variable speed equipment. Uses the ideal algorithm to determine capacity, but the dynamic algorithm for
       EIR.
     For more details, see:
     D. Cutler (2013) Improved Modeling of Residential Air Conditioners and Heat Pumps for Energy Calculations
     https://www1.eere.energy.gov/buildings/publications/pdfs/building_america/modeling_ac_heatpump.pdf
     Section 2.2.1, Equations 7-8 and 11-13
    """

    def __init__(self, control_type='Time', **kwargs):
        # Get number of speeds
        self.n_speeds = kwargs.get(self.hvac_type + 'ing number of speeds', 1)
        if self.n_speeds not in SPEED_TYPES:
            raise EquipmentException('Unknown number of speeds ({}). Should be one of: {}'.format(self.n_speeds,
                                                                                                  SPEED_TYPES))
        speed_type = SPEED_TYPES[self.n_speeds]

        # 2-speed control type and timing variables
        self.control_type = control_type  # 'Time', 'Time2', or 'Setpoint'
        self.disable_speeds = np.zeros(self.n_speeds, dtype=bool)  # if True, disable that speed
        self.time_in_speed = dt.timedelta(0)
        min_time_in_low = kwargs.get(self.hvac_type + 'ing Minimum Low Time', 5)
        min_time_in_high = kwargs.get(self.hvac_type + 'ing Minimum High Time', 5)
        self.min_time_in_speed = [dt.timedelta(minutes=min_time_in_low), dt.timedelta(minutes=min_time_in_high)]

        # Load biquadratic parameters - only keep those with the correct speed type
        biquadratic_file = kwargs.get('biquadratic_file', 'biquadratic_parameters.csv')
        biquad_params = self.initialize_parameters(biquadratic_file, val_col=None, **kwargs).to_dict()
        self.biquad_params = [{
            'eir_t': np.array([val[x + '_eir_T'] for x in 'abcdef'], dtype=float),
            'eir_ff': np.array([val[x + '_EIR_FF'] for x in 'abc'], dtype=float),
            'cap_t': np.array([val[x + '_Qcap_T'] for x in 'abcdef'], dtype=float),
            'cap_ff': np.array([val[x + '_Qcap_FF'] for x in 'abc'], dtype=float),
            'min_Twb': val.get('min_Twb', -100),
            'max_Twb': val.get('max_Twb', 100),
            'min_Tdb': val.get('min_Tdb', -100),
            'max_Tdb': val.get('max_Tdb', 100)}
            for col, val in biquad_params.items() if speed_type == col.split('_')[0]
        ]

        if not self.biquad_params:
            raise EquipmentException('Biquadratic parameters not found for {} speed {}'.format(speed_type, self.name))
        if len(self.biquad_params) != self.n_speeds:
            raise EquipmentException('Number of speeds ({}) does not match number of biquadratic equations ({})'
                                     .format(self.n_speeds, len(self.biquad_params)))

        super().__init__(**kwargs)

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - Disable Speed X: if True, disables speed X (for 2 speed control, X=1 or 2)
        #   - Note: Can be used for ideal equipment (reduces max capacity) or dynamic equipment
        #   - Note: Disable Speeds will not reset back to original value
        for idx in range(self.n_speeds):
            self.disable_speeds[idx] = bool(ext_control_args.get('Disable Speed ' + str(idx + 1)))

        return super().update_external_control(schedule, ext_control_args)

    def run_duty_cycle_control(self, schedule, ext_control_args):
        if self.use_ideal_capacity:
            # update max capacity using highest enabled speed
            max_speed = np.nonzero(~ self.disable_speeds)[0][-1]
            self.capacity_max = self.calculate_biquadratic_param(schedule, param='cap', speed_idx=max_speed)

        return super().run_duty_cycle_control(schedule, ext_control_args)

    def run_two_speed_control(self, schedule):
        mode = super().run_thermostat_control(schedule)
        mode = mode if mode is not None else self.mode
        if mode == 'Off':
            self.speed_idx = 0
            self.time_in_speed = dt.timedelta(0)
            return mode

        if self.control_type == 'Time':
            # Time-based 2-speed HVAC control: High speed turns on if temp continues to drop (for heating)
            if self.mode == 'Off':
                speed_idx = 0
            elif self.hvac_mult * (schedule['Indoor'] - self.temp_indoor_prev) < 0:
                speed_idx = 1
            else:
                speed_idx = 0
        elif self.control_type == 'Setpoint':
            # Setpoint-based 2-speed HVAC control: High speed uses setpoint difference of deadband / 2 (overlapping)
            new_schedule = schedule.copy()
            if self.is_heater:
                new_schedule['heating_setpoint'] -= self.temp_deadband / 2
            else:
                new_schedule['cooling_setpoint'] += self.temp_deadband / 2
            high_mode = super().run_thermostat_control(new_schedule)
            if high_mode == 'On':
                speed_idx = 1
            elif high_mode == 'Off':
                speed_idx = 0
            else:
                # keep previous speed
                speed_idx = self.speed_idx
        elif self.control_type == 'Time2':
            # Old time-based 2-speed HVAC control
            if self.mode == 'Off':
                speed_idx = 0
            else:
                speed_idx = 1
        else:
            raise EquipmentException('Unknown control type for {}: {}'.format(self.name, self.control_type))

        # Enforce minimum on times for speed
        prev_speed_idx = self.speed_idx
        if self.time_in_speed < self.min_time_in_speed[prev_speed_idx]:
            speed_idx = prev_speed_idx

        # enforce speed disabling from external control
        if self.disable_speeds[speed_idx]:
            # set to highest allowed speed
            speed_idx = np.nonzero(~ self.disable_speeds)[0][-1]

        if speed_idx != prev_speed_idx or self.mode == 'Off':
            self.time_in_speed = self.time_res
        else:
            self.time_in_speed += self.time_res
        self.speed_idx = speed_idx
        return mode

    def run_thermostat_control(self, schedule):
        if self.use_ideal_capacity:
            raise EquipmentException('Ideal capacity equipment should not be running a thermostat control.')

        if self.n_speeds == 1:
            # Run regular thermostat control
            return super().run_thermostat_control(schedule)
        elif self.n_speeds == 2:
            return self.run_two_speed_control(schedule)
        else:
            raise EquipmentException('Incompatible number of speeds for dynamic equipment:', self.n_speeds)

    def calculate_biquadratic_param(self, schedule, param, speed_idx):
        # runs biquadratic equation for EIR or capacity given the speed index
        # param is 'cap' or 'eir'

        # get rated value based on speed
        if param == 'cap':
            rated = self.rated_capacity_list[speed_idx]
        elif param == 'eir':
            rated = self.rated_eir_list[speed_idx]
        else:
            raise EquipmentException('Unknown biquadratic parameter:', param)

        # use coil input wet bulb for cooling, dry bulb for heating; ambient dry bulb for both
        t_in = self.coil_input_db if self.is_heater else self.coil_input_wb
        t_ext_db = schedule['ambient_dry_bulb']

        # get biquadratic parameters for current speed
        params = self.biquad_params[speed_idx]

        # clip temperatures to stay within bounds
        t_in = np.clip(t_in, params['min_Twb'], params['max_Twb'])
        t_ext_db = np.clip(t_ext_db, params['min_Tdb'], params['max_Tdb'])

        # create vectors based on temperature and flow fraction
        t_list = np.array([1, t_in, t_in ** 2, t_ext_db, t_ext_db ** 2, t_in * t_ext_db], dtype=float)
        t_ratio = np.dot(t_list, params[param + '_t'])

        # assuming rated flow rate for each speed
        flow_fraction = 1
        # flow_fraction = self.flow_rates[speed_idx] / self.max_flow_rate
        ff_list = np.array([1, flow_fraction, flow_fraction ** 2], dtype=float)
        ff_ratio = np.dot(ff_list, params[param + '_ff'])

        return rated * t_ratio * ff_ratio

    def update_capacity(self, schedule):
        if self.use_ideal_capacity:
            # update max capacity using highest enabled speed
            max_speed = np.nonzero(~ self.disable_speeds)[0][-1]
            self.capacity_max = self.calculate_biquadratic_param(schedule, param='cap', speed_idx=max_speed)
            return super().update_capacity(schedule)
        else:
            # Update capacity using biquadratic model
            return self.calculate_biquadratic_param(schedule, param='cap', speed_idx=self.speed_idx)

    def update_eir(self, schedule):
        # Update EIR using biquadratic model
        if self.use_ideal_capacity:
            # determine closest speeds based on capacity ratio
            capacities = [self.calculate_biquadratic_param(schedule, param='cap', speed_idx=speed)
                          for speed in range(self.n_speeds)]
            eirs = [self.calculate_biquadratic_param(schedule, param='eir', speed_idx=speed)
                    for speed in range(self.n_speeds)]

            # check that capacity_ratio increases with speed
            assert (np.diff(capacities) > 0).all()

            # interpolate between the 2 closest speeds to get EIR, save fractional speed index
            if self.capacity <= capacities[0]:
                self.speed_idx = 0
                return eirs[0]
            elif self.capacity >= capacities[-1]:
                self.speed_idx = self.n_speeds - 1
                return eirs[-1]
            else:
                speed_func = scipy.interpolate.interp1d(capacities, list(range(self.n_speeds)))
                self.speed_idx = float(speed_func(self.capacity))
                eir_func = scipy.interpolate.interp1d(capacities, eirs)
                return float(eir_func(self.capacity))
        else:
            speed = self.speed_idx if self.capacity > 0 else 0
            return self.calculate_biquadratic_param(schedule, param='eir', speed_idx=speed)

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)

        if not to_ext and verbosity >= 6:
            results.update({
                self.end_use + ' Speed (-)': self.speed_idx + 1 if 'On' in self.mode else 0,
                # self.end_use + ' EIR Ratio (-)': self.eir_ratio,
                # self.end_use + ' Capacity Ratio (-)': self.capacity_ratio,
            })
        return results


class AirConditioner(DynamicHVAC, Cooler):
    name = 'Air Conditioner'
    crankcase_kw = 0
    crankcase_temp = Units.F2C(55)

    def calculate_power_and_heat(self, schedule):
        super().calculate_power_and_heat(schedule)

        # add 20W of fan power when off and outdoor temp < 55F
        # no impact on sensible heat for now
        if self.crankcase_kw:
            if self.mode == 'Off' and schedule['ambient_dry_bulb'] < self.crankcase_temp:
                self.electric_kw += self.crankcase_kw * self.space_fraction


class RoomAC(AirConditioner):
    name = 'Room AC'

    def __init__(self, **kwargs):
        if kwargs.get('speed_type', 'Single') != 'Single':
            raise EquipmentException('No model for multi-speed {}'.format(self.name))
        super().__init__(**kwargs)


class ASHPCooler(AirConditioner):
    name = 'ASHP Cooler'
    crankcase_kw = 0  # TODO: HACK to remove for now 0.020


class MinisplitHVAC(DynamicHVAC):
    def __init__(self, **kwargs):
        if kwargs.get(self.hvac_type + 'ing number of speeds') == 10:
            # update the number of speeds for MSHP from 10 to 4
            for rated_list in [self.hvac_type + 'ing capacity (W)', self.hvac_type + 'ing EIR',
                               self.hvac_type + 'ing airflow rate (cfm)']:
                values = kwargs[rated_list]
                kwargs[rated_list] = [values[1], values[3], values[5], values[9]]
            kwargs[self.hvac_type + 'ing number of speeds'] = 4

        super().__init__(**kwargs)


class MinisplitAHSPCooler(MinisplitHVAC, AirConditioner):
    name = 'MSHP Cooler'
    crankcase_kw = 0.015
    crankcase_temp = Units.F2C(32)


class HeatPumpHeater(DynamicHVAC, Heater):
    name = 'Heat Pump Heater'
    folder_name = 'ASHP Heater'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Defrost Parameters
        self.defrost = False
        self.power_defrost = 0
        self.defrost_power_mult = 1

    def update_capacity(self, schedule):
        # Update capacity if defrost is required
        capacity = super().update_capacity(schedule)

        t_ext_db = schedule['ambient_dry_bulb']
        omega_ext = schedule['ambient_humidity']
        pres_ext = schedule['ambient_pressure']

        # Based on EnergyPlus Engineering Reference, Frost Adjustment Factors, for on demand, reverse cycle defrost
        self.defrost = t_ext_db < 4.4445
        if self.defrost:
            # Calculate reduced capacity
            T_coil_out = 0.82 * t_ext_db - 8.589
            # omega_ext = psychrolib.GetHumRatioFromRelHum(t_ext_db, rh_ext, pres_ext * 1000)
            omega_sat_coil = psychrolib.GetHumRatioFromTWetBulb(T_coil_out, T_coil_out, pres_ext * 1000)
            delta_omega_coil_out = max(0.000001, omega_ext - omega_sat_coil)
            defrost_time_frac = 1.0 / (1 + (0.01446 / delta_omega_coil_out))
            defrost_capacity_mult = 0.875 * (1 - defrost_time_frac)
            self.defrost_power_mult = 0.954 / 0.875  # increase in power relative to the capacity

            q_defrost = 0.01 * defrost_time_frac * (7.222 - t_ext_db) * (self.capacity / 1.01667)
            defrost_capacity = capacity * defrost_capacity_mult - q_defrost

            # Calculate additional power and EIR
            defrost_eir_temp_mod_frac = 0.1528  # in kW
            self.power_defrost = defrost_eir_temp_mod_frac * (defrost_capacity / 1.01667) * defrost_time_frac
        else:
            defrost_capacity = capacity
            self.defrost_power_mult = 0
            self.power_defrost = 0

        if self.use_ideal_capacity:
            # do not update capacity from defrost - only EIR gets updated
            return capacity
        else:
            return defrost_capacity

    def update_eir(self, schedule):
        # Update EIR from defrost. Assumes update_capacity is already run
        eir = super().update_eir(schedule)
        if self.defrost and self.capacity > 0:
            eir = (eir * self.capacity * self.defrost_power_mult + self.power_defrost) / self.capacity
        return eir


class ASHPHeater(HeatPumpHeater):
    """
    Heat pump heater with a backup electric resistance element
    """
    name = 'ASHP Heater'
    modes = ['HP On', 'HP and ER On', 'ER On', 'Off']
    folder_name = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # backup element parameters
        self.outdoor_temp_limit = kwargs['supplemental heater cut in temp (C)']
        self.er_capacity_rated = kwargs['supplemental heating capacity (W)']
        self.er_eir_rated = 1
        self.er_capacity = 0
        self.er_duty_cycle_capacity = None

        # TODO: add option to disable ER

    def run_duty_cycle_control(self, schedule, ext_control_args):
        # If duty cycles exist, combine duty cycles for HP and ER modes
        er_duty_cycle = ext_control_args.get('ER Duty Cycle', 0)
        if self.use_ideal_capacity:
            # Use ideal HVAC to determine HP mode and HP duty cycle capacity
            hp_mode = super().run_duty_cycle_control(schedule, ext_control_args)

            # determine ER mode and capacity
            assert isinstance(er_duty_cycle, (int, float)) and 0 <= er_duty_cycle <= 1
            self.er_duty_cycle_capacity = er_duty_cycle * self.er_capacity_rated

            # return mode based on HP and ER modes
            if self.er_duty_cycle_capacity > 0:
                if hp_mode == 'On':
                    return 'HP and ER On'
                else:
                    return 'ER On'
            else:
                if hp_mode == 'On':
                    return 'HP On'
                else:
                    return 'Off'

        else:
            hp_duty_cycle = ext_control_args.get('Duty Cycle', 0)
            duty_cycles = [min(hp_duty_cycle, 1 - er_duty_cycle), min(hp_duty_cycle, er_duty_cycle),
                           min(er_duty_cycle, 1 - hp_duty_cycle), 1 - max(hp_duty_cycle, er_duty_cycle)]

            # update control args and determine mode and speed
            ext_control_args['Duty Cycle'] = duty_cycles
            mode = super().run_duty_cycle_control(schedule, ext_control_args)

            # update mode counters
            if mode == 'HP and ER On':
                # update HP only and ER only counters
                self.ext_mode_counters['HP On'] += self.time_res
                self.ext_mode_counters['ER On'] += self.time_res
            elif 'On' in mode:
                # update HP+ER counter
                self.ext_mode_counters['HP and ER On'] = max(self.ext_mode_counters[mode] + self.time_res,
                                                             self.ext_mode_counters['HP On'],
                                                             self.ext_mode_counters['ER On'])
            return mode

    def update_internal_control(self, schedule):
        # turn off duty cycle capacity
        self.duty_cycle_capacity = None
        self.er_duty_cycle_capacity = None

        # Update setpoint from schedule file
        self.update_setpoint(schedule)

        if self.use_ideal_capacity:
            # update max capacity using highest enabled speed
            max_speed = np.nonzero(~ self.disable_speeds)[0][-1]
            self.capacity_max = self.calculate_biquadratic_param(schedule, param='cap', speed_idx=max_speed)

            # run ideal capacity calculation here, just to determine mode
            tot_capacity = self.solve_ideal_capacity()

            if tot_capacity <= 0:
                mode = 'Off'
            elif tot_capacity <= self.capacity_max:
                mode = 'HP On'
            else:
                mode = 'HP and ER On'

        else:
            # get HP and ER modes separately
            hp_mode = super().update_internal_control(schedule)
            hp_on = hp_mode == 'On' if hp_mode is not None else 'HP' in self.mode
            er_mode = self.run_er_thermostat_control(schedule)
            er_on = er_mode == 'On' if er_mode is not None else 'ER' in self.mode

            # combine HP and ER modes
            if er_on:
                if hp_on:
                    mode = 'HP and ER On'
                else:
                    mode = 'ER On'
            else:
                if hp_on:
                    mode = 'HP On'
                else:
                    mode = 'Off'

        # Force HP off if outdoor temp is very cold
        t_ext_db = schedule['ambient_dry_bulb']
        if t_ext_db < self.outdoor_temp_limit and 'HP' in mode:
            mode = 'ER On'

        return mode

    def run_er_thermostat_control(self, schedule):
        # run thermostat control for ER element - lower the setpoint by the deadband
        # TODO: add option to keep setpoint as is, e.g. when using external control
        er_setpoint = self.temp_setpoint - self.temp_deadband
        temp_indoor = schedule['Indoor']

        # On and off limits depend on heating vs. cooling
        temp_turn_on = er_setpoint - self.hvac_mult * self.temp_deadband / 2
        temp_turn_off = er_setpoint + self.hvac_mult * self.temp_deadband / 2

        # Determine mode
        if self.hvac_mult * (temp_indoor - temp_turn_on) < 0:
            return 'On'
        if self.hvac_mult * (temp_indoor - temp_turn_off) > 0:
            return 'Off'

    def update_capacity(self, schedule):
        if 'HP' in self.mode:
            hp_capacity = super().update_capacity(schedule)
        else:
            hp_capacity = 0

        if 'ER' in self.mode:
            if self.er_duty_cycle_capacity is not None:
                er_capacity = self.er_duty_cycle_capacity
            elif self.use_ideal_capacity:
                # get total ideal capacity
                tot_capacity = self.solve_ideal_capacity()
                er_capacity = tot_capacity - hp_capacity
                er_capacity = np.clip(er_capacity, 0, self.er_capacity_rated)
            else:
                er_capacity = self.er_capacity_rated
        else:
            er_capacity = 0

        # save ER capacity
        self.er_capacity = er_capacity
        return hp_capacity + er_capacity

    def update_eir(self, schedule):
        if self.mode == 'HP and ER On':
            # EIR is a weighted average of HP and ER EIRs
            hp_eir = super().update_eir(schedule)
            hp_capacity = self.capacity - self.er_capacity
            return (hp_capacity * hp_eir + self.er_capacity * self.er_eir_rated) / self.capacity
        elif self.mode in ['HP On', 'Off']:
            return super().update_eir(schedule)
        elif self.mode == 'ER On':
            return self.er_eir_rated
        else:
            raise EquipmentException('Unknown mode for {}: {}'.format(self.name, self.mode))

    def calculate_power_and_heat(self, schedule):
        # Update ER capacity if off
        if 'On' not in self.mode:
            self.er_capacity = 0

        super().calculate_power_and_heat(schedule)

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            results.update({
                'HP Mode': int('HP' in self.mode),
                'ER Mode': int('ER' in self.mode),
                'ER Capacity': Units.W2Ton(self.er_capacity_rated),
            })
        if not to_ext and verbosity >= 6:
            tot_power = self.capacity * self.eir * self.space_fraction / 1000
            er_power = self.er_capacity * self.er_eir_rated * self.space_fraction / 1000
            results.update({
                self.end_use + ' Main Power (kW)': tot_power - er_power,
                self.end_use + ' ER Power (kW)': er_power,
            })

        return results


class MinisplitAHSPHeater(MinisplitHVAC, ASHPHeater):
    name = 'MSHP Heater'
