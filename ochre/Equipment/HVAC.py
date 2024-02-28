import datetime as dt
import numpy as np
import psychrolib

from ochre.utils import OCHREException, convert, load_csv
from ochre.utils.units import kwh_to_therms
import ochre.utils.equipment as utils_equipment
from ochre.Equipment import Equipment

SPEED_TYPES = {
    1: 'Single',
    2: 'Double',
    4: 'Variable',
    # 10: 'Mini-split Variable',  # Note: MSHP model uses 4 speeds, not 10
}

cp_air = 1.005  # kJ/kg-K
rho_air = 1.2041  # kg/m^3


class HVAC(Equipment):
    """
    Base HVAC Equipment Class. Options for static and ideal capacity. `end_use` must be specified in child classes.

    The ideal capacity algorithm uses the envelope model to determine the exact HVAC
     capacity to maintain the setpoint temperature. It does not account for heat gains from other equipment in the
     same time step.
    """
    name = 'Generic HVAC'
    n_speeds = 1

    def __init__(self, envelope_model=None, use_ideal_capacity=None, **kwargs):
        # HVAC type (Heating or Cooling)
        if self.end_use == 'HVAC Heating':
            self.is_heater = True
            self.hvac_mult = 1
        elif self.end_use == 'HVAC Cooling':
            self.is_heater = False
            self.hvac_mult = -1
        else:
            raise OCHREException(f'HVAC type for {self.name} Equipment must be "Heating" or "Cooling".')

        # Building envelope parameters - required for calculating ideal capacity
        # FUTURE: For now, require envelope model. In future, could use ext_model to provide all schedule values
        assert self.zone_name == 'Indoor' and envelope_model is not None
        self.envelope_model = envelope_model

        super().__init__(envelope_model=envelope_model, **kwargs)

        # Capacity parameters
        self.speed_idx = 1  # speed index, 0=Off, 1=lowest speed, max=n_speeds
        if isinstance(kwargs['Capacity (W)'], list):
            self.capacity_list = [0] + kwargs['Capacity (W)']  # rated capacities by speed, in W
        else:
            self.capacity_list = [0, kwargs['Capacity (W)']]
        assert (np.diff(self.capacity_list) > 0).all()
        self.capacity = self.capacity_list[self.speed_idx]
        self.capacity_ideal = self.capacity  # capacity to maintain setpoint, for ideal equipment, in W
        self.capacity_max = self.capacity_list[-1]  # varies for dynamic equipment, in W
        self.capacity_min = kwargs.get('Minimum Capacity (W)', 0)  # for ideal equipment, in W
        self.space_fraction = kwargs.get('Conditioned Space Fraction (-)', 1.0)
        self.delivered_heat = 0  # in W, total sensible heat gain, excluding duct losses

        # Efficiency and loss parameters
        if isinstance(kwargs['EIR (-)'], list):
            self.eir_list = kwargs['EIR (-)']  # Energy Input Ratios by speed, unitless
        else:
            self.eir_list = [kwargs['EIR (-)']]
        self.eir_list = [self.eir_list[0]] + self.eir_list  # add lowest speed EIR as 'off' EIR
        self.eir = self.eir_list[self.speed_idx]
        self.eir_max = self.eir_list[-1]  # eir at max capacity (not the largest EIR for multispeed equipment)

        # SHR (sensible heat ratio), cooling only
        shr = kwargs.get('SHR (-)')
        if shr is None:
            shr = 1
        if isinstance(shr, list):
            shr_list = [shr[0]] + shr  # add lowest speed SHR as 'off' EIR
        else:
            shr_list = [0, shr]
        self.shr = shr_list[self.speed_idx]

        # Air flow parameters
        if isinstance(self, DynamicHVAC):
            # calculate flow rates based on capacity and supply air temperature
            if self.is_heater:
                # temp_setpoint = min(max(kwargs['Min Setpoint (C)'], 15), 24)
                temp_setpoint = 20  # in degC, from ASHRAE Standard 152, 6.3.1 Indoor Air Conditions
                delta_t = convert(105, 'degF', 'degC') - temp_setpoint
            else:
                # interpolate to get cooling supply temp based on rated SHR, must be between 54-58 F
                # temp_setpoint = min(max(kwargs['Max Setpoint (C)'], 18), 27)
                temp_setpoint = 25.5  # from ASHRAE Standard 152, 6.3.1 Indoor Air Conditions
                cool_supply_temp = np.clip(54 + (58 - 54) * (shr_list[-1] - 0.8) / (0.85 - 0.80), 54, 58)
                delta_t = temp_setpoint - convert(cool_supply_temp, 'degF', 'degC')
            self.flow_rate_list = [cap / 1000 / rho_air / cp_air / delta_t for cap in self.capacity_list]  # in m^3/s
        else:
            # Use nominal flow rates, values taken from ResStock (see hvac.rb line 2623)
            cfm_per_ton = 350 if self.is_heater else 312
            ratio = convert(cfm_per_ton, 'cubic_feet/min/refrigeration_ton', 'm^3/s/W')
            self.flow_rate_list = [ratio * capacity for capacity in self.capacity_list]  # in m^3/s
        rated_flow_rate = max(self.flow_rate_list)

        # Fan power parameters
        rated_fan_power = kwargs['Rated Auxiliary Power (W)']
        self.fan_power_per_flow_rate = rated_fan_power / rated_flow_rate
        self.fan_power_list = [self.fan_power_per_flow_rate * rate for rate in self.flow_rate_list]  # in W
        self.fan_power = 0  # in W
        self.fan_power_max = max(self.fan_power_list)
        self.fan_power_ratio = self.fan_power_max / (self.capacity_max * self.eir_max)  # For ideal capacity equipment
        initial_setpoint = kwargs['initial_schedule'][f'{self.end_use} Setpoint (C)']
        self.coil_input_db = initial_setpoint  # Dry bulb temperature after increase from fan power
        self.coil_input_wb = initial_setpoint  # Wet bulb temperature after increase from fan power

        # check length of rated lists
        for speed_list in [self.capacity_list, self.eir_list, self.fan_power_list]:
            if len(speed_list) - 1 != self.n_speeds:
                raise OCHREException(f'Number of speeds ({self.n_speeds}) does not match length of list'
                                         f' ({len(speed_list) - 1})')

        # Duct location and distribution system efficiency (DSE)
        ducts = kwargs.get('Ducts', {'DSE (-)': 1})
        self.duct_dse = ducts.get('DSE (-)')  # Duct distribution system efficiency
        self.duct_zone = self.envelope_model.zones.get(ducts.get('Zone'))
        if self.duct_dse is None:
            # Calculate DSE using ASHRAE 152
            self.duct_dse = utils_equipment.calculate_duct_dse(self, ducts, **kwargs)
        if self.duct_dse < 1 and self.duct_zone == self.zone:
            self.warn(f'Ignoring duct DSE because ducts are in {self.zone.name} zone.')
            self.duct_dse = 1
            self.duct_zone = None

        # basement zone heat fraction
        basement_zone = self.envelope_model.zones.get('Foundation')
        if basement_zone:
            default_basement_frac = 0.2 if basement_zone.zone_type == 'Finished Basement' and self.is_heater else 0
            self.basement_heat_frac = kwargs.get('Basement Airflow Ratio (-)', default_basement_frac)
        else:
            self.basement_heat_frac = 0

        # Determine heat fractions per zone (Indoor zone, duct zone, and basement zone)
        self.zone_fractions = {self.zone: self.duct_dse * (1 - self.basement_heat_frac)}
        if self.duct_dse < 1 and self.duct_zone is not None:
            # if duct_zone is None, DSE losses don't get added to another zone
            self.zone_fractions[self.duct_zone] = 1 - self.duct_dse
        if self.basement_heat_frac > 0:
            if basement_zone == self.duct_zone:
                self.zone_fractions[basement_zone] += self.duct_dse * self.basement_heat_frac
            else:
                self.zone_fractions[basement_zone] = self.duct_dse * self.basement_heat_frac

        # Coil Ao factor, cooling only
        if self.is_heater:
            self.Ao_list = None
        elif isinstance(self, DynamicHVAC):
            rated_dry_bulb = convert(80, 'degF', 'degC')  # in degrees C
            rated_wet_bulb = convert(67, 'degF', 'degC')  # in degrees C
            rated_pressure = 101.3  # in kPa
            rated_w = psychrolib.GetHumRatioFromTWetBulb(rated_dry_bulb, rated_wet_bulb, rated_pressure * 1000)
            ao_data = zip(self.capacity_list[1:], self.flow_rate_list[1:], shr_list[1:])
            ao_list = [utils_equipment.coil_ao_factor(rated_dry_bulb, rated_w, rated_pressure, 
                                                           capacity / 1000, flow_rate, shr)
                                                           for capacity, flow_rate, shr in ao_data]
            self.Ao_list = [ao_list[0]] + ao_list
        else:
            # for ideal coolers
            self.Ao_list = [10] * (self.n_speeds + 1)

        # Thermostat Control Parameters
        self.temp_setpoint = initial_setpoint
        self.temp_deadband = kwargs.get('Deadband Temperature (C)', 1)
        self.ext_ignore_thermostat = kwargs.get('ext_ignore_thermostat', False)
        self.setpoint_ramp_rate = kwargs.get('setpoint_ramp_rate')  # max setpoint ramp rate, in C/min
        self.temp_indoor_prev = self.temp_setpoint
        self.ext_capacity = None  # Option to set capacity directly, ideal capacity only
        self.ext_capacity_frac = 1  # Option to limit max capacity, ideal capacity only

        # Results options
        self.show_eir_shr = kwargs.get('show_eir_shr', False)

        # if main simulator, add envelope as sub simulator
        if self.main_simulator:
            self.sub_simulators.append(self.envelope_model)

        # Use ideal or static/dynamic capacity depending on time resolution and number of speeds
        # 4 speeds are used for variable speed equipment, which must use ideal capacity
        if use_ideal_capacity is None:
            use_ideal_capacity = self.time_res >= dt.timedelta(minutes=5) or self.n_speeds >= 4
        self.use_ideal_capacity = use_ideal_capacity

    def initialize_schedule(self, schedule=None, **kwargs):
        # Compile all HVAC required inputs
        required_inputs = [f'{self.end_use} Setpoint (C)']
        if isinstance(self, DynamicHVAC):
            required_inputs.append('Ambient Dry Bulb (C)')
        if isinstance(self, HeatPumpHeater) or (not self.is_heater and self.zone.humidity is None):
            # Required for heat pump heater and dynamic AC if humidity model not included
            required_inputs.append('Ambient Humidity Ratio (-)')
            required_inputs.append('Ambient Pressure (kPa)')

        return super().initialize_schedule(schedule, required_inputs=required_inputs, **kwargs)

    def update_external_control(self, control_signal):
        # Options for external control signals:
        # - Load Fraction: 1 (no effect) or 0 (forces HVAC off)
        # - Setpoint: Updates heating (cooling) setpoint temperature from the dwelling schedule (in C)
        #   - Note: Setpoint must be provided every timestep or it will revert back to the dwelling schedule
        # - Deadband: Updates heating (cooling) deadband temperature (in C)
        #   - Note: Deadband will only be reset if it is in the schedule
        # - Capacity: Sets HVAC capacity directly, ideal capacity only
        #   - Resets every time step
        # - Max Capacity Fraction: Limits HVAC max capacity, ideal capacity only
        #   - For now, does not get reset
        # - Duty Cycle: Forces HVAC on for fraction of external time step (as fraction [0,1]), non-ideal capacity only
        #   - If 0 < Duty Cycle < 1, the equipment will cycle once every 2 external time steps
        #   - For ASHP: Can supply HP and ER duty cycles
        #   - Note: does not use clock on/off time

        ext_setpoint = control_signal.get('Setpoint')
        if ext_setpoint is not None:
            self.current_schedule[f'{self.end_use} Setpoint (C)'] = ext_setpoint
        self.update_setpoint()

        ext_db = control_signal.get('Deadband')
        if ext_db is not None:
            if f'{self.end_use} Deadband (C)' in self.current_schedule:
                self.current_schedule[f'{self.end_use} Deadband (C)'] = ext_db
            else:
                self.temp_deadband = ext_db

        # If load fraction = 0, force off
        load_fraction = control_signal.get('Load Fraction', 1)
        if load_fraction == 0:
            self.speed_idx = 0
            return 'Off'
        elif load_fraction != 1:
            raise OCHREException(f"{self.name} can't handle non-integer load fractions")

        capacity_frac = control_signal.get('Max Capacity Fraction')
        if capacity_frac is not None:
            if not self.use_ideal_capacity:
                raise IOError(
                    f"Cannot set {self.name} Max Capacity Fraction. "
                    'Set `use_ideal_capacity` to True or control "Duty Cycle".'
                )
            self.ext_capacity_frac = capacity_frac

        capacity = control_signal.get('Capacity')
        if capacity is not None:
            if not self.use_ideal_capacity:
                raise IOError(
                    f"Cannot set {self.name} Capacity. "
                    'Set `use_ideal_capacity` to True or control "Duty Cycle".'
                )

            self.ext_capacity = capacity
            # TODO: remove once schedule is incorporated, test with ASHP modes
            return 'On' if self.ext_capacity > 0 else 'Off'

        if any(['Duty Cycle' in key for key in control_signal]):
            if self.use_ideal_capacity:
                raise IOError(
                    f"Cannot set {self.name} Duty Cycle. "
                    'Set `use_ideal_capacity` to False or use "Capacity" control.'
                )
            return self.run_duty_cycle_control(control_signal)

        # if mode isn't set yet, run internal control method
        return self.update_internal_control()

    def run_duty_cycle_control(self, control_signal):
        duty_cycles = control_signal.get('Duty Cycle', 0)
        if duty_cycles == 0:
            self.speed_idx = 0
            return 'Off'
        if duty_cycles == 1:
            self.speed_idx = self.n_speeds  # max speed
            return 'On'

        # Parse duty cycles
        if isinstance(duty_cycles, (int, float)):
            duty_cycles = [duty_cycles]
        assert 0 <= sum(duty_cycles) <= 1

        # Set mode based on duty cycle from external controller
        mode_priority = self.calculate_mode_priority(*duty_cycles)
        thermostat_mode = self.run_thermostat_control()
        thermostat_mode = thermostat_mode if thermostat_mode is not None else self.mode

        # take thermostat mode if it exists in priority stack, or take highest priority mode (usually current mode)
        mode = thermostat_mode if (thermostat_mode in mode_priority and
                                    not self.ext_ignore_thermostat) else mode_priority[0]

        # by default, turn on to max speed
        self.speed_idx = self.n_speeds if 'On' in mode else 0

        return mode

    def update_internal_control(self):
        # Update setpoint from schedule
        self.update_setpoint()

        if self.use_ideal_capacity:
            # TODO: this won't work until it's added to the schedule
            self.ext_capacity = None

            # run ideal capacity calculation here, just to determine mode and speed
            # FUTURE: capacity update is done twice per loop, could but updated to improve speed
            self.capacity = self.update_capacity()

            return 'On' if self.capacity > 0 else 'Off'

        else:
            # Run thermostat controller and set speed
            return self.run_thermostat_control()

    def update_setpoint(self):
        t_set = self.current_schedule[f'{self.end_use} Setpoint (C)']
        if f'{self.end_use} Deadband (C)' in self.current_schedule:
            self.temp_deadband = self.current_schedule[f'{self.end_use} Deadband (C)']

        # updates setpoint with ramp rate constraints
        # TODO: create temp_setpoint_old and update in update_results. 
        # Could get run multiple times per time step in update_model
        if self.setpoint_ramp_rate is not None:
            delta_t = self.setpoint_ramp_rate * self.time_res.total_seconds() / 60  # in C
            self.temp_setpoint = min(max(t_set, self.temp_setpoint - delta_t), self.temp_setpoint + delta_t)
        else:
            self.temp_setpoint = t_set

        # set envelope comfort limits
        if self.envelope_model is not None:
            if self.is_heater:
                self.envelope_model.heating_setpoint = self.temp_setpoint
                self.envelope_model.heating_deadband = self.temp_deadband
            else:
                self.envelope_model.cooling_setpoint = self.temp_setpoint
                self.envelope_model.cooling_deadband = self.temp_deadband

    def run_thermostat_control(self, setpoint=None):
        if setpoint is None:
            setpoint = self.temp_setpoint

        # On and off limits depend on heating vs. cooling
        temp_turn_on = setpoint - self.hvac_mult * self.temp_deadband / 2
        temp_turn_off = setpoint + self.hvac_mult * self.temp_deadband / 2

        # Determine mode
        if self.hvac_mult * (self.zone.temperature - temp_turn_on) < 0:
            # by default, set to max speed
            self.speed_idx = self.n_speeds
            return 'On'
        elif self.hvac_mult * (self.zone.temperature - temp_turn_off) > 0:
            self.speed_idx = 0
            return 'Off'
        else:
            return None

    def solve_ideal_capacity(self):
        # Update capacity using ideal algorithm - maintains setpoint exactly
        x_desired = self.temp_setpoint

        # Solve for desired H_LIV, accounting for heat to other zones from ducts and finished basement fraction
        # Note: all envelope inputs are updated already
        # TODO: use solver set up function, move this to initialization, see RCModel.setup_multi_input_solver
        zone_idxs = [zone.h_idx for zone in self.zone_fractions]
        zone_ratios = list(self.zone_fractions.values())

        # Note: h_desired should be equal to self.delivered_heat
        h_desired = self.envelope_model.solve_for_inputs(self.zone.t_idx, zone_idxs, x_desired, zone_ratios)  # in W

        # Account for fan power and SHR - slightly different for heating/cooling
        # assumes SHR and EIR from previous time step
        if self.is_heater:
            return h_desired / (self.shr + self.eir * self.fan_power_ratio)
        else:
            return -h_desired / (self.shr - self.eir * self.fan_power_ratio)

    def update_capacity(self):
        if self.use_ideal_capacity:
            # Solve for capacity to meet setpoint
            self.capacity_ideal = self.solve_ideal_capacity()
            capacity = self.capacity_ideal
            
            # Update from direct capacity controls
            if self.ext_capacity is not None:
                capacity = self.ext_capacity

            # Enforce min and max capacity limits
            if capacity < self.capacity_min:
                # If capacity < capacity_min (or capacity is negative), force off
                capacity = 0
            elif capacity > self.capacity_max * self.ext_capacity_frac:
                # Clip at maximum capacity, considering max capacity fraction
                # Note: if ideal capacity is out of bounds, setpoint won't be met
                capacity = self.capacity_max * self.ext_capacity_frac

            # set speed (only used for non-dynamic equipment) and return capacity
            self.speed_idx = capacity / self.capacity_max
            return capacity

        else:
            # set to rated value when on, set to 0 when off. speed_idx should already be set
            return self.capacity_list[self.speed_idx]

    def update_shr(self):
        self.coil_input_db = self.zone.temperature

        if self.is_heater:
            return 1

        if self.zone.humidity is not None:
            w_in = self.zone.humidity.w
            pres_int = self.zone.humidity.pressure
        else:
            # use ambient conditions if no humidity model defined
            w_in = self.current_schedule['Ambient Humidity Ratio (-)']
            pres_int = self.current_schedule['Ambient Pressure (kPa)'] * 1000  # in Pa
        if w_in == 0:
            return 1

        # Update coil temperatures (used for SHR and biquadratic calculations)
        if self.fan_power_max:
            # calculate increased dry and wet bulb temperatures due to fan power
            self.coil_input_db += self.fan_power_per_flow_rate / 1000 / rho_air / cp_air
            self.coil_input_wb = psychrolib.GetTWetBulbFromHumRatio(self.coil_input_db, w_in, pres_int)
        elif self.zone.humidity is not None:
            # Don't recalculate wet bulb if already done in humidity model
            self.coil_input_wb = self.zone.humidity.wet_bulb
        else:
            self.coil_input_wb = psychrolib.GetTWetBulbFromHumRatio(self.coil_input_db, w_in, pres_int)

        # Calculate SHR based on speed
        speed_low = int(self.speed_idx // 1)  # 0 is the lowest speed
        shr_low = utils_equipment.calculate_shr(self.coil_input_db, w_in, pres_int / 1000,
                                             self.capacity_list[speed_low] / 1000, 
                                             self.flow_rate_list[speed_low],
                                             self.Ao_list[speed_low])

        frac_high = self.speed_idx % 1
        if frac_high:
            # take a weighted average of 2 closest speeds based on speed_idx. Note speed_idx=0 means off (capacity=0)
            speed_high = speed_low + 1
            shr_high = utils_equipment.calculate_shr(self.coil_input_db, w_in, pres_int / 1000,
                                                    self.capacity_list[speed_high] / 1000, 
                                                    self.flow_rate_list[speed_high],
                                                    self.Ao_list[speed_high])
            shr = ((1 - frac_high) * shr_low + frac_high * shr_high)
        else:
            shr = shr_low

        return shr

    def update_fan_power(self, capacity):
        if self.use_ideal_capacity:
            # Update fan power as proportional to power (power = capacity * eir)
            return capacity * self.eir * self.fan_power_ratio
        else:
            # Fan power set by speed (if only 1 speed, it is set to that speed)
            return self.fan_power_list[self.speed_idx]

    def update_eir(self):
        # set to rated value when on, set to 0 when off
        return self.eir_list[self.n_speeds]

    def calculate_power_and_heat(self):
        # Calculate delivered heat to envelope model
        if 'On' in self.mode:
            self.shr = self.update_shr()
            self.capacity = self.update_capacity()
            self.fan_power = self.update_fan_power(self.capacity)
            self.eir = self.update_eir()
        else:
            # if 'Off', set capacity and fan power to 0
            self.capacity = 0
            self.fan_power = 0
            self.shr = self.update_shr()
            self.eir = self.update_eir()

        heat_gain = self.hvac_mult * self.capacity  # Heat gain in W, positive=heat, negative=cool

        # Calculate total sensible and latent heat
        self.delivered_heat = heat_gain * self.shr + self.fan_power  # SHR=1 for fan
        self.sensible_gain = self.delivered_heat
        self.latent_gain = heat_gain * (1 - self.shr)  # no latent gains from fan

        # Total power: includes fan power when on
        power_kw = abs(heat_gain) / 1000 * self.eir
        if self.is_gas:
            self.gas_therms_per_hour = power_kw * kwh_to_therms
            if self.is_electric:
                self.electric_kw = self.fan_power / 1000
        elif self.is_electric:
            self.electric_kw = power_kw + self.fan_power / 1000

        # reduce delivered heat (only for results) and power output based on space fraction
        # Note: sensible/latent gains to envelope are not updated
        self.delivered_heat *= self.space_fraction
        self.electric_kw *= self.space_fraction
        self.fan_power *= self.space_fraction
        self.gas_therms_per_hour *= self.space_fraction

        # update previous indoor temperature
        # TODO: move to update_results?
        self.temp_indoor_prev = self.zone.temperature

    def add_gains_to_zone(self):
        for zone, fraction in self.zone_fractions.items():
            zone.hvac_sens_gain += self.sensible_gain * fraction
            zone.hvac_latent_gain += self.latent_gain * fraction

    def generate_results(self):
        results = super().generate_results()
        on = 'On' in self.mode

        # Note: using end use, not equipment name, for all results
        if self.verbosity >= 3:
            results[f'{self.end_use} Delivered (W)'] = abs(self.delivered_heat) * self.duct_dse
        if self.verbosity >= 6:
            # recalculate COP to account for any changes in power (e.g. crankcase, pan heater)
            main_power = self.electric_kw + self.gas_therms_per_hour / kwh_to_therms - self.fan_power / 1000
            if on and main_power != 0:
                cop = self.capacity * self.space_fraction / main_power / 1000
            elif self.show_eir_shr:
                cop = 1 / self.eir
            else:
                cop = 0
            results[f'{self.end_use} Duct Losses (W)'] = abs(self.delivered_heat) * (1 - self.duct_dse)
            results[f'{self.end_use} Setpoint (C)'] = self.temp_setpoint
            results[f'{self.end_use} Main Power (kW)'] = main_power
            results[f'{self.end_use} Fan Power (kW)'] = self.fan_power / 1000
            results[f'{self.end_use} Latent Gains (W)'] = self.latent_gain * self.space_fraction
            results[f'{self.end_use} COP (-)'] = cop
            results[f'{self.end_use} SHR (-)'] = self.shr if on or self.show_eir_shr else 0
            results[f'{self.end_use} Speed (-)'] = self.speed_idx
            results[f'{self.end_use} Capacity (W)'] = self.capacity
            results[f'{self.end_use} Max Capacity (W)'] = self.capacity_max

        if self.save_ebm_results:
            results.update(self.make_equivalent_battery_model())

        return results

    def make_equivalent_battery_model(self):
        # returns a dictionary of equivalent battery model parameters
        # Note: separate models for heating and cooling - both use individual deadbands, not the setpoint difference
        # Note: Energy state increases with temperature for heating; decreases for cooling
        # TODO: Baseline power calculation should assume no change in indoor temperature setpoint
        # TODO: update capacitance using 1R1C model
        ref_temp = 10 if self.is_heater else 30  # temperature at Energy=0, in C
        total_capacitance = convert(self.zone.capacitance, 'kJ', 'kWh')  # in kWh/K
        max_temp = self.temp_setpoint + self.hvac_mult * self.temp_deadband / 2  # "turn off" temperature
        min_temp = self.temp_setpoint - self.hvac_mult * self.temp_deadband / 2  # "turn on" temperature
        return {
            f'{self.end_use} EBM Energy (kWh)': total_capacitance * (self.zone.temperature - ref_temp) * self.hvac_mult,
            f'{self.end_use} EBM Min Energy (kWh)': total_capacitance * (min_temp - ref_temp) * self.hvac_mult,
            f'{self.end_use} EBM Max Energy (kWh)': total_capacitance * (max_temp - ref_temp) * self.hvac_mult,
            f'{self.end_use} EBM Max Power (kW)': self.capacity_max * self.eir / 1000,
            f'{self.end_use} EBM Efficiency (-)': 1 / self.eir,
            f'{self.end_use} EBM Baseline Power (kW)': self.capacity_ideal if self.use_ideal_capacity else None,
        }


class Heater(HVAC):
    end_use = 'HVAC Heating'
    name = 'Generic Heater'
    optional_inputs = ['HVAC Heating Deadband (C)']


class Cooler(HVAC):
    end_use = 'HVAC Cooling'
    name = 'Generic Cooler'
    optional_inputs = ['HVAC Cooling Deadband (C)']


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


class GasBoiler(Heater):
    name = 'Gas Boiler'
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

    def update_eir(self):
        # update EIR based on part load ratio, input/output temperatures
        plr = self.speed_idx  # part-load-ratio
        t_in = self.zone.temperature
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
        self.n_speeds = kwargs.get('Number of Speeds (-)', 1)

        # 2-speed control type and timing variables
        self.control_type = control_type  # 'Time', 'Time2', or 'Setpoint'
        self.disable_speeds = np.zeros(self.n_speeds, dtype=bool)  # if True, disable that speed
        self.time_in_speed = dt.timedelta(0)
        min_time_in_low = kwargs.get('Minimum Low Time (minutes)', 5)
        min_time_in_high = kwargs.get('Minimum High Time (minutes)', 5)
        self.min_time_in_speed = [dt.timedelta(minutes=min_time_in_low), dt.timedelta(minutes=min_time_in_high)]

        # Load biquadratic parameters from file - only keep those with the correct speed type
        if not kwargs.get('Disable HVAC Biquadratics', False):
            self.biquad_params = self.initialize_biquad_params(**kwargs)
        else:
            self.biquad_params = None

        # Load multispeed parameters from file
        if self.n_speeds > 1:
            rated_efficiency = kwargs.get('Rated Efficiency', '(Unknown Efficiency)')
            multispeed_file = kwargs.get('multispeed_file', 'HVAC Multispeed Parameters.csv')
            df_speed = load_csv(multispeed_file)
            speed_params = df_speed.loc[(df_speed['HVAC Name'] == self.name) & 
                                        (df_speed['HVAC Efficiency'] == rated_efficiency) &
                                        (df_speed['Number of Speeds'] == self.n_speeds)]
            if not len(speed_params):
                raise OCHREException(f'Cannot find multispeed parameters for {rated_efficiency} {self.name}')
            assert len(speed_params) == 1
            speed_params = speed_params.iloc[0].to_dict()
            
            # update multispeed arguments (capacity ratios, air flow ratio, EIR, SHR)
            kwargs['Capacity (W)'] = [kwargs['Capacity (W)'] * speed_params[f'Capacity Ratio {i + 1}']
                                    for i in range(self.n_speeds)]
            kwargs['EIR (-)'] = [1 / speed_params[f'COP {i + 1}'] for i in range(self.n_speeds)]
            kwargs['SHR (-)'] = [speed_params[f'SHR {i + 1}'] for i in range(self.n_speeds)]
            kwargs['SHR (-)'] = [shr if not np.isnan(shr) else 1 for shr in kwargs['SHR (-)']]

        super().__init__(**kwargs)

    def initialize_biquad_params(self, **kwargs):
        if self.n_speeds not in SPEED_TYPES:
            raise OCHREException('Unknown number of speeds ({}). Should be one of: {}'.format(self.n_speeds,
                                                                                                SPEED_TYPES))
        speed_type = SPEED_TYPES[self.n_speeds]

        biquadratic_file = kwargs.get('biquadratic_file', f'Biquadratic {self.name}.csv')
        biquad_params = self.initialize_parameters(biquadratic_file, value_col=None, **kwargs)
        biquad_params = biquad_params.loc[:, [col for col in biquad_params if speed_type == col.split('_')[0]]]
        if len(biquad_params.columns) != self.n_speeds:
            raise OCHREException(f'Number of speeds ({self.n_speeds}) does not match number of biquadratic '
                                        f'equations ({len(biquad_params.columns)})')
        biquad_params = {idx + 1: {
            'eir_t': np.array([val[f'{x}_eir_t'] for x in 'abcdef'], dtype=float),
            'eir_ff': np.array([val[f'{x}_eir_ff'] for x in 'abc'], dtype=float),
            'eir_plr': np.array([val[f'{x}_eir_plr'] for x in 'abc'], dtype=float),
            'cap_t': np.array([val[f'{x}_cap_t'] for x in 'abcdef'], dtype=float),
            'cap_ff': np.array([val[f'{x}_cap_ff'] for x in 'abc'], dtype=float),
            'cap_plr': np.array([1, 0, 0], dtype=float),
            'min_Twb': val.get('min_Twb', -100),
            'max_Twb': val.get('max_Twb', 100),
            'min_Tdb': val.get('min_Tdb', -100),
            'max_Tdb': val.get('max_Tdb', 100),
            'min_ff': val.get('min_ff', 0),
            'max_ff': val.get('max_ff', 1),
            'min_plf': val.get('min_plf', 0.7),
            'max_plf': val.get('max_plf', 1)}
            for idx, (col, val) in enumerate(biquad_params.items())
        }
        if not biquad_params:
            raise OCHREException(f'Biquadratic parameters not found for {speed_type} speed {self.name}.')

        if kwargs.get('Disable HVAC Part Load Factor', False):
            # for minimal tests, disable PLF
            for key in biquad_params:
                biquad_params[key]['eir_plr'] = np.array([1, 0, 0], dtype=float)

        return biquad_params

    def update_external_control(self, control_signal):
        # Options for external control signals:
        # - Disable Speed X: if True, disables speed X (for 2 speed control, X=1 or 2)
        #   - Note: Can be used for ideal equipment (reduces max capacity) or dynamic equipment
        #   - Note: Disable Speeds will not reset back to original value
        for idx in range(self.n_speeds):
            self.disable_speeds[idx] = bool(control_signal.get(f'Disable Speed {idx + 1}'))

        return super().update_external_control(control_signal)

    def run_two_speed_control(self):
        mode = super().run_thermostat_control()  # Can be On, Off, or None
        if self.speed_idx == 0:
            # equipment is off
            self.time_in_speed = dt.timedelta(0)
            return mode

        # mode = mode if mode is not None else self.mode
        prev_speed_idx = self.speed_idx
        if self.control_type == 'Time':
            # Time-based 2-speed HVAC control: High speed turns on if temp continues to drop (for heating)
            if self.mode == 'Off':
                speed = 1
            elif self.hvac_mult * (self.zone.temperature - self.temp_indoor_prev) < 0:
                speed = 2
            else:
                speed = self.speed_idx
        # elif self.control_type == 'Time-old':
        #     # Time-based 2-speed HVAC control: High speed turns on if temp continues to drop (for heating)
        #     if self.mode == 'Off':
        #         speed_idx = 0
        #     elif self.hvac_mult * (self.zone.temperature - self.temp_indoor_prev) < 0:
        #         speed_idx = 1
        #     else:
        #         speed_idx = 0
        elif self.control_type == 'Setpoint':
            # Setpoint-based 2-speed HVAC control: High speed uses setpoint difference of deadband / 2 (overlapping)
            high_mode = super().run_thermostat_control(self.temp_setpoint - self.hvac_mult * self.temp_deadband / 2)
            if high_mode == 'On':
                speed = 2
            elif high_mode == 'Off':
                speed = 1
            else:
                speed = self.speed_idx
        elif self.control_type == 'Time2':
            # Old time-based 2-speed HVAC control
            if self.mode == 'Off':
                speed = 1
            else:
                speed = 2
        else:
            raise OCHREException('Unknown control type for {}: {}'.format(self.name, self.control_type))

        # Enforce minimum on times for speed
        if self.time_in_speed < self.min_time_in_speed[prev_speed_idx - 1]:
            speed = prev_speed_idx

        # enforce speed disabling from external control
        if self.disable_speeds[speed - 1]:
            # set to highest allowed speed
            speed = np.nonzero(~ self.disable_speeds)[0][-1] + 1

        if speed != prev_speed_idx or self.mode == 'Off':
            self.time_in_speed = self.time_res
        else:
            self.time_in_speed += self.time_res
        self.speed_idx = speed
        return mode

    def run_thermostat_control(self, setpoint=None):
        if self.use_ideal_capacity:
            raise OCHREException('Ideal capacity equipment should not be running a thermostat control.')

        if self.n_speeds == 1:
            # Run regular thermostat control
            return super().run_thermostat_control(setpoint=setpoint)
        elif self.n_speeds == 2:
            return self.run_two_speed_control()
        else:
            raise OCHREException('Incompatible number of speeds for dynamic equipment:', self.n_speeds)

    def calculate_biquadratic_param(self, param, speed_idx, flow_fraction=1, part_load_ratio=1):
        # runs biquadratic equation for EIR or capacity given the speed index
        # param is 'cap' or 'eir'

        # get rated value based on speed
        if param == 'cap':
            rated = self.capacity_list[speed_idx]
        elif param == 'eir':
            rated = self.eir_list[speed_idx]
        else:
            raise OCHREException('Unknown biquadratic parameter:', param)

        if speed_idx == 0 or self.biquad_params is None:
            return rated

        # get biquadratic parameters for current speed
        params = self.biquad_params[speed_idx]

        # use coil input wet bulb for cooling, dry bulb for heating; ambient dry bulb for both
        t_in = self.coil_input_db if self.is_heater else self.coil_input_wb
        t_ext_db = self.current_schedule['Ambient Dry Bulb (C)']

        # clip temperatures, flow fraction, part load ratio to stay within bounds
        t_in = min(max(t_in, params['min_Twb']), params['max_Twb'])
        t_ext_db = min(max(t_ext_db, params['min_Tdb']), params['max_Tdb'])
        flow_fraction = min(max(flow_fraction, params['min_ff']), params['max_ff'])

        # create vectors based on temperature, flow fraction, and plr
        t_list = np.array([1, t_in, t_in ** 2, t_ext_db, t_ext_db ** 2, t_in * t_ext_db], dtype=float)
        t_ratio = np.dot(t_list, params[param + '_t'])

        ff_list = np.array([1, flow_fraction, flow_fraction ** 2], dtype=float)
        ff_ratio = np.dot(ff_list, params[param + '_ff'])

        plf_list = np.array([1, part_load_ratio, part_load_ratio ** 2], dtype=float)
        plf_ratio = np.dot(plf_list, params[param + '_plr'])
        plf_ratio = min(max(plf_ratio, params['min_plf']), params['max_plf'])

        return rated * t_ratio * ff_ratio / plf_ratio

    def update_capacity(self):
        # update max capacity using highest enabled speed
        max_speed = np.nonzero(~ self.disable_speeds)[0][-1] + 1
        self.capacity_max = self.calculate_biquadratic_param(param='cap', speed_idx=max_speed)

        if self.use_ideal_capacity:
            # determine capacity for each speed, check that capacity_ratio increases with speed
            capacities = [self.calculate_biquadratic_param(param='cap', speed_idx=speed)
                          for speed in range(self.n_speeds + 1)]
            assert (np.diff(capacities) > 0).all()

            # determine ideal capacity
            capacity = super().update_capacity()

            # set speed_idx based on capacity
            if capacity <= capacities[1]:
                # capacity is below lowest rated capacity, run at lowest speed with part load ratio
                self.speed_idx = capacity / capacities[1]
            elif capacity >= capacities[-1]:
                # capacity is above highest speed, run at max capacity
                self.speed_idx = self.n_speeds
            else:
                # interpolate between the 2 closest capacities, save fractional speed index
                speed_high = np.searchsorted(capacities, capacity)
                assert 1 <= speed_high <= self.n_speeds
                speed_low = speed_high - 1
                frac_high = (capacity - capacities[speed_low]) / (capacities[speed_high] - capacities[speed_low])
                self.speed_idx = speed_low + frac_high

            return capacity
        else:
            # Update capacity using biquadratic model. speed_idx should already be set
            return self.calculate_biquadratic_param(param='cap', speed_idx=self.speed_idx)

    def update_eir(self):
        # Update eir and eir_max using biquadratic model
        max_speed = np.nonzero(~ self.disable_speeds)[0][-1] + 1
        self.eir_max = self.calculate_biquadratic_param(param='eir', speed_idx=max_speed)

        if isinstance(self.speed_idx, int):
            return self.calculate_biquadratic_param(param='eir', speed_idx=self.speed_idx)
        elif self.speed_idx < 1:
            # capacity is below lowest rated capacity, run at lowest speed with part load ratio
            return self.calculate_biquadratic_param(param='eir', speed_idx=1, part_load_ratio=self.speed_idx)
        else:
            # interpolate between the 2 closest speeds to get EIR
            speed_low = int(self.speed_idx // 1)
            frac_high = self.speed_idx % 1
            eir_low = self.calculate_biquadratic_param(param='eir', speed_idx=speed_low)
            if frac_high:
                eir_high = self.calculate_biquadratic_param(param='eir', speed_idx=speed_low + 1)
                eir = eir_low * (1 - frac_high) + eir_high * frac_high
            else:
                eir = eir_low
            return eir


class AirConditioner(DynamicHVAC, Cooler):
    name = 'Air Conditioner'
    crankcase_kw = 0.050  # 50W crankcase for AC and ASHP
    crankcase_temp = convert(55, 'degF', 'degC')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Update PLF parameters for low efficiency equipment
        seer = convert(1 / self.eir, 'W', 'Btu/hour')
        if self.n_speeds == 1 and seer < 13 and self.biquad_params is not None:
            self.biquad_params[1]['eir_plf'] = np.array([0.8, 0.2, 0])

    def calculate_power_and_heat(self):
        super().calculate_power_and_heat()

        # add crankcase power when AC is off and outdoor temp is below threshold
        # no impact on sensible heat for now
        if self.crankcase_kw:
            if self.mode == 'Off' and self.current_schedule['Ambient Dry Bulb (C)'] < self.crankcase_temp:
                self.electric_kw += self.crankcase_kw * self.space_fraction


class RoomAC(AirConditioner):
    name = 'Room AC'

    def __init__(self, **kwargs):
        if kwargs.get('speed_type', 'Single') != 'Single':
            raise OCHREException('No model for multi-speed {}'.format(self.name))
        super().__init__(**kwargs)


class ASHPCooler(AirConditioner):
    name = 'ASHP Cooler'
    # crankcase_kw = 0.020  # Keeping 50W crankcase for AC/ASHP


class MinisplitHVAC(DynamicHVAC):
    def __init__(self, **kwargs):
        if kwargs.get('Number of Speeds (-)') == 10:
            # update the number of speeds for MSHP from 10 to 4
            for speed_list in ['Capacity (W)', 'EIR (-)']:
                values = kwargs[speed_list]
                kwargs[speed_list] = [values[1], values[3], values[5], values[9]]
            kwargs['Number of Speeds (-)'] = 4

        super().__init__(**kwargs)


class MinisplitAHSPCooler(MinisplitHVAC, AirConditioner):
    name = 'MSHP Cooler'
    crankcase_kw = 0.015
    crankcase_temp = convert(32, 'degF', 'degC')


class HeatPumpHeater(DynamicHVAC, Heater):
    name = 'Heat Pump Heater'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Defrost Parameters
        self.defrost = False
        self.power_defrost = 0
        self.defrost_power_mult = 1

        # Update PLF parameters for low efficiency equipment
        hspf = convert(1 / self.eir, 'W', 'Btu/hour')
        if self.biquad_params is not None and self.n_speeds == 1 and hspf >= 7:
            self.biquad_params[1]['eir_plf'] = np.array([0.89, 0.11, 0])

    def update_capacity(self):
        # Update capacity if defrost is required
        capacity = super().update_capacity()

        t_ext_db = self.current_schedule['Ambient Dry Bulb (C)']
        omega_ext = self.current_schedule['Ambient Humidity Ratio (-)']
        if self.zone.humidity is not None:
            pres_ext = self.zone.humidity.pressure
        else:
            pres_ext = self.current_schedule['Ambient Pressure (kPa)'] * 1000  # in Pa

        # Based on EnergyPlus Engineering Reference, Defrost Operation, for on demand, reverse cycle defrost
        # see https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/variable-refrigerant-flow-heat-pumps.html#defrost-operation-201605050925
        self.defrost = t_ext_db < 4.4445
        if self.defrost:
            # Calculate reduced capacity
            T_coil_out = 0.82 * t_ext_db - 8.589
            # omega_ext = psychrolib.GetHumRatioFromRelHum(t_ext_db, rh_ext, pres_ext)
            omega_sat_coil = psychrolib.GetHumRatioFromTWetBulb(T_coil_out, T_coil_out, pres_ext)
            delta_omega_coil_out = max(0.000001, omega_ext - omega_sat_coil)
            defrost_time_frac = 1.0 / (1 + (0.01446 / delta_omega_coil_out))
            defrost_capacity_mult = 0.875 * (1 - defrost_time_frac)
            self.defrost_power_mult = 0.954 / 0.875  # increase in power relative to the capacity
            q_defrost = 0.01 * defrost_time_frac * (7.222 - t_ext_db) * (self.capacity_max / 1.01667)

            # Update actual capacity and max allowable capacity
            self.capacity_max = self.capacity_max * defrost_capacity_mult - q_defrost
            if self.use_ideal_capacity:
                capacity = min(capacity, self.capacity_max * self.ext_capacity_frac)
            else:
                capacity = capacity * defrost_capacity_mult - q_defrost

            # Calculate additional power and EIR
            defrost_eir_temp_mod_frac = 0.1528  # in kW
            self.power_defrost = defrost_eir_temp_mod_frac * (capacity / 1.01667) * defrost_time_frac
        else:
            self.defrost_power_mult = 0
            self.power_defrost = 0

        return capacity

    def update_eir(self):
        # Update EIR from defrost. Assumes update_capacity is already run
        eir = super().update_eir()
        if self.defrost and self.capacity > 0:
            eir = (eir * self.capacity * self.defrost_power_mult + self.power_defrost) / self.capacity
        return eir


class ASHPHeater(HeatPumpHeater):
    """
    Heat pump heater with a backup electric resistance element
    """
    name = 'ASHP Heater'
    modes = ['HP On', 'HP and ER On', 'ER On', 'Off']

    def __init__(self, **kwargs):
        if 'setpoint_ramp_rate' not in kwargs:
            # set default setpoint ramp rate to 0.2 C/min to prevent turning on the ER during setpoint changes
            kwargs['setpoint_ramp_rate'] = 0.2

        super().__init__(**kwargs)

        # backup element parameters
        self.outdoor_temp_limit = kwargs.get('Supplemental Heater Cut-in Temperature (C)')  # temp to shut off HP
        self.er_capacity_rated = kwargs['Supplemental Heater Capacity (W)']
        self.er_eir_rated = kwargs.get('Supplemental Heater EIR (-)', 1)
        self.er_capacity = 0
        self.er_ext_capacity = None  # Option to set ER capacity directly, ideal capacity only
        self.er_ext_capacity_frac = 1  # Option to limit max capacity, ideal capacity only

        # Update minimum time for ER element
        er_on_time = kwargs.get(self.end_use + ' Minimum ER On Time', 0)
        self.min_time_in_mode['HP and ER On'] = dt.timedelta(minutes=er_on_time)
        self.min_time_in_mode['ER On'] = dt.timedelta(minutes=er_on_time)

    def update_external_control(self, control_signal):
        # Additional options for ASHP external control signals:
        # - ER Capacity: Sets ER capacity directly, ideal capacity only
        #   - Resets every time step
        # - Max ER Capacity Fraction: Limits ER max capacity, ideal capacity only
        #   - Recommended to set to 0 to disable ER element
        #   - For now, does not get reset
        # - ER Duty Cycle: Combines with "Duty Cycle" control, see HVAC.update_external_control

        capacity_frac = control_signal.get("Max ER Capacity Fraction")
        if capacity_frac is not None:
            if not self.use_ideal_capacity:
                raise IOError(
                    f"Cannot set {self.name} Max ER Capacity Fraction. "
                    'Set `use_ideal_capacity` to True or control "ER Duty Cycle".'
                )
            self.er_ext_capacity_frac = capacity_frac

        capacity = control_signal.get("ER Capacity")
        if capacity is not None:
            if not self.use_ideal_capacity:
                raise IOError(
                    f"Cannot set {self.name} ER Capacity. "
                    'Set `use_ideal_capacity` to True or control "ER Duty Cycle".'
                )
            self.er_ext_capacity = capacity
        
        return super().update_external_control(control_signal)

    def run_duty_cycle_control(self, control_signal):
        # If duty cycles exist, combine duty cycles for HP and ER modes
        er_duty_cycle = control_signal.get('ER Duty Cycle', 0)
        hp_duty_cycle = control_signal.get('Duty Cycle', 0)
        if er_duty_cycle + hp_duty_cycle > 1:
            combo_duty_cycle = 1 - er_duty_cycle - hp_duty_cycle
            er_duty_cycle -= combo_duty_cycle
            hp_duty_cycle -= combo_duty_cycle
            duty_cycles = [hp_duty_cycle, combo_duty_cycle, er_duty_cycle, 0]
        else:
            duty_cycles = [hp_duty_cycle, 0, er_duty_cycle, 1 - er_duty_cycle - hp_duty_cycle]
        assert sum(duty_cycles) == 1

        # update control args and determine mode and speed
        # TODO: update schedule, not control_signal
        control_signal['Duty Cycle'] = duty_cycles
        mode = super().run_duty_cycle_control(control_signal)

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

    def update_internal_control(self):
        if self.use_ideal_capacity:
            # Note: not calling super().update_internal_control
            # TODO: this won't work until they are added to the schedule
            self.ext_capacity = None
            self.er_ext_capacity = None

            # Update setpoint from schedule
            self.update_setpoint()

            # Update HP capacity (and HP max capacity)
            hp_capacity = HeatPumpHeater.update_capacity(self)
            hp_on = hp_capacity > 0

            er_capacity = self.update_er_capacity(hp_capacity)
            er_on = er_capacity > 0
        else:
            # get HP and ER modes separately
            hp_mode = super().update_internal_control()
            hp_on = hp_mode in ['On', 'HP On'] if hp_mode is not None else 'HP' in self.mode
            er_mode = self.run_er_thermostat_control()
            er_on = er_mode == 'On' if er_mode is not None else 'ER' in self.mode

        # Force HP off if outdoor temp is very cold
        t_ext_db = self.current_schedule['Ambient Dry Bulb (C)']
        if self.outdoor_temp_limit is not None and t_ext_db < self.outdoor_temp_limit and hp_on:
            hp_on = False
            er_on = True

        # combine HP and ER modes
        if er_on:
            if hp_on:
                return 'HP and ER On'
            else:
                return 'ER On'
        else:
            if hp_on:
                return 'HP On'
            else:
                return 'Off'

    def run_er_thermostat_control(self):
        # run thermostat control for ER element - lower the setpoint by the deadband
        # TODO: add option to keep setpoint as is, e.g. when using external control
        er_setpoint = self.temp_setpoint - self.temp_deadband
        temp_indoor = self.zone.temperature

        # On and off limits depend on heating vs. cooling
        temp_turn_on = er_setpoint - self.hvac_mult * self.temp_deadband / 2
        temp_turn_off = er_setpoint + self.hvac_mult * self.temp_deadband / 2

        # Determine mode
        if self.hvac_mult * (temp_indoor - temp_turn_on) < 0:
            return 'On'
        if self.hvac_mult * (temp_indoor - temp_turn_off) > 0:
            return 'Off'

    def update_er_capacity(self, hp_capacity):
        if self.use_ideal_capacity:
            if self.er_ext_capacity is not None:
                er_capacity = self.er_ext_capacity
            else:
                # use total ideal capacity - calculated in HVAC.update_capacity
                er_capacity = self.capacity_ideal - hp_capacity
                er_capacity = min(max(er_capacity, 0), self.er_capacity_rated * self.er_ext_capacity_frac)
        else:
            er_capacity = self.er_capacity_rated

        return er_capacity

    def update_capacity(self):
        # Get HP capacity and update ideal capacity
        hp_capacity = super().update_capacity()
        if 'HP' not in self.mode:
            hp_capacity = 0

        if 'ER' in self.mode:
            self.er_capacity = self.update_er_capacity(hp_capacity)
        else:
            self.er_capacity = 0

        return hp_capacity + self.er_capacity

    def update_fan_power(self, capacity):
        fan_power = super().update_fan_power(capacity)

        # if ER on and using ideal capacity, fan power is fixed at rated value
        # this will cause small changes in indoor temperature
        if self.use_ideal_capacity and 'ER' in self.mode:
            if 'HP' in self.mode:
                fixed_fan_power = self.fan_power_max
            else:
                fixed_fan_power = self.fan_power_max / 2

            # add lost fan power to ER capacity, return updated fan power
            self.er_capacity += (fan_power - fixed_fan_power) / self.shr
            self.capacity += (fan_power - fixed_fan_power) / self.shr
            return fixed_fan_power
        else:
            return fan_power

    def update_eir(self):
        if self.mode == 'HP and ER On':
            # EIR is a weighted average of HP and ER EIRs
            hp_eir = super().update_eir()
            hp_capacity = self.capacity - self.er_capacity
            return (hp_capacity * hp_eir + self.er_capacity * self.er_eir_rated) / self.capacity
        elif self.mode in ['HP On', 'Off']:
            return super().update_eir()
        elif self.mode == 'ER On':
            return self.er_eir_rated
        else:
            raise OCHREException('Unknown mode for {}: {}'.format(self.name, self.mode))

    def calculate_power_and_heat(self):
        # Update ER capacity if off
        if 'On' not in self.mode:
            self.er_capacity = 0

        super().calculate_power_and_heat()

    def generate_results(self):
        results = super().generate_results()

        if self.verbosity >= 6:
            tot_power = self.capacity * self.eir * self.space_fraction / 1000
            er_power = self.er_capacity * self.er_eir_rated * self.space_fraction / 1000
            results[f'{self.end_use} Main Power (kW)'] = tot_power - er_power
            results[f'{self.end_use} ER Power (kW)'] = er_power

        return results


class MinisplitAHSPHeater(MinisplitHVAC, ASHPHeater):
    name = 'MSHP Heater'
    pan_heater_kw = 0.150  # turn on pan heater at 150W when below 0C
    pan_heater_temp = 0  # deg C

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pan_heater_on = False

    def calculate_power_and_heat(self):
        super().calculate_power_and_heat()

        # add pan heater power when outdoor temp < 32F
        # no impact on sensible heat
        t_ext = self.current_schedule['Ambient Dry Bulb (C)']
        self.pan_heater_on = self.pan_heater_kw > 0 and t_ext < self.pan_heater_temp
        if self.pan_heater_on:
            self.electric_kw += self.pan_heater_kw * self.space_fraction
