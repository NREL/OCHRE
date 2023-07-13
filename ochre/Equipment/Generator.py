# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:28:35 2019

@author: mblonsky
"""

import datetime as dt
from scipy.interpolate import interp1d

from ochre.utils.units import kwh_to_therms
from ochre.Equipment import Equipment

CONTROL_TYPES = ['Schedule', 'Self-Consumption', 'Off']


class Generator(Equipment):
    allow_consumption = False
    is_gas = False
    zone_name = None
    optional_inputs = ['net_power']

    """Generic equipment class for load-following equipment, including batteries, gas generators, and gas fuel cells."""

    def __init__(self, control_type='Off', parameter_file='default_parameters.csv', efficiency_type='constant',
                 efficiency_file='efficiency_curve.csv', **kwargs):
        super().__init__(parameter_file=parameter_file, **kwargs)

        # power parameters
        self.power_setpoint = 0  # setpoint from controller, AC side (after losses), in kW
        self.power_input = 0  # input power including losses, equal to gas consumption, in kW
        self.power_chp = 0  # usable output heat for combined heat and power (CHP) uses (not implemented), in kW

        # Electrical parameters
        self.capacity = self.parameters['capacity']  # in kW
        self.capacity_min = self.parameters.get('capacity_min')  # minimum generating power for self-consumption, in kW
        self.ramp_rate = self.parameters.get('ramp_rate')  # max output power ramp rate, generation only, in kW/min

        # Efficiency parameters
        self.efficiency = None  # variable efficiency, unitless
        self.efficiency_rated = self.parameters['efficiency']  # unitless
        self.efficiency_chp = self.parameters.get('efficiency_chp', 0)  # CHP efficiency, for generation only
        self.efficiency_type = efficiency_type  # formula for calculating efficiency
        if self.efficiency_type == 'curve':
            # Load efficiency curve
            df = self.initialize_parameters(efficiency_file, name_col='Capacity Ratio', value_col=None)
            self.efficiency_curve = interp1d(df.index, df['Efficiency Ratio'])
        else:
            self.efficiency_curve = None

        # Control parameters
        if control_type not in CONTROL_TYPES:
            raise Exception('Unknown {} control type: {}'.format(self.name, control_type))
        self.control_type = control_type

    def update_external_control(self, control_signal):
        # Options for external control signals:
        # - P Setpoint: Directly sets power setpoint, in kW
        #   - Note: still subject to SOC limits and charge/discharge limits
        # - Control Type: Sets the control type to one of CONTROL_TYPES
        # - Parameters: Update control parameters, including:
        #   - Schedule: charge/discharge start times
        #   - Schedule: charge/discharge powers
        #   - Self-Consumption: charge/discharge offsets, in kW
        #   - Self-Consumption: charge type (from any solar or from net power)

        if 'Parameters' in control_signal:
            self.parameters.update(control_signal['Parameters'])

        if 'Control Type' in control_signal:
            control_type = control_signal['Control Type']
            if control_type in CONTROL_TYPES:
                self.control_type = control_type
            else:
                self.warn('Unknown control type ({}). Keeping previous control type'.format(control_type))

        self.update_internal_control()

        # set power directly from setpoint
        if 'P Setpoint' in control_signal:
            self.power_setpoint = control_signal['P Setpoint']

        return 'On' if self.power_setpoint != 0 else 'Off'

    def update_internal_control(self):
        # Set power setpoint based on internal control type

        if self.control_type == 'Schedule':
            # Charges or discharges at given power and given time of day
            time = self.current_time.time()
            if time == dt.time(hour=int(self.parameters['charge_start_hour'])):
                self.power_setpoint = self.parameters['charge_power']
            if time == dt.time(hour=int(self.parameters['discharge_start_hour'])):
                self.power_setpoint = -self.parameters['discharge_power']

        elif self.control_type == 'Self-Consumption':
            net_power = self.current_schedule.get('net_power')
            if net_power is not None:
                # account for import/export limits
                if net_power >= 0:
                    # generation only
                    self.power_setpoint = min(-net_power + self.parameters.get('export_limit', 0), 0)
                else:
                    # consumption only
                    self.power_setpoint = max(-net_power - self.parameters.get('import_limit', 0), 0)
            else:
                self.warn('Cannot run Self-Consumption control without net power')
                self.power_setpoint = 0

        elif self.control_type == 'Off':
            self.power_setpoint = 0

        return 'On' if self.power_setpoint != 0 else 'Off'

    def get_power_limits(self):
        # Minimum (i.e. generating) output power limit based on capacity and ramp rate
        min_power = -self.capacity
        if self.ramp_rate is not None and self.electric_kw <= 0:
            # ramp rate only impacts generating power
            minutes = self.time_res.total_seconds() / 60
            min_power = max(min_power, self.electric_kw - self.ramp_rate * minutes)

        # Maximum (usually consuming) output power limit based on capacity. Generators may have a min operating power
        if self.allow_consumption:
            max_power = self.capacity
        elif self.control_type == 'Self-Consumption' and self.capacity_min is not None:
            # min operating power - only for generators in self-consumption mode
            max_power = -self.capacity_min
        else:
            max_power = 0

        if max_power < min_power:
            # rare case - use power closest to 0
            if abs(max_power) < abs(min_power):
                min_power = max_power
            else:
                max_power = min_power
        return min_power, max_power

    def calculate_efficiency(self, electric_kw=None, is_output_power=True):
        if electric_kw is None:
            electric_kw = self.electric_kw

        if electric_kw == 0:
            # set efficiency to 0 when off
            return 0
        # Calculate generator efficiency based on type
        elif self.efficiency_type == 'constant':
            return self.efficiency_rated
        elif self.efficiency_type == 'curve':
            assert is_output_power
            capacity_ratio = abs(electric_kw) / self.capacity
            efficiency_ratio = self.efficiency_curve(capacity_ratio)
            return self.efficiency_rated * efficiency_ratio
        elif self.efficiency_type == 'quadratic':
            # Quadratic efficiency curve from:
            # Vishwanathan G, et al. Techno-economic analysis of high-efficiency natural-gas generators for residential
            # combined heat and power. Appl Energy. https://doi.org/10.1016/j.apenergy.2018.06.013.
            assert is_output_power
            capacity_ratio = abs(electric_kw) / self.capacity
            eff = self.efficiency_rated * (-0.5 * capacity_ratio ** 2 + 1.5 * capacity_ratio)
            return min(eff, 0.001)  # must be positive
        else:
            raise Exception('Unknown efficiency type for {}: {}'.format(self.name, self.efficiency_type))

    def calculate_power_and_heat(self):
        if self.mode == 'Off':
            self.electric_kw = 0
        else:
            # force ac power within limits
            min_power, max_power = self.get_power_limits()
            self.electric_kw = min(max(self.power_setpoint, min_power), max_power)

        # calculate input (gas) power and CHP power
        self.efficiency = self.calculate_efficiency()
        assert 0 <= self.efficiency <= 1
        if self.electric_kw < 0:
            # generating/discharging
            self.power_input = self.electric_kw / self.efficiency
            self.power_chp = self.power_input * self.efficiency_chp
        else:
            # consuming power/charging, or off
            self.power_input = self.electric_kw * self.efficiency
            self.power_chp = 0

        if self.is_gas:
            self.gas_therms_per_hour = - self.power_input * kwh_to_therms

        # calculate power losses, equal to heat gains
        # Note: heat gains are not included by default, since the zone defaults to None
        self.sensible_gain = (self.electric_kw - self.power_input) * 1000  # power losses, in W
        assert self.sensible_gain >= 0

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 6:
            results[f'{self.end_use} Setpoint (kW)'] = self.power_setpoint
            results[f'{self.end_use} Efficiency (-)'] = self.efficiency
        return results


class GasGenerator(Generator):
    name = 'Gas Generator'
    end_use = 'Gas Generator'
    is_gas = True


class GasFuelCell(GasGenerator):
    name = 'Gas Fuel Cell'

    def __init__(self, efficiency_type='curve', **kwargs):
        super().__init__(efficiency_type=efficiency_type, **kwargs)
