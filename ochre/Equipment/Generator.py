# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt
import scipy.interpolate

from . import Units
from . import Equipment

CONTROL_TYPES = ['Schedule', 'Self-Consumption', 'Off']


class Generator(Equipment):
    allow_consumption = False
    is_gas = False

    """Generic equipment class for load-following equipment, including batteries, gas generators, and gas fuel cells."""

    def __init__(self, control_type='Off', parameter_file='default_parameters.csv', zone=None,
                 efficiency_type='constant', efficiency_file='efficiency_curve.csv', **kwargs):
        super().__init__(parameter_file=parameter_file, zone=zone, **kwargs)

        # power parameters
        self.power_setpoint = 0  # setpoint from controller, AC side (after losses), in kW
        self.power_input = 0  # input power including losses, equal to gas consumption, in kW
        self.power_chp = 0  # usable output heat for combined heat and power (CHP) uses (not implemented), in kW

        # Electrical parameters
        self.capacity = self.parameters['capacity']  # in kW
        self.capacity_min = self.parameters.get('capacity_min')  # minimum generating power for self-consumption, in kW
        self.ramp_rate = self.parameters.get('ramp_rate')  # max output power ramp rate, generation only, in kW/min

        # Efficiency parameters
        self.efficiency_type = efficiency_type  # formula for calculating efficiency
        self.efficiency = self.parameters['efficiency']  # Generation/discharge efficiency, unitless
        self.efficiency_charge = self.parameters.get('efficiency_charge', self.efficiency)  # Load/charge efficiency
        self.efficiency_chp = self.parameters.get('efficiency_chp', 0)  # CHP efficiency, for generation only
        assert self.efficiency + self.efficiency_chp <= 1
        if self.efficiency_type == 'curve':
            # Load efficiency curve
            df = self.initialize_parameters(efficiency_file, name_col='Capacity Ratio', val_col=None)
            self.efficiency_curve = scipy.interpolate.interp1d(df.index, df['Efficiency Ratio'])
        else:
            self.efficiency_curve = None

        # Control parameters
        if control_type not in CONTROL_TYPES:
            raise Exception('Unknown {} control type: {}'.format(self.name, control_type))
        self.control_type = control_type

    def update_external_control(self, schedule, ext_control_args):
        # Options for external control signals:
        # - P Setpoint: Directly sets power setpoint, in kW
        #   - Note: still subject to SOC limits and charge/discharge limits
        # - Control Type: Sets the control type to one of CONTROL_TYPES
        # - Parameters: Update control parameters, including:
        #   - Schedule: charge/discharge start times
        #   - Schedule: charge/discharge powers
        #   - Self-Consumption: charge/discharge offsets, in kW
        #   - Self-Consumption: charge type (from any solar or from net power)

        if 'Parameters' in ext_control_args:
            self.parameters.update(ext_control_args['Parameters'])

        if 'Control Type' in ext_control_args:
            control_type = ext_control_args['Control Type']
            if control_type in CONTROL_TYPES:
                self.control_type = control_type
            else:
                self.warn('Unknown control type ({}). Keeping previous control type'.format(control_type))

        # set power directly from setpoint
        if 'P Setpoint' in ext_control_args:
            self.power_setpoint = ext_control_args['P Setpoint']
            return 'On' if self.power_setpoint != 0 else 'Off'

        return self.update_internal_control(schedule)

    def update_internal_control(self, schedule):
        # Set power setpoint based on internal control type

        if self.control_type == 'Schedule':
            # Charges or discharges at given power and given time of day
            time = self.current_time.time()
            if time == dt.time(hour=int(self.parameters['charge_start_hour'])):
                self.power_setpoint = self.parameters['charge_power']
            if time == dt.time(hour=int(self.parameters['discharge_start_hour'])):
                self.power_setpoint = -self.parameters['discharge_power']

        elif self.control_type == 'Self-Consumption':
            if 'net_power' not in schedule:
                self.warn('Cannot run Self-Consumption control without net power')
                self.power_setpoint = 0
            else:
                net_power = schedule.get('net_power', 0)

                # account for import/export limits
                if net_power >= 0:
                    # generation only
                    self.power_setpoint = min(-net_power + self.parameters.get('export_limit', 0), 0)
                else:
                    # consumption only
                    self.power_setpoint = max(-net_power - self.parameters.get('import_limit', 0), 0)

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

    def get_efficiency(self):
        if self.efficiency_type == 'constant':
            return self.efficiency
        elif self.efficiency_type == 'curve':
            capacity_ratio = abs(self.electric_kw) / self.capacity
            efficiency_ratio = self.efficiency_curve(capacity_ratio)
            return efficiency_ratio * self.efficiency
        elif self.efficiency_type == 'quadratic':
            # Quadratic efficiency curve from:
            # Vishwanathan G, et al. Techno-economic analysis of high-efficiency natural-gas generators for residential
            # combined heat and power. Appl Energy. https://doi.org/10.1016/j.apenergy.2018.06.013.
            capacity_ratio = abs(self.electric_kw) / self.capacity
            eff = self.efficiency * (-0.5 * capacity_ratio ** 2 + 1.5 * capacity_ratio)
            return min(eff, 0.001)  # must be positive
        else:
            raise Exception('Unknown efficiency type for {}: {}'.format(self.name, self.efficiency_type))

    def calculate_power_and_heat(self, schedule):
        if self.mode == 'Off':
            self.electric_kw = 0
        else:
            # force ac power within limits
            min_power, max_power = self.get_power_limits()
            self.electric_kw = np.clip(self.power_setpoint, min_power, max_power)

        # calculate input (gas) power and CHP power
        if self.electric_kw < 0:
            # generating, or off
            self.power_input = self.electric_kw / self.get_efficiency()
            self.power_chp = self.power_input * self.efficiency_chp
        elif self.electric_kw > 0:
            # consuming power/charging
            self.power_input = self.electric_kw * self.efficiency_charge
            self.power_chp = 0
        else:
            self.power_input = 0
            self.power_chp = 0
        if self.is_gas:
            self.gas_therms_per_hour = - Units.kWh2therms(self.power_input)

        # calculate power losses, equal to heat gains
        # Note: heat gains are not included by default, since the zone defaults to None
        self.sensible_gain = (self.electric_kw - self.power_input) * 1000  # power losses, in W
        assert self.sensible_gain >= 0

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            if self.mode == 'On':
                mode = 0 if self.power_setpoint >= 0 else 1
            else:
                mode = 2
            return {self.name: {'Mode': mode}}
        else:
            if verbosity >= 6:
                results.update({self.name + ' Setpoint (kW)': self.power_setpoint})
        return results


class GasGenerator(Generator):
    name = 'Gas Generator'
    end_use = 'Gas Generator'
    is_gas = True


class GasFuelCell(GasGenerator):
    name = 'Gas Fuel Cell'

    def __init__(self, efficiency_type='curve', **kwargs):
        super().__init__(efficiency_type=efficiency_type, **kwargs)
