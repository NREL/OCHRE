# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:28:35 2019

@author: mblonsky
"""

from scipy.interpolate import interp1d

from ochre.utils import OCHREException
from ochre.utils.units import kwh_to_therms
from ochre.Equipment import Equipment


class Generator(Equipment):
    allow_consumption = False
    is_gas = False
    zone_name = None
    optional_inputs = ["net_power"]

    """Generic equipment class for load-following equipment, including batteries, gas generators, and gas fuel cells."""

    def __init__(
        self,
        self_consumption_mode=False,
        parameter_file="default_parameters.csv",
        efficiency_type="constant",
        efficiency_file="efficiency_curve.csv",
        **kwargs,
    ):
        super().__init__(parameter_file=parameter_file, **kwargs)

        # power parameters
        self.power_setpoint = 0  # setpoint from controller, AC side (after losses), in kW
        self.power_input = 0  # input power including losses, equal to gas consumption, in kW
        # not implemented:
        # self.power_chp = 0  # usable output heat for combined heat and power (CHP) uses, in kW

        # Electrical parameters
        self.capacity = self.parameters['capacity']  # in kW
        # minimum generating power for self-consumption
        self.capacity_min = self.parameters.get('capacity_min')  # in kW
        # max output power ramp rate, generation only
        self.ramp_rate = self.parameters.get('ramp_rate')  # in kW/min

        # Efficiency parameters
        self.efficiency = None  # variable efficiency, unitless
        self.efficiency_rated = self.parameters["efficiency"]  # unitless
        # CHP efficiency, for generation only
        self.efficiency_chp = self.parameters.get("efficiency_chp", 0)  
        self.efficiency_type = efficiency_type  # formula for calculating efficiency
        if self.efficiency_type == "curve":
            # Load efficiency curve
            df = self.initialize_parameters(
                efficiency_file, name_col="Capacity Ratio", value_col=None
            )
            self.efficiency_curve = interp1d(df.index, df["Efficiency Ratio"])
        else:
            self.efficiency_curve = None

        # Control parameters
        self.self_consumption_mode = self_consumption_mode
        self.import_limit = self.parameters.get("import_limit", 0)
        self.export_limit = self.parameters.get("export_limit", 0)

    def parse_control_signal(self, control_signal):
        # Options for external control signals:
        # - P Setpoint: Directly sets power setpoint, in kW
        #   - Note: still subject to SOC limits and charge/discharge limits
        # - Self Consumption Mode: Toggle self consumption mode, does not reset
        # - Max Import Limit: Max dwelling import power for self-consumption control
        # - Max Export Limit: Max dwelling export power for self-consumption control

        if "Self Consumption Mode" in control_signal:
            self.self_consumption_mode = bool(control_signal["Self Consumption Mode"])

        import_limit = control_signal.get("Max Import Limit")
        if import_limit is not None:
            if f"{self.end_use} Max Import Limit (kW)" in self.current_schedule:
                self.current_schedule[f"{self.end_use} Max Import Limit (kW)"] = import_limit
            else:
                self.import_limit = import_limit

        export_limit = control_signal.get("Max Export Limit")
        if export_limit is not None:
            if f"{self.end_use} Max Export Limit (kW)" in self.current_schedule:
                self.current_schedule[f"{self.end_use} Max Export Limit (kW)"] = export_limit
            else:
                self.export_limit = export_limit

        # Note: this overrides self consumption mode, it will always set the setpoint directly
        power_setpoint = control_signal.get("P Setpoint")
        if power_setpoint is not None:
            self.current_schedule[f"{self.end_use} Electric Power (kW)"] = power_setpoint

    def run_internal_control(self):
        if f"{self.end_use} Max Import Limit (kW)" in self.current_schedule:
            self.import_limit = self.current_schedule[f"{self.end_use} Max Import Limit (kW)"]
        if f"{self.end_use} Max Export Limit (kW)" in self.current_schedule:
            self.export_limit = self.current_schedule[f"{self.end_use} Max Export Limit (kW)"]

        # Set power setpoint based on internal control type
        if self.self_consumption_mode:
            net_power = self.current_schedule.get("net_power")
            if net_power is not None:
                # account for import/export limits
                desired_power = max(min(net_power, self.import_limit), -self.export_limit)
                self.power_setpoint = desired_power - net_power
            else:
                self.warn("Cannot run Self-Consumption control without net power")
                self.power_setpoint = 0

        else:
            # Charges or discharges based on schedule
            self.power_setpoint = self.current_schedule.get(
                f"{self.end_use} Electric Power (kW)", 0
            )

        return 1 if self.power_setpoint != 0 else 0

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
        elif self.capacity_min is not None:
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
        elif self.efficiency_type == "constant":
            return self.efficiency_rated
        elif self.efficiency_type == "curve":
            assert is_output_power
            capacity_ratio = abs(electric_kw) / self.capacity
            efficiency_ratio = self.efficiency_curve(capacity_ratio)
            return self.efficiency_rated * efficiency_ratio
        elif self.efficiency_type == "quadratic":
            # Quadratic efficiency curve from:
            # Vishwanathan G, et al. Techno-economic analysis of high-efficiency natural-gas generators for residential
            # combined heat and power. Appl Energy. https://doi.org/10.1016/j.apenergy.2018.06.013.
            assert is_output_power
            capacity_ratio = abs(electric_kw) / self.capacity
            eff = self.efficiency_rated * (-0.5 * capacity_ratio**2 + 1.5 * capacity_ratio)
            return min(eff, 0.001)  # must be positive
        else:
            raise OCHREException(
                "Unknown efficiency type for {}: {}".format(self.name, self.efficiency_type)
            )

    def calculate_power_and_heat(self):
        if not self.on:
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
            self.gas_therms_per_hour = -self.power_input * kwh_to_therms

        # calculate power losses, equal to heat gains
        # Note: heat gains are not included by default, since the zone defaults to None
        self.sensible_gain = (self.electric_kw - self.power_input) * 1000  # power losses, in W
        assert self.sensible_gain >= 0

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 6:
            results[f"{self.end_use} Setpoint (kW)"] = self.power_setpoint
            results[f"{self.end_use} Efficiency (-)"] = self.efficiency
        return results


class GasGenerator(Generator):
    name = "Gas Generator"
    end_use = "Gas Generator"
    is_gas = True


class GasFuelCell(GasGenerator):
    name = "Gas Fuel Cell"

    def __init__(self, efficiency_type="curve", **kwargs):
        super().__init__(efficiency_type=efficiency_type, **kwargs)
