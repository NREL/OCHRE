import os
import pandas as pd

from ochre.utils.units import kwh_to_therms
from ochre.utils import load_csv
import ochre.utils.schedule as utils_schedule
from ochre.Equipment import Equipment


# TODO: Add option to put heat gains in multiple zones (e.g. basement MELs)


class ScheduledLoad(Equipment):
    """
    Equipment with a pre-defined schedule for power. Schedule may come from the main schedule file or a separate file
    named `equipment_schedule_file`. The schedule must have one or more columns named `<equipment_name> (<unit>)`, where
    the unit can be 'kW' for electric equipment and 'therms/hour' for gas equipment. Combo equipment should have
    two columns, one for electric and one for gas power.
    """

    def __init__(self, zone_name=None, **kwargs):
        # Update zone based on name. Zone defaults to Indoor
        if zone_name is None:
            if 'Exterior' in self.name:
                zone_name = 'Outdoor'
            elif 'Garage' in self.name:
                zone_name = 'Garage'
            elif 'Basement' in self.name:
                zone_name = 'Foundation'

        self.electric_name = None
        self.gas_name = None

        super().__init__(zone_name=zone_name, **kwargs)

        self.p_set_point = 0  # in kW
        self.gas_set_point = 0  # in therms/hour

        self.is_electric = self.electric_name in self.schedule
        self.is_gas = self.gas_name in self.schedule

        # Sensible and latent gain fractions, unitless
        # FUTURE: separate convection and radiation, move radiation gains to the surfaces around the zone
        self.sensible_gain_fraction = (kwargs.get('Convective Gain Fraction (-)', 0) +
                                       kwargs.get('Radiative Gain Fraction (-)', 0))
        self.latent_gain_fraction = kwargs.get('Latent Gain Fraction (-)', 0)
        assert self.sensible_gain_fraction + self.latent_gain_fraction <= 1.001  # computational errors possible

    def initialize_schedule(self, schedule=None, equipment_schedule_file=None,
                            schedule_rename_columns=None, schedule_scale_factor=1, **kwargs):
        self.electric_name = f'{self.name} (kW)'
        self.gas_name = f'{self.name} (therms/hour)'

        if equipment_schedule_file is not None:
            # load schedule from separate schedule file - used for scheduled PV and EV
            schedule = load_csv(equipment_schedule_file, sub_folder=self.end_use)
            schedule = schedule.loc[:, schedule_rename_columns]
            schedule = utils_schedule.set_annual_index(schedule, self.start_time.year, timezone=self.start_time.tzinfo)
            schedule = utils_schedule.resample_and_reindex(schedule, **kwargs)
            if schedule_rename_columns is not None:
                schedule = schedule.rename(columns=schedule_rename_columns)
            schedule *= schedule_scale_factor

        if schedule is None:
            raise Exception(f'Schedule required for {self.name}')
        
        required_inputs = [name for name in [self.electric_name, self.gas_name] if name in schedule]
        if not required_inputs:
            raise Exception(f'Cannot find any schedule columns for {self.name}')

        # set schedule columns to zero if month multiplier exists and is zero (for ceiling fans)
        multipliers = kwargs.get('month_multipliers', [])
        zero_months = [i for i, m in enumerate(multipliers) if m == 0]
        if zero_months:
            schedule.loc[schedule.index.month.isin(zero_months), required_inputs] = 0

        return super().initialize_schedule(schedule, required_inputs=required_inputs, **kwargs)

    def update_external_control(self, control_signal):
        # Control options for changing power:
        #  - Load Fraction: gets multiplied by power from schedule, unitless (applied to electric AND gas)
        #  - P Setpoint: overwrites electric power from schedule, in kW
        #  - Gas Setpoint: overwrites gas power from schedule, in therms/hour
        self.update_internal_control()

        load_fraction = control_signal.get('Load Fraction')
        if load_fraction is not None:
            self.p_set_point *= load_fraction
            self.gas_set_point *= load_fraction

        p_set_ext = control_signal.get('P Setpoint')
        if p_set_ext is not None:
            self.p_set_point = p_set_ext

        gas_set_ext = control_signal.get('Gas Setpoint')
        if gas_set_ext is not None:
            self.gas_set_point = gas_set_ext

        return 'On' if self.p_set_point + self.gas_set_point != 0 else 'Off'

    def update_internal_control(self):
        if self.is_electric:
            self.p_set_point = self.current_schedule[self.electric_name]
            if abs(self.p_set_point) > 20:
                self.warn(f'High electric power warning: {self.p_set_point} kW.')
                if abs(self.p_set_point) > 40:
                    raise Exception(f'{self.name} electric power is too large: {self.p_set_point} kW.')

        if self.is_gas:
            self.gas_set_point = self.current_schedule[self.gas_name]
            if abs(self.gas_set_point) > 0.5:
                self.warn(f'High gas power warning: {self.gas_set_point} therms/hour.')
                if abs(self.gas_set_point) > 1:
                    raise Exception(f'{self.name} gas power is too large: {self.gas_set_point} therms/hour.')

        return 'On' if self.p_set_point + self.gas_set_point != 0 else 'Off'

    def calculate_power_and_heat(self):
        if self.mode == 'On':
            self.electric_kw = self.p_set_point if self.is_electric else 0
            self.gas_therms_per_hour = self.gas_set_point if self.is_gas else 0
        else:
            # Force power to 0
            self.electric_kw = 0
            self.gas_therms_per_hour = 0

        total_power_w = (self.electric_kw + self.gas_therms_per_hour / kwh_to_therms) * 1000  # in W
        self.sensible_gain = total_power_w * self.sensible_gain_fraction
        self.latent_gain = total_power_w * self.latent_gain_fraction

    def generate_results(self):
        results = super().generate_results()

        if self.verbosity >= 6 and self.name != self.end_use:
            if self.is_electric:
                results[f'{self.results_name} Electric Power (kW)'] = self.electric_kw
                results[f'{self.results_name} Reactive Power (kVAR)'] = self.reactive_kvar
            if self.is_gas:
                results[f'{self.results_name} Gas Power (therms/hour)'] = self.gas_therms_per_hour
        return results


class LightingLoad(ScheduledLoad):
    end_use = 'Lighting'
