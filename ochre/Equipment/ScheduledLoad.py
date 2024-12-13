from ochre.utils import OCHREException
from ochre.Equipment import Equipment


# TODO: Add option to put heat gains in multiple zones (e.g. basement MELs)


class ScheduledLoad(Equipment):
    """
    Equipment with a pre-defined schedule for power. Schedule may come from
    the building schedule file or separately. The schedule must have one or
    more columns named `<equipment_name> (<unit>)`, where the unit can be 'kW'
    for electric equipment and 'therms/hour' for gas equipment. Combo
    equipment should have two columns, one for electric and one for gas power.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_set_point = 0  # in kW
        self.gas_set_point = 0  # in therms/hour

        self.is_electric = "Power (kW)" in self.schedule
        self.is_gas = "Gas (therms/hour)" in self.schedule

    def initialize_schedule(self, schedule=None, **kwargs):
        # Get power and gas columns from schedule, if they exist
        schedule_cols = {
            f"{self.name} (kW)": "Power (kW)",
            f"{self.name} (therms/hour)": "Gas (therms/hour)",
        }
        optional_inputs = list(schedule_cols.keys())

        schedule = super().initialize_schedule(schedule, optional_inputs=optional_inputs, **kwargs)
        schedule = schedule.rename(columns=schedule_cols)

        # set schedule columns to zero if month multiplier exists and is zero (for ceiling fans)
        multipliers = kwargs.get("month_multipliers", [])
        zero_months = [i for i, m in enumerate(multipliers) if m == 0]
        if zero_months:
            schedule.loc[schedule.index.month.isin(zero_months), :] = 0

        return schedule

    def update_external_control(self, control_signal):
        # Control options for changing power:
        #  - Load Fraction: gets multiplied by power from schedule, unitless (applied to electric AND gas)
        #  - P Setpoint: overwrites electric power from schedule, in kW
        #  - Gas Setpoint: overwrites gas power from schedule, in therms/hour
        self.update_internal_control()

        load_fraction = control_signal.get("Load Fraction")
        if load_fraction is not None:
            self.p_set_point *= load_fraction
            self.gas_set_point *= load_fraction

        p_set_ext = control_signal.get("P Setpoint")
        if p_set_ext is not None:
            self.p_set_point = p_set_ext

        gas_set_ext = control_signal.get("Gas Setpoint")
        if gas_set_ext is not None:
            self.gas_set_point = gas_set_ext

        return "On" if self.p_set_point + self.gas_set_point != 0 else "Off"

    def update_internal_control(self):
        if self.is_electric:
            self.p_set_point = self.current_schedule["Power (kW)"]
            if abs(self.p_set_point) > 20:
                self.warn(f"High electric power warning: {self.p_set_point} kW.")
                if abs(self.p_set_point) > 40:
                    raise OCHREException(
                        f"{self.name} electric power is too large: {self.p_set_point} kW."
                    )

        if self.is_gas:
            self.gas_set_point = self.current_schedule["Gas (therms/hour)"]
            if abs(self.gas_set_point) > 0.5:
                self.warn(f"High gas power warning: {self.gas_set_point} therms/hour.")
                if abs(self.gas_set_point) > 1:
                    raise OCHREException(
                        f"{self.name} gas power is too large: {self.gas_set_point} therms/hour."
                    )

        return "On" if self.p_set_point + self.gas_set_point != 0 else "Off"

    def calculate_power_and_heat(self):
        if self.mode == "On":
            self.electric_kw = self.p_set_point if self.is_electric else 0
            self.gas_therms_per_hour = self.gas_set_point if self.is_gas else 0
        else:
            # Force power to 0
            self.electric_kw = 0
            self.gas_therms_per_hour = 0

        super().calculate_power_and_heat()


class LightingLoad(ScheduledLoad):
    end_use = "Lighting"
