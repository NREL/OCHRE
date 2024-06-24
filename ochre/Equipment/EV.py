import numpy as np
import datetime as dt
import pandas as pd

from ochre.utils import OCHREException, load_csv
from ochre.Equipment import EventBasedLoad, ScheduledLoad

# For EVI-Pro assumptions, see Section 1.2 and 1.3:
# https://afdc.energy.gov/evi-pro-lite/load-profile/assumptions
EV_FUEL_ECONOMY = 1 / 325 * 1000  # miles per kWh, used to calculate capacity, for sedans
EV_EFFICIENCY = 0.9  # unitless, charging efficiency
EV_MAX_POWER = {  # max AC charging power, by vehicle number (PHEV20, PHEV50, BEV100, BEV250)
    'Level0': [1.4, 1.4, 1.4, 1.4],  # For testing only
    'Level1': [1.4, 1.4, 1.4, 1.4],
    'Level2': [3.6, 3.6, 7.2, 11.5],
}


# TODO: change mode to be on only when power > 0 (or remove mode entirely?)
class ElectricVehicle(EventBasedLoad):
    """
    Electric Vehicle model using an event-based schedule. The schedule is based on residential charging data from
    EVI-Pro. Uses a joint probability density function (PDF) to get the arrival time, arrival SOC, and parking duration
    of the EV charging event. Forces an overnight charging event to occur every night.
    """
    name = 'EV'
    end_use = 'EV'
    zone_name = None
    delay_event_end = False
    required_inputs = ['Ambient Dry Bulb (C)']
    optional_inputs = [
        "EV Max Power (kW)",
        "EV SOC (-)",
        "EV Max SOC (-)",
    ]

    def __init__(self, vehicle_type, charging_level, capacity=None, mileage=None, max_power=None, 
                 enable_part_load=None, equipment_event_file=None, **kwargs):
        # get EV battery capacity and mileage
        if capacity is None and mileage is None:
            raise OCHREException('Must specify capacity or mileage for {}'.format(self.name))
        elif capacity is not None:
            self.capacity = capacity  # in kWh
            mileage = self.capacity * EV_FUEL_ECONOMY  # in mi
        else:
            # determine capacity using mileage
            self.capacity = mileage / EV_FUEL_ECONOMY  # in kWh

        # get charging level and set option for part load setpoints
        charging_level = charging_level.replace(' ', '')
        if str(charging_level) not in ['Level0', 'Level1', 'Level2']:
            raise OCHREException('Unknown vehicle type for {}: {}'.format(self.name, charging_level))
        self.charging_level = str(charging_level)
        if enable_part_load is None:
            enable_part_load = self.charging_level == 'Level2'
        self.enable_part_load = enable_part_load

        # get vehicle number (1-4) based on type and mileage. Used for choosing event file
        self.vehicle_type = vehicle_type
        if vehicle_type == 'PHEV':
            vehicle_num = 1 if mileage < 35 else 2
        elif vehicle_type == 'BEV':
            vehicle_num = 3 if mileage < 175 else 4
        else:
            raise OCHREException('Unknown vehicle type for {}: {}'.format(self.name, vehicle_type))
        if equipment_event_file is None:
            equipment_event_file = "pdf_Veh{}_{}.csv".format(vehicle_num, self.charging_level)

        # charging model
        if max_power is None:
            self.max_power = EV_MAX_POWER[self.charging_level][vehicle_num - 1]
        else:
            self.max_power = max_power
        self.max_power_ctrl = self.max_power
        self.setpoint_power = None
        self.soc = 1  # unitless
        self.next_soc = 1  # unitless
        self.soc_max_ctrl = 1  # unitless
        self.unmet_load = 0  # lost charging from delays, in kW

        # initialize events
        super().__init__(equipment_event_file=equipment_event_file, **kwargs)

    def import_probabilities(self, equipment_pdf_file=None, equipment_event_file=None, **kwargs):
        if equipment_pdf_file is not None:
            # load PDF file
            return super().import_probabilities(equipment_pdf_file, **kwargs)

        # for EV event files, not PDFs
        assert equipment_event_file is not None
        df = load_csv(equipment_event_file, sub_folder=self.end_use)

        # update column formats
        if 'weekday' in df.columns:
            df['weekday'] = df['weekday'].astype(bool)
        if 'temperature' in df.columns:
            df['temperature'] = df['temperature'].astype(int)
        df['start_time'] = df['start_time'].astype(float)
        df['duration'] = df['duration'].astype(float)
        df['start_soc'] = df['start_soc'].astype(float)

        # sort by day_id and start_time
        df = df.sort_values(['day_id', 'start_time'])
        df = df.set_index('day_id')

        group_types = [col for col in ["temperature", "weekday"] if col in df.columns]
        if group_types:
            # group by weekday and/or temperature
            day_ids = df.groupby(group_types).groups
            day_ids = {key: val.unique() for key, val in day_ids.items()}
        else:
            # return all day ids without grouping
            day_ids = df.index
        return day_ids, df

    def generate_all_events(
        self,
        probabilities,
        event_data,
        eq_schedule,
        ambient_ev_temp=20,
        ev_daily_charge_ratio=None,
        **kwargs,
    ):
        # Get ratio of days with charging event if not provided
        if ev_daily_charge_ratio is None:
            if self.charging_level != "Level2":
                # Level 1 plug charges most days
                ev_daily_charge_ratio = 0.9
            elif self.capacity >= 70:
                # for large EVs (>~200 mi range), charge every 5 days, on average
                ev_daily_charge_ratio = 0.2
            elif self.capacity >= 35:
                # for smaller EVs (>~100 mi range), charge every 3 days, on average
                ev_daily_charge_ratio = 0.33
            else:
                # for the smallest EVs (mostly PHEV), charge every 2 days, on average
                ev_daily_charge_ratio = 0.5

        if eq_schedule is not None:
            # get average daily ambient temperature for generating events and round to nearest 5 C
            if 'Ambient Dry Bulb (C)' not in eq_schedule:
                raise OCHREException('EV model requires ambient dry bulb temperature in schedule.')
            temps_by_day = eq_schedule['Ambient Dry Bulb (C)']
            temps_by_day = temps_by_day.groupby(temps_by_day.index.date).mean()  # in C
            temps_by_day = ((temps_by_day / 5).round() * 5).astype(int).clip(lower=-20, upper=40)
        else:
            # use constant ambient temperature
            dates = pd.date_range(self.start_time.date(), (self.start_time + self.duration).date(),
                                  freq=dt.timedelta(days=1))
            temps_by_day = pd.Series([ambient_ev_temp] * len(dates), index=dates)

        temps_by_day.index = pd.to_datetime(temps_by_day.index)
        wdays = temps_by_day.index.weekday < 5
        keys = {'temperature': temps_by_day.values, 
                'weekday': wdays,
        }
        keys = [key for name, key in keys.items() if name in event_data.columns]
        if not keys:
            # randomly sample IDs
            day_ids = [np.random.choice(probabilities) for _ in range(len(temps_by_day))]
        else:
            # randomly sample IDs by weekday and/or temp
            keys = keys[0] if len(keys) == 1 else list(zip(*keys))
            day_ids = [np.random.choice(probabilities[key]) for key in keys]

        # assign charging events for some simulation days
        df_events = []
        for day_id, date in zip(day_ids, temps_by_day.index):
            if np.random.rand() <= ev_daily_charge_ratio:
                df = event_data.loc[event_data.index == day_id].reset_index()
                df['start_time'] = date + pd.to_timedelta(df['start_time'], unit='minute')
                df_events.append(df)
        df_events = pd.concat(df_events)
        df_events = df_events.reset_index(drop=True)

        # set end times
        df_events['end_time'] = df_events['start_time'] + pd.to_timedelta(df_events['duration'], unit='minute')

        # set maximum ending SOC
        df_events['start_soc'] /= 100
        max_soc = df_events['start_soc'] + self.max_power * EV_EFFICIENCY * df_events['duration'] / self.capacity
        df_events['end_soc'] = max_soc.clip(upper=1)

        # fix overlaps - if gap betwen 2 events < 1 hour, then move the end time of first event earlier
        new_day_event = df_events['day_id'] != df_events['day_id'].shift(-1)
        overlap_time = df_events['end_time'] + dt.timedelta(hours=1) - df_events['start_time'].shift(-1)
        bad_events = new_day_event & (overlap_time > dt.timedelta(0))
        if bad_events.any():
            df_events.loc[bad_events, 'end_time'] -= overlap_time

            # remove updated events if they last for less than 1 hour
            short_events = bad_events & (df_events['end_time'] - df_events['start_time'] < dt.timedelta(hours=1))
            df_events = df_events.loc[~ short_events]

        return df_events

    def start_event(self):
        # update SOC when event starts
        super().start_event()
        self.soc = self.event_schedule.loc[self.event_index, 'start_soc']

    def end_event(self):
        # reduce next starting SOC by the reduction in current ending SOC
        soc_reduction = self.event_schedule.loc[self.event_index, 'end_soc'] - self.soc
        super().end_event()

        next_start_soc = self.event_schedule.loc[self.event_index, 'start_soc'] - soc_reduction
        if next_start_soc < 0:
            # Unmet loads exist, set unmet loads for 1 time step only
            self.unmet_load = -next_start_soc
            self.event_schedule.loc[self.event_index, 'start_soc'] = 0
        else:
            self.event_schedule.loc[self.event_index, 'start_soc'] = min(next_start_soc, 1)

        # recalculate expected ending SOC
        next_event = self.event_schedule.loc[self.event_index]
        end_soc = next_event['start_soc'] + self.max_power * EV_EFFICIENCY * next_event['duration'] / self.capacity
        self.event_schedule.loc[self.event_index, 'end_soc'] = np.clip(end_soc, 0, 1)

    def update_external_control(self, control_signal):
        # Options for external control signals:
        # - P Setpoint: Directly sets power setpoint, in kW
        # - SOC: Solves for power setpoint to achieve desired SOC, unitless
        # - SOC Rate: Solves for power setpoint to achieve desired SOC Rate, in 1/hour
        # - Max Power: Updates maximum allowed power (in kW)
        #   - Note: Max Power will only be reset if it is in the schedule
        # - Max SOC: Maximum SOC limit for charging
        # - See additional controls in EventBasedLoad.update_external_control

        max_power = control_signal.get("Max Power")
        if max_power is not None:
            if "EV Max Power (kW)" in self.current_schedule:
                self.current_schedule["EV Max Power (kW)"] = max_power
            else:
                self.max_power_ctrl = max_power

        max_soc = control_signal.get("Max SOC")
        if max_soc is not None:
            if "EV Max SOC (-)" in self.current_schedule:
                self.current_schedule["EV Max SOC (-)"] = max_soc
            else:
                self.soc_max_ctrl = max_soc

        mode = super().update_external_control(control_signal)

        # update power setpoint directly or through SOC or SOC Rate
        if 'P Setpoint' in control_signal:
            setpoint = control_signal['P Setpoint']
        elif 'SOC' in control_signal:
            soc = control_signal.get("SOC")
            power_dc = (soc - self.soc) * self.capacity / self.time_res.total_seconds() * 3600
            setpoint = power_dc / EV_EFFICIENCY
        elif 'SOC Rate' in control_signal:
            power_dc = control_signal['SOC Rate'] * self.capacity  # in kW
            setpoint = power_dc / EV_EFFICIENCY
        else:
            setpoint = None

        if setpoint is not None:
            setpoint = max(setpoint, 0)
            if mode != 'On' and setpoint > 0:
                self.warn('Cannot set power when not parked.')
            elif self.enable_part_load:
                self.setpoint_power = setpoint
            else:
                # set to max power if setpoint > half of max
                self.setpoint_power = (
                    self.max_power if setpoint >= self.max_power / 2 else 0
                )

        return mode

    def update_internal_control(self):
        self.setpoint_power = None
        self.unmet_load = 0

        # update control parameters from schedule
        if "EV Max Power (kW)" in self.current_schedule:
            self.max_power_ctrl = self.current_schedule["EV Max Power (kW)"]
        if "EV Max SOC (-)" in self.current_schedule:
            self.soc_max_ctrl = self.current_schedule["EV Max SOC (-)"]

        return super().update_internal_control()

    def calculate_power_and_heat(self):
        # Note: this is copied from the battery model, but they are not linked at all
        if self.mode == 'Off':
            return super().calculate_power_and_heat()

        # force ac power within kw capacity and SOC limits, no discharge allowed
        hours = self.time_res.total_seconds() / 3600
        max_power = self.setpoint_power if self.setpoint_power is not None else self.max_power_ctrl
        ac_power = (self.soc_max_ctrl - self.soc) * self.capacity / hours / EV_EFFICIENCY
        ac_power = min(max(ac_power, 0), max_power)
        self.electric_kw = ac_power

        # update SOC for next time step, check with upper and lower bound of usable SOC
        dc_power = ac_power * EV_EFFICIENCY
        hours = self.time_res.total_seconds() / 3600
        self.next_soc = self.soc + dc_power * hours / self.capacity
        assert 1.001 >= self.next_soc >= -0.001  # small computational errors possible

        # update remaining time needed for charging, maximum final SOC, and unmet charging load
        # remaining_hours = (self.event_end - self.current_time).total_seconds() / 3600
        # max_final_soc = min(self.soc + self.max_power * EV_CHARGER_EFFICIENCY * remaining_hours / self.capacity, 1)

        # calculate internal battery power and power loss
        self.sensible_gain = ac_power - dc_power  # = power losses
        assert self.sensible_gain >= 0

    def generate_results(self):
        results = super().generate_results()

        if self.verbosity >= 3:
            results[f'{self.end_use} SOC (-)'] = self.soc
            results[f'{self.end_use} Parked'] = self.in_event
            results[f'{self.end_use} Unmet Load (kW)'] = self.unmet_load
        if self.verbosity >= 6:
            # results[f'{self.end_use} Setpoint Power (kW)'] = self.setpoint_power or 0
            results[f'{self.end_use} Start Time'] = self.event_start
            results[f'{self.end_use} End Time'] = self.event_end
        if self.verbosity >= 7:
            remaining_charge_minutes = (
                (1 - self.soc) * self.capacity / (self.max_power_ctrl * EV_EFFICIENCY) * 60
            )
            results[f'{self.end_use} Remaining Charge Time (min)'] = remaining_charge_minutes
        return results

    def update_results(self):
        current_results = super().update_results()

        # Update next time step SOC
        self.soc = self.next_soc

        return current_results


class ScheduledEV(ScheduledLoad):
    """
    Electric Vehicle as a scheduled load. Load profile must be defined by the equipment schedule file. This model is not
    controllable.
    """
    name = 'Scheduled EV'
    end_use = 'EV'
    zone_name = None

    def __init__(self, vehicle_num=None, equipment_schedule_file=None, **kwargs):
        if equipment_schedule_file is None:
            equipment_schedule_file = 'EV Profiles.csv'
            kwargs['schedule_rename_columns'] = {vehicle_num: 'Scheduled EV (kW)'}

        super().__init__(equipment_schedule_file=equipment_schedule_file, **kwargs)
