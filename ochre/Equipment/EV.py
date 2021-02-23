import os
import numpy as np
import datetime as dt
import pandas as pd

from ochre.FileIO import default_input_path
from ochre.Equipment import EventBasedLoad, ScheduledLoad, EquipmentException

EV_FUEL_ECONOMY = 3.076923076923077  # miles per kWh, used to calculate capacity
EV_MAX_POWER = {  # max AC charging power, by vehicle type and charge level
    ('PHEV', 'Level0'): 1.4,  # For testing only
    ('PHEV', 'Level1'): 1.4,
    ('PHEV', 'Level2'): 3.6,
    ('BEV', 'Level1'): 1.4,
    ('BEV', 'Level2'): 9.0,
}


class ElectricVehicle(EventBasedLoad):
    """
    Electric Vehicle model using an event-based schedule. The schedule is based on residential charging data from
    EVI-Pro. Uses a joint probability density function (PDF) to get the arrival time, arrival SOC, and parking duration
    of the EV charging event. Forces an overnight charging event to occur every night.
    """
    name = 'EV'
    end_use = 'EV'

    def __init__(self, vehicle_type, charging_level, capacity=None, mileage=None, enable_part_load=None, **kwargs):
        # get EV battery capacity and mileage
        if capacity is None and mileage is None:
            raise EquipmentException('Must specify capacity or mileage for {}'.format(self.name))
        elif capacity is not None:
            self.capacity = capacity  # in kWh
            mileage = self.capacity * EV_FUEL_ECONOMY  # in mi
        else:
            # determine capacity using mileage
            self.capacity = mileage / EV_FUEL_ECONOMY  # in kWh

        # get charging level and set option for part load setpoints
        charging_level = charging_level.replace(' ', '')
        if str(charging_level) not in ['Level0', 'Level1', 'Level2']:
            raise EquipmentException('Unknown vehicle type for {}: {}'.format(self.name, charging_level))
        self.charging_level = str(charging_level)
        if enable_part_load is None:
            enable_part_load = self.charging_level == 'Level2'
        self.enable_part_load = enable_part_load

        # get vehicle number (1-4) based on type and mileage
        if vehicle_type == 'PHEV':
            vehicle_num = 1 if mileage < 35 else 2
        elif vehicle_type == 'BEV':
            vehicle_num = 3 if mileage < 175 else 4
        else:
            raise EquipmentException('Unknown vehicle type for {}: {}'.format(self.name, vehicle_type))
        self.vehicle_type = vehicle_type

        # get average daily ambient temperature for generating events
        temps = kwargs['full_schedule']['ambient_dry_bulb']
        self.temps_by_day = temps.groupby(temps.index.date).mean()  # in C
        # round to nearest 5 C
        self.temps_by_day = ((self.temps_by_day / 5).round() * 5).astype(int).clip(lower=-20, upper=40)

        # charging model
        self.max_power = EV_MAX_POWER[(self.vehicle_type, self.charging_level)]
        self.setpoint_power = None
        self.soc = 0  # unitless
        self.efficiency = 0.9  # unitless, based on EVI-pro assumption
        self.unmet_load = 0  # lost charging from delays, in kW

        # initialize events
        equipment_event_file = 'pdf_Veh{}_{}.csv'.format(vehicle_num, self.charging_level)
        super().__init__(equipment_event_file=equipment_event_file, **kwargs)

    def import_probabilities(self, equipment_pdf_file=None, equipment_event_file=None, input_path=default_input_path,
                             **kwargs):
        if equipment_pdf_file is not None:
            # load PDF file
            return super().import_probabilities(equipment_pdf_file, input_path=input_path, **kwargs)

        # for EV event files, not PDFs
        assert equipment_event_file is not None

        if not os.path.isabs(equipment_event_file):
            equipment_event_file = os.path.join(input_path, self.name, equipment_event_file)
        df = pd.read_csv(equipment_event_file)

        # update column formats
        df['weekday'] = df['weekday'].astype(bool)
        df['temperature'] = df['temperature'].astype(int)
        df['start_time'] = df['start_time'].astype(float)
        df['duration'] = df['duration'].astype(float)
        df['start_soc'] = df['start_soc'].astype(float)

        # sort by day_id and start_time
        df = df.sort_values(['day_id', 'start_time'])
        df = df.set_index('day_id')

        # group by weekday and temperature
        day_ids = df.groupby(['temperature', 'weekday']).groups
        day_ids = {key: val.unique() for key, val in day_ids.items()}
        return day_ids, df

    def generate_all_events(self, probabilities, event_data, **kwargs):
        # sample once per day
        dates = pd.to_datetime(self.temps_by_day.index)
        wdays = dates.weekday < 5

        # randomly sample IDs by weekday and temp
        keys = list(zip(self.temps_by_day.values, wdays))
        day_ids = [np.random.choice(probabilities[key]) for key in keys]

        # get event info and add date
        df_events = []
        for day_id, date in zip(day_ids, dates):
            df = event_data.loc[event_data.index == day_id].reset_index()
            df['start_time'] = date + pd.to_timedelta(df['start_time'], unit='minute')
            df_events.append(df)
        df_events = pd.concat(df_events)
        df_events = df_events.reset_index(drop=True)

        # set end times
        df_events['end_time'] = df_events['start_time'] + pd.to_timedelta(df_events['duration'], unit='minute')

        # set maximum ending SOC
        df_events['start_soc'] /= 100
        max_soc = df_events['start_soc'] + self.max_power * self.efficiency * df_events['duration'] / self.capacity
        df_events['end_soc'] = max_soc.clip(upper=1)

        # fix overlaps - if gap betwen 2 events < 1 hour, then move the end time of first event earlier
        new_day_event = df_events['day_id'] != df_events['day_id'].shift(-1)
        overlap_time = df_events['end_time'] + dt.timedelta(hours=1) - df_events['start_time'].shift(-1)
        bad_events = new_day_event & (overlap_time > dt.timedelta(0))
        if bad_events.any():
            df_events.loc[bad_events, 'end_time'] -= overlap_time

        return df_events

    def start_event(self):
        # update SOC when event starts
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
        end_soc = next_event['start_soc'] + self.max_power * self.efficiency * next_event['duration'] / self.capacity
        self.event_schedule.loc[self.event_index, 'end_soc'] = np.clip(end_soc, 0, 1)

    def update_external_control(self, schedule, ext_control_args):
        # override external control: only update start time, not end time
        delay = ext_control_args.get('Delay', False)

        if delay and self.mode == 'On':
            delay = False

        if delay:
            if isinstance(delay, (int, bool)):
                delay = self.time_res * delay
            if not isinstance(delay, dt.timedelta):
                raise EquipmentException('Unknown delay for {}: {}'.format(self.name, delay))

            # ensure that start time doesn't exceed end time
            if self.event_start + delay > self.event_end:
                self.warn('Event is delayed beyond event end time. Ignoring event.')
                delay = self.event_end - self.event_start

            self.event_start += delay

        mode = self.update_internal_control(schedule)

        # update with power setpoint
        setpoint = ext_control_args.get('P Setpoint')
        if setpoint is not None:
            if mode != 'On' and setpoint > 0:
                self.warn('Cannot set power when not parked.')
            elif self.enable_part_load:
                self.setpoint_power = setpoint
            else:
                # set to max power if setpoint > half of max
                self.setpoint_power = self.max_power if setpoint >= self.max_power / 2 else 0

        return mode

    def update_internal_control(self, schedule):
        self.setpoint_power = None
        self.unmet_load = 0
        return super().update_internal_control(schedule)

    def calculate_power_and_heat(self, schedule):
        # Note: this is copied from the battery model, but they are not linked at all
        if self.mode == 'Off':
            return super().calculate_power_and_heat(schedule)

        # force ac power within kw capacity and SOC limits, no discharge allowed
        hours = self.time_res.total_seconds() / 3600
        ac_power = (1 - self.soc) * self.capacity / hours / self.efficiency
        if self.setpoint_power is not None:
            ac_power = np.clip(ac_power, 0, self.setpoint_power)
        else:
            ac_power = np.clip(ac_power, 0, self.max_power)
        self.electric_kw = ac_power

        # calculate internal battery power and power loss
        dc_power = ac_power * self.efficiency
        self.sensible_gain = ac_power - dc_power  # = power losses
        assert self.sensible_gain >= 0

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)

        if to_ext:
            remaining_charge_minutes = (1 - self.soc) * self.capacity / (self.max_power * self.efficiency) * 60
            results.update({'Is Parked': self.mode == 'On',
                            'Start Time': str(self.event_start),
                            'End Time': str(self.event_end),
                            'SOC': self.soc,
                            'Remaining Charge Time': remaining_charge_minutes,
                            })
        else:
            if verbosity >= 3:
                results.update({self.name + ' SOC (-)': self.soc,
                                self.name + ' Unmet Load (kW)': self.unmet_load,
                                })
            if verbosity >= 6:
                results.update({
                    # self.name + ' Setpoint Power (kW)': self.setpoint_power if self.setpoint_power is not None else 0,
                    self.name + ' Start Time': self.event_start,
                    self.name + ' End Time': self.event_end,
                })
        return results

    def update_model(self, schedule):
        super().update_model(schedule)

        # update SOC, check with upper and lower bound of usable SOC
        dc_power = self.electric_kw * self.efficiency
        hours = self.time_res.total_seconds() / 3600
        self.soc += dc_power * hours / self.capacity
        assert 1.001 >= self.soc >= -0.001  # small computational errors possible

        # update remaining time needed for charging, maximum final SOC, and unmet charging load
        # remaining_hours = (self.event_end - self.current_time).total_seconds() / 3600
        # max_final_soc = min(self.soc + self.max_power * self.efficiency * remaining_hours / self.capacity, 1)


class ScheduledEV(ScheduledLoad):
    """
    Electric Vehicle as a scheduled load. Load profile must be defined by the equipment schedule file. This model is not
    controllable.
    """
    name = 'Scheduled EV'
    end_use = 'EV'

    def __init__(self, vehicle_num=None, zone=None, equipment_schedule_file=None, **kwargs):
        if equipment_schedule_file is None:
            equipment_schedule_file = 'EV Profiles.csv'
            kwargs['val_col'] = vehicle_num
            kwargs['schedule_scale_factor'] = 1 / 1000  # W to kW

        super().__init__(zone=zone, equipment_schedule_file=equipment_schedule_file, **kwargs)
