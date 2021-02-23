import os
import pandas as pd
import datetime as dt
import numpy as np

from ochre.FileIO import default_input_path, save_to_csv
from ochre.Equipment import Equipment, EquipmentException


class EventBasedLoad(Equipment):
    """
    Equipment with a stochastic, event-based schedule. By default, all events are generated during initialization.
    A probability density function or an event list is required to generate the event information, including start time
    and duration.

    By default, event-based equipment can be externally controlled by delaying the event start time. For now, events can
    be delayed indefinitely.
    """
    name = 'Event-Based Load'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # import pdf, convert to cumulative density function (cdf) and normalize to 1
        probabilities, event_data = self.import_probabilities(**kwargs)

        # self.power = 0  # in kW

        # assume heat gains of 0
        self.sensible_gain_fraction = 0  # unitless, gain = gain_fraction * power
        self.latent_gain_fraction = 0  # unitless, gain = gain_fraction * power

        # generate all events
        self.event_schedule = self.generate_all_events(probabilities, event_data, **kwargs)
        self.event_schedule = self.event_schedule.reset_index(drop=True)
        self.event_index = 0

        # for start and end times to be on the simulation time
        self.event_schedule.loc[:, 'start_time'] = self.event_schedule.loc[:, 'start_time'].dt.round(self.time_res)
        self.event_schedule.loc[:, 'end_time'] = self.event_schedule.loc[:, 'end_time'].dt.round(self.time_res)
        self.event_start = self.event_schedule.loc[self.event_index, 'start_time']
        self.event_end = self.event_schedule.loc[self.event_index, 'end_time']

        # check that end time is at or after start time, and events do not overlap
        negative_times = self.event_schedule['end_time'] - self.event_schedule['start_time'] < dt.timedelta(0)
        if negative_times.any():
            bad_event = self.event_schedule.loc[negative_times.idxmax()]
            raise EquipmentException('{} has event with end time before start time. '
                                     'Event details: \n{}'.format(self.name, bad_event))
        overlap = (self.event_schedule['start_time'] - self.event_schedule['end_time'].shift()) < dt.timedelta(0)
        if overlap.any():
            bad_index = overlap.idxmax()
            bad_events = self.event_schedule.loc[bad_index - 1: bad_index + 1]
            raise EquipmentException('{} event overlap. Event details: \n{}'.format(self.name, bad_events))

        if kwargs.get('verbosity', 1) >= 7:
            # save event schedule
            save_to_csv(self.event_schedule, '{}_{}_events.csv'.format(kwargs['house_name'], self.name), **kwargs)

    def import_probabilities(self, equipment_pdf_file=None, equipment_event_file=None, n_header=1, n_index=1,
                             input_path=default_input_path, **kwargs):
        if equipment_pdf_file is not None:
            # assumes each column is a pdf, uses pandas format for multi-index and multi-column csv files
            if not os.path.isabs(equipment_pdf_file):
                equipment_pdf_file = os.path.join(input_path, self.name, equipment_pdf_file)
            pdf = pd.read_csv(equipment_pdf_file, header=list(range(n_header)), index_col=list(range(n_index)))

            # convert pdf to cdf (and normalize)
            cdf = pdf.cumsum() / pdf.sum()

            # split cdfs with event data
            probabilities = cdf.reset_index(drop=True).to_dict(orient='series')
            event_data = cdf.index.to_frame().reset_index(drop=True)
            return probabilities, event_data

        elif equipment_event_file is not None:
            raise NotImplementedError
        else:
            raise EquipmentException('Must specify a PDF or Event file for {}.'.format(self.name))

    def reset_time(self):
        super().reset_time()
        self.event_index = 0
        self.event_start = self.event_schedule.loc[self.event_index, 'start_time']
        self.event_end = self.event_schedule.loc[self.event_index, 'end_time']

    def generate_all_events(self, probabilities, event_data, **kwargs):
        # create event schedule with all event info
        raise NotImplementedError

    def start_event(self):
        # optional function that runs when starting an event
        pass

    def end_event(self):
        # function that runs when ending an event
        self.event_index += 1

        if self.event_index == len(self.event_schedule):
            # no more events - reset to last event index and move start/end times to the end of the simulation
            self.event_index -= 1
            self.event_schedule.loc[self.event_index, 'start_time'] = pd.Timestamp.max
            self.event_schedule.loc[self.event_index, 'end_time'] = pd.Timestamp.max

        self.event_start = self.event_schedule.loc[self.event_index, 'start_time']
        self.event_end = self.event_schedule.loc[self.event_index, 'end_time']

    # deprecated functions
    # def generate_random_event_data(self, key):
    #     # randomly pick parameters for next event
    #     r = np.random.random()
    #     idx = self.probabilities[key].searchsorted(r)
    #     return self.event_data.iloc[idx].to_dict()
    #
    # def generate_event(self):
    #     # function to set next event start and end time
    #     raise NotImplementedError

    def update_external_control(self, schedule, ext_control_args):
        # If Delay=dt.timedelta, extend start time by that time
        # If Delay=True, delay for self.ext_time_res
        # If Delay=int, delay for int * self.ext_time_res
        delay = ext_control_args.get('Delay', False)

        if delay and self.mode == 'On':
            self.warn('Ignoring delay signal, event has already started.')
            delay = False
        if isinstance(delay, (int, bool)):
            delay = self.time_res * delay
        if not isinstance(delay, dt.timedelta):
            raise EquipmentException('Unknown delay for {}: {}'.format(self.name, delay))

        if delay:
            # self.event_schedule.loc[self.event_index, 'start_time'] += delay
            # self.event_schedule.loc[self.event_index, 'end_time'] += delay
            self.event_start += delay
            self.event_end += delay

        return self.update_internal_control(schedule)

    def update_internal_control(self, schedule):
        if self.current_time < self.event_start:
            # waiting for next event to start
            return 'Off'
        elif self.current_time < self.event_end:
            if self.mode == 'Off':
                self.start_event()
            return 'On'
        else:
            # event has ended, move to next event
            self.end_event()
            return 'Off'

    def calculate_power_and_heat(self, schedule):
        if self.mode == 'On':
            power = self.event_schedule.loc[self.event_index, 'power']
        else:
            power = 0

        self.electric_kw = power
        self.sensible_gain = power * self.sensible_gain_fraction
        self.latent_gain = power * self.latent_gain_fraction


class DailyLoad(EventBasedLoad):
    """
    Test equipment with simple event schedule. Assumes 1 event per day. PDF defines the start time with hourly
    resolution. Uses the same PDF for all days.
    """

    def __init__(self, max_power=1, default_duration=dt.timedelta(hours=1), **kwargs):
        self.default_duration = default_duration  # as timedelta, if None, duration should be in cdf
        if self.default_duration % kwargs['time_res'] != dt.timedelta(0):
            new_duration = self.default_duration // self.time_res * self.time_res
            self.warn('Changing default duration ({}) to align with simulation time.'
                      'New duration: {}'.format(self.default_duration, new_duration))
            self.default_duration = new_duration

        self.max_power = max_power  # in kW

        super().__init__(**kwargs)

    def generate_all_events(self, probabilities, event_data, **kwargs):
        # number of events = days of simulation, 1 event per day, plus 1 to be inclusive of end day
        dates = pd.date_range(self.start_time.date(), kwargs['end_time'].date(), freq=dt.timedelta(days=1))
        n_events = len(dates)

        # randomly pick parameters for next event
        r = np.random.random(n_events)
        idx = probabilities['Density'].searchsorted(r)
        df_events = event_data.iloc[idx].reset_index(drop=True)

        # add day to start time, plus random minute
        df_events['start_time'] = dates + pd.to_timedelta(df_events['start_hour'], unit='hour')
        df_events['start_time'] += pd.Series(np.random.random(n_events) * dt.timedelta(hours=1))

        # add end time, power
        df_events['end_time'] = df_events['start_time'] + self.default_duration
        df_events['power'] = self.max_power

        return df_events
