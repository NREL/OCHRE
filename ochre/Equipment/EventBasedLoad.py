import os
import pandas as pd
import datetime as dt
import numpy as np

from ochre.utils import OCHREException, load_csv
from ochre.Equipment import Equipment


class EventBasedLoad(Equipment):
    """
    Equipment with a stochastic, event-based schedule. By default, all events are generated during initialization.
    A probability density function or an event list is required to generate the event information, including start time
    and duration.

    By default, event-based equipment can be externally controlled by delaying the event start time. For now, events can
    be delayed indefinitely.
    """
    delay_event_end = True

    def __init__(self, **kwargs):
        # event data is initialized in initialize_schedule
        self.event_schedule = None
        self.event_index = 0
        self.event_start = None
        self.event_end = None
        self.in_event = False

        super().__init__(**kwargs)

        if kwargs.get('verbosity', 1) >= 7 and self.output_path is not None:
            # save event schedule
            if self.main_sim_name:
                file_name = os.path.join(self.output_path, f'{self.main_sim_name}_{self.name}_events.csv')
            else:
                file_name = os.path.join(self.output_path, f'{self.name}_events.csv')
            self.event_schedule.to_csv(file_name, index=True)

    def import_probabilities(self, equipment_pdf_file=None, equipment_event_file=None, n_header=1, n_index=1, **kwargs):
        if equipment_pdf_file is not None:
            # assumes each column is a pdf, uses pandas format for multi-index and multi-column csv files
            pdf = load_csv(equipment_pdf_file, sub_folder=self.name, header=list(range(n_header)),
                            index_col=list(range(n_index)))

            # convert pdf to cdf (and normalize)
            cdf = pdf.cumsum() / pdf.sum()

            # split cdfs with event data
            probabilities = cdf.reset_index(drop=True).to_dict(orient='series')
            event_data = cdf.index.to_frame().reset_index(drop=True)
            return probabilities, event_data

        elif equipment_event_file is not None:
            raise NotImplementedError
        else:
            raise OCHREException('Must specify a PDF or Event file for {}.'.format(self.name))

    def initialize_schedule(self, **kwargs):
        schedule = super().initialize_schedule(**kwargs)

        # import pdf, convert to cumulative density function (cdf) and normalize to 1
        probabilities, event_data = self.import_probabilities(**kwargs)

        # generate all events
        self.event_schedule = self.generate_all_events(probabilities, event_data, schedule, **kwargs)
        self.event_schedule = self.event_schedule.reset_index(drop=True)

        # for start and end times to be on the simulation time
        self.event_schedule.loc[:, 'start_time'] = self.event_schedule.loc[:, 'start_time'].dt.round(self.time_res)
        self.event_schedule.loc[:, 'end_time'] = self.event_schedule.loc[:, 'end_time'].dt.round(self.time_res)
        self.event_start = self.event_schedule.loc[self.event_index, 'start_time']
        self.event_end = self.event_schedule.loc[self.event_index, 'end_time']

        # check that end time is at or after start time, and events do not overlap
        negative_times = self.event_schedule['end_time'] - self.event_schedule['start_time'] < dt.timedelta(0)
        if negative_times.any():
            bad_event = self.event_schedule.loc[negative_times.idxmax()]
            raise OCHREException('{} has event with end time before start time. '
                                     'Event details: \n{}'.format(self.name, bad_event))
        overlap = (self.event_schedule['start_time'] - self.event_schedule['end_time'].shift()) < dt.timedelta(0)
        if overlap.any():
            bad_index = overlap.idxmax()
            bad_events = self.event_schedule.loc[bad_index - 1: bad_index + 1]
            raise OCHREException(f'{self.name} event overlap. Event details: \n{bad_events}')

        return schedule

    def reset_time(self, start_time=None, **kwargs):
        super().reset_time(start_time, **kwargs)
        self.event_index = 0
        self.in_event = False
        self.event_start = self.event_schedule.loc[self.event_index, 'start_time']
        self.event_end = self.event_schedule.loc[self.event_index, 'end_time']

    def generate_all_events(self, probabilities, event_data, eq_schedule, **kwargs):
        # create event schedule with all event info
        raise NotImplementedError

    def start_event(self):
        # optional function that runs when starting an event
        self.in_event = True

    def end_event(self):
        # function that runs when ending an event
        self.in_event = False
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

    def parse_control_signal(self, control_signal):
        # If Delay=dt.timedelta, extend start time by that time
        # If Delay=True, delay for self.time_res
        # If Delay=int, delay for int * self.time_res
        if 'Delay' in control_signal:
            delay = control_signal['Delay']

            if delay and self.on_frac:
                self.warn('Ignoring delay signal, event has already started.')
                delay = False
            if isinstance(delay, (int, bool)):
                delay = self.time_res * delay
            if not isinstance(delay, dt.timedelta):
                raise OCHREException(f'Unknown delay for {self.name}: {delay}')

            if delay:
                self.event_start += delay

                if self.delay_event_end:
                    self.event_end += delay
                else:
                    # ensure that start time doesn't exceed end time
                    if self.event_start > self.event_end:
                        self.warn('Event is delayed beyond event end time. Ignoring event.')
                        self.event_start = self.event_end

    def run_internal_control(self):
        if self.current_time < self.event_start:
            # waiting for next event to start
            self.on_frac_new = 0
        elif self.current_time < self.event_end:
            if not self.on_frac:
                self.start_event()
            self.on_frac_new = 1
        else:
            # event has ended, move to next event
            self.end_event()
            self.on_frac_new = 0

    def calculate_power_and_heat(self):
        if self.on_frac_new:
            power = self.event_schedule.loc[self.event_index, 'power']
        else:
            power = 0

        self.electric_kw = power


class DailyLoad(EventBasedLoad):
    """
    Test equipment with simple event schedule. Assumes 1 event per day. PDF defines the start time with hourly
    resolution. Uses the same PDF for all days.
    """

    def __init__(self, max_power=1, event_duration=dt.timedelta(hours=1), **kwargs):
        self.event_duration = event_duration  # as timedelta, if None, duration should be in cdf
        if self.event_duration % kwargs['time_res'] != dt.timedelta(0):
            new_duration = self.event_duration // self.time_res * self.time_res
            self.warn('Changing default duration ({}) to align with simulation time.'
                      'New duration: {}'.format(self.event_duration, new_duration))
            self.event_duration = new_duration

        self.max_power = max_power  # in kW

        super().__init__(**kwargs)

    def generate_all_events(self, probabilities, event_data, eq_schedule, **kwargs):
        # number of events = days of simulation, 1 event per day, plus 1 to be inclusive of end day
        end_time = self.start_time + self.duration - self.time_res
        dates = pd.date_range(self.start_time.date(), end_time.date(), freq=dt.timedelta(days=1))
        n_events = len(dates)

        # randomly pick parameters for next event
        r = np.random.random(n_events)
        idx = probabilities['Density'].searchsorted(r)
        df_events = event_data.iloc[idx].reset_index(drop=True)

        # add day to start time, plus random minute
        df_events['start_time'] = dates + pd.to_timedelta(df_events['start_hour'], unit='hour')
        df_events['start_time'] += pd.Series(np.random.random(n_events) * dt.timedelta(hours=1))

        # add end time, power
        df_events['end_time'] = df_events['start_time'] + self.event_duration
        df_events['power'] = self.max_power

        return df_events
