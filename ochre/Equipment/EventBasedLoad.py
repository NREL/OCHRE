import os
import pandas as pd
import datetime as dt
import numpy as np

from ochre.utils import load_csv
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
        self.all_events = None
        self.event_index = 0
        self.event_start = None
        self.event_end = None
        self.in_event = False

        super().__init__(**kwargs)

        if kwargs.get("verbosity", 1) >= 7 and self.output_path is not None:
            # save event schedule
            if self.main_sim_name:
                file_name = os.path.join(
                    self.output_path, f"{self.main_sim_name}_{self.name}_events.csv"
                )
            else:
                file_name = os.path.join(self.output_path, f"{self.name}_events.csv")
            self.all_events.to_csv(file_name, index=True)

    def extract_events(self, eq_powers: pd.DataFrame, random_offset: dt.timedelta | None=None, **kwargs):
        # get event information from time series schedule
        # assumes constant power for all events
        # get start times
        on = eq_powers.sum(axis=1) > 0
        start_times = on & (~on).shift(fill_value=True)
        event_counts = start_times.astype(int).cumsum()
        start_times = start_times.loc[start_times].index
        n_events = len(start_times)

        # get end times
        end_times = ~on & on.shift()
        end_times = end_times.loc[end_times].index
        if len(end_times) < n_events:
            # add final end time if on at end of schedule
            end_times = pd.concat([end_times, self.start_time + self.duration])
        if n_events != len(end_times):
            raise ValueError(f"Cannot parse events for {self.name}")
        
        # apply random offset for start and end times
        if random_offset is not None:
            offsets = np.random.random(n_events) * random_offset
            start_times += offsets
            end_times += offsets

        # get average power from each event
        powers = eq_powers.loc[on].groupby(event_counts.loc[on]).mean()
        assert len(powers) == len(start_times)

        # get total energy from each event, in kWh
        energy = powers["Power (kW)"] * (end_times - start_times).total_seconds() / 3600

        return pd.DataFrame(
            {
                "start_time": start_times,
                "end_time": end_times,
                "power": powers["Power (kW)"],
                # "power_gas": powers["Gas (therms/hour)"],
                "energy": energy,
            },
        ).reset_index()


    def generate_events(self, probabilities, event_data, **kwargs):
        # create event schedule with all event info
        raise NotImplementedError

    def import_from_event_list(self, equipment_event_file, **kwargs):
        raise NotImplementedError

    def import_from_pdf(self, equipment_pdf_file, n_header=1, n_index=1, **kwargs):
        # assumes each column is a pdf, uses pandas format for multi-index and multi-column csv files
        pdf = load_csv(
            equipment_pdf_file,
            sub_folder=self.name,
            header=list(range(n_header)),
            index_col=list(range(n_index)),
        )

        # convert pdf to cdf (and normalize)
        cdf = pdf.cumsum() / pdf.sum()

        # split cdfs with event data
        probabilities = cdf.reset_index(drop=True).to_dict(orient="series")
        event_data = cdf.index.to_frame().reset_index(drop=True)

        return probabilities, event_data


    def initialize_schedule(self, **kwargs):
        # Get power and gas columns from schedule, if they exist (copied from ScheduledLoad)
        schedule_cols = {
            f"{self.name} (kW)": "Power (kW)",
            f"{self.name} (therms/hour)": "Gas (therms/hour)",
        }
        optional_inputs = list(schedule_cols.keys())
        eq_powers = super().initialize_schedule(optional_inputs=optional_inputs, **kwargs)
        eq_powers = eq_powers.rename(columns=schedule_cols)

        # set schedule columns to zero if month multiplier exists and is zero (for ceiling fans)
        multipliers = kwargs.get("month_multipliers", [])
        zero_months = [i for i, m in enumerate(multipliers) if m == 0]
        if zero_months:
            eq_powers.loc[eq_powers.index.month.isin(zero_months), :] = 0

        # generate event schedule
        if not eq_powers.empty:
            self.all_events = self.extract_events(eq_powers, **kwargs)
        elif "equipment_event_file" in kwargs:
            probabilities, event_data = self.import_from_event_list(**kwargs)
            self.all_events = self.generate_events(probabilities, event_data, **kwargs)
        elif "equipment_pdf_file" in kwargs:
            probabilities, event_data = self.import_from_pdf(**kwargs)
            self.all_events = self.generate_events(probabilities, event_data, **kwargs)
        else:
            raise IOError(
                f"Must specify {self.name} schedule, or provide an `equipment_event_file` or `equipment_pdf_file`"
            )

        # set start and end times to be on the simulation time
        self.all_events["start_time"] = self.all_events["start_time"].dt.round(self.time_res)
        self.all_events["end_time"] = self.all_events["end_time"].dt.round(self.time_res)
        self.event_start = self.all_events.loc[self.event_index, "start_time"]
        self.event_end = self.all_events.loc[self.event_index, "end_time"]

        # check that end time is at or after start time, and events do not overlap
        negative_times = self.all_events["end_time"] < self.all_events["start_time"]
        if negative_times.any():
            bad_event = self.all_events.loc[negative_times.idxmax()]
            raise ValueError(
                f"{self.name} has event with end time before start time. "
                f"Event details: \n{bad_event}"
            )
        overlap = self.all_events["start_time"] < self.all_events["end_time"].shift()
        if overlap.any():
            bad_index = overlap.idxmax()
            bad_events = self.all_events.loc[bad_index - 1 : bad_index + 1]
            raise ValueError(f"{self.name} event overlap. Event details: \n{bad_events}")

        return eq_powers

    def reset_time(self, start_time=None, **kwargs):
        super().reset_time(start_time, **kwargs)
        self.event_index = 0
        self.in_event = False
        self.event_start = self.all_events.loc[self.event_index, "start_time"]
        self.event_end = self.all_events.loc[self.event_index, "end_time"]

    def start_event(self):
        # optional function that runs when starting an event
        self.in_event = True

    def end_event(self):
        # function that runs when ending an event
        self.in_event = False
        self.event_index += 1

        if self.event_index == len(self.all_events):
            # no more events - reset to last event index and move start/end times to the end of the simulation
            self.event_index -= 1
            self.all_events.loc[self.event_index, "start_time"] = pd.Timestamp.max
            self.all_events.loc[self.event_index, "end_time"] = pd.Timestamp.max

        self.event_start = self.all_events.loc[self.event_index, "start_time"]
        self.event_end = self.all_events.loc[self.event_index, "end_time"]

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

    def update_external_control(self, control_signal):
        # If Delay=dt.timedelta, extend start time by that time
        # If Delay=True, delay for self.time_res
        # If Delay=int, delay for int * self.time_res
        if "Delay" in control_signal:
            delay = control_signal["Delay"]

            if delay and self.mode == "On":
                self.warn("Ignoring delay signal, event has already started.")
                delay = False
            if isinstance(delay, (int, bool)):
                delay = self.time_res * delay
            if not isinstance(delay, dt.timedelta):
                raise TypeError(f"Unknown delay for {self.name}: {delay}")

            if delay:
                if self.delay_event_end:
                    self.event_end += delay
                else:
                    # ensure that start time doesn't exceed end time
                    if self.event_start + delay > self.event_end:
                        self.warn("Event is delayed beyond event end time. Ignoring event.")
                        delay = self.event_end - self.event_start

                self.event_start += delay

        return self.update_internal_control()

    def update_internal_control(self):
        if self.current_time < self.event_start:
            # waiting for next event to start
            return "Off"
        elif self.current_time < self.event_end:
            if self.mode == "Off":
                self.start_event()
            return "On"
        else:
            # event has ended, move to next event
            self.end_event()
            return "Off"

    def calculate_power_and_heat(self):
        if self.mode == "On":
            power = self.all_events.loc[self.event_index, "power"]
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
        if self.event_duration % kwargs["time_res"] != dt.timedelta(0):
            new_duration = self.event_duration // self.time_res * self.time_res
            self.warn(
                "Changing default duration ({}) to align with simulation time."
                "New duration: {}".format(self.event_duration, new_duration)
            )
            self.event_duration = new_duration

        self.max_power = max_power  # in kW

        super().__init__(**kwargs)

    def generate_events(self, probabilities, event_data, **kwargs):
        # number of events = days of simulation, 1 event per day, plus 1 to be inclusive of end day
        end_time = self.start_time + self.duration - self.time_res
        dates = pd.date_range(self.start_time.date(), end_time.date(), freq=dt.timedelta(days=1))
        n_events = len(dates)

        # randomly pick parameters for next event
        r = np.random.random(n_events)
        idx = probabilities["Density"].searchsorted(r)
        df_events = event_data.iloc[idx].reset_index(drop=True)

        # add day to start time, plus random minute
        df_events["start_time"] = dates + pd.to_timedelta(df_events["start_hour"], unit="hour")
        df_events["start_time"] += pd.Series(np.random.random(n_events) * dt.timedelta(hours=1))

        # add end time, power
        df_events["end_time"] = df_events["start_time"] + self.event_duration
        df_events["power"] = self.max_power

        return df_events
