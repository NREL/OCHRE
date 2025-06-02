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
        self.p_setpoint = 0
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

    def extract_events(
        self, eq_powers: pd.DataFrame, random_offset: dt.timedelta | None = None, **kwargs
    ):
        # get event information from time series schedule
        # assumes constant power for all events
        # get start times
        # TODO: could try to determine start/end times from power in
        #   first/last time step of each event
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
            end_times = end_times.append(pd.DatetimeIndex([self.start_time + self.duration]))
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

        return pd.DataFrame(
            {
                "start_time": start_times,
                "end_time": end_times,
                "power": powers["Power (kW)"],
                # "power_gas": powers["Gas (therms/hour)"],
            },
        ).reset_index(drop=True)

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

    def initialize_schedule(self, event_schedule=None, **kwargs):
        # Get power and gas columns from time-series schedule, if they exist (copied from ScheduledLoad)
        schedule_cols = {
            f"{self.name} Electric Power (kW)": "Power (kW)",
            f"{self.name} Gas Power (therms/hour)": "Gas (therms/hour)",
        }
        optional_inputs = list(schedule_cols.keys())
        ts_schedule = super().initialize_schedule(optional_inputs=optional_inputs, **kwargs)
        ts_schedule = ts_schedule.rename(columns=schedule_cols)

        # set schedule columns to zero if month multiplier exists and is zero (for ceiling fans)
        multipliers = kwargs.get("month_multipliers", [])
        zero_months = [i for i, m in enumerate(multipliers) if m == 0]
        if zero_months:
            ts_schedule.loc[ts_schedule.index.month.isin(zero_months), :] = 0

        # generate event schedule
        if event_schedule is not None:
            self.all_events = event_schedule
        elif not ts_schedule.empty:
            self.all_events = self.extract_events(ts_schedule, **kwargs)
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

        # if no events, add event at end time
        if self.all_events.empty:
            self.all_events.loc[0] = {
                "start_time": self.start_time + self.duration,
                "end_time": self.start_time + self.duration,
                "power": 0,
            }

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

        # add duration and total energy from each event, in kWh
        self.all_events["duration"] = self.all_events["end_time"] - self.all_events["start_time"]
        if "power" in self.all_events.columns:
            self.all_events["energy"] = (
                self.all_events["power"] * self.all_events["duration"].dt.total_seconds() / 3600
            )

        return ts_schedule

    def reset_time(self, start_time=None, **kwargs):
        if self.in_event:
            self.end_event()

        super().reset_time(start_time, **kwargs)

        # get next event index based on new current_time
        future_events = self.all_events["end_time"] > self.current_time
        self.event_index = future_events.idxmax()

        # update event data
        self.event_start = self.all_events.loc[self.event_index, "start_time"]
        self.event_end = self.all_events.loc[self.event_index, "end_time"]
        if self.current_time > self.event_start:
            self.start_event()

    def start_event(self):
        # optional function that runs when starting an event
        self.in_event = True
        if "power" in self.all_events.columns:
            self.p_setpoint = self.all_events.loc[self.event_index, "power"]

    def end_event(self):
        # function that runs when ending an event
        self.in_event = False
        self.p_setpoint = 0
        self.event_index += 1

        if self.event_index == len(self.all_events):
            # no more events - reset to last event index and move start/end times to the end of the simulation
            self.event_index -= 1
            self.all_events.loc[self.event_index, "start_time"] = pd.Timestamp.max
            self.all_events.loc[self.event_index, "end_time"] = pd.Timestamp.max

        self.event_start = self.all_events.loc[self.event_index, "start_time"]
        self.event_end = self.all_events.loc[self.event_index, "end_time"]

    def update_external_control(self, control_signal):
        # Control options for changing power:
        #  - Delay: delays event start time, can be timedelta, boolean, or int
        #    - If True, delays for self.time_res. If int, delays for int * self.time_res
        #  - Load Fraction: gets multiplied by power from schedule, unitless (applied to electric AND gas)
        #  - P Setpoint: overwrites electric power from schedule, in kW
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

        mode = self.update_internal_control()

        p_set_ext = control_signal.get("P Setpoint")
        if p_set_ext is not None:
            self.p_setpoint = p_set_ext

        # If load fraction = 0, force off
        load_fraction = control_signal.get("Load Fraction", 1)
        if load_fraction == 0:
            return "Off"
        elif load_fraction != 1:
            raise IOError(f"{self.name} can't handle non-integer load fractions")

        return mode

    def update_internal_control(self):
        if self.current_time < self.event_start:
            # waiting for next event to start
            return "Off"
        elif self.current_time < self.event_end:
            if not self.in_event:
                self.start_event()
            return "On"
        else:
            # event has ended, move to next event
            self.end_event()
            return "Off"

    def calculate_power_and_heat(self):
        self.electric_kw = self.p_setpoint if self.mode == "On" else 0

        super().calculate_power_and_heat()


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


class EventDataLoad(EventBasedLoad):
    """
    Equipment with event-based time-series data. Each event type has a unique
    schedule, and one or more event types are allowed. Schedules can be taken
    from defaults or provided as an input. If more than one event type is
    provided, one will be chosen randomly for each event.
    """

    def __init__(self, **kwargs):
        self.event_ts_data = None
        self.event_schedule = None

        super().__init__(**kwargs)

    def add_event_types(self):
        # assign an event type to each event
        #  - maintain total number of events, event duration, and total energy
        #  - minimize differences in duration and energy per event
        total_energy = self.all_events["energy"].sum()
        if total_energy == 0:
            # no events, set event_type to None
            self.all_events["event_type"] = None
            return

        # get duration and energy for each event type
        # max power isn't necessary, power is taken from schedule time series
        duration_by_type = pd.Series(
            (self.event_ts_data == 0)[::-1].idxmin() + self.time_res,
            index=self.event_ts_data.columns,
        )
        energy_by_type = pd.Series(
            self.event_ts_data.sum() * self.time_res.total_seconds() / 3600,  # in kWh
            index=self.event_ts_data.columns,
        )

        # determine event type for each event
        self.all_events["event_type"] = None
        self.all_events["duration_multiplier"] = None
        self.all_events["energy_multiplier"] = None
        for i, event in self.all_events.iterrows():
            # compare duration and energy for each event type
            duration_ratio = event["duration"] / duration_by_type
            duration_mult = duration_ratio.round().clip(lower=1)
            duration_error = abs(duration_ratio - duration_mult) * duration_by_type
            energy_error = (event["energy"] / duration_mult - energy_by_type) / energy_by_type
            
            # determine event type based on duration and energy "scores"
            # 50% error in energy ~= 30 minutes of error in duration
            duration_score = duration_error.dt.total_seconds() / 60
            energy_score = abs(energy_error) ** 2 * 120
            event_type = (duration_score + energy_score).idxmin()
            self.all_events.loc[i, "event_type"] = event_type

            # update multipliers and other data
            d_mult = duration_mult[event_type]
            self.all_events.loc[i, "duration_multiplier"] = int(d_mult)
            self.all_events.loc[i, "duration"] = duration_by_type[event_type] * d_mult
            if abs(energy_error[event_type]) > 0.5:
                self.warn(f"Adjusting power by {energy_error[event_type] * 100}% for {event_type}.")
            self.all_events.loc[i, "energy_multiplier"] = 1 + energy_error[event_type]
            check = energy_by_type[event_type] * d_mult * (1 + energy_error[event_type])
            assert abs(self.all_events.loc[i, "energy"] - check) < 0.01

        # revise event end times
        self.all_events["end_time"] = self.all_events["start_time"] + self.all_events["duration"]

    def initialize_schedule(self, event_schedule_file=None, **kwargs):
        # load event schedule data
        if event_schedule_file is None:
            event_schedule_file = "Event Schedules.csv"
        self.event_ts_data = load_csv(
            event_schedule_file, sub_folder=self.name, index_col="Seconds"
        )
        self.event_ts_data.index = pd.to_timedelta(self.event_ts_data.index, unit="s")

        # resample event data to time_res
        time_res_file = self.event_ts_data.index[1] - self.event_ts_data.index[0]
        if time_res_file < self.time_res:
            # downsample - average
            self.event_ts_data = self.event_ts_data.resample(self.time_res).mean()
        elif time_res_file > self.time_res:
            # upsample - ffill
            self.event_ts_data = self.event_ts_data.resample(self.time_res).ffill()

        ts_schedule = super().initialize_schedule(**kwargs)

        if "event_type" not in self.all_events:
            self.add_event_types()
        if "energy_multiplier" not in self.all_events:
            self.all_events["energy_multiplier"] = 1
        if "duration_multiplier" not in self.all_events:
            self.all_events["duration_multiplier"] = 1

        return ts_schedule

    def setup_event_schedule(self, duration_passed=None):
        # create schedule for new event
        event = self.all_events.loc[self.event_index]
        event_schedule = self.event_ts_data[event["event_type"]]
        schedule_duration = event["duration"] / event["duration_multiplier"] - self.time_res
        event_schedule = event_schedule.loc[:schedule_duration].values

        # apply energy and duration multipliers
        event_schedule *= self.all_events.loc[self.event_index, "energy_multiplier"]
        event_schedule = np.tile(event_schedule, event["duration_multiplier"])
        event_schedule = np.concatenate((event_schedule, [0]))

        if duration_passed is not None:
            event_schedule = event_schedule.loc[duration_passed:]

        return iter(event_schedule)

    def reset_time(self, start_time=None, **kwargs):
        super().reset_time(start_time, **kwargs)

        if self.in_event:
            # set schedule to current time
            duration_passed = self.current_time - self.all_events[self.event_index, "start_time"]
            self.event_schedule = self.setup_event_schedule(duration_passed)
            
    def start_event(self):
        super().start_event()

        self.event_schedule = self.setup_event_schedule()

        # set power for first time step in event
        self.p_setpoint = next(self.event_schedule)

    def end_event(self):
        super().end_event()

        # reset event schedule
        self.event_schedule = None

    def update_inputs(self, schedule_inputs=None):
        super().update_inputs(schedule_inputs)

        # update power setpoint from event schedule
        if self.in_event:
            self.p_setpoint = next(self.event_schedule)
