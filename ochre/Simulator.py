import os
import re
import datetime as dt
import numpy as np
import pandas as pd
import hashlib

from ochre import __version__
from ochre.utils import OCHREException
import ochre.utils.schedule as utils_schedule


class Simulator:
    name = 'OCHRE'
    required_inputs = []
    optional_inputs = []

    def __init__(self, start_time, time_res, duration, name=None, main_sim_name=None, seed=None,
                 verbosity=1, save_results=None, save_status=None, output_path=None, output_to_parquet=False,
                 initialization_time=None, export_res=None, **kwargs):
        if name is not None:
            self.name = name
        self.main_sim_name = main_sim_name
        self.main_simulator = self.main_sim_name is None
        self.sub_simulators = []  # list of simulators included within self (e.g., Equipment in a Dwelling)

        # Timing parameters
        self.start_time = start_time  # Note: may be updated later with time zone
        self.current_time = start_time
        self.time_res = time_res
        self.duration = duration
        if self.duration < self.time_res:
            raise OCHREException(f'Duration ({duration}) must be longer than time resolution ({time_res}).')
        self.initialization_time = initialization_time
        self.sim_times = pd.date_range(self.start_time, self.start_time + self.duration, freq=self.time_res,
                                       inclusive='left')
                                       
        # Results parameters
        self.results = []
        self.verbosity = verbosity
        if self.main_simulator and self.verbosity >= 3:
            self.print(f'Initializing {self.name} (OCHRE v{__version__})')

        # Output file parameters
        if save_results is None:
            save_results = self.main_simulator and self.verbosity > 0
        self.save_results = save_results
        if save_status is None:
            save_status = self.main_simulator and save_results
        self.save_status = save_status
        self.output_path = output_path
        self.output_to_parquet = output_to_parquet
        self.export_res = export_res
        self.results_file = None
        if self.save_results:
            self.set_up_results_files(**kwargs)

        # Set random seed based on output path. Only sets seed if seed or output_path is specified
        if self.main_simulator:
            if seed is None:
                seed = self.output_path
            if seed is not None:
                if isinstance(seed, str):
                    seed = int(hashlib.md5(seed.encode()).hexdigest(), 16) % 2 ** 32
                np.random.seed(seed)

        # Define model schedule and time resolution
        self.all_schedule_inputs = None
        self.schedule = self.initialize_schedule(**kwargs)
        self.current_schedule = self.schedule.iloc[0].to_dict()
        self.schedule_iterable = None
        self.reset_time()

    def set_up_results_files(self, hpxml_file=None, **kwargs):
        if self.output_path is None:
            if hpxml_file is not None:
                self.output_path = os.path.dirname(hpxml_file)
            else:
                raise OCHREException('Must specify output_path, or set save_results=False.')
        if not os.path.isabs(self.output_path):
            self.output_path = os.path.abspath(self.output_path)
        os.makedirs(self.output_path, exist_ok=True)

        # save result file path 
        file_name = self.name if not self.main_sim_name else f'{self.name}_{self.main_sim_name}'
        extn = '.parquet' if self.output_to_parquet else '.csv'
        self.results_file = os.path.join(self.output_path, file_name + extn)

        # Remove existing results files
        for f in os.listdir(self.output_path):
            if f == f'{file_name}.csv' or (self.name in f and '.parquet' in f):
                self.print('Removing previous results file:', os.path.join(self.output_path, f))
                os.remove(os.path.join(self.output_path, f))


        # remove existing status files
        statuses = ['failed', 'complete']
        for status in statuses:
            file_name = os.path.join(self.output_path, f'{self.name}_{status}')
            if os.path.exists(file_name):
                os.remove(file_name)

            
    def initialize(self, extra_hours=None):
        # run for initialization time, then reset time. don't generate results
        if self.verbosity >= 3:
            self.print('Running initialization for', self.initialization_time)
        tmp = self.verbosity
        self.verbosity = 0

        end_time = self.start_time + self.initialization_time
        init_times = pd.date_range(self.start_time, end_time, freq=self.time_res, inclusive='left')
        for _ in init_times:
            self.update()
        self.reset_time()

        # run for additional hours at initial time to reduce initial errors
        if extra_hours is not None:
            for _ in range(dt.timedelta(hours=extra_hours) // self.time_res):
                self.update()
                self.reset_time()

        # reset verbosity
        self.verbosity = tmp

    def initialize_schedule(self, schedule=None, required_inputs=None, optional_inputs=None, **kwargs):
        # Saves schedule as a DataFrame with required and optional columns
        if required_inputs is None:
            required_inputs = self.required_inputs
        if optional_inputs is None:
            optional_inputs = self.optional_inputs
        self.all_schedule_inputs = required_inputs + optional_inputs
        assert len(self.all_schedule_inputs) == len(set(self.all_schedule_inputs))  # columns should be unique

        # Load schedule from file if necessary
        if isinstance(schedule, str):
            schedule = pd.read_csv(schedule)
            if 'Time' in schedule.columns:
                schedule = schedule.set_index('Time')
                schedule.index = pd.to_datetime(schedule.index)

        if schedule is None:
            schedule = pd.DataFrame(index=self.sim_times)

        if not isinstance(schedule.index, pd.DatetimeIndex):
            raise OCHREException(f'{self.name} schedule index must be a DateTime index, not {type(schedule.index)}.'
                            f' If loading schedule from a file, try setting index column to "Time".')

        # Print warning if all required inputs are not in schedule
        missing_inputs = [name for name in required_inputs if name not in schedule.columns]
        if missing_inputs:
            self.warn(f'Schedule is missing required inputs: {missing_inputs}')
         
        # Only keep specified inputs
        schedule_cols = [name for name in self.all_schedule_inputs if name in schedule.columns]
        schedule = schedule.loc[:, schedule_cols]

        # resample and reindex schedule if necessary
        schedule = utils_schedule.resample_and_reindex(
            schedule,
            self.time_res,
            self.start_time,
            self.duration,
            **kwargs,
        )

        return schedule

    def update_inputs(self, schedule_inputs=None):
        # Update schedule at current time
        if not self.schedule.empty:
            self.current_schedule = next(self.schedule_iterable)
        else:
            self.current_schedule = {}

        # Update schedule with external schedule inputs
        if isinstance(schedule_inputs, dict):
            for key, val in schedule_inputs.items():
                if key in self.all_schedule_inputs:
                    self.current_schedule[key] = val

        # Update inputs for all sub simulators
        for sub in self.sub_simulators:
            assert sub.current_time >= self.current_time
            if sub.current_time == self.current_time:
                sub.update_inputs(schedule_inputs)

    def start_sub_update(self, sub, control_signal):
        # Used to update main simulator before sub simulator update_model starts
        if control_signal:
            return control_signal.get(sub.name)
        else:
            return None

    def finish_sub_update(self, sub):
        # Used to update main simulator after sub simulator update_model finishes
        pass

    def update_model(self, control_signal=None):
        # update models for all sub simulators
        for sub in self.sub_simulators:
            sub_control_signal = self.start_sub_update(sub, control_signal)
            if sub.current_time == self.current_time:
                sub.update_model(sub_control_signal)
            self.finish_sub_update(sub)

    def generate_results(self):
        current_results = {}

        if self.save_results or (self.main_simulator and self.verbosity > 0):
            current_results['Time'] = self.current_time

        return current_results

    def export_results(self):
        df = pd.DataFrame(self.results).set_index('Time') if self.results else None
        
        if not self.save_results or df is None:
            # Do nothing if not saving results to file or there are no results to save
            pass
        elif self.output_to_parquet:
            # create a new parquet file with timestamp
            time_str = self.current_time.strftime('%Y%m%d-%H%M%S')
            file_name = self.results_file.replace('.parquet', f'_{time_str}.parquet')
            df.to_parquet(file_name)
        else:
            # if a csv, append to existing results or create a new file
            if os.path.exists(self.results_file):
                df.reset_index().to_csv(self.results_file, index=False, header=False, mode='a')
            else:
                df.reset_index().to_csv(self.results_file, index=False)

        # Remove results from memory
        self.results.clear()

        return df
        
    def update_results(self):
        current_results = self.generate_results()

        # Update sub simulators and get sub results (keep separate or add to main results)
        for sub in self.sub_simulators:
            if sub.current_time == self.current_time:
                # Note: if sub runs slower than main, sub results won't be added for every time step
                sub_results = sub.update_results()
                if not sub_results:
                    pass
                elif sub.save_results:
                    sub.results.append(sub_results)
                else:
                    current_results.update(sub_results)

        if current_results and self.main_simulator:
            self.results.append(current_results)

        # Update current time
        self.current_time += self.time_res

        if self.export_res is not None and (self.current_time - self.start_time) % self.export_res == dt.timedelta(0):
            self.export_results()

        return current_results

    def update(self, control_signal=None, schedule_inputs=None):
        # Function to update Simulator by one time step. Splits the update into 3 sections
        #  - update_inputs(): prepares model update, should only get called once per time step 
        #  - update_model(): runs the model update, can get called multiple times for co-optimization
        #  - update_results(): collects all results and updates the time, should only get called once per time step
        
        self.update_inputs(schedule_inputs)

        self.update_model(control_signal)

        return self.update_results()

    def reset_time(self, start_time=None, remove_results=True, **kwargs):
        if start_time is None:
            start_time = self.start_time

        if remove_results:
            self.results.clear()

        self.current_time = start_time

        # reset schedule_iterable
        if not self.schedule.empty:
            schedule = self.schedule.loc[self.current_time:]
            self.schedule_iterable = iter(schedule.to_dict('records'))

        for sub in self.sub_simulators:
            sub.reset_time(start_time=start_time, remove_results=remove_results, **kwargs)

    def finalize(self, failed=False):
        # load all results and save to files
        if not self.save_results:
            if self.results:
                df = pd.DataFrame(self.results).set_index('Time')
                self.results.clear()
            else:
                df = None

        elif self.output_to_parquet:
            output_files = [os.path.join(self.output_path, f) for f in os.listdir(self.output_path)
                            if re.match(f'{self.name}.*\\.parquet', f) and '_schedule.parquet' not in f]
            dfs = [pd.read_parquet(f) for f in sorted(output_files)]
            if self.results:
                # add recent results that haven't been saved to a parquet file
                dfs.append(pd.DataFrame(self.results).set_index('Time'))
                self.results.clear()
            df = pd.concat(dfs) if dfs else None

            # save parquet results file and remove intermediate files
            if df is not None and len(df):
                df.to_parquet(self.results_file)
                for f in output_files:
                    os.remove(f)

        else:
            # using csv results files
            dfs = []
            if os.path.exists(self.results_file):
                dfs = [pd.read_csv(self.results_file, index_col='Time', parse_dates=True)]
            dfs.append(self.export_results())
            df = pd.concat(dfs) if any([df is not None for df in dfs]) else None

        # Print status and save to file
        status = 'failed' if failed else 'complete'
        if self.main_simulator and self.verbosity >= 3:
            if df is None:
                results = 'no results'
            elif self.save_results:
                results = f'time series results saved to: {self.results_file}'
            else:
                results = f'time series results saved in memory (not to a file)'
            self.print(f'Simulation {status}, {results}')
        if self.save_status:
            status_file = os.path.join(self.output_path, f'{self.name}_{status}')
            with open(status_file, 'a'):
                pass

        # finalize sub_simulators (only used if sub.save_results is True). Don't return sub results
        for sub in self.sub_simulators:
            sub.finalize(failed=failed)

        return df

    def simulate(self, start_time=None, duration=None, verbosity=None):
        if start_time is not None:
            self.start_time = start_time
        if self.start_time != self.current_time:
            self.reset_time(self.start_time)
        if duration is not None:
            self.duration = duration
        if verbosity is not None:
            self.verbosity = verbosity

        # determine simulation run times
        if self.verbosity >= 3:
            self.print('Running Simulation for', self.duration)
        self.sim_times = pd.date_range(self.start_time, self.start_time + self.duration, freq=self.time_res,
                                       inclusive='left')
        try:
            for _ in self.sim_times:
                self.update()

        except Exception as e:
            self.print('****** ERROR ******')
            self.finalize(failed=True)
            raise e

        return self.finalize()

    def print(self, *msg):
        print(f'{dt.datetime.now()} - {self.name} at {self.current_time}:', *msg)

    def warn(self, *msg):
        self.print('WARNING:', *msg)
