import os
import pandas as pd
import math
import datetime as dt

from ochre.FileIO import default_input_path, default_output_path


class EquipmentException(Exception):
    pass


class Equipment:
    name = 'Generic Equipment'
    end_use = 'Other'
    folder_name = None
    is_electric = True
    is_gas = False
    modes = ['On', 'Off']  # On and Off assumed as default modes

    def __init__(self, start_time, time_res, name=None, zip_model=None, zone='LIV', ext_time_res=None, **kwargs):
        """
        Base class for all equipment in a dwelling.
        All equipment must have:
         - A set of modes (default is ['On', 'Off'])
         - Fuel variables (by default, is_electric=True, is_gas=False)
         - A control algorithm to determine the mode (update_internal_control)
         - A method to determine the power output (calculate_power) and heat output (calculate_heat)
        Optional features for equipment include:
         - A control algorithm to use for external control (update_external_control)
         - A ZIP model for voltage-dependent real and reactive power
         - A parameters file to get loaded as self.parameters
        Equipment can use data from:
         - The dwelling schedule (or from a player file)
         - Any other information from the dwelling (passed through house_args)
        """
        if name is not None:
            self.name = name

        self.start_time = start_time
        self.current_time = start_time
        self.time_res = time_res

        # General parameters
        self.parameters = self.initialize_parameters(**kwargs) if 'parameter_file' in kwargs else {}
        self.zone = zone

        # Power parameters
        self.electric_kw = 0  # in kW
        self.reactive_kvar = 0  # in kVAR
        self.gas_therms_per_hour = 0  # in therms/hour
        if zip_model is not None and self.name in zip_model:
            self.zip_data = zip_model[self.name]
            self.zip_data['pf_mult'] = math.tan(math.acos(self.zip_data['pf']))
        else:
            self.zip_data = None

        # Heat parameters - if a number, it is injected into self.zone. Can be a dict of {zone_label: gain}
        self.sensible_gain = 0  # in W
        self.latent_gain = 0  # in W

        # Mode and controller parameters (assuming a duty cycle)
        self.mode = 'Off'
        self.time_in_mode = dt.timedelta(minutes=0)
        # self.tot_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
        self.mode_cycles = {mode: 0 for mode in self.modes}

        self.ext_time_res = ext_time_res
        self.ext_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
        self.duty_cycle_by_mode = {mode: 0 for mode in self.modes}  # fraction of time per mode, should sum to 1
        self.duty_cycle_by_mode['Off'] = 1

    def initialize_parameters(self, parameter_file, name_col='Name', val_col='Value', input_path=default_input_path,
                              **kwargs):
        # assumes a parameters file with columns for name and value
        folder_name = self.folder_name if self.folder_name is not None else self.name
        if not os.path.isabs(parameter_file):
            parameter_file = os.path.join(input_path, folder_name, parameter_file)
        if not os.path.exists(parameter_file):
            self.print('WARNING: Cannot find parameter file. '
                       'Using default OCHRE file instead of {}'.format(parameter_file))
            parameter_file = os.path.join(default_input_path, folder_name, parameter_file)

        df = pd.read_csv(parameter_file, index_col=name_col)
        if val_col is None:
            return df
        else:
            parameters = df[val_col].to_dict()

            # update parameters from kwargs (overrides the parameters file values)
            parameters.update({key: val for key, val in kwargs.items() if key in parameters})
            return parameters

    def reset_time(self):
        self.current_time = self.start_time
        self.mode = 'Off'
        self.time_in_mode = dt.timedelta(minutes=0)
        self.mode_cycles = {mode: 0 for mode in self.modes}
        self.ext_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}
        # self.tot_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}

    def update(self, voltage, schedule, ext_control_args):
        # run equipment controller to determine mode
        if ext_control_args:
            mode = self.update_external_control(schedule, ext_control_args)
        else:
            mode = self.update_internal_control(schedule)

        if mode is None or mode == self.mode:
            self.time_in_mode += self.time_res
        else:
            if mode not in self.modes:
                raise EquipmentException(
                    "Can't set {} mode to {}. Valid modes are: {}".format(self.name, mode, self.modes))
            self.mode = mode
            self.time_in_mode = self.time_res
            self.mode_cycles[self.mode] += 1
        if ext_control_args:
            self.ext_mode_counters[self.mode] += self.time_res
        # self.tot_mode_counters[self.mode] += self.time_res

        # calculate electric and gas power
        self.calculate_power_and_heat(schedule)
        if self.zip_data:
            # Update electric real/reactive power with ZIP model
            self.run_zip(voltage)

    def update_external_control(self, schedule, ext_control_args):
        # Overwrite if external control might exist
        raise EquipmentException('Must define external control algorithm for {}'.format(self.name))

    def update_internal_control(self, schedule):
        # Returns the equipment mode; can return None if the mode doesn't change
        # Overwrite if internal control exists
        raise NotImplementedError()

    def calculate_power_and_heat(self, schedule):
        raise NotImplementedError()

    def generate_results(self, verbosity, to_ext=False):
        # Saves results for OCHRE and for the external controller. By default, saves the mode and powers
        if to_ext:
            return {}
        else:
            if verbosity >= 6:
                return {self.name + ' Mode': self.mode}
            else:
                return {}

    def update_model(self, schedule):
        # Update time
        self.current_time += self.time_res

    def update_duty_cycles(self, *duty_cycles):
        duty_cycles = list(duty_cycles)
        if len(duty_cycles) == len(self.modes) - 1:
            duty_cycles.append(1 - sum(duty_cycles))
        if len(duty_cycles) != len(self.modes):
            raise EquipmentException('Error parsing duty cycles. Expected a list of length equal or 1 less than ' +
                                     'the number of modes ({}): {}'.format(len(self.modes), duty_cycles))

        self.duty_cycle_by_mode = dict(zip(self.modes, duty_cycles))

    def calculate_mode_priority(self, *duty_cycles):
        """
        Calculates the mode priority based on duty cycles from external controller. Always prioritizes current mode
        first. Other modes are prioritized based on the order of Equipment.modes. Excludes modes that have already
        "used up" their time in the external control cycle.
        :param duty_cycles: iterable of duty cycles from external controller, as decimals. Order should follow the order
        of Equipment.modes. Length of list must be equal to or 1 less than the number of modes. If length is 1 less, the
        final mode duty cycle is equal to 1 - sum(duty_cycles).
        :return: list of mode names in order of priority
        """
        if self.ext_time_res is None:
            raise EquipmentException('External control time resolution is not defined for {}.'.format(self.name))
        if duty_cycles:
            self.update_duty_cycles(*duty_cycles)

        if (self.current_time - self.start_time) % self.ext_time_res == 0 or \
                sum(self.ext_mode_counters.values(), dt.timedelta(0)) >= self.ext_time_res:
            # reset mode counters
            self.ext_mode_counters = {mode: dt.timedelta(minutes=0) for mode in self.modes}

        modes_with_time = [mode for mode in self.modes
                           if self.ext_mode_counters[mode] / self.ext_time_res < self.duty_cycle_by_mode[mode]]

        # move previous mode to top of priority list
        if self.mode in modes_with_time and modes_with_time[0] != self.mode:
            modes_with_time.pop(modes_with_time.index(self.mode))
            modes_with_time = [self.mode] + modes_with_time

        if not len(modes_with_time):
            self.warn('No available modes, keeping the current mode. '
                      'Duty cycles: {}; Time per mode: {}'.format(duty_cycles, self.ext_mode_counters))
            modes_with_time.append(self.mode)

        return modes_with_time

    def run_zip(self, v, v0=1):
        if self.electric_kw == 0:
            self.reactive_kvar = 0
            return

        # pf_mult = 1 for inductive load, -1 for capacitive
        if v == v0:
            self.reactive_kvar = self.electric_kw * self.zip_data['pf_mult']
        else:
            self.reactive_kvar = self.electric_kw * self.zip_data['pf_mult'] * (self.zip_data['Zq'] * (v / v0) ** 2 +
                                                                                self.zip_data['Iq'] * (v / v0) +
                                                                                self.zip_data['Pq'])
            self.electric_kw = self.electric_kw * (self.zip_data['Zp'] * (v / v0) ** 2 +
                                                   self.zip_data['Ip'] * (v / v0) +
                                                   self.zip_data['Pp'])

    def simulate(self, schedule=None, duration=None, voltage=1, equip_from_ext=None, verbosity=7, name=None,
                 output_path=default_output_path):
        # function to run individual equipment, without dwelling
        if schedule is None and duration is None:
            raise EquipmentException('Must specify schedule or duration to simulate equipment')
        if equip_from_ext is None:
            equip_from_ext = {}

        if schedule is None:
            times = pd.date_range(self.start_time, freq=self.time_res, end=self.start_time + duration, closed='left')
            schedule = pd.DataFrame({'NA': [0] * len(times)}, index=times)
        elif duration is not None:
            # shorten schedule if necessary
            assert isinstance(schedule.index, pd.DatetimeIndex)
            assert schedule.index[0] == self.start_time
            schedule = schedule.loc[schedule.index < self.start_time + duration]
        else:
            duration = schedule.index[-1] - self.start_time + self.time_res

        # update simulation name for output file
        if name is None:
            name = self.name

        print('Running {} Simulation for {}'.format(name, duration))
        all_results = []
        try:
            for current_schedule in schedule.itertuples():
                self.update(voltage, current_schedule._asdict(), equip_from_ext)

                results = {}
                if self.is_electric:
                    results[self.name + ' Electric Power (kW)'] = self.electric_kw
                    results[self.name + ' Reactive Power (kVAR)'] = self.reactive_kvar
                if self.is_gas:
                    results[self.name + ' Gas Power (therms/hour)'] = self.gas_therms_per_hour
                results.update(self.generate_results(verbosity))
                all_results.append(results)

                # Update model (e.g. WH tank) after results
                self.update_model(schedule)

        except Exception as e:
            print('ERROR: {} Simulation failed at time {}'.format(name, self.current_time))

            # save results before raising error
            df = pd.DataFrame(all_results, index=schedule.index)
            results_file = os.path.join(output_path, name + '.csv')
            df.to_csv(results_file)
            raise e

        # save results and return data frame
        df = pd.DataFrame(all_results, index=schedule.index)
        results_file = os.path.join(output_path, name + '.csv')
        df.to_csv(results_file)
        print('{} Simulation Complete, results saved to {}'.format(name, results_file))
        return df

    def print(self, *msg):
        print('{} - {} at {}:'.format(dt.datetime.now(), self.name, self.current_time), *msg)

    def warn(self, *msg):
        self.print('WARNING,', *msg)
