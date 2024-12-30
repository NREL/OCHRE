# -*- coding: utf-8 -*-
"""
Created on Mon Apr 02 13:24:32 2018

@author: mblonsky, kmckenna, jmaguire
"""

import os
import datetime as dt
import pandas as pd
import numpy as np

from ochre import Simulator, Analysis
from ochre.utils import OCHREException, load_hpxml, load_schedule, nested_update, update_equipment_properties, save_json
from ochre.Models import Envelope
from ochre.Equipment import *


class Dwelling(Simulator):
    """
    A Dwelling is a collection of Equipment and an Envelope Model. All Equipment contribute to energy usage and most
    contribute to heat gains in the envelope. The Dwelling class also handles all input and output files, and defines
    the timing of the simulation.
    """

    def __init__(self, metrics_verbosity=6, save_schedule_columns=None, save_args_to_json=False, 
                 **house_args):
        super().__init__(**house_args)
        house_args.pop('name', None)  # remove name from kwargs
        house_args['main_sim_name'] = self.name

        # Time parameters
        if self.initialization_time is not None:
            # TODO: use times before start for initialization, if possible
            house_args['duration'] = max(self.duration, self.initialization_time)  # used for generating schedules

        # voltage-dependency parameters
        self.voltage = 1

        # Power inputs
        self.total_p_kw = 0
        self.total_q_kvar = 0
        self.total_gas_therms_per_hour = 0

        # Results parameters
        self.metrics_verbosity = metrics_verbosity
        _ = house_args.pop('save_results', None)  # remove save_results from args to prevent saving all Equipment files
        if self.output_path is not None:
            # remove existing output files
            for file_type in ['metrics', 'hourly', 'schedule']:
                for extn in ['parquet', 'csv']:
                    f = os.path.join(self.output_path, f'{self.name}_{file_type}.{extn}')
                    if os.path.exists(f):
                        self.print('Removing previous results file:', f)
                        os.remove(f)

            # save file locations
            extn = '.parquet' if self.output_to_parquet else '.csv'
            self.metrics_file = os.path.join(self.output_path, self.name + '_metrics.csv')
            if self.verbosity >= 3:
                self.hourly_output_file = os.path.join(self.output_path, self.name + '_hourly' + extn)
            else:
                self.hourly_output_file = None
            if self.verbosity >= 7 or save_schedule_columns:
                ochre_schedule_file = os.path.join(self.output_path, self.name + '_schedule' + extn)
            else:
                ochre_schedule_file = None
        else:
            self.metrics_file = None
            self.hourly_output_file = None
            ochre_schedule_file = None

        # Load properties from HPXML file
        properties, weather_station = load_hpxml(**house_args)

        # Load occupancy schedule and weather files
        schedule, location = load_schedule(properties, weather_station=weather_station, **house_args)
        properties['location'] = location
        self.start_time = self.start_time.replace(tzinfo=schedule.index.tzinfo)

        # Save HPXML properties to json file
        if self.save_results and (save_args_to_json or self.verbosity >= 3):
            json_file = os.path.join(self.output_path, self.name + '.json')
            properties_to_save = {key.capitalize(): val for key, val in properties.items()}
            if save_args_to_json:
                # add house_args to json file
                properties_to_save = nested_update(properties_to_save, house_args)
            save_json(properties_to_save, json_file)

        # Save schedule file
        if ochre_schedule_file is not None:
            if save_schedule_columns:
                schedule_to_save = schedule.loc[:, [col for col in schedule.columns if col in save_schedule_columns]]
            else:
                schedule_to_save = schedule
            if self.output_to_parquet:
                schedule_to_save.to_parquet(ochre_schedule_file)
            else:
                schedule_to_save.reset_index().to_csv(ochre_schedule_file, index=False)
            self.print('Saved schedule to:', ochre_schedule_file)

        # Update args for initializing Envelope and Equipment
        sim_args = {
            **house_args,
            'start_time': self.start_time,  # updates time zone if necessary
            'schedule': schedule,
            'initial_schedule': schedule.loc[self.start_time].to_dict(),
            'output_path': self.output_path,
        }

        # Initialize Envelope
        envelope_args = {
            **properties,
            **sim_args,
            **house_args.get('Envelope', {})}
        self.envelope = Envelope(**envelope_args)
        # initial_schedule.update(self.envelope.get_main_states())
        sim_args['envelope_model'] = self.envelope

        # Add detailed equipment properties, including ZIP parameters
        equipment_dict = update_equipment_properties(properties, **sim_args)

        # Create all equipment
        self.equipment = {}
        for equipment_name, equipment_args in equipment_dict.items():
            equipment_args = {**sim_args, **equipment_args}
            eq = EQUIPMENT_BY_NAME[equipment_name](name=equipment_name, **equipment_args)
            self.equipment[equipment_name] = eq
            self.sub_simulators.append(eq)

        # sort equipment by end use
        self.equipment_by_end_use = {
            end_use: [e for e in self.equipment.values() if e.end_use == end_use] for end_use in ALL_END_USES
        }
        for end_use, eq in self.equipment_by_end_use.items():
            # check if there is more than 1 equipment per end use. Raise error for HVAC/WH, else print a warning
            if len(eq) > 1:
                if end_use in ['HVAC Heating', 'HVAC Cooling', 'Water Heating']:
                    raise OCHREException(f'More than 1 equipment defined for {end_use}: {eq}')
                elif end_use not in ['Lighting', 'Other']:
                    self.warn(f'More than 1 equipment defined for {end_use}: {eq}')

        for name, eq in self.equipment.items():
            # if time step is large, check that ideal equipment is being used
            ideal = eq.use_ideal_capacity if isinstance(eq, (HVAC, WaterHeater)) else True
            if not ideal:
                if self.time_res >= dt.timedelta(minutes=15):
                    raise OCHREException(f'Cannot use non-ideal equipment {name} with large time step of'
                                             f' {self.time_res}')
                if self.time_res >= dt.timedelta(minutes=5):
                    self.warn(f'Using non-ideal equipment {name} with large time step of {self.time_res}')

        # get list of zone temperatures needed for equipment schedules
        self.zones_for_schedule = []
        for eq in self.equipment.values():
            if 'Zone Temperature (C)' in eq.all_schedule_inputs and eq.zone and eq.zone not in self.zones_for_schedule:
                self.zones_for_schedule.append(eq.zone)

        # force ideal HVAC equipment to go last - so all heat from other equipment is known during update
        for eq in self.equipment.values():
            if isinstance(eq, HVAC) and eq.use_ideal_capacity:
                self.sub_simulators.pop(self.sub_simulators.index(eq))
                self.sub_simulators.append(eq)
        # force generator/battery to go last - so it can run self-consumption controller
        for eq in self.equipment.values():
            if isinstance(eq, Generator):
                self.sub_simulators.pop(self.sub_simulators.index(eq))
                self.sub_simulators.append(eq)

        # add envelope to sub_simulators after all equipment
        self.sub_simulators.append(self.envelope)

        # Run initialization to get realistic initial state
        if self.initialization_time is not None:
            self.initialize()

        if self.verbosity >= 3:
            self.print('Dwelling Initialized')

    def get_equipment_by_end_use(self, end_use):
        # returns equipment in end use. Also works by equipment name
        # if multiple equipment in end use, will return a list of equipment
        if end_use in self.equipment_by_end_use:
            if len(self.equipment_by_end_use[end_use]) == 1:
                return self.equipment_by_end_use[end_use][0]
            elif len(self.equipment_by_end_use[end_use]) > 1:
                return self.equipment_by_end_use[end_use]
            else:
                return None
        elif end_use in self.equipment:
            return self.equipment[end_use]
        else:
            raise OCHREException(f'Unknown end use: {end_use}')

    def update_inputs(self, schedule_inputs=None):
        if schedule_inputs is None:
            schedule_inputs = {}

        # check voltage from external model
        self.voltage = schedule_inputs.get('Voltage (-)', 1)
        if np.isnan(self.voltage) or self.voltage < 0:
            raise OCHREException(f'Error reading voltage for house {self.name}: {self.voltage}')
        if self.voltage == 0:
            # Enter resilience mode when voltage is 0. Assumes home generator maintains voltage at 1 p.u.
            schedule_inputs['Voltage (-)'] = 1

        # Add zone temperatures (dry and wet bulb) to schedule for equipment that use it
        for zone in self.zones_for_schedule:
            schedule_inputs[f'{zone.name} Temperature (C)'] = zone.temperature
            if zone.humidity is not None:
                # TODO: only add wet bulb when it's required
                schedule_inputs[f'{zone.name} Wet Bulb Temperature (C)'] = zone.humidity.wet_bulb

        # Reset house power
        self.total_p_kw = 0
        self.total_q_kvar = 0
        self.total_gas_therms_per_hour = 0

        super().update_inputs(schedule_inputs)

    def update_model(self, control_signal=None):
        if control_signal is None:
            control_signal = {}

        # Parse data from external controller - move end-use data to each equipment with given end-use
        for key in list(control_signal.keys()):
            for equipment in self.equipment_by_end_use.get(key, []):
                if equipment.name not in control_signal:
                    control_signal[equipment.name] = control_signal[key]

        super().update_model(control_signal)

        # Check if grid is connected or house can run in islanded mode
        if self.voltage > 0 or abs(self.total_p_kw) < 0.001:
            # running in grid-connected or islanded mode with no import/export
            # FUTURE: ensure that reactive power is small too
            pass
        else:
            # grid is disconnected and load isn't met - reset power and envelope inputs
            if self.verbosity >= 7:
                self.print('Grid disconnect and cannot meet power requirements')
            self.total_p_kw = 0
            self.total_q_kvar = 0
            self.total_gas_therms_per_hour = 0
            for zone in self.envelope.zones.values():
                zone.internal_sens_gain = 0
                zone.internal_latent_gain = 0
                zone.hvac_sens_gain = 0
                zone.hvac_latent_gain = 0
                for surface in zone.surfaces:
                    surface.internal_gain = 0

            # turn all electric equipment off and rerun house model (gas power can be non-zero for some equipment)
            for equipment in self.equipment.values():
                if equipment.is_electric:
                    equipment.current_schedule['Voltage (-)'] = 0
            self.update_model(control_signal)

    def start_sub_update(self, sub, control_signal):
        sub_control_signal = super().start_sub_update(sub, control_signal)
        
        # Add house net_power to schedule for Generator
        if isinstance(sub, Generator) and 'net_power' not in sub.current_schedule:
            sub.current_schedule["net_power"] = self.total_p_kw

        # Add pv_power to schedule for Battery
        if isinstance(sub, Battery) and "pv_power" not in sub.current_schedule:
            pv_power = sum([e.electric_kw for e in self.equipment_by_end_use['PV']])
            sub.current_schedule["pv_power"] = pv_power

        return sub_control_signal
    
    def finish_sub_update(self, sub):
        if isinstance(sub, Equipment):
            # update total electric and gas powers
            self.total_p_kw += sub.electric_kw
            self.total_q_kvar += sub.reactive_kvar
            self.total_gas_therms_per_hour += sub.gas_therms_per_hour

    def generate_results(self):
        # Results columns are in this order (minimum verbosity level):
        # 1. Total house power in kW: P, Q, Gas (0)
        # 2. Total house energy in kWh: P, Q, Gas (1)
        # 3. Electric and/or gas power by end use (2)
        # 4. House voltage and reactive power by end use (5)
        # 5. Envelope results:
        #    - Air temperatures from main zones, includes wet bulb (1)
        #    - Humidity, infiltration, and convection results (4)
        #    - Detailed model results (8)
        # 6. Specific equipment results, including:
        #    - HVAC heat delivered (3)
        #    - Water tank main results (3)
        #    - Battery and EV SOC (3)
        #    - All other equipment results (6)
        # 8. Water tank model detailed results (9)
        results = super().generate_results()

        if self.verbosity >= 0:
            results.update({
                'Total Electric Power (kW)': self.total_p_kw,
                'Total Reactive Power (kVAR)': self.total_q_kvar,
                'Total Gas Power (therms/hour)': self.total_gas_therms_per_hour
            })

        if self.verbosity >= 1:
            hours_per_step = self.time_res / dt.timedelta(hours=1)
            results.update({
                'Total Electric Energy (kWh)': self.total_p_kw * hours_per_step,
                'Total Reactive Energy (kVARh)': self.total_q_kvar * hours_per_step,
                'Total Gas Energy (therms)': self.total_gas_therms_per_hour * hours_per_step,
            })

        if self.verbosity >= 2:
            for end_use, equipment in self.equipment_by_end_use.items():
                if equipment and any([e.is_electric for e in equipment]):
                    results[end_use + ' Electric Power (kW)'] = sum([e.electric_kw for e in equipment])
            for end_use, equipment in self.equipment_by_end_use.items():
                if equipment and any([e.is_gas for e in equipment]):
                    results[end_use + ' Gas Power (therms/hour)'] = sum([e.gas_therms_per_hour for e in equipment])
        if self.verbosity >= 5:
            results['Grid Voltage (-)'] = self.voltage
            for end_use, equipment in self.equipment_by_end_use.items():
                if equipment and any([e.is_electric for e in equipment]):
                    results[end_use + ' Reactive Power (kVAR)'] = sum([e.reactive_kvar for e in equipment])

        return results

    def finalize(self, failed=False):
        # save final results
        df = super().finalize(failed)

        if df is not None:
            # calculate metrics
            metrics = Analysis.calculate_metrics(df, dwelling=self, metrics_verbosity=self.metrics_verbosity)

            # Save metrics to file (as single row df)
            if self.metrics_file is not None:
                df_metrics = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
                df_metrics.to_csv(self.metrics_file, index=False)
                self.print('Post-processing metrics saved to:', self.metrics_file)

            # Convert to hourly data and save
            if self.hourly_output_file is not None:
                # aggregate using mean or sum based on units
                agg_funcs = {col: Analysis.get_agg_func(col) for col in df.columns}
                agg_funcs = {col: func for col, func in agg_funcs.items() if func is not None}
                df_hourly = df.resample(dt.timedelta(hours=1)).aggregate(agg_funcs)
                if self.output_to_parquet:
                    df_hourly.to_parquet(self.hourly_output_file)
                else:
                    df_hourly.reset_index().to_csv(self.hourly_output_file, index=False)
                self.print('Hourly results saved to:', self.hourly_output_file)
            else:
                df_hourly = None
        else:
            metrics = None
            df_hourly = None

        return df, metrics, df_hourly

    def simulate(self, metrics_verbosity=None, **kwargs):
        if metrics_verbosity is not None:
            self.metrics_verbosity = metrics_verbosity

        return super().simulate(**kwargs)
