import os
import re
import json
import datetime as dt
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import numba  # required for array-based psychrolib
import psychrolib
from numpy.polynomial.polynomial import Polynomial

from ochre.utils import OCHREException, convert, load_csv, ZONES
from ochre.Equipment import ALL_END_USES

psychrolib.SetUnitSystem(psychrolib.SI)

FIND_FILE_KWARGS = ['path', 'ending', 'priority_list', 'dirs_to_include']
        

def get_unit(column):
    matches = re.findall(r'\(.*\)', column)  # Finds strings in parenthesis
    return matches[-1][1:-1] if matches else None


def get_agg_func(column, agg_type='Time'):
    unit = get_unit(column)
    if unit is None:
        return None
    if agg_type == 'Time':
        units_to_sum = ['kWh', 'kVARh', 'therms', '$', 'lbs']
        return 'sum' if unit in units_to_sum else 'mean'
    if agg_type == 'House':
        units_to_mean = ['-', 'C']
        return 'mean' if unit in units_to_mean else 'sum'


def load_timeseries_file(file_name, columns=None, resample_res=None, ignore_errors=True, **kwargs):
    # Loads OCHRE-defined timeseries files, csv and parquet options
    # option to specify columns to load as a list. Will add 'Time' index
    # option to resample the file at a given timedelta, will take the mean of all columns
    if not os.path.exists(file_name):
        return None
    if columns is not None and isinstance(columns, list):
        columns = ['Time'] + columns

    extn = os.path.splitext(file_name)[1]
    if extn == '.parquet':
        if ignore_errors and columns:
            # check that all columns exist, see:
            # https://stackoverflow.com/questions/65705660/ignore-columns-not-present-in-parquet-with-pyarrow-in-pandas
            parquet_file = pq.ParquetFile(file_name)
            columns = [c for c in columns if c in parquet_file.schema.names]
        df = pd.read_parquet(file_name, columns=columns, **kwargs)
    elif extn == '.csv':
        df = pd.read_csv(file_name, index_col='Time', parse_dates=True, usecols=columns, **kwargs)
    else:
        print(f'Unknown time series file extension: {extn}')
        return None

    if resample_res is not None:
        df = df.resample(resample_res).mean(numeric_only=True)

    return df


def load_ochre(ochre_path, ochre_name, load_main=True, load_hourly=True, combine_schedule=False, remove_tz=True):
    # load metrics file
    metrics_file = os.path.join(ochre_path, ochre_name + '_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f'Missing OCHRE metrics file: {metrics_file}')
        return None, None, None
    metrics = pd.read_csv(metrics_file, index_col='Metric')['Value'].to_dict()

    def find_and_load_file(suffix='', extns=['.csv', '.parquet']):
        # return csv or parquet file if it exists, otherwise return None
        file_names = [os.path.join(ochre_path, ochre_name + suffix + extn) for extn in extns]
        for file_name in file_names:
            df = load_timeseries_file(file_name)
            if df is not None:
                return df
        
    # load main, hourly, and schedule files
    main = find_and_load_file() if load_main else None
    hourly = find_and_load_file('_hourly') if load_hourly else None
    schedule = find_and_load_file('_schedule') if combine_schedule else None

    # Combine with schedule file if it exists
    if schedule is not None:
        if main is not None:
            schedule = schedule.drop(columns=[col for col in schedule.columns if col in main.columns])
            main = main.join(schedule)
        if hourly is not None:
            schedule = schedule.drop(columns=[col for col in schedule.columns if col in hourly.columns])
            schedule_hourly = schedule.resample(dt.timedelta(hours=1)).mean()
            hourly = hourly.join(schedule_hourly)

    if remove_tz:
        if main is not None:
            main.index = main.index.tz_localize(None)
        if hourly is not None:
            hourly.index = hourly.index.tz_localize(None)

    return main, metrics, hourly


    # Load hourly output file from annual EnergyPlus simulation. Takes 3 types of file formats
def load_eplus_file(file_name, eplus_format='ResStock', variable_names_file='Variable names and units.csv', year=2019):
    if eplus_format == 'BEopt':
        # skip header rows, remove "My Design - ", and replace "|"
        df = pd.read_csv(file_name, skiprows=[0, 2, 3, 4], low_memory=False)
        df.columns = [' - '.join(col.split(' - ')[1:]) if ' - ' in col else col for col in df.columns]
        df.columns = [col.replace('|', ':') for col in df.columns]
        eplus_format = 'OS-HPXML'
    elif eplus_format == 'ResStock':
        df = pd.read_csv(file_name, skiprows=[1])
        eplus_format = 'OS-HPXML'
    elif eplus_format == 'Eplus Detailed':
        df = pd.read_csv(file_name)
    else:
        raise OCHREException(f'Unknown EnergyPlus output file format: {eplus_format}')

    if len(df) != 8760:
        raise OCHREException(f'Comparison file should have 8760 rows: {file_name}')

    # Load variable names and units file
    df_names = load_csv(variable_names_file)
    df_names = df_names.loc[df_names[f'{eplus_format} Name'].notna(),
                            ['OCHRE Name', 'OCHRE Units', f'{eplus_format} Name', f'{eplus_format} Units']]

    # Rename E+ column names and convert units
    missing = pd.Series([0.0] * len(df), index=df.index)
    for ochre_name, ochre_unit, eplus_name, eplus_unit in df_names.itertuples(index=False):
        try:
            # name can be a list of names - take a sum of all variables in list that exist
            eplus_list = eval(eplus_name)
            assert isinstance(eplus_list, list)
        except (NameError, SyntaxError, AssertionError):
            eplus_list = [eplus_name]

        # subtracts columns if '~' is the first character in the column name
        eplus_cols = pd.Series({col[1:] if col[0] == '~' else col: -1 if col[0] == '~' else 1 for col in eplus_list})
        eplus_cols = eplus_cols.loc[eplus_cols.index.isin(df.columns)]
        if len(eplus_cols):
            data = (df.loc[:, eplus_cols.index] * eplus_cols).sum(axis=1)
            if ochre_unit != eplus_unit:
                data = convert(data.values, eplus_unit, ochre_unit)
            df[ochre_name] = data
        else:
            pass
            # df[ochre_name] = missing

    def replace_nans(s):
        return s.replace([np.nan, np.inf, -np.inf], 0)

    # add gas power to heating main power
    if 'HVAC Heating Gas Power (therms/hour)' in df.columns:
        gas_kw = convert(df['HVAC Heating Gas Power (therms/hour)'].values, 'therm/hour', 'kW')
        if 'HVAC Heating Main Power (kW)' in df.columns:
            df['HVAC Heating Main Power (kW)'] += gas_kw
        else:
            df['HVAC Heating Main Power (kW)'] = gas_kw

    # TODO: no longer used. Need verify if this is needed for OS-HPXML format
    if eplus_format == 'BEopt':
        # add HVAC COP and SHR (note, excludes fan power and duct losses) - BEopt only
        df['HVAC Heating COP (-)'] = replace_nans(
            df['HVAC Heating Capacity (W)'] / 1000 / (
                df['HVAC Heating Main Power (kW)'] + df.get('HVAC Heating ER Power (kW)', missing)))
        df['HVAC Cooling COP (-)'] = replace_nans(df['HVAC Cooling Capacity (W)'] / 1000 / df['HVAC Cooling Main Power (kW)'])
        df['HVAC Cooling SHR (-)'] = replace_nans(df['HVAC Cooling Sensible Capacity (W)'] /
                                                  df['HVAC Cooling Capacity (W)'])

        # calculate indoor wet bulb - BEopt only
        df['Temperature - Indoor Wet Bulb (C)'] = psychrolib.GetTWetBulbFromRelHum(
            df['Temperature - Indoor (C)'].values,
            df['Relative Humidity - Indoor (-)'],
            convert(df['Weather|Atmospheric Pressure'].values, 'atm', 'Pa'),
        )

        # add unmet HVAC loads - BEopt only
        df['Unmet HVAC Load (C)'] = df['Temperature - Indoor (C)'] - df['Temperature - Indoor (C)'].clip(
            convert(df['Living Space|Heating Setpoint'].values, 'degF', 'degC'),
            convert(df['Living Space|Cooling Setpoint'].values, 'degF', 'degC'))

    # update index to datetime
    df.index = pd.date_range(dt.datetime(year, 1, 1), periods=8760, freq=dt.timedelta(hours=1))

    return df


def add_eplus_detailed_results(df, df_ochre, ochre_properties):
    # get time series of film coefficients from E+. Assumes a minimal building with an attic and raised floor
    surface_names = {
        'Exterior Wall Ext.': [(f'WALL{i}', 'Outside') for i in range(1, 5)],
        'Exterior Wall Indoor': [(f'WALL{i}', 'Inside') for i in range(1, 5)],
        'Interior Wall Indoor': [('LIVING SPACE LIVING PARTITION', 'Inside')],
        'Attic Wall Ext.': [(f'WALL{i}', 'Outside') for i in range(5, 7)],
        'Attic Wall Attic': [(f'WALL{i}', 'Inside') for i in range(5, 7)],
        'Attic Roof Ext.': [(f'ROOF{i}', 'Outside') for i in range(1, 3)],
        'Attic Roof Attic': [(f'ROOF{i}', 'Inside') for i in range(1, 3)],
        'Attic Floor Attic': [('SURFACE 1 REVERSED', 'Inside')],  # Might not be the right surface
        'Attic Floor Indoor': [('FRAMEFLOOR2', 'Inside')],
        'Raised Floor Ext.': [('FRAMEFLOOR1', 'Outside')],
        'Raised Floor Indoor': [('FRAMEFLOOR1', 'Inside')],
        'Indoor Furniture Indoor': [('FURNITURE MASS LIVING SPACE LIVING', 'Inside')],
    }
    films = pd.DataFrame(index=df.index)
    for surface, name_list in surface_names.items():
        cols = [f'{s}:Surface {f} Face Convection Heat Transfer Coefficient [W/m2-K](Hourly)' for s, f in name_list]
        if all([col in df.columns for col in cols]):
            films[f'{surface} Film Coefficient (m^2-K/W)'] = 1 / df.loc[:, cols].mean(axis=1)
    df = df.join(films)

    # calculate functions for interior and exterior films:
    #  - exterior film vs. wind speed
    #  - interior film vs. deltaT (T_boundary - T_zone), not implemented
    wind_speed = df_ochre['Wind Speed (m/s)']
    for col, f in films.items():
        if 'Ext.' in col:
            poly_fit = Polynomial.fit(wind_speed, f, 1)
            print(col, poly_fit)
        else:
            pass

    # TODO: get time series of radiation power from E+ for each surface

    return df


def calculate_metrics(results=None, results_file=None, dwelling=None, metrics_verbosity=8):
    # Included in verbosity level:
    #  1. Total energy metrics (without kVAR)
    #  2. End-use energy metrics (without kVAR)
    #  3. Average zone temperatures
    #  4. HVAC metrics (most), water heater metrics, battery/generator
    #     metrics, EV metrics, outage metrics
    #  5. Equipment-level energy metrics (without kVAR) and cycling metrics
    #  6. Peak electric power
    #  7. Reactive energy metrics
    #  8. Zone temperature std. dev., other equipment metrics
    #  9. Average value metrics for most time-series results
    if results is None:
        if results_file is None:
            results_file = dwelling.results_file
        results = load_timeseries_file(results_file)

    if len(results) < 2:
        # results are empty (or almost empty), likely due to error. Return empty dict
        return {}

    time_res = results.index[1] - results.index[0]
    hr_per_step = time_res / dt.timedelta(hours=1)
    metrics = {}

    # Total power metrics
    power_names = [('Electric Power (kW)', 'Electric Energy (kWh)'),
                   ('Gas Power (therms/hour)', 'Gas Energy (therms)')]
    if metrics_verbosity >= 7:
        power_names += [('Reactive Power (kVAR)', 'Reactive Energy (kVARh)')]
    for power_name, energy_name in power_names:
        col = 'Total ' + power_name
        if col in results:
            metrics[col.replace(power_name, energy_name)] = results[col].sum(skipna=False) * hr_per_step

    # Average and peak electrical power
    if metrics_verbosity >= 6:
        p = results['Total Electric Power (kW)']
        metrics.update({
            'Average Electric Power (kW)': p.mean(),
            'Peak Electric Power (kW)': p.max(),
            'Peak Electric Power - 15 min avg (kW)': p.resample('15min').mean().max(),
            'Peak Electric Power - 30 min avg (kW)': p.resample('30min').mean().max(),
            'Peak Electric Power - 1 hour avg (kW)': p.resample('1h').mean().max(),
        })

    # End use power metrics
    if metrics_verbosity >= 2:
        for power_name, energy_name in power_names:
            for end_use in ALL_END_USES:
                col = f'{end_use} {power_name}'
                if col in results:
                    metrics[col.replace(power_name, energy_name)] = results[col].sum(skipna=False) * hr_per_step

    # Envelope metrics
    if metrics_verbosity >= 3:
        # Average and std. dev. of zone temperatures
        for node in ZONES.values():
            col = 'Temperature - {} (C)'.format(node)
            if col in results:
                metrics['Average ' + col] = results[col].mean()
                if metrics_verbosity >= 8:
                    metrics[f'Std. Dev. Temperature - {node} (C)'] = results[col].std()
    if metrics_verbosity >= 6:
        # Total component load values
        # Note: component loads are pos for inducing heating, opposite sign of heat gain results
        component_load_names = [
            ('Internal Heat Gain - Indoor (W)', 'Component Load - Internal Gains (kWh)'),
            ('Infiltration Heat Gain - Indoor (W)', 'Component Load - Infiltration (kWh)'),
            ('Forced Ventilation Heat Gain - Indoor (W)', 'Component Load - Forced Ventilation (kWh)'),
            ('Natural Ventilation Heat Gain - Indoor (W)', 'Component Load - Natural Ventilation (kWh)'),
            ('HVAC Heating Duct Losses (W)', 'Component Load - Ducts, Heating (kWh)'),
            ('HVAC Cooling Duct Losses (W)', 'Component Load - Ducts, Cooling (kWh)'),
        ]
        for result_name, metric_name in component_load_names:
            if result_name in results:
                metrics[metric_name] = -results[result_name].sum() * hr_per_step / 1000

    # HVAC metrics
    if metrics_verbosity >= 4:
        if 'Unmet HVAC Load (C)' in results:
            unmet_hvac = results['Unmet HVAC Load (C)']
            metrics['Unmet Heating Load (C-hours)'] = -unmet_hvac.clip(upper=0).sum(skipna=False) * hr_per_step
            metrics['Unmet Cooling Load (C-hours)'] = unmet_hvac.clip(lower=0).sum(skipna=False) * hr_per_step

        for end_use, hvac_mult in [('HVAC Heating', 1), ('HVAC Cooling', -1)]:
            # Delivered heating/cooling
            if end_use + ' Delivered (W)' in results:
                delivered = results[end_use + ' Delivered (W)'] / 1000  # in kW
                delivered_sum = delivered.sum(skipna=False)
                metrics['Total {} Delivered (kWh)'.format(end_use)] = delivered_sum * hr_per_step
            else:
                delivered_sum = 0

            if end_use + ' Capacity (W)' in results:
                capacity = results[end_use + ' Capacity (W)'] / 1000  # in kW
                capacity_sum = capacity.sum()

                # FUTURE: maybe add: fan power ratio = total power / main power;
                #  and fan heat ratio = net delivered / delivered without fan

                # COP = capacity / power
                power_sum = results[f'{end_use} Main Power (kW)'].sum()
                if f'{end_use} ER Power (kW)' in results:
                    power_sum += results[f'{end_use} ER Power (kW)'].sum()
                if power_sum != 0:
                    metrics[f'Average {end_use} COP (-)'] = capacity_sum / power_sum

                # SHR = sensible capacity / capacity
                if f'{end_use} SHR (-)' in results:
                    sens_capacity_sum = (capacity * results[f'{end_use} SHR (-)']).sum()
                    if capacity_sum != 0:
                        metrics[f'Average {end_use} SHR (-)'] = sens_capacity_sum / capacity_sum
                else:
                    sens_capacity_sum = capacity_sum

                # Duct losses - note: excludes latent gains and fan power
                # DSE = (sensible delivered - fan power * DSE) / (sensible capacity)
                if f'{end_use} Fan Power (kW)' in results:
                    fan_heat = results[f'{end_use} Fan Power (kW)'].sum() * hvac_mult
                else:
                    fan_heat = 0
                if sens_capacity_sum != 0:
                    metrics[f'Average {end_use} Duct Efficiency (-)'] = delivered_sum / (sens_capacity_sum + fan_heat)

                # HVAC capacity - only when device is on
                if metrics_verbosity >= 8:  
                    metrics['Average {} Capacity (kW)'.format(end_use)] = capacity[capacity > 0].mean()

    # Water heater and hot water metrics
    if metrics_verbosity >= 4 and "Water Heating Delivered (W)" in results:
        heat = results['Water Heating Delivered (W)'] / 1000  # in kW
        metrics['Total Water Heating Delivered (kWh)'] = heat.sum(skipna=False) * hr_per_step

        # COP - weighted average only when device is on
        if 'Water Heating COP (-)' in results and heat.sum(skipna=False) != 0:
            cop = results['Water Heating COP (-)']
            metrics['Average Water Heating COP (-)'] = (cop * heat).sum(skipna=False) / heat.sum(skipna=False)

        # Unmet hot water demand
        if 'Hot Water Unmet Demand (kW)' in results:
            metrics['Total Hot Water Unmet Demand (kWh)'] = \
                results['Hot Water Unmet Demand (kW)'].sum(skipna=False) * hr_per_step

        # Hot water delivered
        if 'Hot Water Delivered (L/min)' in results:
            # FUTURE: Down with imperial units!
            metrics['Total Hot Water Delivered (gal/day)'] = convert(
                results['Hot Water Delivered (L/min)'].mean(skipna=False), 'L/min', 'gallon/day')
        if 'Hot Water Delivered (W)' in results:
            metrics['Total Hot Water Delivered (kWh)'] = \
                results['Hot Water Delivered (W)'].sum(skipna=False) / 1000 * hr_per_step

    # EV metrics
    if "EV SOC (-)" in results:
        metrics["Average EV SOC (-)"] = results["EV SOC (-)"].mean(skipna=False)
    if "EV Unmet Load (kW)" in results:
        metrics["Total EV Unmet Load (kWh)"] = (
            results["EV Unmet Load (kW)"].sum(skipna=False) * hr_per_step
        )

    # Battery metrics
    if metrics_verbosity >= 4 and 'Battery Electric Power (kW)' in results:
        batt_energy = results['Battery Electric Power (kW)'] * hr_per_step
        metrics['Battery Charging Energy (kWh)'] = batt_energy.clip(lower=0).sum(skipna=False)
        metrics['Battery Discharging Energy (kWh)'] = -batt_energy.clip(upper=0).sum(skipna=False)
        if metrics['Battery Charging Energy (kWh)'] != 0:
            metrics['Battery Round-trip Efficiency (-)'] = (metrics['Battery Discharging Energy (kWh)'] /
                                                            metrics['Battery Charging Energy (kWh)'])
            
        if all([r in results for r in ['Battery Energy to Discharge (kWh)', 'Total Electric Energy (kWh)']]):
            cumulative_energy = (results['Total Electric Energy (kWh)'] - batt_energy).cumsum()
            end_energy = cumulative_energy + results['Battery Energy to Discharge (kWh)']
            last_time = results.index[-1] + time_res
            islanding_times = []
            for t, energy in end_energy.items():
                future_energies = cumulative_energy[t : t + dt.timedelta(days=7)]
                end_time = (future_energies >= energy).idxmax()
                islanding_time = (end_time - t).total_seconds() / 3600 if end_time > t else 24 * 7
                islanding_times.append(islanding_time)
            results["Islanding Time (hours)"] = islanding_times

            metrics['Average Islanding Time (hours)'] = results['Islanding Time (hours)'].mean()

    # Gas generator metrics
    if metrics_verbosity >= 4 and 'Gas Generator Electric Energy (kWh)' in metrics:
        metrics['Gas Generator Efficiency (-)'] = (-metrics['Gas Generator Electric Energy (kWh)'] /
                                                   convert(metrics['Gas Generator Gas Energy (therms)'], 'therm',
                                                           'kWh'))

    # Outage metrics
    if metrics_verbosity >= 4 and 'Grid Voltage (-)' in results:
        outage = results["Grid Voltage (-)"] == 0
        if outage.any():
            outage_sum = outage.sum(skipna=False) * hr_per_step
            outage_diff = np.diff(outage.values, prepend=0, append=0)
            outage_starts = np.nonzero(outage_diff.clip(min=0))[0]
            outage_ends = np.nonzero(-outage_diff.clip(max=0))[0]
            metrics['Number of Outages'] = len(outage_starts)
            metrics['Average Outage Duration (hours)'] = outage_sum / len(outage_starts)
            metrics['Longest Outage Duration (hours)'] = (outage_ends - outage_starts).max() * hr_per_step

    # Equipment power metrics
    if metrics_verbosity >= 5:
        for power_name, energy_name in power_names:
            power_cols = [col for col in results.columns if power_name in col]
            metrics.update({col.replace(power_name, energy_name): results[col].sum(skipna=False) * hr_per_step
                            for col in power_cols})

    # Equipment cycling metrics
    if metrics_verbosity >= 5:
        mode_cols = [col for col in results if " On-Time Fraction (-)" in col]
        for mode_col in mode_cols:
            name = re.fullmatch("(.*) On-Time Fraction (-)", mode_col).group(1)
            on_frac = results[mode_col].astype(bool)
            cycle_starts = on_frac & (~on_frac).shift()
            cycles = cycle_starts.sum()
            if cycles > 0:
                metrics[f"{name} Cycles"] = cycles

    # FUTURE: add rates, emissions, other post processing
    # print('Loading rate file...')
    # rate_file = os.path.join(main_path, 'Inputs', 'Rates', 'Utility Rates.csv')
    # df_rates = Input_File_Functions.import_generic(rate_file, keep_cols=locations, annual_output=True, **default_args)
    # df_rates.index.name = 'Time'
    # df_rates = df_rates.reset_index().melt(id_vars='Time', var_name='Location', value_name='Rate')
    #
    # print('Calculating annual costs...')
    # df_all = df_all.reset_index().merge(df_rates, how='left', on=['Time', 'Location']).set_index('Time')
    # df_all['Cost'] = df_all['Rate'] * df_all['Total Electric Energy (kWh)']
    # annual_costs = df_all.groupby(['Location', 'Setpoint Difference'])['Cost'].sum()
    # print(annual_costs)
    # annual_costs.to_csv(os.path.join(main_path, 'Outputs', 'poster_results.csv'))
    # df_all.reset_index().to_feather(os.path.join(main_path, 'Outputs', 'poster_all_data.feather'))

    if metrics_verbosity >= 9:
        # The kitchen sink approach: sum all power columns (e.g. Main Power, Fan Power)
        power_names = [('Power (kW)', 'Energy (kWh)'),
                       ('Power (therms/hour)', 'Energy (therms)')]
        for power_name, energy_name in power_names:
            metrics.update({col.replace(power_name, energy_name): results[col].sum(skipna=False) * hr_per_step
                            for col in results.columns if power_name in col})

        # The kitchen sink approach: average all unitless results (e.g. relative humidity)
        metrics.update({f'Average {col}': results[col].mean() for col in results.columns
                        if ' (-)' in col and f'Average {col}' not in metrics})

    return metrics


def create_comparison_metrics(ochre, eplus, ochre_metrics, eplus_metrics, include_mean=False, include_rmse=True,
                              **kwargs):
    # Functions for calculating error comparison metrics
    def rmse(x, y):
        return ((x.values - y.values) ** 2).mean() ** 0.5

    def pct_error(new, base):
        if ~np.isnan(new) and ~np.isnan(base) and base != 0:
            return (new - base) / base * 100.0
        else:
            return 100.0

    def abs_error(new, base):
        return new - base

    # create data frame with metric values for OCHRE, E+, absolute error, and percentage error
    # aggregate metrics: record metric and calculate absolute and percent difference
    compare_metrics = {name: {
        'OCHRE': ochre_metrics[name],
        'EnergyPlus': eplus_metrics[name],
        'Absolute Error': abs_error(ochre_metrics[name], eplus_metrics[name]),
        'Percent Error (%)': pct_error(ochre_metrics[name], eplus_metrics[name]),
    } for name in eplus_metrics
        if name in ochre_metrics and not np.isnan(ochre_metrics[name]) and eplus_metrics[name] != 0
    }

    if include_mean:
        # calculate time series mean
        ochre_mean = ochre.mean(numeric_only=True)
        eplus_mean = eplus.mean(numeric_only=True)
        compare_metrics.update({name + ' MEAN': {
            'OCHRE': ochre_mean[name],
            'EnergyPlus': eplus_mean[name],
            'Absolute Error': abs_error(ochre_mean[name], eplus_mean[name]),
            'Percent Error (%)': pct_error(ochre_mean[name], eplus_mean[name]),
        } for name in eplus_mean.index
            if name in ochre_mean.index and eplus_mean[name] != 0 and ochre_mean[name] != 0
        })

    if include_rmse:
        # calculate time series RMSE
        ochre_std = ochre.std(numeric_only=True)
        eplus_std = eplus.std(numeric_only=True)
        compare_metrics.update({name + ' RMSE': {
            'OCHRE': ochre_std[name],
            'EnergyPlus': eplus_std[name],
            'Absolute Error': rmse(ochre[name], eplus[name]),
            'Percent Error (%)': rmse(ochre[name], eplus[name]) / eplus_std[name] * 100,
        } for name in eplus_std.index
            if name in ochre_std.index and eplus_std[name] != 0 and ochre_std[name] != 0
        })

    compare_metrics = pd.DataFrame(compare_metrics).T
    return compare_metrics


def get_parent_folders(file_path, dirs_to_include=1):
    folder_list = os.path.dirname(file_path).split(os.sep)
    return os.sep.join(folder_list[-dirs_to_include:])


def find_subfolders(root_folder, includes_file_patterns=None, excludes_file_patterns=None):
    # returns list of absolute folder paths for folders that includes files that match all of the includes file patterns
    # the folder will be ignored if there are any matches in the excludes_file_patterns
    if includes_file_patterns is None:
        includes_file_patterns = []
    if excludes_file_patterns is None:
        excludes_file_patterns = []

    subfolders = []
    for (root, _, files) in os.walk(root_folder):
        if any([any([re.match(pattern, f) for f in files]) for pattern in excludes_file_patterns]):
            continue
        if not all([any([re.match(pattern, f) for f in files]) for pattern in includes_file_patterns]):
            continue
        subfolders.append(root)

    return subfolders


def find_files_from_ending(path, ending, priority_list=None, **kwargs):
    # returns a dictionary of {run_name: file_path} for files in path with given ending
    # run name is derived from directory names
    if priority_list is None:
        priority_list = []

    all_files = {}
    for root, _, file_names in os.walk(path):
        matches = [f for f in file_names if f.endswith(ending)]
        if not matches:
            continue
        if len(matches) > 1:
            # select file match in priority list. If not found, throw an error
            matches = [f for f in priority_list if f in matches]
            if len(matches) != 1:
                raise OCHREException(f'{len(matches)} files found matching {ending} in {root}: {matches}')
        
        file_path = os.path.join(root, matches[0])
        run_name = get_parent_folders(file_path, **kwargs)
        if run_name in all_files:
            raise OCHREException(f'Multiple files found with same run name ({run_name}).'
                             'Try increasing dirs_to_include. Error from:', file_path)

        all_files[run_name] = file_path        

    return all_files


def combine_json_files(json_files=None, **kwargs):
    # combine input json files from multiple OCHRE simulations into one DataFrame
    # if files are not specified, calls Analysis.find_files_from_ending(ending='.json', **kwargs)
    if json_files is None:
        json_files = find_files_from_ending(ending='.json', **kwargs)

    # load all files, combine to a df
    all_properties = {}
    for run_name, json_file in json_files.items():
        if not os.path.exists(json_file):
            print(f'No json file found for {run_name}. ')
            continue

        with open(json_file) as f:
            properties = json.load(f)
        properties = pd.json_normalize(properties).loc[0].to_dict()
        all_properties[run_name] = properties

    # combine metrics to DataFrame, 1 row for each file
    df = pd.DataFrame(all_properties).T
    return df

def combine_metrics_files(metrics_files=None, **kwargs):
    # combine metrics from multiple OCHRE simulations into one DataFrame
    # if files are not specified, calls Analysis.find_files_from_ending(ending='_metrics.csv', **kwargs)
    if metrics_files is None:
        metrics_files = find_files_from_ending(ending='_metrics.csv', **kwargs)

    # load all metrics, combine to 1 file
    all_metrics = {}
    for run_name, metrics_file in metrics_files.items():
        if not os.path.exists(metrics_file):
            print(f'No metrics file found for {run_name}.')
            continue

        df = pd.read_csv(metrics_file, index_col='Metric')
        metrics = df['Value'].to_dict()
        all_metrics[run_name] = metrics

    # combine metrics to DataFrame, 1 row for each file
    df = pd.DataFrame(all_metrics).T
    return df


def combine_time_series_column(column, results_files=None, **kwargs):
    # combines time series results from multiple OCHRE simulations into single DataFrame
    # Only keeps 1 column and sets the file names as the new column names
    # results_files is a dictionary of {run_name: file_path}
    if results_files is None:
        find_kwargs = {arg: kwargs.pop(arg) for arg in FIND_FILE_KWARGS}
        results_files = find_files_from_ending(ending='ochre.csv', **find_kwargs)

    data = []
    for run_name, results_file in results_files.items():
        if not os.path.exists(results_file):
            print(f'No time series file found for {run_name}.')
            continue

        s = load_timeseries_file(results_file, columns=[column], **kwargs)[column]
        s.name = run_name
        data.append(s)

    df = pd.concat(data, axis=1)
    return df


def combine_time_series_files(results_files=None, agg_func=None, **kwargs):
    # combines time series results from multiple OCHRE simulations into single DataFrame
    # results_files is a dictionary of {run_name: file_path}
    # columns specifies the columns to load (using pd.read_csv(use_cols=columns))
    # agg_func will aggregate the results, see GroupBy.agg for options
    # Otherwise, will return a DataFrame with a MultiIndex for (Run Name, Time)
    # Other arguments are passed to read_csv (e.g., use_cols)
    if results_files is None:
        find_kwargs = {arg: kwargs.pop(arg) for arg in FIND_FILE_KWARGS}
        results_files = find_files_from_ending(ending='ochre.csv', **find_kwargs)

    dfs = []
    for run_name, results_file in results_files.items():
        if not os.path.exists(results_file):
            print(f'No time series file found for {run_name}.')
            continue

        df = load_timeseries_file(results_file,  **kwargs)
        df['House'] = run_name
        dfs.append(df.reset_index())

    df = pd.concat(dfs, ignore_index=True)

    if agg_func is not None:
        # See get_agg_func for recommended options
        df = df.groupby('Time').agg(agg_func)
    else:
        df = df.set_index(['Time', 'House'])

    return df
