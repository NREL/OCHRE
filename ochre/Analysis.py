import os
import pandas as pd
import datetime as dt
import numpy as np
import psychrolib

from ochre import Units, default_year

psychrolib.SetUnitSystem(psychrolib.SI)

pd.set_option('precision', 3)  # precision in print statements
pd.set_option('expand_frame_repr', False)  # Keeps results on 1 line


# pd.set_option('max_columns', None)  # Prints all columns


def create_metrics(df, time_res=dt.timedelta(hours=1)):
    hours = time_res.total_seconds() / 3600
    metrics = {
        'Total Electric Energy (kWh)': df['Total Electric Power (kW)'].sum() * hours,
        'HVAC Heating Electric Energy (kWh)': df['HVAC Heating Electric Power (kW)'].sum() * hours,
        'HVAC Cooling Electric Energy (kWh)': df['HVAC Cooling Electric Power (kW)'].sum() * hours,
        'Water Heating Electric Energy (kWh)': df['Water Heating Electric Power (kW)'].sum() * hours,
        'Other Electric Energy (kWh)': df['Other Electric Power (kW)'].sum() * hours,
        'Total Gas Energy (therms)': df['Total Gas Power (therms/hour)'].sum() * hours,
        'HVAC Heating Gas Energy (therms)': df['HVAC Heating Gas Power (therms/hour)'].sum() * hours,
        'Water Heating Gas Energy (therms)': df['Water Heating Gas Power (therms/hour)'].sum() * hours,
        'Other Gas Energy (therms)': df['Other Gas Power (therms/hour)'].sum() * hours,

        'Average Temperature - Indoor (C)': df['Temperature - Indoor (C)'].mean(),
        'Average Temperature - Garage (C)': df['Temperature - Garage (C)'].mean(),
        'Average Temperature - Foundation (C)': df['Temperature - Foundation (C)'].mean(),
        'Average Temperature - Attic (C)': df['Temperature - Attic (C)'].mean(),

        'Total HVAC Heating Delivered (kWh)': df['HVAC Heating Delivered (kW)'].sum() * hours,
        'Total HVAC Cooling Delivered (kWh)': df['HVAC Cooling Delivered (kW)'].sum() * hours,
        'Total Hot Water Delivered (kWh)': df['Hot Water Delivered (kW)'].sum() * hours,
    }
    if metrics['Total HVAC Heating Delivered (kWh)'] != 0 and df['HVAC Heating COP (-)'].sum() != 0:
        metrics.update({
            'Average HVAC Heating COP (-)': (df['HVAC Heating COP (-)'] * df[
                'HVAC Heating Delivered (kW)']).sum() / df['HVAC Heating Delivered (kW)'].sum(),
        })
    if metrics['Total HVAC Cooling Delivered (kWh)'] != 0 and df['HVAC Cooling COP (-)'].sum() != 0:
        metrics.update({
            'Average HVAC Cooling COP (-)': (df['HVAC Cooling COP (-)'] * df[
                'HVAC Cooling Delivered (kW)']).sum() / df['HVAC Cooling Delivered (kW)'].sum(),
            'Average HVAC Cooling SHR (-)': (df['HVAC Cooling SHR (-)'] * df[
                'HVAC Cooling Delivered (kW)']).sum() / df['HVAC Cooling Delivered (kW)'].sum(),
        })
    return metrics


# Root mean squared error between expected and actual
def rmse(x, y): return ((x.values - y.values) ** 2).mean() ** 0.5


def pct_error(new, base):
    if ~np.isnan(new) and ~np.isnan(base) and base != 0:
        return (new - base) / base * 100.0
    else:
        return 100.0


def abs_error(new, base):
    return new - base


def load_ochre(ochre_folder, ochre_name, load_main=False, combine_schedule=False, resample=None):
    # load hourly file
    hourly_file = os.path.join(ochre_folder, ochre_name + '_hourly.csv')
    hourly = pd.read_csv(hourly_file, index_col='Time', parse_dates=True)

    # load metrics file
    metrics_file = os.path.join(ochre_folder, ochre_name + '_metrics.csv')
    metrics = pd.read_csv(metrics_file, index_col='Metric')['Value'].to_dict()

    if load_main:
        # load main time series file
        main_file = os.path.join(ochre_folder, ochre_name + '.csv')
        main = pd.read_csv(main_file, index_col='Time', parse_dates=True)
    else:
        main = None

    # Combine with schedule file if it exists
    schedule_file = os.path.join(ochre_folder, ochre_name + '_schedule.csv')
    if combine_schedule and os.path.exists(schedule_file):
        schedule = pd.read_csv(schedule_file, index_col='Time', parse_dates=True)
        hourly = pd.concat([hourly, schedule.resample(dt.timedelta(hours=1)).mean()], axis=1)
        if main is not None:
            main = pd.concat([main, schedule], axis=1)

    if resample is not None and main is not None:
        main = main.resample(resample)

    return hourly, metrics, main


def load_eplus_file(file_name):
    # loads hourly output file from annual EnergyPlus simulation
    df = pd.read_csv(file_name, skiprows=[1])
    df.columns = [' - '.join(col.split(' - ')[1:]) if ' - ' in col else col for col in df.columns]

    # rename E+ column names, tuples are (InSPIRE name, E+ name, unit conversion function)
    eplus_names_and_units = [
        ('Total Electric Power (kW)', 'Site Energy|Total (E)', None),
        ('HVAC Heating Electric Power (kW)', ['Site Energy|Heating (E)',
                                              'Site Energy|Heating - Suppl. (E)',
                                              'Site Energy|Heating Fan/Pump (E)'], None),
        ('HVAC Heating Main Power (kW)', 'Site Energy|Heating (E)', None),
        ('HVAC Heating ER Power (kW)', 'Site Energy|Heating - Suppl. (E)', None),
        ('HVAC Heating Fan Power (kW)', 'Site Energy|Heating Fan/Pump (E)', None),
        ('HVAC Cooling Electric Power (kW)', ['Site Energy|Cooling (E)',
                                              'Site Energy|Cooling Fan/Pump (E)'], None),
        ('HVAC Cooling Main Power (kW)', 'Site Energy|Cooling (E)', None),
        ('HVAC Cooling Fan Power (kW)', 'Site Energy|Cooling Fan/Pump (E)', None),
        ('Water Heating Electric Power (kW)', 'Site Energy|Hot Water (E)', None),
        ('EV Electric Power (kW)', 'Site Energy|EV (E)', None),  # not yet tested
        ('PV Electric Power (kW)', 'Site Energy|PV (E)', None),  # not yet tested
        ('Battery Electric Power (kW)', 'Site Energy|Battery (E)', None),  # not yet tested
        ('Other Electric Power (kW)', ['Site Energy|Lights (E)', 'Site Energy|Lg. Appl. (E)',
                                       'Site Energy|Vent Fan (E)', 'Site Energy|Misc. (E)'], None),
        ('Total Gas Power (therms/hour)', 'Site Energy|Total (G)', Units.Btu2therm),
        ('HVAC Heating Gas Power (therms/hour)', 'Site Energy|Heating (G)', Units.Btu2therm),
        ('Water Heating Gas Power (therms/hour)', 'Site Energy|Hot Water (G)', Units.Btu2therm),
        ('Other Gas Power (therms/hour)', ['Site Energy|Lg. Appl. (G)', 'Site Energy|Misc. (G)'], Units.Btu2therm),

        ('Temperature - Indoor (C)', 'Living Space|Indoor Temperature', Units.F2C),
        ('Temperature - Garage (C)', 'Garage|Indoor Temperature', Units.F2C),
        ('Temperature - Foundation (C)', ['Unfinished Basement|Indoor Temperature',
                                          'Finished Basement|Indoor Temperature',
                                          'Crawlspace|Indoor Temperature'], Units.F2C),
        ('Temperature - Attic (C)', 'Unfinished Attic|Indoor Temperature', Units.F2C),
        ('Relative Humidity - Indoor (-)', 'Living Space|Indoor Relative Humidity', lambda x: x / 100),
        ('Humidity Ratio - Indoor (-)', 'Living Space|Indoor Humidity', None),

        ('HVAC Heating Delivered (kW)', ['Delivered Energy|Heating Delivered (main)',
                                         'Delivered Energy|Heating Delivered (suppl.)'], Units.Btu2kWh),
        # ('HVAC Heating EIR Ratio (-)', 'HP_HEAT-EIR-FT:Performance Curve Output Value [](Hourly)', None),
        # ('HVAC Heating Capacity Ratio (-)', 'HP_HEAT-CAP-FT:Performance Curve Output Value [](Hourly)', None),
        ('HVAC Cooling Delivered (kW)', 'Delivered Energy|Cooling Delivered (sensible)', Units.Btu2kWh),
        ('HVAC Cooling Latent Gains (kW)', 'Delivered Energy|Cooling Delivered (latent)', Units.Btu2kWh),
        # ('HVAC Cooling EIR Ratio (-)', 'COOL-EIR-FT:Performance Curve Output Value [](Hourly)', None),
        # ('HVAC Cooling Capacity Ratio (-)', 'COOL-CAP-FT:Performance Curve Output Value [](Hourly)', None),
        ('Hot Water Delivered (kW)', 'Delivered Energy|DHW Delivered Energy', Units.Btu2kWh),
        ('Hot Water Outlet Temperature (C)', 'Domestic Water System|Delivered Hot Water Temperature', Units.F2C),
        ('Hot Water Delivered (L/min)', 'Domestic Water System|Hot Water Use Flow Rate',
         lambda x: Units.pint2liter(x) * 8 / 60),  # Gal/hour to L/min

        ('Indoor Ventilation Flow Rate (m^3/s)', 'Living Space|Mech. Vent. (rated)', Units.cfm2m3_s),
        ('Indoor Infiltration Flow Rate (m^3/s)', 'Living Space|Infiltration', Units.cfm2m3_s),
        ('Foundation Infiltration Flow Rate (m^3/s)', ['Basement|Infiltration',
                                                       'Crawlspace|Infiltration'], Units.cfm2m3_s),
        ('Garage Infiltration Flow Rate (m^3/s)', 'Garage|Infiltration', Units.cfm2m3_s),
        ('Attic Infiltration Flow Rate (m^3/s)', 'Unfinished Attic|Infiltration', Units.cfm2m3_s),

        ('Temperature - Outdoor (C)', 'Weather|Outdoor Drybulb', Units.F2C),
        ('Ambient Humidity Ratio (-)', ' Weather|Outdoor Humidity Ratio', None),
        ('Wind Speed (m/s)', 'Weather|Wind Speed (at Weather Station)', None),
        ('DNI (W/m^2)', 'Weather|Direct Normal Solar', lambda x: x / Units.W_m22Btu_ft2(1)),
        ('DHI (W/m^2)', 'Weather|Diffuse Horizontal Solar', lambda x: x / Units.W_m22Btu_ft2(1)),
        ('GHI (W/m^2)', 'Weather|Total Horizontal Solar', lambda x: x / Units.W_m22Btu_ft2(1)),
        ('solar_WD', 'Living Space|Zone Glazing Transmitted Solar', Units.Btu2Wh),
    ]
    missing = pd.Series([0.0] * len(df), index=df.index)
    for col, eplus_col, unit_func in eplus_names_and_units:
        if isinstance(eplus_col, list):
            if any([col in df for col in eplus_col]):
                data = df.loc[:, [col for col in eplus_col if col in df]].sum(axis=1)
            else:
                data = None
        else:
            data = df.get(eplus_col)
        if data is not None:
            df[col] = unit_func(data) if unit_func is not None else data
        else:
            df[col] = missing

    # add gas power to heating main power
    df['HVAC Heating Main Power (kW)'] += Units.therms2kWh(df['HVAC Heating Gas Power (therms/hour)'])

    # add HVAC COP and SHR (note, excludes fan power and duct losses)
    heat_power = df['HVAC Heating Main Power (kW)'] + df['HVAC Heating ER Power (kW)']
    heat_duct_loss = Units.Btu2kWh(df.get('Delivered Energy|Heating Duct Losses', missing))
    df['HVAC Heating COP (-)'] = ((df['HVAC Heating Delivered (kW)'] + heat_duct_loss) / heat_power).fillna(0)
    cool_sensible = (df['HVAC Cooling Delivered (kW)'] +
                     Units.Btu2kWh(df.get('Delivered Energy|Cooling Duct Losses (sensible)', missing)))
    cool_latent = (df['HVAC Cooling Latent Gains (kW)'] +
                   Units.Btu2kWh(df.get('Delivered Energy|Cooling Duct Losses (latent)', missing)))
    df['HVAC Cooling COP (-)'] = ((cool_sensible + cool_latent) / df['HVAC Cooling Main Power (kW)']).fillna(0)
    df['HVAC Cooling SHR (-)'] = (cool_sensible / (cool_sensible + cool_latent)).fillna(0)

    # update HVAC delivered heat with fan power and duct losses
    df['HVAC Heating Delivered (kW)'] += df.get('Site Energy|Heating Fan/Pump (E)', missing)
    df['HVAC Cooling Delivered (kW)'] -= df.get('Site Energy|Cooling Fan/Pump (E)', missing)

    # calculate indoor wet bulb
    df['Temperature - Indoor Wet Bulb (C)'] = [
        psychrolib.GetTWetBulbFromRelHum(t, rh, p) for (t, rh, p) in zip(
            df['Temperature - Indoor (C)'], df['Relative Humidity - Indoor (-)'],
            Units.atm2Pa(df['Weather|Atmospheric Pressure']))
    ]

    # replace large negative numbers (usually an error or NA) to NA
    df = df.where(df > -100, np.nan)

    # update index to datetime
    assert len(df) == 8760
    df.index = pd.date_range(dt.datetime(default_year, 1, 1), periods=8760, freq=dt.timedelta(hours=1))

    return df


def create_comparison_metrics(df_hourly, eplus_hourly, df_metrics, eplus_metrics, **kwargs):
    # create data frame with metric values for OCHRE, E+, absolute error, and percentage error

    # aggregate metrics: record metric and calculate absolute and percent difference
    compare_metrics = {}
    for name in eplus_metrics:
        if name not in df_metrics or np.isnan(df_metrics[name]) or eplus_metrics[name] == 0:
            continue
        compare_metrics[name] = {
            'OCHRE': df_metrics[name],
            'EnergyPlus': eplus_metrics[name],
            'Absolute Error': abs_error(df_metrics[name], eplus_metrics[name]),
            'Percent Error (%)': pct_error(df_metrics[name], eplus_metrics[name]),
        }

    # calculate time series RMSE
    compare_metrics.update({name + ' RMSE': {
        'OCHRE': df_hourly[name].mean(),
        'EnergyPlus': eplus_hourly[name].mean(),
        'Absolute Error': rmse(df_hourly[name], eplus_hourly[name]),
        'Percent Error (%)': rmse(df_hourly[name], eplus_hourly[name]) / eplus_hourly[name].mean() * 100,
    } for name in eplus_hourly if name in df_hourly and eplus_hourly[name].sum() != 0
    })

    compare_metrics = pd.DataFrame(compare_metrics).T
    return compare_metrics
