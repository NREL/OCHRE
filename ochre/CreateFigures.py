import pandas as pd
import datetime as dt
import string

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

register_matplotlib_converters()
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator, offset_formats=['', '', '%b', '%b-%d', '%b-%d', '%b-%d %H:%M'])
formatter2 = mdates.ConciseDateFormatter(locator, show_offset=False)
formatter3 = mdates.ConciseDateFormatter(locator)

all_power_colors = {
    'PV': 'yellow',
    'Gas Generator': 'purple',
    'Battery': 'c',
    'Other': 'grey',
    'EV': 'm',
    'Water Heating': 'g',
    'HVAC Heating': 'r',
    'HVAC Cooling': 'b',
}
zones = {
    'Indoor': 'k',
    'Indoor Wet Bulb': 'grey',
    'Garage': 'g',
    'Foundation': 'orange',
    'Attic': 'purple',
    'Outdoor': 'y',
    'Ground': 'brown',
}
zone_data = [('Temperature - {} (C)'.format(zone), zone + ' Temp', color, False, 1) for zone, color in zones.items()]
ls_list = ['-', '--', ':', '-.']


def valid_file(s):
    # removes special characters like ",:$#*" from file names
    valid_chars = "-_.() {}{}".format(string.ascii_letters, string.digits)
    return ''.join(c for c in s if c in valid_chars)


# **** Time-based figures ****
def plot_daily_profile(df_raw, column, plot_average=True, plot_singles=True, plot_min=True, plot_max=True,
                       plot_sd=False, **kwargs):
    # sets datetime index to time, by default, plots the average, min, max, and individual days (singles).
    # plot_sd: plots a 95% confidence interval, uses average +/- 2 * standard dev.
    df = df_raw.copy()

    assert isinstance(df.index, pd.DatetimeIndex)
    df['Time of Day'] = df.index.time
    df['Date'] = df.index.date
    time_res = df.index[1] - df.index[0]
    # use arbitrary date for plotting
    times = pd.date_range(dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2),
                          freq=time_res, closed='left').to_pydatetime()

    fig, ax = plt.subplots()

    if plot_singles:
        df_singles = pd.pivot(df, 'Time of Day', 'Date', column)
        alpha = kwargs.pop('singles_alpha', 1 / len(df_singles.columns))
        for col in df_singles.columns:
            ax.plot(times, df_singles[col], 'k', alpha=alpha, label=None)

    df_agg = df.groupby('Time of Day')[column].agg(['min', 'max', 'mean', 'std'])
    if plot_max:
        ax.plot(times, df_agg['max'], 'k', label='Maximum')

    if plot_average:
        ax.plot(times, df_agg['mean'], 'b--', label='Average')

    if plot_min:
        ax.plot(times, df_agg['min'], 'k', label='Minimum')

    if plot_sd:
        df_agg['min'] = df_agg['mean'] - 2 * df_agg['std']
        df_agg['max'] = df_agg['mean'] + 2 * df_agg['std']
        alpha = kwargs.get('std_alpha', 0.4)
        ax.fill_between(times, df_agg['min'], df_agg['max'], alpha=alpha, label='95% C.I.')

    ax.legend()
    ax.set_ylabel(column)
    ax.set_xlabel('Time of Day')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(locator)

    return fig


def plot_power_stack(df, add_gas=False, **kwargs):
    # plots power columns in stacked line plot
    if add_gas:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', **kwargs)
    else:
        fig, ax1 = plt.subplots(**kwargs)
        ax2 = None
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex='all', constrained_layout=True,
    #                                          gridspec_kw={'height_ratios': [3, 1, 1, 1]}, figsize=(6, 8))

    # stacked electric power plot
    power_cols = {key: color for key, color in all_power_colors.items() if key +
                  ' Electric Power (kW)' in df.columns and df[key + ' Electric Power (kW)'].sum() != 0}
    df_power = df.loc[:, [key + ' Electric Power (kW)' for key in power_cols]]
    ax1.stackplot(df_power.index, df_power.clip(lower=0).values.T,
                  colors=power_cols.values(), labels=power_cols.keys())
    ax1.stackplot(df_power.index, df_power.clip(upper=0).loc[:, ::-1].values.T,
                  colors=list(power_cols.values())[::-1])

    ax1.plot(df.index, df['Total Electric Power (kW)'], 'k:', label='Net Power')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc='upper left')
    ax1.set_ylabel('Power (kW)')
    #     ax1.set_ylim((-4.9, None))
    #     ax1.axvspan(peak_start, peak_end, alpha=0.1, color='k')
    #     ax1.xaxis.set_major_formatter(formatter)

    # stacked gas power plot
    if add_gas:
        gas_cols = {key: color for key, color in all_power_colors.items() if key +
                    ' Gas Power (therms/hour)' in df.columns and df[key + ' Gas Power (therms/hour)'].sum() != 0}
        df_gas = df.loc[:, [key + ' Gas Power (therms/hour)' for key in gas_cols]]
        ax2.stackplot(df_gas.index, df_gas.clip(lower=0).values.T, colors=gas_cols.values(), labels=gas_cols.keys())
        # ax1.stackplot(df_gas.index, df_gas.clip(upper=0).loc[:, ::-1].values.T, colors=list(gas_cols.values())[::-1])

        # ax2.plot(df.index, df['Total Gas Power (therms/hour)'], 'k:', label='Net Power')
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[::-1], labels[::-1], loc='upper left')
        ax2.set_ylabel('Gas Power (therms/hour)')

    # ax2.plot(df.index, df['Temperature - Indoor (C)'], 'k', label='Indoor')
    # ax2.plot(df.index, df['Temperature - Outdoor (C)'], 'b', label='Outdoor')
    # ax2.legend(loc='upper left')
    # ax2.set_ylabel('Air\nTemperature ($^\circ$C)')
    #
    # ax3.plot(df.index, df.get('Hot Water Outlet Temperature (C)', missing), 'k',
    #          label='Water Temperature ($^\circ$C)')
    # ax3.legend(loc='upper left')
    # ax3.set_ylabel('Water\nTemperature ($^\circ$C)')
    #
    # ax4.plot(df.index, df.get('Battery SOC (-)', missing), 'k', label='Battery SOC (-)')
    # ax4.legend(loc='upper left')
    # ax4.set_ylabel('Battery\nSOC (-)')
    # ax4.xaxis.set_major_formatter(formatter)
    # ax4.set_xlim((start_day, end_day))
    ax1.xaxis.set_major_formatter(formatter)

    return fig


def make_comparison_plot(dfs_to_plot, plot_info=None, add_diff=None, **kwargs):
    # dfs_to_plot is a dict of {df_label: df}, where df is a time series DataFrame
    # plot_info is list of tuples: (col_name, col_label, color, twin axis, multiplier).
    #  - if None, defaults to all columns in dfs_to_plot[0]
    # add_diff adds a line showing the difference between the first and second df in dfs_to_plot
    if plot_info is None:
        df = list(dfs_to_plot.values())[0]
        plot_info = [(col_name, col_name, None, False, 1) for i, col_name in enumerate(df.columns)]

    ls_diff = ls_list[len(dfs_to_plot)]
    if add_diff is None:
        add_diff = []

    # remove data that doesn't exist in all of the dfs
    plot_info = [dat for dat in plot_info
                 if any([dat[0] in df and df[dat[0]].sum() != 0 for df in dfs_to_plot.values()])]
    if not plot_info:
        return None, (None, None)

    fig, ax1 = plt.subplots(**kwargs)
    if any([dat[3] for dat in plot_info]):
        ax2 = ax1.twinx()
    else:
        ax2 = None

    for (col_name, col_label, color, is_twin, mult) in plot_info:
        ax = ax2 if is_twin else ax1
        for i, (df_label, df) in enumerate(dfs_to_plot.items()):
            if col_name in df:
                label = col_label + ', ' + df_label if df_label else col_label
                ax.plot(df[col_name] * mult, color=color, ls=ls_list[i], label=label)
        if col_name in add_diff:
            dfs_list = list(dfs_to_plot.values())
            diff = (dfs_list[0][col_name] - dfs_list[1][col_name]) * mult
            ax.plot(diff, color=color, ls=ls_diff, label=col_label + ', Diff')

    ax1.legend(loc='upper left')
    if ax2 is not None:
        ax2.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(formatter)
    return fig, (ax1, ax2)


def plot_time_series(df, **kwargs):
    # df is a time series DataFrame
    return make_comparison_plot({'': df}, **kwargs)


def plot_external(dfs_to_plot, raw_weather=None):
    # plot irradiance: DNI, DHI, GHI
    plot_info = [
        ('DNI (W/m^2)', 'DNI', 'b', False, 1),
        ('DHI (W/m^2)', 'DHI', 'r', False, 1),
        ('GHI (W/m^2)', 'GHI', 'g', False, 1),
    ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None and raw_weather is not None:
        ax1.plot(raw_weather['directNormalIrradianceWsqm'], 'c', label='DNI, Raw EPW')
        ax1.plot(raw_weather['diffuseHorizontalRadiationWsqm'], 'm', label='DHI, Raw EPW')
        ax1.plot(raw_weather['downwardSolarRadiationWsqm'], 'y', label='GHI, Raw EPW')
        ax1.set_ylabel('Irradiance (W/m$^2$)')

    # outdoor temperature, wind speed, humidity, and pressure
    plot_info = [
        ('Temperature - Outdoor (C)', 'Ambient Temp', 'b', False, 1),
        ('Ambient Relative Humidity (-)', 'Rel. Humidity', 'm', True, 1),
        ('Ambient Humidity Ratio (-)', 'Humidity Ratio', 'm', True, 100),
        ('Wind Speed (m/s)', 'Wind Speed', 'y', True, 1 / 10),
        ('Ambient Pressure (kPa)', 'Pressure', 'g', True, 1 / 100),
    ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        if raw_weather is not None:
            ax1.plot(raw_weather.index, raw_weather['surfaceTemperatureCelsius'], 'b:', label='Ambient Temp, Raw EPW')
            ax2.plot(raw_weather.index, raw_weather['windSpeed'], 'g:', label='Wind Speed, Raw EPW')
        ax1.set_ylabel(r'Temperature ($^\circ$C)')
        ax2.set_ylabel('Rel. Humidity (-), Humidity Ratio (x100), Speed (m/s/10), Pressure (bar)')

    # plot wall and roof irradiance
    # if df is not None and eplus is not None:
    #     fig, ax3 = plt.subplots()
    #     ax3.plot(df.index, df['Horizontal Irradiance (W/m^2)'], 'r', label='Horizontal, OCHRE')
    #     ax3.plot(df.index, df['Wall Irradiance - Front (W/m^2)'], 'b', label='Front Wall, OCHRE')
    #     ax3.plot(df.index, df['Wall Irradiance - Right (W/m^2)'], 'm', label='Right Wall, OCHRE')
    #     ax3.plot(df.index, df['Wall Irradiance - Back (W/m^2)'], 'y', label='Back Wall, OCHRE')
    #     ax3.plot(df.index, df['Wall Irradiance - Left (W/m^2)'], 'g', label='Left Wall, OCHRE')
    #     # assumes front is south (true for Ft. Collins tests)
    #     hor = 'HORIZONTAL:Surface Outside Face Incident Solar Radiation Rate per Area [W/m2](TimeStep) '
    #     ax3.plot(df.index, eplus[hor].values, 'r--', label='Horizontal, E+')
    #     east = 'WALLEAST:Surface Outside Face Incident Solar Radiation Rate per Area [W/m2](TimeStep)'
    #     ax3.plot(df.index, eplus[east].values, 'b--', label='East Wall, E+')
    #     south = 'WALLSOUTH:Surface Outside Face Incident Solar Radiation Rate per Area [W/m2](TimeStep)'
    #     ax3.plot(df.index, eplus[south].values, 'm--', label='South Wall, E+')
    #     west = 'WALLWEST:Surface Outside Face Incident Solar Radiation Rate per Area [W/m2](TimeStep)'
    #     ax3.plot(df.index, eplus[west].values, 'y--', label='West Wall, E+')
    #     north = 'WALLNORTH:Surface Outside Face Incident Solar Radiation Rate per Area [W/m2](TimeStep)'
    #     ax3.plot(df.index, eplus[north].values, 'g--', label='North Wall, E+')
    #     ax3.legend(loc='upper right')
    #     ax3.set_ylabel('Irradiance (W/m$^2$)')
    #     ax3.xaxis.set_major_formatter(formatter)

    # Plot solar angles
    # plt.plot(df.index, df['Solar Time Angle'], 'r', label='Solar Time Angle')
    # plt.plot(df.index, df['Solar Zenith'], 'g', label='Solar Zenith')
    # plt.plot(df.index, df['Solar Azimuth'], 'b', label='Solar Azimuth')
    # plt.legend()


def plot_envelope(dfs_to_plot):
    # Plot internal temperatures, HVAC delivered, heat gains, window transmittance
    plot_info = zone_data + [
        ('Humidity Ratio - Indoor (-)', 'Humidity Ratio', 'b', True, 100),
        ('Relative Humidity - Indoor (-)', 'Indoor RH', 'c', True, 1),
        # ('Ambient Relative Humidity (-)', 'Outdoor RH', 'm', True, 1),
    ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel(r'Temperature ($^\circ$C)')
        ax2.set_ylabel('Relative Humidity (-) and Humidity Ratio (kg/kg x100)')

    # plot living space heat injections from boundaries and infiltration
    df = dfs_to_plot.get('OCHRE, exact', dfs_to_plot['OCHRE'])
    heats = [col for col in df.columns if 'Convection from' in col] + ['Indoor Infiltration Heat Gain (W)']
    if 'H_LIV' in df:
        df['Other Injected Heat Gains (W)'] = df['H_LIV'] - (
                df['Indoor Infiltration Heat Gain (W)'] +
                df['HVAC Heating Delivered (kW)'] * 1000 -
                df['HVAC Cooling Delivered (kW)'] * 1000
        )
        heats.append('Other Injected Heat Gains (W)')
    fig, ax3 = plt.subplots()
    df.loc[:, heats].plot(ax=ax3, legend=True)
    ax3.set_ylabel('Power (W)')

    # plot solar transmitted through windows (in W)
    plot_info = [
        ('solar_WD', 'Window Transmitted Solar', 'g', False, 1),
    ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Solar Power (W)')

    # plot all envelope temperatures
    temps = [col for col in df.columns if col[:2] == 'T_' and 'WH' not in col]
    if temps:
        fig, ax5 = plt.subplots()
        df.loc[:, temps].plot(ax=ax5, legend=True)
        ax5.set_ylabel(r'Temperature ($^\circ$C)')

    # plot infiltration and ventilation flow rates
    plot_info = [('Indoor Ventilation Flow Rate (m^3/s)', 'Ventilation', 'b', False, 1)] + [
        (zone + ' Infiltration Flow Rate (m^3/s)', zone + ' Infiltration', color, False, 1)
        for zone, color in zones.items()
    ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Flow Rate (m$^3$/s')


def plot_hvac(dfs_to_plot):
    # plot HVAC delivered heat and temperatures
    plot_info = zone_data + [
        ('HVAC Heating Delivered (kW)', 'Heating Delivered', 'r', True, 1),
        ('HVAC Cooling Delivered (kW)', 'Cooling Delivered', 'b', True, 1),
    ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info, add_diff=['HVAC Heating Delivered (kW)',
                                                                             'HVAC Cooling Delivered (kW)'])
    if ax1 is not None:
        ax1.set_ylabel(r'Temperature ($^\circ$C)')
    if ax2 is not None:
        ax2.set_ylabel('Power (kW)')

    # plot HVAC COP and SHR, Humidity Ratio, Temperatures
    plot_info = [('HVAC Heating COP (-)', 'Heating COP', 'r', False, 1),
                 ('HVAC Cooling COP (-)', 'Cooling COP', 'b', False, 1),
                 ('HVAC Cooling SHR (-)', 'Cooling SHR', 'g', False, 1),
                 ('HVAC Heating Speed (-)', 'Heating Speed', 'm', False, 1),
                 ('HVAC Cooling Speed (-)', 'Cooling Speed', 'c', False, 1),
                 # ('HVAC Heating EIR Ratio (-)', 'Heating EIR Ratio', 'm', False, 1),
                 # ('HVAC Heating Capacity Ratio (-)', 'Heating Capacity Ratio', 'orange', False, 1),
                 # ('HVAC Cooling EIR Ratio (-)', 'Cooling EIR Ratio', 'c', False, 1),
                 # ('HVAC Cooling Capacity Ratio (-)', 'Cooling Capacity Ratio', 'purple', False, 1),
                 ('Relative Humidity - Indoor (-)', 'Relative Humidity', 'k', True, 1),
                 ('Humidity Ratio - Indoor (-)', 'Humidity Ratio', 'grey', True, 100),
                 ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('COP or SHR (unitless)')
    if ax2 is not None:
        ax2.set_ylabel('Relative Humidity (-) and Humidity Ratio (kg/kg x100)')

    # plot HVAC-specific powers, electric only
    plot_info = [('HVAC Heating Main Power (kW)', 'Heating Main', 'r', False, 1),
                 ('HVAC Heating ER Power (kW)', 'Heating ER', 'orange', False, 1),
                 ('HVAC Heating Fan Power (kW)', 'Heating Fan', 'm', False, 1),
                 ('HVAC Cooling Main Power (kW)', 'Cooling Main', 'b', False, 1),
                 ('HVAC Cooling Fan Power (kW)', 'Cooling Fan', 'c', False, 1),
                 ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Power (kW)')


def plot_wh(dfs_to_plot):
    # plot water heater delivered heat, power, flow rate if they exist
    plot_info = [('Hot Water Delivered (kW)', 'Delivered Heat', 'c', False, 1),
                 ('Hot Water Delivered (L/min)', 'Flow Rate', 'b', False, 1),
                 ('Water Heating Delivered (kW)', 'WH Heat', 'r', False, 1),
                 ('Hot Water Outlet Temperature (C)', 'Outlet Temp', 'g', True, 1),
                 ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Hot Water Delivered (kW and L/min)')
        ax2.set_ylabel('Temperature ($^\circ$C)')

    # plot water heater power and COP
    plot_info = [('Water Heating Electric Power (kW)', 'Electric Power', 'm', False, 1),
                 ('Water Heating Gas Power (therms/hour)', 'Gas Power', 'purple', False, 1),
                 ('Water Heating COP (-)', 'COP', 'g', True, 1),
                 ]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Power (kW or therms/hour) and Flow Rate (L/min)')
    if ax2 is not None:
        ax2.set_ylabel('COP (-)')

    # plot water heater model temperatures
    df = dfs_to_plot.get('OCHRE, exact', dfs_to_plot['OCHRE'])
    wh_temps = [col for col in df.columns if col in ['T_WH' + str(i) for i in range(1, 13)]]
    if wh_temps:
        fig, ax = plt.subplots()
        df.loc[:, wh_temps].plot(ax=ax, legend=True)
        ax.set_ylabel(r'Temperature ($^\circ$C)')


def plot_powers(dfs_to_plot):
    # plot total and individual powers
    power_colors = [('Total', 'k'), ('HVAC Heating', 'r'), ('HVAC Cooling', 'b'), ('Water Heating', 'm'),
                    ('EV', 'g'), ('PV', 'y'), ('Battery', 'c'), ('Other', 'purple')]
    plot_info = [(end_use + ' Electric Power (kW)', end_use + ' Power', color, False, 1) for end_use, color in
                 power_colors]
    plot_info += [(end_use + ' Gas Power (therms/hour)', end_use + ' Power', color, True, 1)
                  for end_use, color in power_colors]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Electric Power (kW)')
        if ax2 is not None:
            ax2.set_ylabel('Gas Power (therms/hour)')


def plot_monthly_powers(dfs_to_plot):
    # aggregate power columns to energy by month
    def agg_monthly(df):
        hours = (df.index[1] - df.index[0]).total_seconds() / 3600
        df = df.loc[:, [col for col in df.columns if ' Electric Power (kW)' in col]] * hours
        df.columns = [col.replace(' Electric Power (kW)', ' Electric Energy (kWh)') for col in df.columns]
        df = df.resample('MS').sum()
        return df

    dfs_to_plot = {key: agg_monthly(val) for key, val in dfs_to_plot.items()}

    power_colors = [('Total', 'k'), ('HVAC Heating', 'r'), ('HVAC Cooling', 'b'), ('Water Heating', 'm'),
                    ('EV', 'g'), ('PV', 'y'), ('Battery', 'c'), ('Other', 'purple')]
    plot_info = [(end_use + ' Electric Energy (kWh)', end_use, color, False, 1) for end_use, color in
                 power_colors]
    # plot_info += [(end_use + ' Gas Power (therms/hour)', end_use + ' Power', color, True, 1)
    #               for end_use, color in power_colors]
    fig, (ax1, ax2) = make_comparison_plot(dfs_to_plot, plot_info)
    if ax1 is not None:
        ax1.set_ylabel('Monthly Energy Usage (kWh)')
        # if ax2 is not None:
        #     ax2.set_ylabel('Gas Power (therms/hour)')
