import pandas as pd
import datetime as dt
import string

from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

default_colors = cm.get_cmap('tab10').colors  # discrete color map with 10 colors
register_matplotlib_converters()
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
formatter2 = mdates.ConciseDateFormatter(locator, offset_formats=['', '', '%b', '%b-%d', '%b-%d', '%b-%d %H:%M'])
formatter3 = mdates.ConciseDateFormatter(locator)

# TODO: option to reverse order of plot lines, so first color stays on top, remains visible

all_power_colors = {
    'PV': 'yellow',
    'Gas Generator': 'purple',
    'Battery': 'c',
    'Other': 'grey',
    'Lighting': 'orange',
    'EV': 'm',
    'HVAC Heating': 'r',
    'HVAC Cooling': 'b',
    'Water Heating': 'g',
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
zone_data = [('Temperature - {} (C)'.format(zone), zone + ' Temp', color) for zone, color in zones.items()]
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
                          freq=time_res, inclusive='left').to_pydatetime()

    fig, ax = plt.subplots()

    if plot_singles:
        df_singles = pd.pivot(df, index="Time of Day", columns="Date", values=column)
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
    gas_cols = {key: color for key, color in all_power_colors.items() if key +
                ' Gas Power (therms/hour)' in df.columns and df[key + ' Gas Power (therms/hour)'].sum() != 0}
    if add_gas and not gas_cols:
        # no gas outputs, don't add gas plot
        add_gas = False
        if 'gridspec_kw' in kwargs:
            kwargs.pop('gridspec_kw')

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


def parse_plot_data(data, df=None):
    # data includes info for a single curve. Options include:
    #  - Tuple: (series, label, color, axis_left, axis_num, ls, mult)
    #  - Dict with same keys as above
    # df is optional DataFrame, should have data as a column name
    data_defaults = {'series': None,
                     'label': None,
                     'color': None,
                     'axis_left': True,
                     'axis_num': 0,
                     'ls': None,
                     'mult': 1}

    out = data_defaults.copy()
    if isinstance(data, (str, pd.Series)):
        data = (data,)
    if isinstance(data, tuple):
        data = dict(zip(list(data_defaults.keys())[:len(data)], data))
    out.update(data)

    if isinstance(out['series'], str):
        if df is None or out['series'] not in df:
            return {}
        out['series'] = df[out['series']]

    if out['label'] is None:
        out['label'] = out['series'].name
    return out


def plot_time_series_detailed(*plot_data, df_plot=None, step=None, colors=default_colors, **kwargs):
    # need: series with time index (or df with column name), label, axis #, twin axis, color, ls,
    # plot_data is a list of tuples or dicts with info for each line. See parse_plot_data for details
    # colors is a list of valid colors, used if color in plot data is an int
    if not plot_data:
        return None, (None, None)

    df = pd.DataFrame([parse_plot_data(data, df_plot) for data in plot_data])

    n = df['axis_num'].max() + 1
    fig, axes_left = plt.subplots(n, 1, sharex='all', **kwargs)
    if n == 1:
        axes_left = [axes_left]
    axes_right = [ax.twinx() if (~ df.loc[df['axis_num'] == i, 'axis_left']).any() else None
                  for i, ax in enumerate(axes_left)]

    for _, row in df.iterrows():
        ax = axes_left[row['axis_num']] if row['axis_left'] else axes_right[row['axis_num']]
        s = row['series']
        c = row['color'] if not isinstance(row['color'], int) else colors[row['color'] % len(colors)]
        if s is not None:
            if step is None:
                time_res = s.index[1] - s.index[0]
                use_step = time_res >= dt.timedelta(minutes=15)
            else:
                use_step = step
            if use_step:
                ax.step(s.index, s * row['mult'], color=c, ls=row['ls'], label=row['label'], where='post')
            else:
                ax.plot(s.index, s * row['mult'], color=c, ls=row['ls'], label=row['label'])

    for ax in axes_left:
        ax.legend(loc='upper left')
    for ax in axes_right:
        if ax is not None:
            ax.legend(loc='upper right')
    axes_left[-1].xaxis.set_major_formatter(formatter)

    if len(axes_left) == 1:
        axes_left = axes_left[0]
        axes_right = axes_right[0]

    return fig, (axes_left, axes_right)


def multi_comparison_plot(dfs_to_plot, plot_info=None, add_diff=None, update_ls=True, **kwargs):
    # creates a plot to compare multiple time series results across multiple data sets
    # dfs_to_plot is a dict of {df_label: df}, where df is a time series DataFrame
    # plot_info is list of column names, tuples, or dicts, see parse_plot_data for details
    #  - if None, defaults to all column names in first df from dfs_to_plot
    # add_diff adds a line showing the difference between the first and second df in dfs_to_plot
    # update_ls modifies the line style for each df. If False, will plot each comparison on separate axes
    # step=True plots the data as a step function. Defaults to False
    # Additional kwargs sent to plt.subplots
    if isinstance(dfs_to_plot, pd.DataFrame):
        dfs_to_plot = {'': dfs_to_plot}
    df_first = list(dfs_to_plot.values())[0]
    if plot_info is None:
        plot_info = df_first.columns.to_list()

    if add_diff is None:
        add_diff = []

    plot_data = []
    for i, info in enumerate(plot_info):
        for j, (name, df) in enumerate(dfs_to_plot.items()):
            data = parse_plot_data(info, df)
            if not data:
                continue

            # update label
            if name:
                data['label'] += f', {name}'

            # update axis or ls
            if update_ls:
                data['ls'] = ls_list[j] if j < len(ls_list) else ls_list[0]
                # update color so all are the same
                if data['color'] is None:
                    data['color'] = i
            else:
                data['axis_num'] = i

            plot_data.append(data)

        # Add difference between first 2 dfs
        data = parse_plot_data(info, df_first)
        col_name = data.get('label')
        if col_name is not None and col_name in add_diff:
            dfs_list = list(dfs_to_plot.values())
            if len(dfs_list) >= 2 and col_name in dfs_list[0] and col_name in dfs_list[1]:
                data['series'] = (dfs_list[0][col_name] - dfs_list[1][col_name])
                data['label'] += ', Diff'
                if update_ls:
                    n = len(dfs_to_plot)
                    data['ls'] = ls_list[n] if n < len(ls_list) else ls_list[0]
                    # update color so all are the same
                    if data['color'] is None:
                        data['color'] = i
                else:
                    data['axis_num'] = i
                plot_data.append(data)

    return plot_time_series_detailed(*plot_data, **kwargs)


def plot_time_series(df, **kwargs):
    # df is a time series DataFrame
    return multi_comparison_plot({'': df}, **kwargs)


def plot_external(dfs_to_plot, **kwargs):
    # plot irradiance: DNI, DHI, GHI
    plot_info = [
        ('DNI (W/m^2)', 'DNI'),
        ('DHI (W/m^2)', 'DHI'),
        ('GHI (W/m^2)', 'GHI'),
    ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel(r'Irradiance (W/m$^2$)')

    # outdoor temperature, wind speed, humidity, and pressure
    plot_info = [
        ('Temperature - Outdoor (C)', 'Ambient Temp', 'b', True, 0),
        ('Ambient Relative Humidity (-)', 'Rel. Humidity', 'm', False, 0),
        ('Ambient Humidity Ratio (-)', 'Humidity Ratio', 'r', False, 0, None, 100),
        ('Wind Speed (m/s)', 'Wind Speed', 'c', True, 1),
        ('Ambient Pressure (kPa)', 'Pressure', 'g', False, 1),
    ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1[0] is not None:
        ax1[0].set_ylabel(r'Temperature ($^\circ$C)')
    if ax2[0] is not None:
        ax2[0].set_ylabel('Rel. Humidity (-), Humidity Ratio (x100)')
    if ax1[1] is not None:
        ax1[1].set_ylabel('Speed (m/s)')
    if ax2[1] is not None:
        ax2[1].set_ylabel('Pressure (kPa)')

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

    return fig


def plot_envelope(dfs_to_plot, **kwargs):
    # Basic envelope plots

    # plot HVAC delivered heat and temperatures
    plot_info = zone_data + [
        ('HVAC Heating Delivered (W)', 'Heating Delivered', 'r', False),
        ('HVAC Cooling Delivered (W)', 'Cooling Delivered', 'b', False),
    ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, add_diff=['HVAC Heating Delivered (W)',
                                                                              'HVAC Cooling Delivered (W)'], **kwargs)
    if ax1 is not None:
        ax1.set_ylabel(r'Temperature ($^\circ$C)')
    if ax2 is not None:
        ax2.set_ylabel('Heat Delivered (W)')

    # plot all component loads
    plot_info = [
        ('Forced Ventilation Heat Gain - Indoor (W)', 'Forced Ventilation', 'lightgreen'),
        ('Natural Ventilation Heat Gain - Indoor (W)', 'Natural Ventilation', 'darkgreen'),
        ('Infiltration Heat Gain - Indoor (W)', 'Infiltration', 'm'),
        ('Internal Heat Gain - Indoor (W)', 'Internal Gains', 'c'),
        ('HVAC Heating Duct Losses (W)', 'Ducts (Heating)', 'r'),
        ('HVAC Cooling Duct Losses (W)', 'Ducts (Cooling)', 'b'),
        # ('HVAC Heating Delivered (W)', 'Heating Delivered', 'r', False),
        # ('HVAC Cooling Delivered (W)', 'Cooling Delivered', 'b', False),
        # ('Temperature - Indoor (C)', 'Indoor Temp', 'k')
        # ('Temperature - Outdoor (C)', 'Ambient Temp', 'k', True, 1),
        # ('Wind Speed (m/s)', 'Wind Speed', 'r', False, 1),
    ]
    fig, (ax1, _) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel('Component Load (W)')

    # plot infiltration and ventilation flow rates with ambient temp and wind speed
    plot_info = [('Forced Ventilation Flow Rate - Indoor (m^3/s)', 'Forced Ventilation', 'b'),
                 ('Natural Ventilation Flow Rate - Indoor (m^3/s)', 'Natural Ventilation', 'c'),
                 *[(f'Infiltration Flow Rate - {zone} (m^3/s)', zone + ' Infiltration', color)
                   for zone, color in zones.items()],
                 ('Temperature - Outdoor (C)', 'Ambient Temp', 'k', True, 1),
                 ('Wind Speed (m/s)', 'Wind Speed', 'r', False, 1),
                 ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1[0].set_ylabel('Flow Rate (m$^3$/s)')
        ax1[1].set_ylabel(r'Temperature ($^\circ$C)')
    if ax2[1] is not None:
        ax2[1].set_ylabel('Wind Speed (m/s)')

    return fig

def plot_envelope_detailed(dfs_to_plot, **kwargs):
    # Detailed envelope plots

    # Plot internal temperatures, and humidity
    plot_info = zone_data + [
        ('Humidity Ratio - Indoor (-)', 'Humidity Ratio', 'b', False, 0, None, 100),
        ('Relative Humidity - Indoor (-)', 'Indoor RH', 'c', False),
        # ('Ambient Relative Humidity (-)', 'Outdoor RH', 'm', True, 1),
    ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel(r'Temperature ($^\circ$C)')
    if ax2 is not None:
        ax2.set_ylabel('Relative Humidity (-) and Humidity Ratio (kg/kg x100)')

    # plot living space heat injections from boundaries, infiltration, occupants, and HVAC
    df = dfs_to_plot.get('OCHRE, exact', list(dfs_to_plot.values())[0])
    for zone in zones.keys():
        convection_heat_cols = [col for col in df.columns if ('Convection from' in col and f'to {zone}' in col)]
        plot_info = [(col,) for col in convection_heat_cols]
        # (series, label, color, axis_left, axis_num, ls, mult)
        if convection_heat_cols:
            plot_info.append((df.loc[:, convection_heat_cols].sum(axis=1), f'Net Convection Heat Gain - {zone} (W)'))
        if zone == 'Indoor' and ('HVAC Heating Delivered (W)' in df.columns or
                                 'HVAC Cooling Delivered (W)' in df.columns):
            df[f'HVAC Heat Gain - {zone} (W)'] = (df.get('HVAC Heating Delivered (W)', 0) -
                                                  df.get('HVAC Cooling Delivered (W)', 0))
        injection_heat_cols = [col for col in [
            f'Occupancy Heat Gain - {zone} (W)',
            f'Infiltration Heat Gain - {zone} (W)',
            f'Radiation Heat Gain - {zone} (W)',
            f'HVAC Heat Gain - {zone} (W)',
        ] if col in df.columns]
        plot_info.extend([(col,) for col in injection_heat_cols])
        if f'Net Sensible Heat Gain - {zone} (W)' in df.columns:
            net = df[f'Net Sensible Heat Gain - {zone} (W)']
            other = net - df.loc[:, injection_heat_cols].sum(axis=1)
            plot_info.append((other, f'Other Injected Heat Gain - {zone} (W)'))
            plot_info.append((net, f'Net Injected Heat Gain - {zone} (W)'))
        if plot_info:
            fig, (ax1, ax2) = plot_time_series_detailed(*plot_info, df_plot=df)
            ax1.set_ylabel(f'{zone} Heat Gain (W)')

    # plot solar transmitted through windows (in W)
    plot_info = [
        ('Window Transmitted Solar Gain (W)', 'Window Transmitted Solar', 'g'),
    ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel('Solar Power (W)')

    # plot all envelope temperatures
    temps = [col for col in df.columns if col[:2] == 'T_' and 'WH' not in col]
    if temps:
        fig, ax5 = plt.subplots()
        df.loc[:, temps].plot(ax=ax5, legend=True)
        ax5.set_ylabel(r'Temperature ($^\circ$C)')

    return fig


def plot_hvac(dfs_to_plot, **kwargs):
    # plot HVAC COP and SHR, Humidity Ratio, Temperatures
    plot_info = [('HVAC Heating COP (-)', 'Heating COP', 'r', True),
                 ('HVAC Cooling COP (-)', 'Cooling COP', 'b', True),
                 ('HVAC Cooling SHR (-)', 'Cooling SHR', 'g', True),
                 ('HVAC Heating Speed (-)', 'Heating Speed', 'm', True),
                 ('HVAC Cooling Speed (-)', 'Cooling Speed', 'c', True),
                 # ('HVAC Heating EIR Ratio (-)', 'Heating EIR Ratio', 'm', False),
                 # ('HVAC Heating Capacity Ratio (-)', 'Heating Capacity Ratio', 'orange', False),
                 # ('HVAC Cooling EIR Ratio (-)', 'Cooling EIR Ratio', 'c', False),
                 # ('HVAC Cooling Capacity Ratio (-)', 'Cooling Capacity Ratio', 'purple', False),
                 ('Relative Humidity - Indoor (-)', 'Relative Humidity', 'k', False),
                 ('Humidity Ratio - Indoor (-)', 'Humidity Ratio', 'grey', False, 0, None, 100),
                 ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel('COP or SHR (unitless)')
    if ax2 is not None:
        ax2.set_ylabel('Relative Humidity (-) and Humidity Ratio (kg/kg x100)')

    # plot HVAC-specific powers, electric only
    plot_info = [('HVAC Heating Main Power (kW)', 'Heating Main', 'r'),
                 ('HVAC Heating ER Power (kW)', 'Heating ER', 'orange'),
                 ('HVAC Heating Fan Power (kW)', 'Heating Fan', 'm'),
                 ('HVAC Cooling Main Power (kW)', 'Cooling Main', 'b'),
                 ('HVAC Cooling Fan Power (kW)', 'Cooling Fan', 'c'),
                 ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel('Power (kW)')

    return fig


def plot_wh(dfs_to_plot, **kwargs):
    # plot water heater delivered heat, power, flow rate if they exist
    plot_info = [('Hot Water Delivered (W)', 'Delivered Heat', 'c'),
                 ('Water Heating Delivered (W)', 'WH Heat', 'r'),
                 ('Hot Water Delivered (L/min)', 'Flow Rate', 'b', False),
                 ('Hot Water Outlet Temperature (C)', 'Outlet Temp', 'g', True, 1),
                 ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1[0].set_ylabel('Hot Water Delivered (W)')
        ax2[0].set_ylabel('Hot Water Delivered (L/min)')
        ax1[1].set_ylabel('Temperature ($^\circ$C)')

    # plot water heater power and COP
    plot_info = [('Water Heating Electric Power (kW)', 'Electric Power', 'm'),
                 ('Water Heating Gas Power (therms/hour)', 'Gas Power', 'purple'),
                 ('Water Heating COP (-)', 'COP', 'g', False),
                 ]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel('Power (kW or therms/hour)')
    if ax2 is not None:
        ax2.set_ylabel('COP (-)')

    # plot water heater model temperatures
    df = dfs_to_plot.get('OCHRE, exact', list(dfs_to_plot.values())[0])
    wh_temps = [col for col in df.columns if col in ['T_WH' + str(i) for i in range(1, 13)]]
    if wh_temps:
        fig, ax = plt.subplots()
        df.loc[:, wh_temps].plot(ax=ax, legend=True)
        ax.set_ylabel(r'Temperature ($^\circ$C)')

    return fig


def plot_end_use_powers(dfs_to_plot, **kwargs):
    # plot total and individual powers
    power_colors = all_power_colors.copy()
    power_colors['Total'] = 'k'
    plot_info = [(end_use + ' Electric Power (kW)', end_use, color) for end_use, color in
                 power_colors.items()]
    plot_info += [(end_use + ' Gas Power (therms/hour)', end_use, color, True, 1)
                  for end_use, color in power_colors.items()]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
    if ax1 is not None:
        ax1[0].set_ylabel('Electric Power (kW)')
        if ax1[1] is not None:
            ax1[1].set_ylabel('Gas Power (therms/hour)')

    return fig


def plot_all_powers(dfs_to_plot, **kwargs):
    # plot individual equipment powers
    for fuel_text in [' Electric Power (kW)', ' Gas Power (therms/hour)']:
        cols = list({col for df in dfs_to_plot.values() for col in df.columns if fuel_text in col})
        labels = {col.replace(fuel_text, ''): col for col in cols}
        labels.pop('Total', None)
        labels.pop('Other', None)
        labels.pop('Lighting', None)
        plot_info = [(col, label, None) for label, col in labels.items()]
        fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, **kwargs)
        if ax1 is not None:
            ax1.set_ylabel(fuel_text[1:])


def plot_monthly_powers(dfs_to_plot, add_gas=False, **kwargs):
    # aggregate power columns to energy by month
    def agg_monthly(df):
        hours = (df.index[1] - df.index[0]).total_seconds() / 3600
        keep_cols = [col for col in df.columns if ' Electric Power (kW)' in col]
        if add_gas:
            keep_cols += [col for col in df.columns if ' Gas Power (therms/hour)' in col]
        df = df.loc[:, keep_cols]
        df = df.resample('MS').sum() * hours
        return df

    dfs_to_plot = {key: agg_monthly(val) for key, val in dfs_to_plot.items()}

    power_colors = all_power_colors.copy()
    power_colors['Total'] = 'k'
    plot_info = [(end_use + ' Electric Power (kW)', end_use, color)
                 for end_use, color in power_colors.items()]
    if add_gas:
        plot_info += [(end_use + ' Gas Power (therms/hour)', end_use, color, False)
                      for end_use, color in power_colors.items()]
    fig, (ax1, ax2) = multi_comparison_plot(dfs_to_plot, plot_info, step=True, **kwargs)
    if ax1 is not None:
        ax1.set_ylabel('Monthly Energy Usage (kWh)')
    if ax2 is not None:
        ax2.set_ylabel('Gas Energy Usage (therms)')

    return fig
