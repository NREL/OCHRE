import os
import numpy as np
import pandas as pd
import datetime as dt
import psychrolib

from ochre import Units

psychrolib.SetUnitSystem(psychrolib.SI)

this_path = os.path.dirname(__file__)
main_path = os.path.abspath(os.path.join(this_path, os.pardir))
default_input_path = os.path.join(main_path, 'defaults')
default_output_path = os.path.join(main_path, 'outputs')

header_file = os.path.join(this_path, 'Models', 'sam_weather_header.csv')
default_sam_weather_file = os.path.join(this_path, 'Models', 'SAM_weather_{}.csv')  # has placeholder for house name

SCHEDULE_COLUMNS = {
    'Heating Setpoint (C)': 'heating_setpoint',
    'Cooling Setpoint (C)': 'cooling_setpoint',
    'Mains Temperature (C)': 'mains_temperature',
    'Ground Temperature (C)': 'ground_temperature',
    'Ambient Dry Bulb (C)': 'ambient_dry_bulb',
    'Sky Temperature (C)': 'sky_temperature',
    'Ambient Humidity Ratio (-)': 'ambient_humidity',
    'Ambient Pressure (kPa)': 'ambient_pressure',
    'Wind Speed (m/s)': 'wind_speed',
    'Ventilation Rate': 'ventilation_rate',
}


def get_rc_params(**properties):
    # create RC params
    # Capacitors - convert kJ/K to J/K (Assumes H is in Watts)
    res = {key: val for key, val in properties.items() if key[:2] == 'R_'}
    cap = {key: val * 1000 for key, val in properties.items() if key[:2] == 'C_'}

    # Add params for furniture nodes
    if 'Living Furniture properties' in properties:
        cap['C_IM1'] = properties['Living Furniture properties']['Capacitance (kJ/m^2-K)']  # in kJ/m^2-K
        res['R_IM1'] = properties['Living Furniture properties']['R-value (m^2-K/W)']  # m^2-K/W
        res['R_IM_ext'] = 0.440334654337  # m^2-K/W
        res['R_IM_int'] = 0.440334654337  # m^2-K/W
    if 'Garage Furniture properties' in properties:
        cap['C_GM1'] = properties['Garage Furniture properties']['Capacitance (kJ/m^2-K)']  # in kJ/m^2-K
        res['R_GM1'] = properties['Garage Furniture properties']['R-value (m^2-K/W)']  # m^2-K/W
        res['R_GM_ext'] = 0.440334654337  # m^2-K/W
        res['R_GM_int'] = 0.440334654337  # m^2-K/W
    if 'Basement Furniture properties' in properties:
        cap['C_BM1'] = properties['Basement Furniture properties']['Capacitance (kJ/m^2-K)']  # in kJ/m^2-K
        res['R_BM1'] = properties['Basement Furniture properties']['R-value (m^2-K/W)']  # m^2-K/W
        res['R_BM_ext'] = 0.440334654337  # m^2-K/W
        res['R_BM_int'] = 0.440334654337  # m^2-K/W
    return res, cap


def import_properties(properties_file, input_path=default_input_path, **kwargs):
    # Assumes initial file structure, e.g. "city          = CA_RIVERSIDE-MUNI"
    out = {}
    if not os.path.isabs(properties_file):
        properties_file = os.path.join(input_path, 'BEopt Files', properties_file)

    # Open file and parse
    with open(properties_file, 'r') as prop_file:
        for line in prop_file:
            if line[0] == '#' or ' = ' not in line:
                # line doesn't include a property
                continue
            line_split = line.split(' = ')
            key = line_split[0].strip()

            if len(line_split) == 2:
                # convert to string, float, or list
                val = line_split[1].strip()
                try:
                    out[key] = eval(val)
                except (NameError, SyntaxError):
                    out[key] = val

            elif len(line_split) > 2:
                # convert to dict (all should be floats)
                line_list = '='.join(line_split[1:]).split('   ')
                line_list = [tuple(s.split('=')) for s in line_list]
                out[key] = {k.strip(): float(v.strip()) for (k, v) in line_list}

    out['building volume (m^3)'] = out['finished floor area (m^2)'] * out['ceiling height (m)']

    return out


def import_zip_model(zip_file='ZIP_loads.csv', input_path=default_input_path, **kwargs):
    if not os.path.isabs(zip_file):
        zip_file = os.path.join(input_path, zip_file)
    df = pd.read_csv(zip_file, index_col='Load', usecols=range(10))
    return df.to_dict('index')


def import_generic(file_name, keep_cols=None, remove_cols=None, fillna=None, **kwargs):
    if keep_cols is None:
        df = pd.read_csv(file_name)
    else:
        df = pd.read_csv(file_name, usecols=keep_cols)

    if remove_cols is not None:
        df = df.loc[:, [col for col in df.columns if col not in remove_cols]]
    if fillna is not None:
        df = df.fillna(fillna)

    # keep times within simulation
    df = resample(df, **kwargs)
    return df


def import_schedule(schedule_file, **kwargs):
    df = import_generic(schedule_file, remove_cols=['DayOfWeek'], **kwargs)
    df.columns = [col.strip() for col in df.columns]

    # update temperature units
    temp_cols = [col for col in df.columns if '(F)' in col]
    df.loc[:, temp_cols] = Units.F2C(df.loc[:, temp_cols])
    df = df.rename(columns={col: col.replace('(F)', '(C)') for col in temp_cols})

    # check if cool-heat setpoint difference is large enough, if not throw a warning
    setpoint_diff = df['Cooling Setpoint (C)'] - df['Heating Setpoint (C)']
    if min(setpoint_diff) < 1:
        # if min(setpoint_diff) < 0:
        #     raise Exception('ERROR: Cooling setpoint is equal or less than heating setpoint in schedule file')
        print('WARNING: Cooling setpoint is within 1C of heating setpoint in schedule file.',
              'Separating setpoints by at least 1C.')
        setpoint_avg = (df['Cooling Setpoint (C)'] + df['Heating Setpoint (C)']) / 2
        df['Cooling Setpoint (C)'] = df['Cooling Setpoint (C)'].clip(lower=setpoint_avg + 0.5)
        df['Heating Setpoint (C)'] = df['Heating Setpoint (C)'].clip(upper=setpoint_avg - 0.5)

    return df


def run_solar_calcs(df, **kwargs):
    # Calculate solar time from time, longitude, time zone
    # See Duffie and Beckman, Section 1.4 - 1.6
    times = df.index - df.index.floor(dt.timedelta(days=1))
    yday = df.index.dayofyear.values
    b = (yday - 1) * 2 * np.pi / 365
    e = 229.2 * (0.000075 + 0.001868 * np.cos(b) - 0.032077 * np.sin(b) -
                 0.014615 * np.cos(2 * b) - 0.04089 * np.sin(2 * b))  # in minutes
    solar_time = (times.total_seconds().values / 3600 - 12) * np.pi / 12
    solar_time += kwargs['timezone'] * np.pi / 12 - kwargs['longitude'] * np.pi / 180
    solar_time += e / 60 / 12 * np.pi
    # solar_time += 30 / 60 / 12 * np.pi  # HACK, shifts solar time by 30 minutes
    df['Solar Time Angle'] = np.rad2deg(solar_time)

    # Calculate solar zenith angle from azimuth, latitude, declination
    # See Duffie and Beckman, Section 1.4 - 1.6
    latitude = kwargs['latitude'] * np.pi / 180
    # declination = 23.45 * np.pi / 180 * np.cos(2 * np.pi / 365 * (yday + 284))
    declination = (0.006918 - 0.399912 * np.cos(b) + 0.070257 * np.sin(b) - 0.006758 * np.cos(2 * b) +
                   0.000907 * np.sin(2 * b) - 0.002697 * np.cos(3 * b) + 0.00148 * np.sin(3 * b))
    zenith = np.arccos(np.sin(latitude) * np.sin(declination) +
                       np.cos(latitude) * np.cos(declination) * np.cos(solar_time))
    df['Solar Zenith'] = np.rad2deg(zenith)
    azimuth = np.sign(solar_time) * abs(np.arccos((np.cos(zenith) * np.sin(latitude) - np.sin(declination)) /
                                                  np.sin(zenith) / np.cos(latitude)))
    df['Solar Azimuth'] = np.rad2deg(azimuth)

    # Add irradiance columns if missing
    if 'GHI (W/m^2)' not in df.columns:
        df['GHI (W/m^2)'] = (df['DHI (W/m^2)'] + df['DNI (W/m^2)'] * np.cos(zenith)).clip(lower=0)
    if 'DHI (W/m^2)' not in df.columns:  # True for some irradiance sensors
        df['DHI (W/m^2)'] = (df['GHI (W/m^2)'] - df['DNI (W/m^2)'] * np.cos(zenith)).clip(lower=0)

    # Add Extraterrestrial Radiation
    df['Extraterrestrial Radiation'] = 1367 * (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b) +
                                               0.000719 * np.cos(2*b) + 0.000077 * np.sin(2*b))

    # force all irradiance to 0 at night
    # nighttime = zenith > np.pi / 2
    # df.loc[nighttime, 'GHI (W/m^2)'] = 0
    # df.loc[nighttime, 'DNI (W/m^2)'] = 0
    # df.loc[nighttime, 'DHI (W/m^2)'] = 0
    return df


def import_weather(weather_file, return_raw=False, create_sam_file=False, **kwargs):
    required_cols = ['Ambient Dry Bulb (C)', 'Ambient Relative Humidity (-)', 'Ambient Pressure (kPa)',
                     'GHI (W/m^2)', 'DNI (W/m^2)', 'DHI (W/m^2)', 'Wind Speed (m/s)', 'Sky Temperature (C)']
    is_epw = os.path.splitext(weather_file)[1] == '.epw'

    if is_epw:
        epw_columns = [
            'yr', 'mo', 'day', 'hr', 'min', 'flags',
            'surfaceTemperatureCelsius', 'surfaceDewpointTemperatureCelsius', 'relativeHumidityPercent',
            'surfaceAirPressurePascals', 'Extraterrestrial HorRad', 'Extra DNR', 'Horiz infra Rad from Sky',
            'downwardSolarRadiationWsqm', 'directNormalIrradianceWsqm',
            'diffuseHorizontalRadiationWsqm', 'GlobalHorizIll', 'DirectNormIll', 'DiffHorizIll', 'ZenithLum', 'WindDir',
            'windSpeed', 'TotalSkyCover',
            'OpaqueSkyCover', 'Visiblilty', 'CeilingHeight', 'PresentWeatherObserved', 'PresentWeatherCodes',
            'PrecipWater',
            'AerosolOp', 'SnowDepth', 'Days',
            'Unknown1', 'Unknown2', 'Unknown3'
        ]
        epw_column_to_keep = ['surfaceTemperatureCelsius', 'relativeHumidityPercent', 'surfaceAirPressurePascals',
                              'downwardSolarRadiationWsqm', 'directNormalIrradianceWsqm',
                              'diffuseHorizontalRadiationWsqm', 'windSpeed', 'Sky Temperature (C)']
        df = pd.read_csv(weather_file, header=None, skiprows=8)
        df = df.iloc[:, :len(epw_columns)]
        df.columns = epw_columns[:len(df.columns)]

        if return_raw:
            return df

        # add sky temperature
        # see https://www.energyplus.net/sites/default/files/docs/site_v8.3.0/EngineeringReference/05-Climate/index.html
        #  - EnergyPlus Sky Temperature Calculation
        df['Sky Temperature (C)'] = Units.K2C((df['Horiz infra Rad from Sky'] / 5.6697E-8) ** 0.25)

        # create datetime index, remove leap day if it exists
        df = df.loc[(df['mo'] != 2) | (df['day'] != 29)]
        year = kwargs['start_time'].year
        df['yr'] = year
        df['hr'] = df['hr'] - 1
        date_info = df.iloc[:, :5]
        date_info.columns = ['year', 'month', 'day', 'hour', 'minute']
        df.index = pd.to_datetime(date_info)

        # remove and rename columns
        df = df.loc[:, epw_column_to_keep]
        df.columns = required_cols

        # fix units: Pressure Pa to kPa, RH percentage to fraction
        df['Ambient Pressure (kPa)'] /= 1000
        df['Ambient Relative Humidity (-)'] /= 100

    else:
        # assumes csv is in NSRDB file structure
        nsrdb_cols = ['Temperature', 'Relative Humidity', 'Pressure', 'GHI', 'DNI', 'DHI', 'Wind Speed',
                      'Sky Temperature']

        # load header for lat/lon and time zone
        header = pd.read_csv(weather_file, nrows=1).iloc[0].to_dict()
        if kwargs.get('latitude') is None:
            kwargs['latitude'] = header['Latitude']
        elif abs(kwargs['latitude'] - header['Latitude']) > 5:
            print('WARNING: Weather data latitude ({}) is off from simulation latitude ({})'.format(
                header['Latitude'], kwargs['latitude']))
        if kwargs.get('longitude') is None:
            kwargs['longitude'] = header['Longitude']
        elif abs(kwargs['longitude'] - header['Longitude']) > 5:
            print('WARNING: Weather data longitude ({}) is off from simulation longitude ({})'.format(
                header['Longitude'], kwargs['longitude']))
        if kwargs.get('timezone') is None:
            kwargs['timezone'] = header['Time Zone']
        elif kwargs['timezone'] != header['Time Zone']:
            print('WARNING: Weather data time zone ({}) is off from simulation time zone ({})'.format(
                header['Time Zone'], kwargs['timezone']))

        # load weather data
        df = pd.read_csv(weather_file, skiprows=2)
        if return_raw:
            return df

        # remove leap day if it exists
        df = df.loc[(df['Month'] != 2) | (df['Day'] != 29)]
        year = kwargs['start_time'].year
        if df.loc[0, 'Year'] != year:
            print('WARNING: Simulation year ({}) not equal to Weather data year ({}).'.format(year, df.loc[0, 'Year']))
        df['Year'] = year

        # add time index
        date_info = df.loc[:, ['Year', 'Month', 'Day', 'Hour', 'Minute']]
        df.index = pd.to_datetime(date_info)

        # Set sky temperature to NaN
        df['Sky Temperature'] = np.nan

        # keep required columns and rename
        df = df.loc[:, nsrdb_cols]
        df.columns = required_cols

        # fix units: Pressure mbar to kPa, RH percentage to fraction
        df['Ambient Pressure (kPa)'] /= 10
        df['Ambient Relative Humidity (-)'] /= 100

    # add humidity ratio
    df['Ambient Humidity Ratio (-)'] = psychrolib.GetHumRatioFromRelHum(df['Ambient Dry Bulb (C)'].values,
                                                                        df['Ambient Relative Humidity (-)'].values,
                                                                        df['Ambient Pressure (kPa)'].values * 1000)

    # interpolate before running solar calculations
    offset = dt.timedelta(minutes=30) if is_epw else kwargs.get('weather_offset')
    if create_sam_file:
        df = resample(df, interpolate=True, offset=offset, annual_output=True, **kwargs)
    else:
        df = resample(df, interpolate=True, offset=offset, **kwargs)

    # calculate solar angles and irradiance
    df = run_solar_calcs(df, **kwargs)

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ImportError('Weather file missing required columns:', missing)

    # save annual weather file for SAM
    if create_sam_file:
        create_sam_weather_file(df, **kwargs)

        # keep simulation times only
        df = resample(df, **kwargs)

    return df


def create_sam_weather_file(df_input, sam_weather_file=None, **kwargs):
    # Convert weather data to SAM readable format
    if sam_weather_file is None:
        sam_weather_file = default_sam_weather_file.format(kwargs['house_name'])

    # load header file
    header = pd.read_csv(header_file)

    # update params
    header['Latitude'] = kwargs['latitude']
    header['Longitude'] = kwargs['longitude']
    header['Time Zone'] = kwargs['timezone']
    header['Elevation'] = kwargs.get('elevation', 0)  # TODO: not in properties file
    header['Local Time Zone'] = kwargs['timezone']

    # build main weather data
    columns = ['Year', 'Month', 'Day', 'Hour', 'Minute',
               'DNI (W/m^2)', 'GHI (W/m^2)', 'DHI (W/m^2)', 'Temperature', 'Wind Speed (m/s)']
    df = df_input.loc[:, ['DNI (W/m^2)', 'GHI (W/m^2)', 'DHI (W/m^2)', 'Wind Speed (m/s)']]
    df['Year'] = df_input.index.year
    df['Month'] = df_input.index.month
    df['Day'] = df_input.index.day
    df['Hour'] = df_input.index.hour
    df['Minute'] = df_input.index.minute
    df['Temperature'] = df_input['Ambient Dry Bulb (C)']
    df = df.loc[:, columns]
    df.columns = [col[:col.index('(') - 1] if '(' in col else col for col in df.columns]

    # Save new weather data to csv, with header
    header.to_csv(sam_weather_file, index=False)
    df.to_csv(sam_weather_file, mode='a', index=False)


def calculate_plane_irradiance(df, tilt, panel_azimuth, albedo=0.2, window_shgc=None, separate=False):
    # All units are in W / m**2
    ghi = df['GHI (W/m^2)']
    dni = df['DNI (W/m^2)']
    dhi = df['DHI (W/m^2)']
    solar_zenith = np.deg2rad(df['Solar Zenith'])
    solar_azimuth = np.deg2rad(df['Solar Azimuth'])
    er = df['Extraterrestrial Radiation']
    tilt = np.deg2rad(tilt)
    panel_azimuth = np.deg2rad(panel_azimuth)

    # Total irradiance = Direct + Diffuse + Ground reflection
    # See https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/irradiance-and-insolation-2/
    incidence_angle = np.arccos(np.cos(solar_zenith) * np.cos(tilt) +
                                np.sin(solar_zenith) * np.sin(tilt) * np.cos(solar_azimuth - panel_azimuth))
    incidence_cos = np.cos(incidence_angle).clip(lower=0)
    irr_direct = dni * incidence_cos
    if window_shgc is not None:
        # see https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/window-calculation-module.html
        # see step-4.-determine-layer-solar-transmittance
        # Note: Only meant for windows with U < 3.4 and SHGC > 0.15
        t_sol = 0.085775 * window_shgc ** 2 + 0.963954 * window_shgc - 0.084958

        # see angular-properties-for-simple-glazing-systems
        irr_direct *= 1 - (0.768 + 0.817 * window_shgc ** 4) * np.sin(incidence_angle) ** 3

        # see step-7.-determine-angular-performance
        # using transmittance curve E
        # t_params = [2.883, -5.873, 2.489, 1.51, -0.002577][::-1]
        # t = np.dot(t_params, [np.cos(incidence_angle).clip(lower=0) ** i for i in range(len(t_params))])
        # irr_direct *= t
    else:
        t_sol = 1

    irr_direct *= t_sol

    # irr_diffuse = dhi * (1 + np.cos(tilt)) / 2 * t_sol  # Isotropic Sky Diffuse Model
    # irr_diffuse += ghi * (0.012 * np.rad2deg(solar_zenith) - 0.04) * (1 - np.cos(tilt)) / 2 # Sandia Sky Diffuse Model
    # Hay and Davies Model
    a_i = dni / er
    b = np.cos(solar_zenith).clip(lower=np.cos(85 * np.pi / 180))
    irr_diffuse = dhi * (a_i * incidence_cos / b + (1 - a_i) * (1 + np.cos(tilt)) / 2)
    # Reindl Model
    # a_i = dni / er
    # h_brightening = (dni * np.cos(solar_zenith) / ghi) ** 0.5 * np.sin(tilt / 2) ** 3
    # irr_diffuse = dhi * (a_i * incidence_cos + (1 - a_i) * (1 + np.cos(tilt)) / 2 * (1 + h_brightening.fillna(0)))
    irr_diffuse *= t_sol

    irr_ground = ghi * albedo * (1 - np.cos(tilt)) / 2 * t_sol
    # irr_ground = (dni * np.cos(solar_zenith).clip(lower=0) + dhi) * albedo * (1 - np.cos(tilt)) / 2 * t_sol

    if separate:
        # return data frame with separate columns for each irradiance type
        return pd.DataFrame({'Direct': irr_direct, 'Diffuse': irr_diffuse, 'Ground': irr_ground,
                             'Incidence Angle': incidence_cos})
    else:
        return irr_direct + irr_diffuse + irr_ground


def resample(df, start_time=None, end_time=None, time_res=None, annual_input=True, annual_output=False,
             interpolate=False, offset=None, repeat_years=False, preserve_sum=False, **kwargs):
    # TODO: use times before start for initialization, if possible
    # update end_time to include duration of initialization
    if kwargs.get('initialization_time') is not None:
        end_time = max(end_time, start_time + kwargs['initialization_time'])
    df = df.copy()

    if annual_input:
        year = start_time.year
        if len(df) % 8760 == 0 and 525600 % len(df) == 0:
            init_time_res = dt.timedelta(minutes=525600 // len(df))
            df.index = pd.date_range(dt.datetime(year, 1, 1), periods=len(df), freq=init_time_res)
        elif len(df) % 8760 == 1 and 525600 % (len(df) - 1) == 0:
            init_time_res = dt.timedelta(minutes=525600 // (len(df) - 1))
            df.index = pd.date_range(dt.datetime(year, 1, 1), periods=len(df), freq=init_time_res)
        else:
            raise ImportError('File length of {} is incompatible for annual input'.format(len(df)))
        end_check = end_time - init_time_res
        if repeat_years and end_check.year != year:
            # copy df for future years
            end_year = (end_time - init_time_res).year
            if len(df) % 8760 == 1:
                # remove last time step before repeating
                df = df.iloc[:-1]
            df = pd.concat([df] * (end_year - year + 1), axis=0)
            df.index = pd.date_range(dt.datetime(year, 1, 1), periods=len(df), freq=init_time_res)

    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise Exception('Resampling requires a dataframe with a DatetimeIndex.'
                            'Current index type: {}'.format(type(df.index)))
        init_time_res = df.index[1] - df.index[0]
    df.index.name = 'Time'

    if annual_output:
        # update start and end times to be at the start/end of the year
        year = start_time.year
        start_time = dt.datetime(year, 1, 1)
        end_time = dt.datetime(year + 1, 1, 1)

    # shift times by offset
    if offset is not None:
        df.index = df.index + offset

    # check if all data is available - extend by up to 2 time steps
    if df.index[0] <= start_time:
        pass
    elif df.index[0] <= start_time + 2 * init_time_res:
        first_row = df.iloc[0].copy()
        first_row.name = first_row.name - 2 * init_time_res
        df = df.append(first_row)
        df = df.sort_values('Time')
    else:
        raise Exception('Start of input data ({})'
                        'is after the required start time ({})'.format(df.index[0], start_time))

    if df.index[-1] >= end_time:
        pass
    elif df.index[-1] >= end_time - 2 * init_time_res:
        last_row = df.iloc[-1].copy()
        last_row.name = last_row.name + 2 * init_time_res
        df = df.append(last_row)
    else:
        raise Exception('End of input data ({}) is before the required end time ({})'.format(df.index[-1], end_time))

    # shorten df before resampling (improves speed)
    keep = (df.index >= start_time - 2 * init_time_res) & (df.index <= end_time + 2 * init_time_res)
    df = df.loc[keep]

    # resample the data
    # see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    if time_res < init_time_res or start_time not in df.index:
        # upsample - either interpolate, pad, or preserve sum by dividing
        if interpolate:
            if offset is not None and (offset % time_res != dt.timedelta(0)):
                resample_time_res = np.gcd(int(time_res.total_seconds()), int(offset.total_seconds()))
                resample_time_res = dt.timedelta(seconds=int(resample_time_res))
            else:
                resample_time_res = time_res
            df = df.resample(resample_time_res).interpolate()
        else:
            # normally, just use pad (forward fill)
            df = df.resample(time_res).pad()
            if preserve_sum:
                # multiply by sample time ratio
                df *= time_res / init_time_res
    else:
        # downsample - either sum or average
        if preserve_sum:
            df = df.resample(time_res).sum()
        else:
            df = df.resample(time_res).mean()

    # only keep simulation times
    times = pd.date_range(start_time, end_time, freq=time_res, closed='left')
    df = df.reindex(times)

    return df


def import_all_time_series(schedule_output_file, schedule_file, weather_file, water_draw_file=None,
                           input_path=default_input_path, **kwargs):
    # Import weather
    if not os.path.isabs(weather_file):
        weather_file = os.path.join(input_path, 'Weather', weather_file)
    weather = import_weather(weather_file, **kwargs)

    all_time_series = []

    # Import Schedule
    if not os.path.isabs(schedule_file):
        schedule_file = os.path.join(input_path, 'BEopt Files', schedule_file)
    main_schedule = import_schedule(schedule_file, **kwargs)
    all_time_series.append(main_schedule)

    # Get solar gains on roof and walls
    roof_tilt = kwargs['roof pitch']
    wall_tilt = 90
    front_dir = kwargs.get('building orientation', np.random.randint(0, 360))  # random orientation if not specified
    front_dir = (front_dir + 180) % 360  # BeOpt: 180=South; OCHRE: 0=South

    # Roof Solar Injection
    flat_roof = calculate_plane_irradiance(weather, 0, 0)
    if roof_tilt == 0:
        front_roof, back_roof = flat_roof, flat_roof
    else:
        front_roof = calculate_plane_irradiance(weather, roof_tilt, front_dir)
        back_roof = calculate_plane_irradiance(weather, roof_tilt, front_dir + 180)
    weather['Horizontal Irradiance (W/m^2)'] = flat_roof
    weather['solar_RF'] = (kwargs['front roof area (m^2)'] * front_roof +
                           kwargs['back roof area (m^2)'] * back_roof)  # in W

    # Detailed wall outputs
    front_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir, separate=True)
    right_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir + 90, separate=True)
    back_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir + 180, separate=True)
    left_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir + 270, separate=True)
    weather = weather.join(pd.DataFrame({'Wall Irradiance - Front ' + key: val for key, val in front_wall.items()}))
    weather = weather.join(pd.DataFrame({'Wall Irradiance - Right ' + key: val for key, val in right_wall.items()}))
    weather = weather.join(pd.DataFrame({'Wall Irradiance - Back ' + key: val for key, val in back_wall.items()}))
    weather = weather.join(pd.DataFrame({'Wall Irradiance - Left ' + key: val for key, val in left_wall.items()}))
    # Wall Solar Injection
    front_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir)
    right_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir + 90)
    back_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir + 180)
    left_wall = calculate_plane_irradiance(weather, wall_tilt, front_dir + 270)
    weather['Wall Irradiance - Front (W/m^2)'] = front_wall
    weather['Wall Irradiance - Right (W/m^2)'] = right_wall
    weather['Wall Irradiance - Back (W/m^2)'] = back_wall
    weather['Wall Irradiance - Left (W/m^2)'] = left_wall
    # subtracting window area from wall area
    weather['solar_EW'] = ((kwargs['front wall area (m^2)'] - kwargs['front window area (m^2)']) * front_wall +
                           (kwargs['right wall area (m^2)'] - kwargs['right window area (m^2)']) * right_wall +
                           (kwargs['back wall area (m^2)'] - kwargs['back window area (m^2)']) * back_wall +
                           (kwargs['left wall area (m^2)'] - kwargs['left window area (m^2)']) * left_wall)  # in W

    if 'right gable wall area (m^2)' in kwargs:
        weather['solar_RG'] = ((kwargs['right gable wall area (m^2)']) * right_wall +
                               (kwargs['left gable wall area (m^2)']) * left_wall)  # in W

    # Window Solar Injection
    factor = kwargs['Windows properties']['SummerInteriorShading']
    shgc = kwargs['Windows properties']['SHGC']
    front_window = calculate_plane_irradiance(weather, wall_tilt, front_dir, window_shgc=shgc, separate=True)
    right_window = calculate_plane_irradiance(weather, wall_tilt, front_dir + 90, window_shgc=shgc, separate=True)
    back_window = calculate_plane_irradiance(weather, wall_tilt, front_dir + 180, window_shgc=shgc, separate=True)
    left_window = calculate_plane_irradiance(weather, wall_tilt, front_dir + 270, window_shgc=shgc, separate=True)
    weather['Window Irradiance - Front (W/m^2)'] = front_window.iloc[:, :3].sum(axis=1)
    weather['Window Irradiance - Right (W/m^2)'] = right_window.iloc[:, :3].sum(axis=1)
    weather['Window Irradiance - Back (W/m^2)'] = back_window.iloc[:, :3].sum(axis=1)
    weather['Window Irradiance - Left (W/m^2)'] = left_window.iloc[:, :3].sum(axis=1)
    weather['solar_WD'] = 0
    for solar_type in ['Direct', 'Diffuse', 'Ground']:
        weather['Window Transmitted Solar, {} (W)'.format(solar_type)] = factor * (
                kwargs['front window area (m^2)'] * front_window[solar_type] +
                kwargs['right window area (m^2)'] * right_window[solar_type] +
                kwargs['back window area (m^2)'] * back_window[solar_type] +
                kwargs['left window area (m^2)'] * left_window[solar_type])  # in W
        weather['solar_WD'] += weather['Window Transmitted Solar, {} (W)'.format(solar_type)]

    # Garage Wall and Roof Solar Injection
    if 'garage floor area (m^2)' in kwargs:
        if roof_tilt == 0:
            gar_roof = flat_roof
        else:
            right_roof = calculate_plane_irradiance(weather, roof_tilt, front_dir + 90)
            left_roof = calculate_plane_irradiance(weather, roof_tilt, front_dir + 270)
            gar_roof = (right_roof + left_roof) / 2
        weather['solar_GR'] = kwargs['garage floor area (m^2)'] * gar_roof  # in W

        weather['solar_GW'] = (kwargs['garage front wall area (m^2)'] * front_wall +
                               kwargs['garage right wall area (m^2)'] * right_wall +
                               kwargs['garage back wall area (m^2)'] * back_wall +
                               kwargs['garage left wall area (m^2)'] * left_wall)  # in W

    # Crawlspace Wall and Solar Injection
    if 'crawlspace above grade wall area (m^2)' in kwargs:
        # make proportional to the wall irradiance
        wall_irradiance = (kwargs['front wall area (m^2)'] * front_wall +
                           kwargs['right wall area (m^2)'] * right_wall +
                           kwargs['back wall area (m^2)'] * back_wall +
                           kwargs['left wall area (m^2)'] * left_wall)
        weather['solar_CW'] = (wall_irradiance / kwargs['total wall area (m^2)'] *
                               kwargs['crawlspace above grade wall area (m^2)'])  # in W

    all_time_series.append(weather)

    # Import water draw profile
    # FUTURE: convert water draw profiles to sparse file format (use scipy.sparse, npz format)
    if water_draw_file is not None:
        if not os.path.isabs(water_draw_file):
            water_draw_file = os.path.join(input_path, 'Water Draw Profiles', water_draw_file)
        water = import_generic(water_draw_file, fillna=0, **kwargs)

        # Scale water draws by max flow rates from properties file, convert to L/min
        if 'Shower max flow rate (m^3/s)' in kwargs:
            water_cols = ['Showers', 'Sinks', 'CW', 'DW', 'Baths']
            assert all([col in water for col in water_cols])
            water['Showers'] *= kwargs['Shower max flow rate (m^3/s)']
            water['Sinks'] *= kwargs['Sink max flow rate (m^3/s)']
            water['CW'] *= kwargs['Clothes washer max flow rate (m^3/s)']
            water['DW'] *= kwargs['Dishwasher max flow rate (m^3/s)']
            water['Baths'] *= kwargs['Bath max flow rate (m^3/s)']
            water *= 1000 * 60  # converts m^3/s to L/min
        else:
            # TODO: convert to error once all properties files are updated
            print('WARNING: max flow rate parameters not in properties file. Assuming water draw has units of L/min.')

        all_time_series.append(water)

    # Check if all columns are different
    all_cols = [col for df in all_time_series for col in df.columns]
    if len(all_cols) != len(set(all_cols)):
        duplicates = [col for col in set(all_cols) if all_cols.count(col) > 1]
        raise Exception('Duplicate columns found in time series inputs: {}'.format(duplicates))

    # Combine all time series files into schedule
    schedule = pd.concat(all_time_series, axis=1)
    schedule.index.name = 'Time'

    # Save schedule file
    if schedule_output_file is not None:
        print('Saving Schedule to: {}'.format(schedule_output_file))
        schedule.reset_index().to_csv(schedule_output_file, index=False)

    # Rename columns
    schedule = schedule.rename(columns=SCHEDULE_COLUMNS)
    return schedule


def save_to_csv(df, filename, output_path=default_output_path, index=True, **kwargs):
    if not os.path.isabs(filename):
        filename = os.path.join(output_path, filename)

    df.to_csv(filename, index=index)
