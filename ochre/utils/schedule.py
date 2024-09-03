import math
import os
import numpy as np
import pandas as pd
import datetime as dt
import collections.abc
import xmltodict
# import re
import numba  # required for array-based psychrolib
import psychrolib
import pytz
import pvlib

from ochre.utils import OCHREException, default_input_path, load_csv, convert
from ochre.utils.envelope import calculate_solar_irradiance

# List of variables and functions for loading and parsing schedule files

# TODO: move to simple schedule parameters file?
SCHEDULE_NAMES = {
    "Occupancy": {
        "occupants": "Occupancy",
    },
    "Power": {
        "clothes_washer": "Clothes Washer",
        "clothes_dryer": "Clothes Dryer",
        "dishwasher": "Dishwasher",
        "refrigerator": "Refrigerator",
        "cooking_range": "Cooking Range",
        "lighting_interior": "Indoor Lighting",
        "lighting_exterior": "Exterior Lighting",
        "lighting_basement": "Basement Lighting",
        "lighting_garage": "Garage Lighting",
        "plug_loads_other": "MELs",
        "plug_loads_tv": "TV",
        "plug_loads_well_pump": "Well Pump",
        # 'plug_loads_vehicle': 'electric vehicle charging',  # Not using scheduled EV load
        "fuel_loads_grill": "Gas Grill",
        "fuel_loads_fireplace": "Gas Fireplace",
        "fuel_loads_lighting": "Gas Lighting",
        "pool_pump": "Pool Pump",
        "pool_heater": "Pool Heater",
        "permanent_spa_pump": "Spa Pump",
        "permanent_spa_heater": "Spa Heater",
        "ceiling_fan": "Ceiling Fan",
        # 'vent_fan': 'Ventilation Fan',  # not included in schedule
        # 'basement_mels': 'Basement MELs',  # not modeled
    },
    "Water": {
        "hot_water_fixtures": "Water Fixtures",
        # "hot_water_showers": "Showers",  # for unmet loads only
        "hot_water_clothes_washer": "Clothes Washer",
        "hot_water_dishwasher": "Dishwasher",
    },
    "Setpoint": {
        "heating_setpoint": "HVAC Heating",
        "cooling_setpoint": "HVAC Cooling",
        "water_heater_setpoint": "Water Heating",
    },
    "Ignore": {
        "extra_refrigerator": None,
        "freezer": None,
        "clothes_dryer_exhaust": None,
        "lighting_exterior_holiday": None,
        "plug_loads_vehicle": None,
        "battery": None,
        "vacancy": None,
        "water_heater_operating_mode": None,
        "Vacancy": None,
        "Power Outage": None,
    },
}

ALL_SCHEDULE_NAMES = {
    hpxml_name: (category, ochre_name)
    for category, data in SCHEDULE_NAMES.items() for hpxml_name, ochre_name in data.items()
}

WEATHER_NAMES = {  # Column names required in weather file
    'temp_air': 'Ambient Dry Bulb (C)',
    'relative_humidity': 'Ambient Relative Humidity (-)',
    'pressure': 'Ambient Pressure (kPa)',
    'ghi': 'GHI (W/m^2)',
    'dni': 'DNI (W/m^2)',
    'dhi': 'DHI (W/m^2)',
    'wind_speed': 'Wind Speed (m/s)',
    'sky_temperature': 'Sky Temperature (C)',
}


def set_annual_index(df, start_year, offset=None, timezone=None):
    # sets DataFrame index to DatetimeIndex assuming annual data. Determines time_res based on length of data
    # assumes df includes data for 365 days
    n = len(df)
    start_time = dt.datetime(start_year, 1, 1)
    end_time = dt.datetime(start_year + 1, 1, 1)
    duration = end_time - start_time  # 365 or 366 days
    if duration.days != 365:
        raise OCHREException('Cannot parse data for a leap year.')
    if n % 8760 == 0 and 525600 % n == 0:
        init_time_res = dt.timedelta(minutes=525600 // n)
        df.index = pd.date_range(start_time, end_time, freq=init_time_res, inclusive='left')
    elif n % 8760 == 1 and 525600 % (n - 1) == 0:
        init_time_res = dt.timedelta(minutes=525600 // (n - 1))
        df.index = pd.date_range(start_time, end_time, freq=init_time_res)
    else:
        raise OCHREException(f'File length of {n} is incompatible for annual input')

    # shift times by offset, update year and re-sort index if necessary
    if offset is not None:
        df.index = df.index + offset
        year_diff = df.index.year - start_year
        if year_diff.any():
            df.index -= dt.timedelta(days=365) * (year_diff - start_year)
            df = df.sort_index()

    if timezone is not None:
        df.index = df.index.tz_localize(timezone)

    df.index.name = 'Time'
    return df


# FUTURE: could get epw file from API, ResStock uses https://data.nrel.gov/system/files/156/BuildStock_TMY3_FIPS.zip
def import_weather(weather_file=None, weather_path=None, weather_station=None, weather_metadata=None, **kwargs):
    # get weather station and weather file name
    if weather_file is not None and weather_station is not None:
        if weather_station not in weather_file:
            print(f'WARNING: Properties file weather station ({weather_station}) may be different from weather file'
                  f' used: {weather_file}')
    elif weather_file is None and weather_station is not None:
        # take weather file from HPXML Weather Station name
        weather_file = weather_station + '.epw'
    elif weather_file is None and weather_station is None:
        raise OCHREException('Must define a weather file.')

    # get weather file path
    if not os.path.isabs(weather_file):
        if weather_path is None:
            weather_path = os.path.join(default_input_path, 'Weather')
        weather_file = os.path.join(weather_path, weather_file)

    start_year = kwargs['start_time'].year
    ext = os.path.splitext(weather_file)[-1]
    if weather_metadata is not None:
        # Assumes OCHRE weather file format
        offset = weather_metadata.get('Offset')
        df = load_csv(weather_file, sub_folder='Weather', index_col='Time', parse_dates=True)
        df['month'] = df.index.month
        df['day'] = df.index.day

        # Set timezone if not already set
        if df.index.tzinfo is None:
            weather_timezone = weather_metadata.get('timezone')
            if weather_timezone is None:
                raise OCHREException('Timezone must be specified either in weather_metadata.')
            tz = dt.timezone(dt.timedelta(hours=weather_timezone))
            df = df.tz_localize(tz)

        # check that lat/lon is in weather_metadata
        missing = [data for data in ['latitude', 'longitude'] if data not in weather_metadata]
        if missing:
            raise OCHREException(f'Missing required location data (must be specified in house Location parameters): {missing}')
        location = weather_metadata
    elif ext == '.epw':
        offset = dt.timedelta(minutes=30)
        df, location = pvlib.iotools.read_epw(weather_file)

        # Update year and save time zone info
        df = set_annual_index(df, start_year, offset=offset, timezone=df.index.tzinfo)
        location['timezone'] = location.get('TZ')

        # Convert GHI Infrared to Sky Temperature
        # see https://www.energyplus.net/sites/default/files/docs/site_v8.3.0/EngineeringReference/05-Climate/index.html
        #  - EnergyPlus Sky Temperature Calculation
        df['sky_temperature'] = convert((df['ghi_infrared'].values / 5.6697E-8) ** 0.25, 'K', 'degC')

        # fix column names and units: Pressure Pa to kPa, RH percentage to fraction
        df['pressure'] = df['atmospheric_pressure'] / 1000
        df['relative_humidity'] /= 100
    elif ext == '.csv':
        # assumes csv is in NSRDB file structure
        offset = None
        df, location = pvlib.iotools.read_psm3(weather_file, map_variables=True)
        # df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        if df.index[0].year != start_year:
            print(f'WARNING: Simulation year ({start_year}) not equal to Weather data year ({df.index[0].year}).')
            df = set_annual_index(df, start_year, timezone=df.index.tzinfo)

        # fix units: Pressure mbar to kPa, RH percentage to fraction
        df['pressure'] /= 10
        df['relative_humidity'] /= 100

        # Set sky temperature to NaN
        df['sky_temperature'] = np.nan
    else:
        raise OCHREException(f'Unknown weather file extension: {ext}')

    # check for leap day
    if ((df.index.month == 2) & (df.index.day == 29)).any():
        print('WARNING: weather data includes leap day')

    # remove unnecessary columns
    df = df.loc[:, list(WEATHER_NAMES.keys())].rename(columns=WEATHER_NAMES)

    # add average weather data to location
    location.update({
        'Weather Station': weather_station,
        'Average Wind Speed (m/s)': df['Wind Speed (m/s)'].mean(),  # For film resistance
        'Average Ambient Temperature (C)': df['Ambient Dry Bulb (C)'].mean(),  # For film resistance
    })

    # Add water mains and ground temperature - annual data only!
    if len(df.index.month.unique()) == 12:
        # Get annual/monthly temperature averages (in F)
        t_amb = pd.Series(convert(df['Ambient Dry Bulb (C)'].values, 'degC', 'degF'), index=df.index)
        t_amb_avg = float(t_amb.mean())
        t_monthly_avg = t_amb.groupby(df.index.month).mean()
        dt_monthly = (t_monthly_avg.max() - t_monthly_avg.min()) / 2

        # get water mains temp,
        # See Burch and Christensen, "Towards development of an algorithm for mains water temperature"
        yday = df.index.dayofyear.values
        tmains_ratio = 0.4 + 0.01 * (t_amb_avg - 44)
        tmains_lag = 35 - (t_amb_avg - 44)
        sign = -1 if location.get('latitude') > 0 else 1
        t_mains = t_amb_avg + 6 + tmains_ratio * dt_monthly * \
            np.sin(np.pi / 180 * (0.986 * (yday - 15 - tmains_lag) + sign * 90))
        df['Mains Temperature (C)'] = convert(t_mains, 'degF', 'degC')

        # get ground temperature
        # same correlation as DOE-2's src\WTH.f file, subroutine GTEMP.
        month = df.index.month
        mid_month = [0, 15, 46, 74, 95, 135, 166, 196, 227, 258, 288, 319, 349]
        mday = month.to_series().replace(dict(enumerate(mid_month))).values
        beta = (np.pi / (8760 * 0.025)) ** 0.5 * 10
        x = np.exp(-beta)
        y = (x ** 2 - 2 * x * np.cos(beta) + 1) / (2 * beta ** 2)
        gm = y ** 0.5
        z = (1 - x * (np.cos(beta) + np.sin(beta))) / (1 - x * (np.cos(beta) - np.sin(beta)))
        # t_ground = t_amb_avg - dt_monthly * gm * np.cos(2 * np.pi / 365 * yday - 0.6 - np.arctan(z))
        t_ground = t_amb_avg - dt_monthly * gm * np.cos(2 * np.pi / 365 * mday - 0.6 - np.arctan(z))
        df['Ground Temperature (C)'] = convert(t_ground, 'degF', 'degC')
    else:
        # for non-annual weather data, mains and ground temps need to be specified or they won't be included in schedule
        if 'Mains Temperature (C)' in kwargs:
            df['Mains Temperature (C)'] = kwargs['Mains Temperature (C)']
        if 'Ground Temperature (C)' in kwargs:
            df['Ground Temperature (C)'] = kwargs['Ground Temperature (C)']

    if 'Ground Temperature (C)' in df:
        # For film resistance calc
        location['Average Ground Temperature (C)'] = float(df['Ground Temperature (C)'].mean())

    # add humidity ratio and wet bulb
    df['Ambient Humidity Ratio (-)'] = psychrolib.GetHumRatioFromRelHum(df['Ambient Dry Bulb (C)'].values,
                                                                        df['Ambient Relative Humidity (-)'].values,
                                                                        df['Ambient Pressure (kPa)'].values * 1000)
    df['Ambient Wet Bulb (-)'] = psychrolib.GetTWetBulbFromHumRatio(df['Ambient Dry Bulb (C)'].values,
                                                                    df['Ambient Humidity Ratio (-)'].values,
                                                                    df['Ambient Pressure (kPa)'].values * 1000)

    return df, location


def create_simple_schedule(weekday_fractions, weekend_fractions=None, month_multipliers=None, **kwargs):
    # converts weekday/weekend/month fractions into time series schedule
    if weekend_fractions is None:
        weekend_fractions = weekday_fractions
    if month_multipliers is None:
        month_multipliers = [1] * 12

    df_hour = pd.DataFrame({
        'hour': range(24),
        True: weekday_fractions,
        False: weekend_fractions,
    }).melt('hour', var_name='weekday', value_name='w_fracs')
    df_month = pd.DataFrame({
        'month': range(1, 13),
        'm_fracs': month_multipliers,
    })
    df = pd.merge(df_hour, df_month, how='cross')
    df = df.set_index(['month', 'hour', 'weekday'])
    return df['w_fracs'] * df['m_fracs']


def convert_power_column(s_hpxml, ochre_name, properties):
    # try getting from max power or from annual energy, priority goes to max power
    if 'Max Electric Power (W)' in properties:
        max_value = properties['Max Electric Power (W)'] / 1000  # W to kW
    elif 'Annual Electric Energy (kWh)' in properties:
        annual_mean = properties['Annual Electric Energy (kWh)'] / 8760
        schedule_mean = s_hpxml.mean()
        max_value = annual_mean / schedule_mean if schedule_mean != 0 else 0
    else:
        max_value = None
    if max_value is not None:
        out = s_hpxml * max_value
        out.name = f'{ochre_name} (kW)'
    else:
        out = None

    # check for gas (max power and annual energy), and copy schedule
    if 'Max Gas Power (therms/hour)' in properties:
        max_value = properties['Max Gas Power (therms/hour)']  # in therms/hour
    elif 'Annual Gas Energy (therms)' in properties:
        annual_mean = properties['Annual Gas Energy (therms)'] / 8760  # in therms/hour
        schedule_mean = s_hpxml.mean()
        max_value = annual_mean / schedule_mean if schedule_mean != 0 else 0
    else:
        max_value = None
    if max_value is None:
        pass
    elif out is None:
        out = s_hpxml * max_value
        out.name = f'{ochre_name} (therms/hour)'
    else:
        # combine 2 series into data frame
        s_gas = s_hpxml * max_value
        s_gas.name = f'{ochre_name} (therms/hour)'
        out = pd.concat([out, s_gas], axis=1)

    if out is None:
        raise OCHREException(f'Could not determine max value for {s_hpxml.name} schedule ({ochre_name}).')

    return out


def convert_water_column(s_hpxml, ochre_name, equipment):
    if ochre_name in ["Water Fixtures", "Showers"]:
        # Fixtures include sinks, showers, and baths (SSB), all combined
        # Showers are only included for unmet loads calculation
        equipment_name = "Water Heating"
    else:
        equipment_name = ochre_name

    if equipment_name not in equipment:
        return None
    
    properties = equipment[equipment_name]
    avg_water_draw = properties.get('Average Water Draw (L/day)', 0)
    annual_mean = avg_water_draw / 1440  # in L/min
    
    schedule_mean = s_hpxml.mean()
    max_value = annual_mean / schedule_mean if schedule_mean != 0 else 0
    out = s_hpxml * max_value
    out.name = f'{ochre_name} (L/min)'

    return out


def import_occupancy_schedule(occupancy, equipment, start_time, schedule_input_file=None,
                              simple_schedule_file='Simple Schedule Parameters.csv', **kwargs):
    # Import stochastic occupancy schedule file. Note that initial values are normalized to max_value=1
    # FUTURE: for sub-annual schedules, create annual schedule and then shorten to simulation time
    if schedule_input_file is not None:
        df_norm = load_csv(schedule_input_file, sub_folder='Input Files')
    else:
        # create empty, hourly DataFrame
        df_norm = pd.DataFrame(index=range(8760))
    df_norm = set_annual_index(df_norm, start_time.year)

    # Copy zone-specific schedules if not in the schedule file
    if 'Basement Lighting' in equipment and 'lighting_basement' not in df_norm and 'lighting_interior' in df_norm:
        df_norm['lighting_basement'] = df_norm['lighting_interior']

    # Load simple schedule parameters file
    df_simple = load_csv(simple_schedule_file, index_col='Name')

    # Add normalized simple schedules from HPXML to df_norm
    schedules_to_merge = []
    for hpxml_name, (category, ochre_name) in ALL_SCHEDULE_NAMES.items():
        ochre_dict = occupancy if category == 'Occupancy' else equipment.get(ochre_name, {})
        if not ochre_dict:
            continue

        # Add setpoint simple schedules
        if category == 'Setpoint':
            if hpxml_name in df_norm:
                # convert setpoints to degC
                df_norm[hpxml_name] = convert(df_norm[hpxml_name].values, 'degF', 'degC')
            elif ochre_dict.get('Weekday Setpoints (C)') is not None:
                # create simple setpoint schedule
                s_hpxml = create_simple_schedule(ochre_dict.get('Weekday Setpoints (C)'),
                                                 ochre_dict.get('Weekend Setpoints (C)'))
                s_hpxml.name = hpxml_name
                schedules_to_merge.append(s_hpxml)
            elif 'Setpoint Temperature (C)' in ochre_dict:
                # create a constant setpoint schedule
                df_norm[hpxml_name] = ochre_dict['Setpoint Temperature (C)']
            else:
                raise OCHREException(f'Must specify {ochre_name} setpoints in schedule input file or include parameters '
                              '"Weekday/Weekend Setpoints (C)" or "Setpoint Temperature (C)".')

        # Add occupancy/power/water draw simple schedules
        elif hpxml_name not in df_norm:
            if ochre_dict.get('weekday_fractions') is None:
                # add data from simple schedule defaults file
                data = df_simple.loc[ochre_name].to_dict()
                ochre_dict.update({key: eval(val) for key, val in data.items() if isinstance(val, str)})
            s_hpxml = create_simple_schedule(**ochre_dict)
            s_hpxml.name = hpxml_name
            schedules_to_merge.append(s_hpxml)

    if schedules_to_merge:
        df_to_merge = pd.concat(schedules_to_merge, axis=1)
        df_norm = df_norm.join(df_to_merge, on=[df_norm.index.month, df_norm.index.hour, df_norm.index.weekday < 5])
        df_norm = df_norm.drop(columns=['key_0', 'key_1', 'key_2'], errors='ignore')

    # Calculate max value for each column and add to new DataFrame
    schedule_data = []
    for hpxml_name, s_hpxml in df_norm.items():
        # Get max value from properties and update  - either directly or using annual value
        category, ochre_name = ALL_SCHEDULE_NAMES[hpxml_name]
        if category == 'Occupancy':
            s_ochre = s_hpxml * occupancy['Number of Occupants (-)']
            s_ochre.name = f'{ochre_name} (Persons)'
            schedule_data.append(s_ochre)
        elif category == "Power":
            if ochre_name in equipment:
                schedule_data.append(convert_power_column(s_hpxml, ochre_name, equipment[ochre_name]))
        elif category == "Water":
            s_ochre = convert_water_column(s_hpxml, ochre_name, equipment)
            if s_ochre is not None:
                schedule_data.append(s_ochre)
        elif category == 'Setpoint':
            # Already in the correct units
            s_ochre = s_hpxml
            s_ochre.name = f'{ochre_name} Setpoint (C)'
            schedule_data.append(s_ochre)
        elif category == 'Ignore':
            # Schedule is not used in OCHRE
            continue
        else:
            raise OCHREException(f'Unknown column in schedule file: {hpxml_name}')

    schedule = pd.concat(schedule_data, axis=1)

    # Add ventilation fan power and flow rate - constant schedule
    if 'Ventilation Fan' in equipment:
        schedule['Ventilation Fan (kW)'] = equipment['Ventilation Fan']['Power (W)'] / 1000
        schedule['Ventilation Rate (cfm)'] = equipment['Ventilation Fan']['Ventilation Rate (cfm)']

    return schedule


def resample_and_reindex(df, time_res, start_time=None, duration=None, interpolate=False, offset=None,
                         preserve_sum=False, **kwargs):
    # Resamples time series data and sets the index to the simulation duration. df must have a DateTimeIndex
    # Options to select resample method, e.g. interpolate vs. pad
    # Will repeat annual data and extend start and end rows by up to 2 time steps if necessary
    assert isinstance(df.index, pd.DatetimeIndex)
    df.index.name = 'Time'
    init_start_time = df.index[0]
    init_time_res = df.index[1] - df.index[0]

    # use existing start/end times if not specified
    if start_time is None:
        start_time = init_start_time
    if duration is None:
        duration = (df.index[-1] + init_time_res) - start_time
    end_time = start_time + duration

    # set time zone based on start_time
    if df.index.tzinfo != start_time.tzinfo:
        df = df.tz_localize(start_time.tzinfo)

    start_year = start_time.year
    end_year = (end_time - time_res).year
    if end_year > start_year:
        # copy df for future years
        if len(df) % 8760 == 1:
            # remove last time step before duplicating annual
            df = df.iloc[:-1]
        if len(df) % 8760 != 0:
            raise OCHREException(f'Simulation spans multiple years ({start_year}-{end_year}). Must provide annual data.')

        print(f'Simulation spans multiple years ({start_year}-{end_year}). Duplicating time series data.')
        df = pd.concat([df] * (end_year - start_year + 1), axis=0)
        df.index = pd.date_range(init_start_time, periods=len(df), freq=init_time_res)

    # check if all data is available - extend by up to 2 time steps
    if df.index[0] <= start_time:
        pass
    elif df.index[0] <= start_time + 2 * init_time_res:
        first_row = df.iloc[:1].copy()
        first_row.index -= 2 * init_time_res
        df = pd.concat([first_row, df], axis=0)
    else:
        raise OCHREException(f'Start of input data ({df.index[0]}) is after the required start time ({start_time})')

    if df.index[-1] >= end_time:
        pass
    elif df.index[-1] >= end_time - 2 * init_time_res:
        last_row = df.iloc[-1:].copy()
        last_row.index += 2 * init_time_res
        df = pd.concat([df, last_row])
    else:
        raise OCHREException(f'End of input data ({df.index[-1]}) is before the required end time ({end_time})')

    # shorten df before resampling (improves speed)
    df = df.loc[start_time - 2 * init_time_res: end_time + 2 * init_time_res]

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
            df = df.resample(time_res).ffill()
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
    times = pd.date_range(start_time, end_time, freq=time_res, inclusive='left')
    df = df.reindex(times)
    df.index.name = 'Time'
    return df


def load_schedule(properties, schedule=None, time_zone=None, **house_args):
    # Load weather file and update Location properties
    df_weather, location = import_weather(**house_args)

    # Import occupancy schedule - either a csv file or a properties file
    df_occupancy = import_occupancy_schedule(properties['occupancy'], properties['equipment'], **house_args)

    # Check that all columns are different
    duplicates = [col for col in df_weather.columns if col in df_occupancy.columns]
    if duplicates:
        raise OCHREException(f'Duplicate column names found in weather and schedule inputs: {duplicates}')

    # resample weather and occupancy schedules using simulation parameters
    weather_tz = df_weather.index.tzinfo
    df_weather = resample_and_reindex(df_weather, **house_args)  # loses weather timezone info
    df_occupancy = resample_and_reindex(df_occupancy, **house_args)

    # add solar calculations to weather (more accurate if done after resampling)
    df_weather = calculate_solar_irradiance(df_weather, weather_tz, location, properties['boundaries'], **house_args)

    # combine weather and main schedule
    schedule_init = pd.concat([df_weather, df_occupancy], axis=1)

    # modify OCHRE schedule from house_args
    if schedule:
        df_modify = pd.DataFrame(schedule)
        bad_cols = [col for col in df_modify.columns if col not in schedule_init.columns]
        if bad_cols:
            print('WARNING: Skipping schedule columns not in OCHRE schedule:', bad_cols)
            df_modify = df_modify.drop(columns=bad_cols)

        df_modify = resample_and_reindex(df_modify, **house_args)
        schedule_init.update(df_modify)
    schedule = schedule_init

    # check if cooling-heating setpoint difference is large enough, if not throw a warning and fix
    if 'HVAC Cooling Setpoint (C)' in schedule and 'HVAC Heating Setpoint (C)' in schedule:
        setpoint_diff = schedule['HVAC Cooling Setpoint (C)'] - schedule['HVAC Heating Setpoint (C)']
        if setpoint_diff.min() < 1:
            # if min(setpoint_diff) < 0:
            #     raise OCHREException('ERROR: Cooling setpoint is equal or less than heating setpoint in schedule file')
            print('WARNING: Cooling setpoint is within 1C of heating setpoint in schedule file.'
                  ' Separating setpoints by at least 1C.')
            setpoint_avg = (schedule['HVAC Cooling Setpoint (C)'] + schedule['HVAC Heating Setpoint (C)']) / 2
            schedule['HVAC Cooling Setpoint (C)'] = schedule['HVAC Cooling Setpoint (C)'].clip(lower=setpoint_avg + 0.5)
            schedule['HVAC Heating Setpoint (C)'] = schedule['HVAC Heating Setpoint (C)'].clip(upper=setpoint_avg - 0.5)

    # Check for missing data - should not have any NA values except for airmass and sky temperature
    if schedule.drop(columns=['airmass', 'Sky Temperature (C)']).isna().any().any():
        check = schedule.drop(columns=['airmass', 'Sky Temperature (C)'])
        bad_cols = check.columns[check.isna().any()]
        first_na = check.isna().any(axis=1).idxmax()
        raise OCHREException(f'Missing data found in schedule columns {bad_cols}. See time step {first_na}')

    # update time zone, if specified
    if time_zone == 'DST':
        # Use weather timezone offset to determine US timezone with daylight savings
        us_dst_timezones = {
            -5: 'US/Eastern',
            -6: 'US/Central',
            -7: 'US/Mountain',
            -8: 'US/Pacific',
        }
        time_zone = pytz.timezone(us_dst_timezones[location['timezone']])
    elif time_zone == 'noDST':
        # Use weather timezone offset to determine US timezone without daylight savings
        time_zone = dt.timezone(dt.timedelta(hours=location['timezone']))
    elif time_zone in pytz.all_timezones:
        # Use specified time zone
        time_zone = pytz.timezone(time_zone)
    elif time_zone is not None:
        raise OCHREException(f'Unknown time zone parameter: {time_zone}')
    schedule.index = schedule.index.tz_localize(time_zone)

    return schedule, location
