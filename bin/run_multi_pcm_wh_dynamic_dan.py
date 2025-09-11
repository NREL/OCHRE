import datetime as dt
import numpy as np
import pandas as pd
import os
import shutil

from ochre import (
    HeatPumpWaterHeater,
    CreateFigures,
    ElectricResistanceWaterHeater,
)
from ochre.Models import TankWithMultiPCM
from ochre.utils import convert
import time
from bin.run_dwelling import dwelling_args
import multiprocessing
import copy


# ANSI Terminal color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"



GAL_TO_L = 3.78541


start_node = 4
end_node = 9
DEFAULT_PCM_PROPERTIES = {
    "t_m1": 50,  # C
    "t_m2": 55,  # C
    "h_fus": 226,  # J/g
    "h": 600,  # W/m^2K
    "sa_ratio": 15,  # m^2/m^3 of total pcm volume
    "h_conv": 100,  # W/K, accounts for surface area (ha)
    "setpoint_temp": 50,  # C
    "solid": {
        "pcm_density": 0.904,  # g/cm**3
        "pcm_cp": 1.20,  # J/g-C # adjusted by real measurements average from 0-45c
        "pcm_conductivity": 0.28,  # W/m-C, not used
        # "pcm_c": 1717.6,  # J/m**3-C, not used
    },
    "liquid": {
        "pcm_density": 0.829,  # g/cm**3
        "pcm_cp": 1.33,  # J/g-C # adjusted by real measurements average from 55-100c
        "pcm_conductivity": 0.16,  # W/m-C, not used
        # "pcm_c": 1823.8,  # J/m**3-C, not used
    },
    "enthalpy_lut": "cp_h-T_data_shifted_120F.csv",
}

num_points = 10

sa_ratios = [15]
# sa_ratios = [5, 10, 15, 20, 25, 30]
h_values = [500]
# h_values = np.logspace(np.log10(50), np.log10(5000), num_points)


pcm_file_names = [
    "cp_h-T_data_shifted_120F.csv",
    "cp_h-T_data_shifted_125F.csv",
    "cp_h-T_data_shifted_130F.csv",
    "cp_h-T_data_shifted_135F.csv",
    "cp_h-T_data_shifted_140F.csv",
]


pcm_file_names = ['cp_h-T_data_shifted_120F.csv']
# pcm_file_names = ['cp_h-T_data_52_6C.csv']s

setpoint_temps_f = [140]
# setpoint_temps_f = [125, 140]
setpoint_temps_c = [
    (setpoint_temp - 32) * (5 / 9) for setpoint_temp in setpoint_temps_f
]

# tank_volume_gal = [40,50, 65]
tank_volume_gal = [50]


vol_fract = 0.00000001  # 1.540e-06 kg
vol_fract = 0.0001  # 1.540e-02 kg
vol_fract = 0.5  # 7.700e+01 kg
vol_fracs = [0.5]
# vol_fracs = [0.5]

# pcm_vol_fractions = [{i: vol_fract for i in range(1, n + 1)} for n in range(1, num_nodes + 1)]
# pcm_vol_fractions = [
#                     {7: vol_fract},
#                     ]

# pcm_vol_fractions = [
#                     {12: vol_fract},
#                     ]
pcm_vol_fractions = []
for vol_fract in vol_fracs:
    pcm_vol_fractions.append(
        {node: vol_fract for node in range(start_node, end_node + 1)}
    )


load_profile = "2.00gpm30min_0gpm180min_cycling.csv"
load_profile = "2.00gpm30min_0gpm600min_cycling.csv"
load_profile = "2.00gpm120min_0gpm600min_cycling.csv"
load_profile = "MediumUseL.csv"
load_profile = "2.00gpm_1200minStartIdle_2cycles_30minDraw_240minOff_0minEndIdle.csv"
load_profile = "2.00gpm_1200minStartIdle_2cycles_30minDraw_240minOff_0minEndIdle_Single_draw.csv"
load_profiles = ["MediumUseL.csv", "2.00gpm30min_0gpm600min_cycling.csv"]


def convert_dict_to_name(dict):
    # Check if all values are the same
    values = list(dict.values())
    if len(values) > 0 and all(v == values[0] for v in values):
        # Check if keys are sequential integers
        keys = sorted(dict.keys())
        if all(isinstance(k, int) for k in keys) and keys == list(
            range(min(keys), max(keys) + 1)
        ):
            # Return the compact format
            return f"{values[0]}_pcm{min(keys)}-{max(keys)}"

    # Fallback to original format if conditions aren't met
    return "".join([str(key) + "_" + str(value) + "_" for key, value in dict.items()])


def add_pcm_model(default_args, name, pcm_vol_fractions, pcm_properties):
    
    default_args['model_class'] = TankWithMultiPCM
    default_args['water_nodes'] = 12
    default_args['Water Tank'] = {
        'pcm_node_vol_fractions': pcm_vol_fractions,
        'pcm_properties': pcm_properties,
    }

    default_args["name"] = name
    return default_args

default_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    "time_res": dt.timedelta(minutes=1),
    "duration": dt.timedelta(days=2),
    "verbosity": 9,
    "save_results": None,  # if True, must specify output_path # None Merges the simulator results into 1 file
    "output_path": '../OCHRE_output/OCHRE_results/results/',
    "name": "ZDefault_ElectricResistanceWaterHeater",
    "schedule_input_file": load_profile,
}

def import_water_heating_schedule(schedule_file):
    # Import water heating schedule file
    try:

        df = pd.read_csv(f"ochre/defaults/Input Files/{schedule_file}")
        
        hot_water_schedule = df.loc[:, 'hot_water_fixtures']
        #clip to 2880 points
        hot_water_schedule = hot_water_schedule.iloc[0:2880]
    
    except Exception as e:
        print(f"Error importing water heating schedule file: {e}")
    finally:
        print(os.getcwd())
        
    return hot_water_schedule
        
    
def simulate_first_hour_test(wh, enable_first_hour_test=True, disable_heating_during_draw=False, first_hour_duration=60, draw_rate_gpm=3, allow_setpoint_start=False, hot_water_temp_f=110, setpoint_temp_f=140):
    
    test_active = False
    test_completed = False
    test_initialized = False
    test_timer = 0.0
    draw_active = False
    total_gallons_delivered = 0.0
    final_draw_triggered = False  # New flag to track if final draw has been triggered

    # These constants would typically be defined globally or passed in.
    # For the function to run stand-alone, they'd need to be defined here or passed.
    # Assuming GAL_TO_L and convert are available in the scope this function is called from.
    # e.g. GAL_TO_L = 3.78541
    # e.g. from some_utils import convert 

    times = wh.sim_times
    prev_t = None

    setpoint_temp = convert(setpoint_temp_f, 'degF', 'degC')
    hot_water_temp = convert(hot_water_temp_f, 'degF', 'degC')
    
    for t_idx, t in enumerate(times):
        control_signal = {}

        if prev_t is not None:
            delta_min = (t - prev_t).total_seconds() / 60.0
        else:
            delta_min = 0.0
        prev_t = t

        if enable_first_hour_test:
            if t_idx == 1:
                test_initialized = True

            if test_initialized and not test_active and not test_completed:
                if wh.mode == 'Off':
                    test_active = True
                    test_timer = float(first_hour_duration)
                    draw_active = True
                    control_signal = {
                        'Water Heating (L/min)': draw_rate_gpm * GAL_TO_L
                    }
                    if disable_heating_during_draw:
                        control_signal['Water Heating Setpoint (C)'] = 5
                    print(f"[{t}] → FIRST-HOUR test STARTED, drawing {draw_rate_gpm} gpm")

            elif test_active:
                test_timer -= delta_min

                if wh.mode == 'Off' and not draw_active:
                    draw_active = True
                    # control_signal will be set in accumulation step if draw_active remains true
                    print(f"[{t}] → MODE=Off, RESTARTING draw")

                # Check if we've reached the test duration and should trigger final draw
                if test_timer <= 0 and not final_draw_triggered:
                    final_draw_triggered = True
                    # If not already drawing, start the draw
                    if not draw_active:
                        draw_active = True
                        print(f"[{t}] → FINAL DRAW: Timer elapsed ({test_timer:.2f}), initiating final draw")
                    else:
                        print(f"[{t}] → FINAL DRAW: Timer elapsed ({test_timer:.2f}), draw already active, continuing")
                
                if draw_active and wh.model.outlet_temp < hot_water_temp:
                    draw_active = False
                    control_signal = {} # Explicitly stop draw signal for this step
                    print(f"[{t}] → temp fell ({wh.model.outlet_temp:.1f}C vs {hot_water_temp:.1f}C limit), STOPPING draw")

                if (allow_setpoint_start and not draw_active and wh.model.outlet_temp >= setpoint_temp):
                    draw_active = True
                    # control_signal will be set in accumulation step
                    print(f"[{t}] → reached setpoint ({wh.model.outlet_temp:.1f}C), STARTING draw")

                if draw_active:
                    control_signal['Water Heating (L/min)'] = draw_rate_gpm * GAL_TO_L
                    if disable_heating_during_draw:
                        control_signal['Water Heating Setpoint (C)'] = 5
                    total_gallons_delivered += draw_rate_gpm * delta_min

                # Only complete the test if the timer is up AND 
                # either: 1) final draw has been completed (not active) or 2) outlet temp fell below limit
                if test_timer <= 0 and final_draw_triggered and not draw_active:
                    test_active = False
                    test_completed = True
                    print(f"[{t}] → TEST COMPLETE: delivered {total_gallons_delivered:.2f} gallons")
                    # We don't break immediately to ensure the final state is properly updated
                    # Instead, we'll break at the end of this iteration
        
        _ = wh.update(schedule_inputs=control_signal)

        # If test completed inside the 'if enable_first_hour_test' block, break from the outer loop
        if test_completed:
            break 

    return total_gallons_delivered, wh 

def create_water_schedule(
    withdraw_rate_lpm=None,
    withdraw_rate_gpm=3.0,
    setpoint_default=51.667,
    deadband_default=5.56,
    zone_temp_c=19.722222,
    zone_wet_bulb_temp_c=15.0,
    mains_temp_c=14.4444,
    start_idle_time_min=0,
    draw_time_min=30,
    off_time_min=240,
    draw_repeats=2,
    end_idle_time_min=0,
    no_heating_during_draw=False,
    times=pd.date_range(
        dt.datetime(2018, 1, 1, 0, 0),
        dt.datetime(2018, 1, 1, 0, 0) + dt.timedelta(days=2)+ dt.timedelta(minutes=1),
        freq=dt.timedelta(minutes=1),
        inclusive="left",
    )
):
    """
    Create a water schedule DataFrame with adjustable parameters for all variables.
    
    Parameters:
    -----------
    withdraw_rate_gpm : float
        Water withdrawal rate in gal/min during draw periods
    setpoint_default : float
        Default water heating setpoint temperature in Celsius
    deadband_default : float
        Default water heating deadband in Celsius
    zone_temp_c : float
        Zone temperature in Celsius
    zone_wet_bulb_temp_c : float
        Zone wet bulb temperature in Celsius
    mains_temp_c : float
        Mains water temperature in Celsius
    start_idle_time_min : int
        Duration of initial idle period in minutes
    draw_time_min : int
        Duration of each draw period in minutes
    off_time_min : int
        Duration of each off period in minutes
    draw_repeats : int
        Number of draw cycles to repeat
    end_idle_time_min : int
        Duration of final idle period in minutes
    no_heating_during_draw : bool
        If True, the setpoint will drop below mains temperature during water draws
        to prevent heating. If False, setpoint remains constant.
    times : pd.DatetimeIndex
        Datetime index for the schedule (default is 2 days starting at January 1, 2018)
        
    Returns:
    --------
    pd.DataFrame
        Water schedule with all variables
    """
    GAL_TO_L = 3.78541
    
    withdraw_rate = withdraw_rate_gpm * GAL_TO_L
    
    total_minutes = len(times)
    
    # Initialize arrays for each variable
    water_withdraw = []
    setpoint_temp = []
    
    # Define setpoint temperature during draw (slightly below mains temperature)
    draw_setpoint = mains_temp_c - 2.0  # 2°C below mains temperature
        
    if withdraw_rate_lpm is None:
        # Build the pattern for water withdrawal and setpoint temperatures
        # 1. Start idle period
        water_withdraw.extend([0] * start_idle_time_min)
        setpoint_temp.extend([setpoint_default] * start_idle_time_min)
        
        # 2. Draw profile repeated as many times as specified
        for _ in range(draw_repeats):
            # Draw period (on) - set withdrawal rate and possibly lower setpoint
            water_withdraw.extend([withdraw_rate] * draw_time_min)
            
            # Set setpoint based on no_heating_during_draw flag
            if no_heating_during_draw:
                setpoint_temp.extend([draw_setpoint] * draw_time_min)
            else:
                setpoint_temp.extend([setpoint_default] * draw_time_min)
            
            # Off period (idle) - no withdrawal and restore default setpoint
            water_withdraw.extend([0] * off_time_min)
            setpoint_temp.extend([setpoint_default] * off_time_min)
        
        # 3. End idle period
        water_withdraw.extend([0] * end_idle_time_min)
        setpoint_temp.extend([setpoint_default] * end_idle_time_min)
        
        # Ensure the patterns reach the required length
        if len(water_withdraw) >= total_minutes:
            water_withdraw = water_withdraw[:total_minutes]
            setpoint_temp = setpoint_temp[:total_minutes]
        else:
            # If the constructed patterns are shorter than required, pad the end
            remaining = total_minutes - len(water_withdraw)
            water_withdraw.extend([0] * remaining)
            setpoint_temp.extend([setpoint_default] * remaining)
    else:
        water_withdraw = list(withdraw_rate_lpm)
        setpoint_temp = [setpoint_default] * total_minutes
        # If no_heating_during_draw is True, set the setpoint to draw_setpoint 
        # during periods when water is being withdrawn (non-zero withdrawal rate)
        if no_heating_during_draw:
            for i in range(total_minutes):
                if water_withdraw[i] > 0:
                    setpoint_temp[i] = draw_setpoint
        
    # Fill in other variables with their default values
    deadband = [deadband_default] * total_minutes
    zone_temp = [zone_temp_c] * total_minutes
    zone_wet_bulb = [zone_wet_bulb_temp_c] * total_minutes
    mains_temp = [mains_temp_c] * total_minutes
    
    # Create the DataFrame with all variables
    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": water_withdraw,
            "Water Heating Setpoint (C)": setpoint_temp,
            "Water Heating Deadband (C)": deadband,
            "Zone Temperature (C)": zone_temp,
            "Zone Wet Bulb Temperature (C)": zone_wet_bulb,
            "Mains Temperature (C)": mains_temp,
        },
        index=times,
    )
    
    return schedule

def f_to_c(f):
    return (f - 32) * 5 / 9

def calculate_net_PCM_heat(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = "Total PCM Heat Injected (W)"

    if pcm_column not in df.columns:
        return 0

    # Sum the PCM enthalpy columns row-wise.

    net_PCM_to_water_heat_Transfer = df[pcm_column].sum()

    return net_PCM_to_water_heat_Transfer

def calculate_net_PCM_enthalpy(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = "Total PCM Enthalpy (J)"

    if pcm_column not in df.columns:
        return 0

    # Sum the PCM enthalpy columns row-wise.

    net_PCM_enthalpy = df[pcm_column].iloc[-1] - df[pcm_column].iloc[0]

    return net_PCM_enthalpy

def calculate_net_water_temp(df):
    water_temp_column = ["Hot Water Average Temperature (C)"]
    valid_columns = [col for col in water_temp_column if col in df.columns]

    if not valid_columns:
        return 0

    # Compute the average of the first and last row for all available columns
    average_start_temp = df[valid_columns].iloc[0].mean()
    average_end_temp = df[valid_columns].iloc[-1].mean()

    return average_end_temp - average_start_temp




def lookup_water_properties(T):
    """
    Returns interpolated water properties for a given temperature T (in °C).
    The returned properties are based on approximate data for pure water at 1 atm.

    Properties returned:
      - density: in kg/m³
      - specific_weight: in N/m³ (calculated as density * 9.81)
      - thermal_expansion: volumetric thermal expansion coefficient in 1/°C

    Parameters:
      T (float): Temperature in °C (must be within 0 to 100)

    Raises:
      ValueError: If the temperature is outside the 0 to 100°C range.
    """
    # Predefined lookup arrays for water properties at 1 atm.
    _TEMPS = np.array([0, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    _DENSITIES = np.array(
        [
            999.8,
            1000.0,
            999.7,
            998.2,
            995.7,
            992.2,
            988.1,
            983.2,
            977.8,
            971.8,
            965.3,
            958.4,
        ],
        dtype=float,
    )
    _THERMALEXPANSIONS = np.array(
        [
            -1.0e-4,
            0.0,
            1.1e-4,
            2.1e-4,
            2.6e-4,
            3.1e-4,
            3.6e-4,
            4.1e-4,
            4.7e-4,
            5.3e-4,
            5.9e-4,
            6.6e-4,
        ],
        dtype=float,
    )
    
    if T < _TEMPS[0] or T > _TEMPS[-1]:
        raise ValueError("Temperature out of range (must be between 0 and 100 °C)")

    density = np.interp(T, _TEMPS, _DENSITIES)
    thermal_expansion = np.interp(T, _TEMPS, _THERMALEXPANSIONS)
    specific_weight = density * 9.81

    return {
        "temperature": T,
        "density": density,
        "specific_weight": specific_weight,
        "thermal_expansion": thermal_expansion,
    }


def calculate_net_water_energy(volume, temperature, temperature_difference):
    water_properties_start = lookup_water_properties(temperature - temperature_difference)
    water_properties_end = lookup_water_properties(temperature)
    density_start = water_properties_start["density"]
    density_end = water_properties_end["density"]
    density_average = (density_start + density_end) / 2
    # volume is in Liters
    water_weight_average = density_average * volume / 1e3  # in kg

    return water_weight_average * temperature_difference * 4184  # J


def calculate_uef(df, water_volume):
    # calculate UEF of the water tank
    Q_cons = (
        df["Water Heating Electric Power (kW)"].sum() * 1000
    )  # not sure if this is the correct term that I should be pulling
    Q_load = df[
        "Hot Water Delivered (W)"
    ].sum()  # not sure if this is the correct term that I should be pulling

    PCM_Q_Heat_to_Water = calculate_net_PCM_heat(df)  # make sure in W*min
    PCM_net_enthalpy = calculate_net_PCM_enthalpy(df)
    PCM_net_heat_loss = PCM_net_enthalpy / 60  # make sure in W*min
    water_net_temp_delta = calculate_net_water_temp(df)
    water_net_energy = (
        calculate_net_water_energy(
            water_volume,
            df["Hot Water Average Temperature (C)"].iloc[-1],
            water_net_temp_delta,
        )
        / 60
    )  # make sure in W*min
    Q_cons_total = Q_cons - PCM_net_heat_loss - water_net_energy  # make sure in W*min
    UEF = Q_load / Q_cons_total

    return UEF

def run_water_heater_electric(default_args, setpoint_temp, tank_volume):
    # Create water draw schedule


    if default_args.get('schedule_input_file', None) is None:
        schedule = create_water_schedule(setpoint_default=setpoint_temp, withdraw_rate_gpm=0, no_heating_during_draw=False)
    else:
        hot_water_schedule = import_water_heating_schedule(default_args.get('schedule_input_file'))
        times = pd.date_range(dt.datetime(2018, 1, 1, 0, 0), dt.datetime(2018, 1, 1, 0, 0) + dt.timedelta(minutes=len(hot_water_schedule)), freq=dt.timedelta(minutes=1), inclusive="left")
        schedule = create_water_schedule(withdraw_rate_lpm=hot_water_schedule, setpoint_default=setpoint_temp, no_heating_during_draw=False, times=times)
        


    hot_water_temp = convert(110, 'degF', 'degC') 
    equipment_args = {
        # Equipment parameters
        # "Setpoint Temperature (C)": 14.4444,
        "Setpoint Temperature (C)": setpoint_temp,
        "Tank Volume (L)": tank_volume * GAL_TO_L * 0.9,
        "Tank Height (m)": 1.22,
        # "UA (W/K)": 2.17,
        "UA (W/K)": 1e-9, 
        # "schedule": schedule,
        "Capacity (W)": 4500,
        "water_nodes": 12,
        **default_args,
        "time_res": dt.timedelta(minutes=1),
    }

    # Initialize equipment
    wh = ElectricResistanceWaterHeater(schedule=schedule, **equipment_args,)

    # # Simulate equipment
    if default_args.get('schedule_input_file', None) is None:
        hot_water_output_gallons, wh = simulate_first_hour_test(wh, enable_first_hour_test=True, disable_heating_during_draw=False, first_hour_duration=60, draw_rate_gpm=3, allow_setpoint_start=False, hot_water_temp_f=110, setpoint_temp_f=140)
    else:
        wh.simulate()
        
    df = wh.finalize()

    uef = calculate_uef(df, equipment_args['Tank Volume (L)'])
    
    return uef

    # print(df.head())



def run_water_heater_heatpump(default_args, setpoint_temp, tank_volume):
    # Define equipment and simulation parameters
    # setpoint_default = setpoint_temp  # in C
    # deadband_default = 5.56  # in C
    hot_water_temp = convert(110, 'degF', 'degC') 
    equipment_args = {
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": None,  # if True, must specify output_path None Merges the simulator results into 1 file
        "output_path": '../OCHRE_output/OCHRE_results/results/',
        # "Setpoint Temperature (C)": 14.4444,
        "Setpoint Temperature (C)": setpoint_temp,
        "Tank Volume (L)": tank_volume * GAL_TO_L * 0.9,
        "Tank Height (m)": 1.22,
        # "UA (W/K)": 2.17,
        "UA (W/K)": 1e-9,
        "HPWH COP (-)": 4.5,
        **default_args,
        "time_res": dt.timedelta(minutes=1),
        # "hp_only_mode": True
    }

    if default_args.get('schedule_input_file', None) is None:
        schedule = create_water_schedule(setpoint_default=setpoint_temp, withdraw_rate_gpm=0, no_heating_during_draw=False)
    else:
        hot_water_schedule = import_water_heating_schedule(default_args.get('schedule_input_file'))
        times = pd.date_range(dt.datetime(2018, 1, 1, 0, 0), dt.datetime(2018, 1, 1, 0, 0) + dt.timedelta(minutes=len(hot_water_schedule)), freq=dt.timedelta(minutes=1), inclusive="left")
        schedule = create_water_schedule(withdraw_rate_lpm=hot_water_schedule, setpoint_default=setpoint_temp, no_heating_during_draw=False, times=times)
        

    deadband_default = schedule['Water Heating Deadband (C)'].iloc[0]
    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    if default_args.get('schedule_input_file', None) is None:
        hot_water_output_gallons, hpwh = simulate_first_hour_test(hpwh, enable_first_hour_test=True, disable_heating_during_draw=False, first_hour_duration=60, draw_rate_gpm=3, allow_setpoint_start=False, hot_water_temp_f=110, setpoint_temp_f=140)
    else:
        hpwh.simulate()

    df = hpwh.finalize()
    
    uef = calculate_uef(df, equipment_args['Tank Volume (L)'])
    
    return uef

    # # print(df.head())
    # cols_to_plot = [
    #     "Hot Water Outlet Temperature (C)",
    #     "Hot Water Average Temperature (C)",
    #     "Water Heating Deadband Upper Limit (C)",
    #     "Water Heating Deadband Lower Limit (C)",
    #     "Water Heating Electric Power (kW)",
    #     "Hot Water Unmet Demand (kW)",
    #     "Hot Water Delivered (L/min)",
    # ]
    # df.loc[:, cols_to_plot].plot()
    # CreateFigures.plt.show()


main_results_folder = "../OCHRE_output/OCHRE_results/results/"
graphing_results_folder = "../OCHRE_output/results/"


def move_results(results_folder, graphings_results_folder, num_files_to_move):
    # List CSV files that do not have '_schedule' in the name.
    files = [
        f
        for f in os.listdir(results_folder)
        if f.endswith(".csv") and "_schedule" not in f
    ]

    # Sort files by modification time.
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_folder, x)))

    # Select the last num_files_to_move files.
    files_to_move = files[-num_files_to_move:]

    # Clear or create the graphings_results_folder.
    if os.path.exists(graphings_results_folder):
        shutil.rmtree(graphings_results_folder)
    os.mkdir(graphings_results_folder)

    # Move each selected file.
    for file in files_to_move:
        source = os.path.join(results_folder, file)
        destination = os.path.join(graphings_results_folder, file)
        os.rename(source, destination)


def run_water_heater_process(
    default_args, tank_volume, setpoint_temp, filename, submission_time
):
    """
    Function wrapper to execute run_water_heater in a separate process and measure:
    - Queue wait time (time from submission to actual start)
    - Active simulation processing
    - Total time (waiting + processing)
    """
    actual_start_time = time.perf_counter()
    title = default_args.get('name', 'Name_Not_Specified')
    is_heatpump = default_args.get('is_heatpump', False)
    wait_time = actual_start_time - submission_time
    print(
        f"{CYAN}Starting simulation process: {filename}; waited {wait_time:.2f} sec in queue{RESET}"
    )

    # Run the actual simulation
    try:
        if is_heatpump:
            result = run_water_heater_heatpump(default_args, setpoint_temp, tank_volume)
        else:
            result = run_water_heater_electric(default_args, setpoint_temp, tank_volume)
    except Exception as e:
        print(
            f"{RED}{title}Error in simulation process run_water_heater_process: {str(e)}{RESET}"
        )
    end_time = time.perf_counter()
    sim_duration = end_time - actual_start_time  # time spent in simulation function
    total_duration = end_time - submission_time  # includes waiting time

    print(
        f"{CYAN}Process {filename}: {title} completed in {sim_duration:.2f} sec (simulation), "
        f"total {total_duration:.2f} sec (including wait){RESET}"
    )

    return title, result, sim_duration, wait_time, total_duration



if __name__ == "__main__":
    print(
        f"{BOLD}{GREEN}Starting parallel execution of water heater simulations...{RESET}\n"
    )
    _start_time = time.perf_counter()

    no_pcm_electric_title_base = "zDefault_No_PCM_Electric"
    no_pcm_heatpump_title_base = "zDefault_No_PCM_HeatPump"
    # Create a deep copy of dwelling arguments to avoid modifying original
    default_args_default = copy.deepcopy(default_args)
    process_results = {}  # Dictionary to store execution times, wait times and UEF values
    uef_values = []

    # Initialize multiprocessing pool with limited number of processes to avoid resource contention
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Create a list to store all async results, including the default case
        async_results = []

        # Add the default (no PCM) case to the processing queue with submission timestamp
        i = 0
        for tank_volume in tank_volume_gal:
            for setpoint_temp_c, setpoint_temp_f in zip(setpoint_temps_c, setpoint_temps_f):
                # Electric water heater
                submission_time = time.perf_counter()
                current_default_args = copy.deepcopy(default_args_default)
                no_pcm_title = f"{no_pcm_electric_title_base}_setpoint-{setpoint_temp_f:.0f}F_{tank_volume}gal_{i}"
                current_default_args["name"] = no_pcm_title
                no_pcm_future_electric = pool.apply_async(
                    run_water_heater_process,
                    (
                        current_default_args,
                        tank_volume,
                        setpoint_temp_c,
                        no_pcm_title,
                        submission_time,
                    ),
                )
                async_results.append(no_pcm_future_electric)
                print(
                    f"{YELLOW}Submitted default {no_pcm_title} simulation to queue{RESET}"
                )
                
                i += 1
                # Heat pump water heater
                submission_time = time.perf_counter()
                current_default_args = copy.deepcopy(default_args_default)
                current_default_args['is_heatpump'] = True
                no_pcm_title = f"{no_pcm_heatpump_title_base}_setpoint-{setpoint_temp_f:.0f}F_{tank_volume}gal_{i}"
                current_default_args["name"] = no_pcm_title
                no_pcm_future_heatpump = pool.apply_async(
                    run_water_heater_process,
                    (
                        current_default_args,
                        tank_volume,
                        setpoint_temp_c,
                        no_pcm_title,
                        submission_time,
                    ),
                )
                async_results.append(no_pcm_future_heatpump)
                
                i += 1
                
                print(
                    f"{YELLOW}Submitted default {no_pcm_title} simulation to queue{RESET}"
                )

                # Add all PCM variation tasks to the queue
                for pcm_file_name in pcm_file_names:
                    for pcm_vol_fraction in pcm_vol_fractions:
                        for sa_ratio in sa_ratios:
                            for h_value in h_values:
                                # electric water heater
                                # Create fresh copy of default args for each simulation
                                current_default_args = copy.deepcopy(default_args_default)
                                current_pcm_properties = copy.deepcopy(DEFAULT_PCM_PROPERTIES)
                                current_pcm_properties['setpoint_temp'] = setpoint_temp_c
                                current_pcm_properties["sa_ratio"] = sa_ratio
                                current_pcm_properties["h"] = h_value
                                current_pcm_properties['enthalpy_lut'] = pcm_file_name

                                # Add PCM model with specific volume fraction
                                model_name = convert_dict_to_name(pcm_vol_fraction)

                                model_name = f"{model_name}_Electric_SA-{sa_ratio:.2f}_H-{h_value:.2f}_setpoint-{setpoint_temp_f:.0f}F_{pcm_file_name.split('.')[0]}_{tank_volume}gal_{i}"
                                i += 1
                                current_default_args = add_pcm_model(
                                    current_default_args,
                                    model_name,
                                    pcm_vol_fraction,
                                    current_pcm_properties,
                                )

                                # Record submission time for this task
                                submission_time = time.perf_counter()
                                async_result = pool.apply_async(
                                    run_water_heater_process,
                                    (
                                        current_default_args,
                                        tank_volume,
                                        setpoint_temp_c,
                                        model_name,
                                        submission_time,
                                    ),
                                )
                                async_results.append(async_result)
                                print(
                                    f"{YELLOW}Submitted {model_name} simulation to queue{RESET}"
                                )
                                # heat pump water heater
                                 # Create fresh copy of default args for each simulation
                                current_default_args = copy.deepcopy(default_args_default)
                                current_default_args['is_heatpump'] = True
                                current_pcm_properties = copy.deepcopy(DEFAULT_PCM_PROPERTIES)
                                current_pcm_properties['setpoint_temp'] = setpoint_temp_c
                                current_pcm_properties["sa_ratio"] = sa_ratio
                                current_pcm_properties["h"] = h_value
                                current_pcm_properties['enthalpy_lut'] = pcm_file_name

                                # Add PCM model with specific volume fraction
                                model_name = convert_dict_to_name(pcm_vol_fraction)

                                model_name = f"{model_name}_Heatpump_SA-{sa_ratio:.2f}_H-{h_value:.2f}_setpoint-{setpoint_temp_f:.0f}F_{pcm_file_name.split('.')[0]}_{tank_volume}gal_{i}"
                                i += 1
                                current_default_args = add_pcm_model(
                                    current_default_args,
                                    model_name,
                                    pcm_vol_fraction,
                                    current_pcm_properties,
                                )

                                # Record submission time for this task
                                submission_time = time.perf_counter()
                                async_result = pool.apply_async(
                                    run_water_heater_process,
                                    (
                                        current_default_args,
                                        tank_volume,
                                        setpoint_temp_c,
                                        model_name,
                                        submission_time,
                                    ),
                                )
                                async_results.append(async_result)
                                print(
                                    f"{YELLOW}Submitted {model_name} simulation to queue{RESET}"
                                )

        # Collect results from all simulations with improved error handling
        for async_result in async_results:
            try:
                # Now we get five return values: title, result, sim_duration, wait_time, total_duration
                title, result, sim_duration, wait_time, total_duration = (
                    async_result.get()
                )  # 5 minute timeout
                process_results[title] = {
                    "uef": result,
                    "sim_time": sim_duration,
                    "wait_time": wait_time,
                    "total_task_time": total_duration,
                }
                uef_values.append(result)
                print(f"{GREEN}Successfully completed: {title}{RESET}")
            except multiprocessing.TimeoutError:
                print(f"{RED}{title} simulation timed out after 300 seconds{RESET}")
            except Exception as e:
                print(f"{RED}{title} error in simulation process: {str(e)}{RESET}")

    # Print summary with colors
    print(f"\n{BOLD}{YELLOW}Execution Summary:{RESET}")
    for process, data in process_results.items():
        print(
            f"{GREEN}{process}: simulation time = {data['sim_time']:.2f} sec, "
            f"wait time = {data['wait_time']:.2f} sec, total = {data['total_task_time']:.2f} sec, "
            f"UEF: {data['uef']:.6f}{RESET}"
        )

    # --- Calculate and print overall overhead statistics ---
    _end_time = time.perf_counter()
    total_time = _end_time - _start_time

    # Sum up the total active simulation time and waiting time across all tasks
    total_simulation_time = sum(data["sim_time"] for data in process_results.values())
    total_wait_time = sum(data["wait_time"] for data in process_results.values())

    # Note: total_simulation_time is the sum across tasks, which is useful for profiling resource usage,
    # but it will be larger than the wall-clock time since tasks run concurrently.
    print(
        f"\n{BOLD}{GREEN}Total wall-clock execution time: {total_time:.2f} seconds{RESET}"
    )
    print(
        f"{BOLD}{YELLOW}Aggregate simulation (processing) time: {total_simulation_time:.2f} seconds{RESET}"
    )
    print(
        f"{BOLD}{YELLOW}Aggregate waiting (queue delay) time: {total_wait_time:.2f} seconds{RESET}"
    )
    overhead_percentage = (total_wait_time / total_time * 100) if total_time > 0 else 0
    print(
        f"{BOLD}{RED}Overhead due to queue delays: {total_wait_time:.2f} seconds ({overhead_percentage:.2f}% of wall-clock time){RESET}"
    )
    
    _start_time_move_results = time.perf_counter()
    move_results(
        main_results_folder,
        graphing_results_folder,
        num_files_to_move=len(process_results.items()),
    )
    print(
        f"{BOLD}{GREEN}{len(process_results.items())} results moved to {graphing_results_folder} in {time.perf_counter() - _start_time_move_results:.2f} seconds{RESET}"
    )