import datetime as dt


from ochre import Dwelling, CreateFigures
from ochre.Models import TankWithPCM, TankWithMultiPCM
from bin.run_dwelling import dwelling_args
import copy
import matplotlib.pyplot as plt
import time
from itertools import chain, combinations
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import os
import shutil
import glob
from bin.compare_specific_results import load_data, create_heat_exchanger_plots, create_temperature_plots, calculate_hot_water_delivered, plot_draw_event_summary, plot_draw_events, create_capacitance_plots, plot_pcm_enthalpies


start_node = 4
end_node = 9

water_heater_setpoint_temp = 59

vol_fract = 0.00000001 # 1.540e-06 kg
vol_fract = 0.0001 # 1.540e-02 kg
vol_fract = 0.5 # 7.700e+01 kg
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
    pcm_vol_fractions.append({node: vol_fract for node in range(start_node, end_node + 1)})
# no PCM in top and bottom node    

# pcm_vol_fractions[0][3] = 0.00000001

# UEF draw profiles
# LowUseUEF = 'LowUseL.csv'
# MediumUseUEF = 'MediumUseL.csv'
# HighUseUEF = 'HighUseL.csv'

no_pcm_title = "No PCM"
# with_pcm_title = "With PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)

load_profile = "2.00gpm30min_0gpm180min_cycling.csv"
load_profile = "2.00gpm30min_0gpm600min_cycling.csv"
load_profile = "MediumUseL.csv"
load_profile = "2.00gpm120min_0gpm600min_cycling.csv"
load_profiles = ["MediumUseL.csv", "2.00gpm30min_0gpm600min_cycling.csv"]

# updated names if all nodes have same vol_fract
def convert_dict_to_name(dict):
    # Check if all values are the same
    values = list(dict.values())
    if len(values) > 0 and all(v == values[0] for v in values):
        # Check if keys are sequential integers
        keys = sorted(dict.keys())
        if all(isinstance(k, int) for k in keys) and keys == list(range(min(keys), max(keys) + 1)):
            # Return the compact format
            return f"{values[0]}_pcm{min(keys)}-{max(keys)}"
    
    # Fallback to original format if conditions aren't met
    return "".join([str(key) + "_" + str(value) + "_" for key, value in dict.items()])

dwelling_args.update(
    {
        "time_res": dt.timedelta(minutes=1),  # time resolution of the simulation
        "duration": dt.timedelta(days=2),  # duration of the simulation
        "verbosity": 9,
        'Setpoint Temperature (C)': 60,
        "schedule_input_file": load_profile,  # changes the default load profile in run_dwelling.py for this code to call the UEF load_profile
        "output_path": "../OCHRE_output/OCHRE_results/results/",
        "name": "zDefault",
    }
)

def calculate_uef(df, water_volume):
    # calculate UEF of the water tank
    Q_cons = (
        df["Water Heating Electric Power (kW)"].sum() * 1000
    )  # not sure if this is the correct term that I should be pulling
    Q_load = df[
        "Hot Water Delivered (W)"
    ].sum()  # not sure if this is the correct term that I should be pulling
    
    PCM_Q_Heat_to_Water= calculate_net_PCM_heat(df)     # make sure in W*min
    PCM_net_enthalpy = calculate_net_PCM_enthalpy(df)                               
    PCM_net_heat_loss = PCM_net_enthalpy / 60           # make sure in W*min
    water_net_temp_delta = calculate_net_water_temp(df)
    water_net_energy = calculate_net_water_energy(water_volume, df['Hot Water Average Temperature (C)'].iloc[-1], water_net_temp_delta) / 60 # make sure in W*min
    Q_cons_total = Q_cons - PCM_net_heat_loss - water_net_energy          # make sure in W*min                            
    UEF = Q_load / Q_cons_total
    
    return UEF
    

# Predefined lookup arrays for water properties at 1 atm.
_TEMPS = np.array([0, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
_DENSITIES = np.array([999.8, 1000.0, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4], dtype=float)
_THERMALEXPANSIONS = np.array([-1.0e-4, 0.0, 1.1e-4, 2.1e-4, 2.6e-4, 3.1e-4, 3.6e-4, 4.1e-4, 4.7e-4, 5.3e-4, 5.9e-4, 6.6e-4], dtype=float)

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

def calculate_net_water_temp(df):
    water_temp_column = ["Hot Water Average Temperature (C)"]
    valid_columns = [col for col in water_temp_column if col in df.columns]

    if not valid_columns:
        return 0

    # Compute the average of the first and last row for all available columns
    average_start_temp = df[valid_columns].iloc[0].mean()
    average_end_temp = df[valid_columns].iloc[-1].mean()

    return average_end_temp - average_start_temp
    

def calculate_net_PCM_heat(df):

    # Dynamically find all PCM enthalpy columns.
    pcm_column = 'Total PCM Heat Injected (W)'
    
    if pcm_column not in df.columns:
        return 0
    
    # Sum the PCM enthalpy columns row-wise.

    net_PCM_to_water_heat_Transfer = df[pcm_column].sum()
    
    return net_PCM_to_water_heat_Transfer

def calculate_net_water_energy(volume, temperature, temperature_difference):
    
    water_properties = lookup_water_properties(temperature)
    density = water_properties["density"]
    # volume is in Liters
    water_weight = density * volume / 1e3 # in kg
    
    
    return water_weight * temperature_difference * 4184 # J

def calculate_net_PCM_enthalpy(df):
    # Dynamically find all PCM enthalpy columns.
    pcm_column = 'Total PCM Enthalpy (J)'
    
    if pcm_column not in df.columns:
        return 0
    
    # Sum the PCM enthalpy columns row-wise.

    net_PCM_enthalpy = df[pcm_column].iloc[-1] - df[pcm_column].iloc[0]
    
    return net_PCM_enthalpy

def add_pcm_model(dwelling_args, name, pcm_vol_fractions):
    dwelling_args["Equipment"]["Water Heating"] = {
        "model_class": TankWithMultiPCM,
        "water_nodes": 12,
        "Water Tank": {
            "pcm_node_vol_fractions": pcm_vol_fractions,
        },
    }

    dwelling_args["name"] = name
    dwelling_args["output_path"] = "../OCHRE_output/OCHRE_results/results/"
    return dwelling_args


def run_water_heater(dwelling_args, plot_title, load_profile_in, name):
    # Create Dwelling from input files, see bin/run_dwelling.py

    
    dwelling = Dwelling(**dwelling_args)

    # Extract equipment by its end use and update simulation properties
    equipment = dwelling.get_equipment_by_end_use("Water Heating")
    
    equipment.main_simulator = True
    equipment.save_results = dwelling.save_results
    # equipment.model.save_results = True
    # equipment.model.results_file = "test.csv"
    equipment.export_res = dwelling.export_res
    equipment.results_file = dwelling.results_file
    equipment.verbosity = 9
    equipment.model.verbosity = 9
    equipment.metrics_verbosity = 9

    # If necessary, update equipment schedule
    equipment.current_schedule['Water Heating Setpoint (C)'] = 60
    equipment.setpoint_temp = water_heater_setpoint_temp
    # set all values of equipment.schedule to water_heater_setpoint_temp
    equipment.schedule['Water Heating Setpoint (C)'] = water_heater_setpoint_temp
    water_volume = equipment.model.volume
    equipment.model.schedule["Zone Temperature (C)"] = (
        19.722222  # from the UEF standard https://www.energy.gov/eere/buildings/articles/2014-06-27-issuance-test-procedures-residential-and-commercial-water
    )
    # equipment.model.schedule['Water Use Schedule (L/min)'] = load_profile_in #converted the schedule files directly to L/min
    equipment.model.schedule["Mains Temperature (C)"] = 14.4444
    # TODO: 50% RH schedule? Will have some impact on HP performance, but not much
    equipment.reset_time()

    # Simulate equipment
    df = equipment.simulate()
    # print(df.columns)

    # # print(df.head())
    # CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))
    # CreateFigures.plt.title(plot_title)
    # CreateFigures.plt.suptitle(load_profile_in)

    # # print all water tank temperatures
    # cols = [f"T_WH{i}" for i in range(1, 13)]
    # if "With PCM" in plot_title:
    #     cols += ["T_PCM"]

    # df[cols].plot()
    # CreateFigures.plt.show()
    
    # cols = [f"T_PCM{i}" for i in range(1, 13)]
    # if "With PCM" in plot_title:
    #     cols += ["T_PCM"]

    # df[cols].plot()
    # CreateFigures.plt.show()
    

    UEF = calculate_uef(df, water_volume)
    print(f" {plot_title} {load_profile_in} UEF = {UEF}")
    return UEF


main_results_folder = "../OCHRE_output/OCHRE_results/results/"
graphing_results_folder = "../OCHRE_output/results/"


def move_results(results_folder, graphings_results_folder, num_files_to_move):
    # find most recent num_files_to_move files in results_folder
    files = os.listdir(results_folder)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(results_folder, x)))
    files_to_move = files[-num_files_to_move:]
    
    # clear graphings_results_folder
    if os.path.exists(graphings_results_folder):
        shutil.rmtree(graphings_results_folder)
        os.mkdir(graphings_results_folder)
    
    for file in files_to_move:
        # move file to graphings_results_folder
        os.rename(os.path.join(results_folder, file), os.path.join(graphings_results_folder, file))
        
        

    

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"

def run_water_heater_process(dwelling_args, title, load_profile, filename):
    """ Function wrapper to execute run_water_heater in a separate process and measure execution time. """
    start_time = time.perf_counter()
    print(f"{CYAN}Starting simulation process: {filename}{RESET}")
    result = run_water_heater(dwelling_args, title, load_profile, filename)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    print(f"{CYAN}Process {filename}: {title} completed in {duration:.2f} seconds{RESET}")
    
    return title, result, duration

if __name__ == "__main__":
    print(f"{BOLD}{GREEN}Starting parallel execution of water heater simulations...{RESET}\n")
    _start_time = time.perf_counter()

    # Create a deep copy of dwelling arguments to avoid modifying original
    dwelling_args_default = copy.deepcopy(dwelling_args)
    process_results = {}  # Dictionary to store execution times and UEF values
    uef_values = []
    
    # Initialize multiprocessing pool with limited number of processes to avoid resource contention
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Create a list to store all async results, including the default case
        async_results = []
        
        # Add the default (no PCM) case to the processing queue
        no_pcm_future = pool.apply_async(
            run_water_heater_process,
            (dwelling_args_default, no_pcm_title, load_profile, no_pcm_title)
        )
        async_results.append(no_pcm_future)
        print(f"{YELLOW}Submitted default NO_PCM simulation to queue{RESET}")
        
        # Add all PCM variation tasks to the queue
        for pcm_vol_fraction in pcm_vol_fractions:
            # Create fresh copy of default args for each simulation
            current_dwelling_args = copy.deepcopy(dwelling_args_default)
            
            # Add PCM model with specific volume fraction
            model_name = convert_dict_to_name(pcm_vol_fraction)
            current_dwelling_args = add_pcm_model(
                current_dwelling_args, 
                model_name, 
                pcm_vol_fraction
            )
            
            # Submit task to process pool
            async_result = pool.apply_async(
                run_water_heater_process,
                (current_dwelling_args, model_name, load_profile, model_name)
            )
            async_results.append(async_result)
            print(f"{YELLOW}Submitted {model_name} simulation to queue{RESET}")
        
        # Collect results from all simulations with improved error handling
        for async_result in async_results:
            try:
                title, result, duration = async_result.get()  # 5 minute timeout
                process_results[title] = {"uef": result, "time": duration}
                uef_values.append(result)
                print(f"{GREEN}Successfully completed: {title}{RESET}")
            except multiprocessing.TimeoutError:
                print(f"{RED}A simulation timed out after 300 seconds{RESET}")
            except Exception as e:
                print(f"{RED}Error in simulation process: {str(e)}{RESET}")

    # Print summary with colors
    print(f"\n{BOLD}{YELLOW}Execution Summary:{RESET}")
    for process, data in process_results.items():
        print(f"{GREEN}{process}: {data['time']:.2f} seconds, UEF: {data['uef']:.6f}{RESET}")
    
    _start_time_move_results = time.perf_counter()
    move_results(main_results_folder, graphing_results_folder, num_files_to_move=len(process_results.items()))
    print(f"{BOLD}{GREEN}{len(process_results.items())} results moved to {graphing_results_folder} in {time.perf_counter() - _start_time_move_results:.2f} seconds{RESET}")
    
    _start_time_plot_results = time.perf_counter()
    dfs  = load_data(results_folder=graphing_results_folder)
    plot = create_heat_exchanger_plots(dfs)
    temp_plots,_= create_temperature_plots(dfs, uef_values=uef_values, patterns=['T_WH', 'T_PCM'])
    enthalpy_plot = plot_pcm_enthalpies(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "ochre", "Models", "cp_h-T_data.csv"), delimiter=",", skiprows=1))
    capacitance_plots,_ = create_capacitance_plots(dfs, uef_values=uef_values)
    plot.show()
    enthalpy_plot.show()
    for temp_plot in temp_plots:
        temp_plot.show()
    for capacitance_plot in capacitance_plots:
        capacitance_plot.show()
    
    print(f"{BOLD}{GREEN}Plots created in {time.perf_counter() - _start_time_plot_results:.2f} seconds{RESET}")
    
    # Draw data summary
    output = calculate_hot_water_delivered(dfs)
    plot_draw_event_summary(output)
    plot_draw_events(output) 
    
    _end_time = time.perf_counter()
    total_time = _end_time - _start_time
    print(f"\n{BOLD}{RED}Total execution time: {total_time:.2f} seconds{RESET}")
    
    