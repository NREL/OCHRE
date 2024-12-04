import os
import datetime as dt
import pandas as pd

from ochre import Dwelling, Analysis
from ochre.utils import default_input_path

# Script to run multiple simulations. Assumes each simulation has a unique folder with all required inputs

# Download weather files from: https://data.nrel.gov/submissions/156 or https://energyplus.net/weather
weather_path = os.path.join(default_input_path, 'Weather')


def run_single_building(input_path, size, der_type, charging_level, sim_type='ev_control', tech1='Cooking Range', tech2='EV', output_path=None, simulation_name='ochre'):
    # run individual building case
    my_print(f'Running OCHRE for building {simulation_name} ({input_path})')
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
        
    dwelling_args = {
        # Timing parameters
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=2),          # time resolution of the simulation
        "duration": dt.timedelta(days=10),            # duration of the simulation
        "initialization_time": dt.timedelta(days=1),  # used to create realistic starting temperature
        # Input parameters
        "hpxml_file": os.path.join(input_path, "bldg0112631-up11.xml"),
        "schedule_input_file": os.path.join(input_path, "bldg0112631_schedule.csv"),
        "weather_path": weather_path,
        # Output parameters
        "output_path": os.path.join(input_path, "ochre_output", sim_type),
        "output_to_parquet": False,  # saves time series files as parquet files (False saves as csv files)
        "verbosity": 7,  # verbosity of time series files (0-9)
        # 'seed': int(input_path[-3:]),
        "Equipment": {},
    }

    # determine output path, default uses same as input path
    if output_path is None:
        output_path = os.path.join(input_path, 'ochre_output')
    os.makedirs(output_path, exist_ok=True)
    
    # create OCHRE building based on der type
    if der_type == 'ev':
        dwelling = setup_ev_run(dwelling_args, charging_level)
    else: 
        # baseline ResStock dwelling
        dwelling = Dwelling(**dwelling_args)           
                        
    # run simulation
    if 'baseline' in sim_type:
        # run without controls
        dwelling.simulate()
    
    elif sim_type == 'circuit_sharing':
        circuit_sharing_control(dwelling, tech1, tech2, output_path)
    
    elif sim_type == 'circuit_pausing':
        circuit_pausing_control(dwelling, tech1, size, output_path)
        
    elif sim_type == 'ev_control':
        ev_charger_adapter(dwelling, size, output_path)


def setup_ev_run(dwelling_args, charging_level):

    equipment_args = {
        'Electric Vehicle':{
            # Equipment parameters
            'vehicle_type': 'BEV',
            'charging_level': charging_level,
            "capacity": 57.5,
        }
    }

    dwelling_args['Equipment']=equipment_args
    dwelling = Dwelling(**dwelling_args)

    return dwelling


def circuit_sharing_control(dwelling, tech1, tech2, output_path):   
    
    # run simulation with circuit sharing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    N = 0 # total number of delayed cycles
    t_delay = [] # for storing timestamps of delay
       
    for t in times:
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)
    
        # get primary load power
        if tech1 in ['Strip Heat']:
            P_prim = house_status['HVAC Heating ER Power (kW)']
        else:
            P_prim = house_status[tech1+ ' Electric Power (kW)']    
    
        # decide control for secondary load
        P_limit = 0.01 if tech1 == 'Water Heating' else 0
        if P_prim > P_limit:
            P_second = house_status[tech2+ ' Electric Power (kW)'] 
            if tech2 == 'Water Heating':
                control_signal = {tech2: {'Load Fraction': 0}}
                if P_second > 0.01: # rule out standby WH power from number of interruptions
                    N = N+1
                    t_delay.append(dwelling.current_time)
            else: # EV
                control_signal = {tech2: {'P Setpoint': 0}}
            if P_second > 0:
                N = N+1
                t_delay.append(dwelling.current_time)
        else:
            # keep previous control signal (same as forward fill)
            control_signal = None

    df = pd.DataFrame(t_delay, columns=['Timestamp'])
    df.to_csv(os.path.join(output_path, tech2+'_metrics.csv'), index=False)

    dwelling.finalize()
          

def circuit_pausing_control(dwelling, tech1, size, output_path):
          
    # run simulation with circuit pausing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    N = 0 # total number of delayed cycles
    t_delay = [] # for storing timestamps of delay
        
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)
        
        # get total power
        P_prim = house_status['Total Electric Power (kW)']
        
        # control target load
        if P_prim > 0.8*size*240/1000:
            P_second = house_status[tech1+ ' Electric Power (kW)'] 
            if tech1 == 'Water Heating':
                control_signal = {tech1: {'Load Fraction': 0}}
                if P_second > 0.01: # rule out standby WH power from number of interruptions
                    N = N+1
                    t_delay.append(dwelling.current_time)
            else: # EV
                control_signal = {tech1: {'P Setpoint': 0}}
            if P_second > 0:
                N = N+1
                t_delay.append(dwelling.current_time)
        else:
            # keep previous control signal (same as forward fill)
            control_signal = None
         
    df = pd.DataFrame(t_delay, columns=['Timestamp'])
    df.to_csv(os.path.join(output_path, tech1+'_metrics.csv'), index=False)

    dwelling.finalize()


def shed_ev(house_status, size, Pmin=1.44, Pmax=7.68):
    
    if house_status['EV Electric Power (kW)']==0:
        return None
    else:
        P_rest=house_status['Total Electric Power (kW)']-house_status['EV Electric Power (kW)']
        P_ev=0.8*size*240/1000-P_rest
        # Note: the Pmin and Pmax values are for L2 chargers.
        if P_ev<Pmin:
            P_ev=Pmin
        if P_ev>Pmax:
            P_ev=Pmax
        return P_ev
        

def ev_charger_adapter(dwelling, size, output_path):
         
    # run simulation with ev charger adapter controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    clock = dwelling.start_time # record last time EV was stopped
    N = 0 # total number of delayed cycles
    t_delay = [] # for storing timestamps of delay
    
    for t in times:
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)
        
        if house_status['Total Electric Power (kW)'] > 0.8*size*240/1000:
            control_signal = {'EV': {'P Setpoint': shed_ev(house_status, size)}}
            clock = dwelling.current_time
            if shed_ev(house_status, size) is not None:
                N += 1
                t_delay.append(dwelling.current_time)
        elif dwelling.current_time - clock < pd.Timedelta(15, "m"):
            if clock == dwelling.start_time: # no EV load has been shedded
                control_signal = None
            else:
                control_signal = {'EV': {'P Setpoint': shed_ev(house_status, size)}}
                if shed_ev(house_status, size) is not None:
                    N += 1
                    t_delay.append(dwelling.current_time)
        else:
            # Keep previous control signal (same as forward fill)
            control_signal = None

    df = pd.DataFrame(t_delay, columns=['Timestamp'])
    df.to_csv(os.path.join(output_path, 'EV_metrics.csv'), index=False)
    
    dwelling.finalize()


def compile_results(main_path, n_max=None):
    # Sample script to compile results from multiple OCHRE runs

    # set up folder for compiled results
    output_path = os.path.join(main_path, "compiled")
    os.makedirs(output_path, exist_ok=True)
    print("Compiling OCHRE results for:", main_path)

    # find all building folders
    required_files = ["OCHRE_complete"]
    run_paths = Analysis.find_subfolders(main_path, required_files)
    n = len(run_paths)

    # ensure at least 1 run folder found
    if not n:
        print("No buildings found in:", main_path)
        return
    else:
        print(f"Found {n} completed simulations in:", main_path)

    # limit total number of runs
    if n_max is not None and n > n_max:
        print(f"Limiting number of runs to {n_max}")
        run_paths = run_paths[:n_max]
        n = n_max

    run_names = {os.path.relpath(path, main_path): path for path in run_paths}

    # combine input json files
    json_files = {name: os.path.join(path, "OCHRE.json") for name, path in run_names.items()}
    df = Analysis.combine_json_files(json_files)
    df.to_csv(os.path.join(output_path, "all_ochre_inputs.csv"))

    # combine metrics files
    metrics_files = {
        name: os.path.join(path, "OCHRE_metrics.csv") for name, path in run_names.items()
    }
    df = Analysis.combine_metrics_files(metrics_files)
    df.to_csv(os.path.join(output_path, "all_ochre_metrics.csv"))

    # combine single time series column for each house (e.g., total electricity consumption)
    results_files = {name: os.path.join(path, "OCHRE.csv") for name, path in run_names.items()}
    df = Analysis.combine_time_series_column("Total Electric Power (kW)", results_files)
    df.to_csv(os.path.join(output_path, "all_ochre_total_powers.csv"))

    # aggregate outputs for each house (sum or average only)
    _, _, df_single = Analysis.load_ochre(list(run_names.values())[0], "OCHRE", load_main=False)
    agg_func = {col: Analysis.get_agg_func(col, 'House') for col in df_single.columns}
    df = Analysis.combine_time_series_files(results_files, agg_func)
    df.to_csv(os.path.join(output_path, "all_ochre_results.csv"))

    print("Saved OCHRE results to:", output_path)


def my_print(*args):
    # prints with date and other info
    now = dt.datetime.now()
    print(now, *args)


if __name__ == "__main__":
    input_path = os.path.join(default_input_path, 'Input Files')
    size = 100 # amps
    der_type = None
    charging_level = None
    
    # baseline case
    run_single_building(input_path, size, der_type, charging_level, sim_type='baseline')
    
    # case 1, circuit sharing with cooking range (primary) and WH (secondary)
    run_single_building(input_path, size, der_type, charging_level, sim_type='circuit_sharing', tech1='Cooking Range', tech2='Water Heating')
    
    # case 2, circuit pausing with WH
    run_single_building(input_path, size, der_type, charging_level, sim_type='circuit_pausing', tech1='Water Heating')
  
    # baseline case - with EV
    der_type='ev'
    charging_level='Level 2'
    run_single_building(input_path, size, der_type, charging_level, sim_type='baseline_ev')
    
    # case 3, smart EV charging
    run_single_building(input_path, size, der_type, charging_level, sim_type='ev_control', tech1='EV')
        
    # compile results
    compile_results(os.path.join(input_path, 'ochre_output'))
        