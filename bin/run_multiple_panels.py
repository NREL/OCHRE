import os
import shutil             
import sys
import time
import datetime as dt
import subprocess
from multiprocessing import Pool
import pandas as pd

from ochre import Dwelling, Analysis, ElectricVehicle

# Script to run multiple simulations. Assumes each simulation has a unique folder with all required inputs

# Download weather files from: https://data.nrel.gov/submissions/156 or https://energyplus.net/weather
# weather_path = os.path.join('path', 'to', 'weather_files')
weather_path = os.path.join(os.getcwd(), 'ResStockFiles', 'Weather')

end_use_names=['HVAC Heating Electric Power (kW)', 'HVAC Cooling Electric Power (kW)', 'Water Heating Electric Power (kW)',
               'Clothes Washer Electric Power (kW)', 'Clothes Dryer Electric Power (kW)', 'Dishwasher Electric Power (kW)',
               'Cooking Range Electric Power (kW)', 'Refrigerator Electric Power (kW)', 'Lighting Electric Power (kW)',
               'Exterior Lighting Electric Power (kW)', 'MELs Electric Power (kW)', 'EV Electric Power (kW)']

def run_multiple_hpc(main_folder, overwrite='False', n_max=None, *args):
    # runs multiple OCHRE simulations on HPC using slurm
    # args are passed to run_single_building
    overwrite = eval(overwrite)

    # find all building folders
    main_folder = os.path.abspath(main_folder)
    required_files = ['in.xml', 'schedules.csv']
    exclude_files = ['ochre_complete'] if not overwrite else []  # if not overwrite, skip completed runs
    ochre_folders = Analysis.find_subfolders(main_folder, required_files, exclude_files)
    n = len(ochre_folders)
    my_print(f'Found {n} buildings in:', main_folder)
    if n_max is not None and n > n_max:
        my_print(f'Limiting number of runs to {n_max}')
        ochre_folders = ochre_folders[:n_max]

    processes = {}
    for ochre_folder in ochre_folders:
        log_file = os.path.join(ochre_folder, 'ochre.log')

        # run srun command
        # TODO: for small runs (n<18?), might be best to remove --exclusive, or increase cpus and mem
        python_exec = shutil.which("python")
        cmd = ['srun', '--nodes=1', '--ntasks=1', '--exclusive', '-Q', '-o', log_file,
               python_exec, '-u', __file__, 'single', ochre_folder, *args
               ]
        my_print(f'Running subprocess:', ' '.join(cmd))
        p = subprocess.Popen(cmd)
        processes[p] = True  # True when process is running

    my_print('Submitted all processes.')

    n_processes = len(processes)
    n_success = 0
    n_fail = 0
    n_running = n_processes
    while n_running > 0:
        time.sleep(10)
        for p, running in processes.items():
            if not running:
                continue
            if p.poll() is not None:
                processes[p] = False
                n_running -= 1
                code = p.returncode
                if code == 0:
                    n_success += 1
                    my_print(f'Process complete ({n_success}/{n_processes}):', ' '.join(p.args))
                else:
                    n_fail += 1
                    my_print(f'Error in process ({n_fail}/{n_processes}):', ' '.join(p.args))

    my_print('All processes finished, exiting.')


def run_multiple_local(main_folder, overwrite='False', n_parallel=1, n_max=None, *args):
    # runs multiple OCHRE simulations on local machine (can run in parallel or not)
    # args are passed to run_single_building
    overwrite = eval(overwrite)

    # get all building folders
    main_folder = os.path.abspath(main_folder)
    required_files = ['in.xml', 'schedules.csv']
    exclude_files = ['ochre_complete'] if not overwrite else []  # if not overwrite, skip completed runs
    ochre_folders = Analysis.find_subfolders(main_folder, required_files, exclude_files)
    n = len(ochre_folders)
    my_print(f'Found {n} buildings in:', main_folder)
    if n_max is not None and n > n_max:
        my_print(f'Limiting number of runs to {n_max}')
        ochre_folders = ochre_folders[:n_max]

    # run single cases
    # for now, no log file. Could use subprocess.run to save logs
    # log_file = os.path.join(ochre_folder, 'ochre.log')
    
    # get parent folder name
    for ochre_folder in ochre_folders:
        print(os.path.dirname(os.path.abspath(ochre_folder)))
    
    if n_parallel == 1:
        for ochre_folder in ochre_folders:
            run_single_building(os.path.dirname(os.path.abspath(ochre_folder)), *args)
    else:
        map_args = [(os.path.dirname(os.path.abspath(ochre_folder)), *args) for ochre_folder in ochre_folders]
        with Pool(n_parallel) as p:
            p.starmap(run_single_building, map_args)

    my_print('All processes finished, exiting.')


def run_single_building(input_path, der_type='ev', sim_type='baseline', simulation_name='ochre', output_path=None):
    # run individual building case
    my_print(f'Running OCHRE for building {simulation_name} ({input_path})')
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
        
    dwelling_args = {
        # Timing parameters      
        'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        'time_res': dt.timedelta(minutes=15),         # time resolution of the simulation
        'duration': dt.timedelta(days=365),             # duration of the simulation
        'initialization_time': dt.timedelta(days=1),  # used to create realistic starting temperature

        # Input parameters
        'hpxml_file': os.path.join(input_path, 'Input Files', 'in.xml'),
        'schedule_input_file': os.path.join(input_path, 'Input Files', 'schedules.csv'),
        'weather_path': weather_path,
    
        # Output parameters
        'output_path': input_path,
        'output_to_parquet': False,              # saves time series files as parquet files (False saves as csv files)
        'verbosity': 7,                         # verbosity of time series files (0-9)
        
        'seed': int(input_path[-3:]),
        
        'Equipment': {
        },
    }

    # determine output path, default uses same as input path
    if output_path is None:
        output_path = input_path
    os.makedirs(output_path, exist_ok=True)
    
    
    # create OCHRE building based on der type
    if der_type == 'ev':
        dwelling = setup_ev_run(simulation_name, dwelling_args)
    else: 
        # baseline ResStock dwelling
        dwelling = Dwelling(simulation_name,**dwelling_args)           
                            
    # run simulation
    if sim_type == 'baseline':
        # run without controls
        dwelling.simulate()
    
    elif sim_type == 'circuit_sharing':
        assert der_type == 'ev'
        circuit_sharing_control(dwelling)
    
    elif sim_type == 'ev_charger_adapter':
        assert der_type == 'ev'
        ev_charger_adapter(input_path, dwelling)
    
    elif sim_type == 'smart_electrical_panel':
        assert der_type == 'ev'
        smart_electrical_panel(input_path, dwelling)
    
    elif sim_type == 'hems':
        assert der_type == 'ev'
        hems(input_path, dwelling)
    
    
def setup_ev_run(simulation_name, dwelling_args):


    equipment_args = {
        'Electric Vehicle':{
            # Equipment parameters
            'vehicle_type': 'BEV',
            'charging_level': 'Level 2',
            'mileage': 300,
        }
    }

    # Initialize equipment
    # equipment = ElectricVehicle(**equipment_args)

    dwelling_args['Equipment']=equipment_args
    # print(dwelling_args)
    dwelling = Dwelling(simulation_name, **dwelling_args)


    return dwelling


def circuit_sharing_control(dwelling):
    
    dryer = dwelling.get_equipment_by_end_use('Clothes Dryer') 
    
    # run simulation with circuit sharing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        # print(dryer.schedule.loc[t, 'Clothes Dryer (kW)'])
        if dryer.schedule.loc[t, 'Clothes Dryer (kW)'] > 0:
            control_signal = {'EV': {'P Setpoint': 0}}
        else:
            # Keep previous control signal (same as forward fill)
            control_signal = None
        
        house_status = dwelling.update(ext_control_args=control_signal)

    dwelling.finalize()


def total_amp(house_status):
    
    amp=0
    
    # print(house_status)
    for end_use in end_use_names:
        if end_use in house_status:
            if end_use in [end_use_names[j] for j in [0,1,2,4,6,11]]:
                amp+=house_status[end_use]*1000/240
            else:
                amp+=house_status[end_use]*1000/120
            
    return amp


def shed_ev(house_status, size, Pmin=1.44, Pmax=11.5):
    
    if house_status['EV Electric Power (kW)']==0:
        return None
    else:
        P_rest=house_status['Total Electric Power (kW)']-house_status['EV Electric Power (kW)']
        P_ev=0.8*size*240/1000-P_rest
        if P_ev<Pmin:
            P_ev=Pmin
        if P_ev>Pmax:
            P_ev=Pmax
        return P_ev
        

def ev_charger_adapter(input_path, dwelling):
      
    # read panel sizes
    panels = pd.read_csv('D:/GitHub/ochre/ResStockFiles/panels.csv', index_col=0)
    size = panels['panel'].loc[int(input_path[-3:])]
    print('panel size:', size)
    
    # run simulation with ev charger adapter controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    clock = dwelling.start_time # record last time EV was stopped
    
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(ext_control_args=control_signal)
        # print('total amp:', t, total_amp(house_status))
        
        if house_status['Total Electric Power (kW)'] > 0.8*size*240/1000:
            control_signal = {'EV': {'P Setpoint': shed_ev(house_status, size)}}
            clock = dwelling.current_time
        elif dwelling.current_time - clock < pd.Timedelta(15, "m"):
            if clock == dwelling.start_time: # no EV load has been shedded
                control_signal = None
            else:
                control_signal = {'EV': {'P Setpoint': shed_ev(house_status, size)}}
        else:
            # Keep previous control signal (same as forward fill)
            control_signal = None
        
        # house_status = dwelling.update(ext_control_args=control_signal)

    dwelling.finalize()


def smart_electrical_panel(input_path, dwelling):
      
    # read panel sizes
    panels = pd.read_csv('D:/GitHub/ochre/ResStockFiles/panels.csv', index_col=0)
    size = panels['panel'].loc[int(input_path[-3:])]
    print('panel size:', size)
    
    # read load order
    order = pd.read_csv('D:/GitHub/ochre/ResStockFiles/load_order.csv', index_col=0)
    print('load order:', order)
    
    # run simulation with smart panel controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    clock = dwelling.start_time # record last time total power exceeds 80%
    
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(ext_control_args=control_signal)
        
        # house_status = dwelling.update(ext_control_args={'Lighting': {'P Setpoint': 0}})
        # control_signal = {'Lighting': {'Load Fraction': 0}}
        
        if house_status['Total Electric Power (kW)'] > 0.8*size*240/1000:
            # turn off the last load on the list
            control_signal = {str(order['load'].iloc[-1]): {str(order['control variable'].iloc[-1]): 0}}
            new_total = house_status['Total Electric Power (kW)'] - house_status[str(order['load'].iloc[-1])+' Electric Power (kW)']
            for i in range(len(order)-1):
                if new_total <= 0.8*size*240/1000:
                    break
                else:
                    control_signal[str(order['load'].iloc[-2-i])] = {str(order['control variable'].iloc[-2-i]): 0}
                    if str(order['load'].iloc[-2-i])+' Electric Power (kW)' in house_status:
                        new_total = new_total - house_status[str(order['load'].iloc[-2-i])+' Electric Power (kW)']
                    else:
                        new_total = new_total
            clock = dwelling.current_time
        elif dwelling.current_time - clock < pd.Timedelta(15, "m"):
            # Keep previous control signal (same as forward fill)
            pass
        else:
            # turn on loads one at a time
            if clock == dwelling.start_time: # no load has been turned off, i is not initialized
                control_signal = None
            elif i == 0: # only the last load has been turned off
                control_signal = None
            else:
                control_signal = {str(order['load'].iloc[-1]): {str(order['control variable'].iloc[-1]): 0}}
                for j in range(i-1):
                    control_signal[str(order['load'].iloc[-2-j])] = {str(order['control variable'].iloc[-2-j]): 0}
                i=i-1
        
        print('clock:', clock, 'control signal:', control_signal)
        
    dwelling.finalize()


def hems(input_path, dwelling):
      
    # read panel sizes
    panels = pd.read_csv('D:/GitHub/ochre/ResStockFiles/panels.csv', index_col=0)
    size = panels['panel'].loc[int(input_path[-3:])]
    print('panel size:', size)

    
    # run simulation with HEMS control
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    clock = dwelling.start_time # record last time total power exceeds 80%
    
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(ext_control_args=control_signal)
        
        if house_status['Total Electric Power (kW)'] > 0.8*size*240/1000:
            # shed EV load first
            control_signal = {'EV': {'P Setpoint': shed_ev(house_status, size)}}
            if shed_ev(house_status, size) is not None:
                new_total = house_status['Total Electric Power (kW)'] - house_status['EV Electric Power (kW)'] + shed_ev(house_status, size)
            else:
                new_total = house_status['Total Electric Power (kW)']
            if new_total > 0.8*size*240/1000:
                    # shed TCLs
                    control_signal['HVAC Heating'] = {'Setpoint': 17.8} # 64 F
                    control_signal['HVAC Cooling'] = {'Setpoint': 29.4} # 85 F
                    control_signal['Water Heating'] = {'Setpoint': 46.1} # 115 F                      
            clock = dwelling.current_time # loads have been changed, the 15 min requirement kicks in
        
        elif dwelling.current_time - clock < pd.Timedelta(15, "m"):
            if clock == dwelling.start_time: # no load has been turned off
                control_signal = None
            else:
                control_signal['EV'] = {'P Setpoint': shed_ev(house_status, size)}
                control_signal['Electric Vehicle'] = {'P Setpoint': shed_ev(house_status, size)}

        else:
            # turn on loads one at a time
            if control_signal is None: # no load has been turned off
                control_signal = None
            elif 'HVAC Heating' in control_signal: # TCLs have been shedded at the last timestep
                control_signal = {'EV': {'P Setpoint': shed_ev(house_status, size)}} # revert setpoints but keep shedding EV
            else:
                control_signal = None # revert all load control
                
        print('clock:', clock, 'control signal:', control_signal)
        
    dwelling.finalize()
    

def compile_results(main_folder):
    # Sample script to compile results from multiple OCHRE runs
    # assumes each run is in a different folder, and all simulation names are 'ochre'
    # dirs_to_include = int(dirs_to_include)

    # set up
    main_folder = os.path.abspath(main_folder)
    output_folder = os.path.join(main_folder, 'compiled')
    os.makedirs(output_folder, exist_ok=True)
    my_print('Compiling OCHRE results for:', main_folder)

    # find all building folders
    required_files = ['ochre.parquet', 'ochre_metrics.csv', 'ochre_complete']
    run_folders = Analysis.find_subfolders(main_folder, required_files)
    n = len(run_folders)
    my_print(f'Found {n} folders with completed OCHRE simulations')

    # combine input json files
    df = Analysis.combine_json_files(path=main_folder)
    df.to_csv(os.path.join(output_folder, 'all_ochre_inputs.csv'))

    # combine metrics files
    df = Analysis.combine_metrics_files(path=main_folder)
    df.to_csv(os.path.join(output_folder, 'all_ochre_metrics.csv'))

    # combine total electricity consumption for each house
    results_files = [os.path.join(f, 'ochre.parquet') for f in run_folders]
    df = Analysis.combine_time_series_column(results_files, 'Total Electric Power (kW)')
    df.to_parquet(os.path.join(output_folder, 'all_ochre_powers.parquet'))

    # aggregate outputs for each house (sum or average only)
    _, _, df_single = Analysis.load_ochre(run_folders[0], 'ochre', load_main=False)
    agg_func = {col: Analysis.get_agg_func(col, 'House') for col in df_single.columns}
    df = Analysis.combine_time_series_files(results_files, agg_func)
    df.to_parquet(os.path.join(output_folder, 'all_ochre_results.parquet'))

    my_print('All compiling complete')


def my_print(*args):
    # prints with date and other info
    now = dt.datetime.now()
    print(now, *args)


if __name__ == "__main__":
    assert len(sys.argv) >= 2
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == 'hpc':
        run_multiple_hpc(*args)
    elif cmd == 'local':
        run_multiple_local(*args)
    elif cmd == 'single':
        run_single_building(*args)
    else:
        my_print(f'Invalid command ({cmd}) for run_ochre.py. Must be "hpc", "local", or "single".')

    # compile results from multi-run
    if args:
        compile_results(args[0])
