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


def run_single_building(input_path, size, der_type=None, sim_type='circuit_sharing', tech1='Cooking Range', tech2='Clothes Dryer', simulation_name='ochre', output_path=None):
    # run individual building case
    my_print(f'Running OCHRE for building {simulation_name} ({input_path})')
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)
        
    dwelling_args = {
        # Timing parameters      
        'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        'time_res': dt.timedelta(minutes=2),         # time resolution of the simulation
        'duration': dt.timedelta(days=365),             # duration of the simulation
        'initialization_time': dt.timedelta(days=1),  # used to create realistic starting temperature

        # Input parameters
        'hpxml_file': os.path.join(input_path, 'in.xml'),
        'schedule_input_file': os.path.join(input_path, 'schedules.csv'),
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
        circuit_sharing_control(sim_type, dwelling, tech1, tech2, output_path)
    
    elif sim_type == 'circuit_pausing':
        circuit_pausing_control(sim_type, input_path, dwelling, tech1, size, output_path)
    
    
def setup_ev_run(simulation_name, dwelling_args):


    equipment_args = {
        'Electric Vehicle':{
            # Equipment parameters
            'vehicle_type': 'BEV',
            'charging_level': 'Level 2',
            "capacity": 57.5,
        }
    }

    # Initialize equipment
    # equipment = ElectricVehicle(**equipment_args)

    dwelling_args['Equipment']=equipment_args
    # print(dwelling_args)
    dwelling = Dwelling(simulation_name, **dwelling_args)

    return dwelling


def circuit_sharing_control(sim_type, dwelling, tech1, tech2, output_path):   
    
    # run simulation with circuit sharing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    
    if tech2 == 'Clothes Dryer':
        # initialize an empty list for cycle recording
        pipeline = [] # for storing cycles waiting to be rescheduled
        n_delay = 0 # number of cycles waiting in the pipeline
        N = 0 # total number of delayed cycles
        t_delay = [] # for storing timestamps of delay
        deltat_delay = [] # for storing delayed timesteps of each delayed cycle
        n_pop = 0 # index for tracking number of cycles poped
    
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)
    
        # get primary load power
        if tech1 in ['Strip Heat']:
            P_prim = house_status['HVAC Heating ER Power (kW)']
        else:
            P_prim = house_status[tech1+ ' Electric Power (kW)']    
    
        # decide control for secondary load
        if tech2 == 'Clothes Dryer':
            n_delay, N, t_delay, deltat_delay, n_pop = dryer_control(sim_type, dwelling, P_prim, t, pipeline, n_delay, N, t_delay, deltat_delay, n_pop)
        else:        
            if P_prim > 0:
                if tech2 == 'Water Heating':
                    control_signal = {tech2: {'Load Fraction': 0}}
                else: # EV
                    control_signal = {tech2: {'P Setpoint': 0}}
            else:
                # keep previous control signal (same as forward fill)
                control_signal = None

    if tech2 == 'Clothes Dryer':
        df = pd.DataFrame(t_delay, columns=['Timestamp'])
        df['Delayed Time (s)'] = deltat_delay
        df.to_csv(os.path.join(output_path, 'Dryer_metrics.csv'), index=True)
        # print('File saved.')

    dwelling.finalize()
    
    
def record_and_remove_cycle(dwelling, schedule, t, pipeline):
    
    # all zero indicies
    zero_indices = schedule.index[schedule == 0].tolist()
    
    # find the last index that the power is zero - cycle start
    cycle_start = max([index for index in zero_indices if index < t], default=None) + dwelling.time_res
        
    # find the next index that the power is zero - cycle end
    cycle_end = min([index for index in zero_indices if index > t], default=None) - dwelling.time_res
    
    # record cycle in pipeline
    pipeline.append(schedule[cycle_start:cycle_end].copy())
    # print(pipeline) 
    
    # remove cycle from schedule
    for i in pd.date_range(cycle_start, cycle_end, freq=dwelling.time_res, inclusive='both'):
        schedule.loc[i] = 0
    
    return(schedule)


def add_first_cycle_to_schedule(dwelling, schedule, pipeline, deltat_delay, n_pop):
    
    # first cycle in pipeline
    # print(pipeline[0])
    
    cycle_start = pipeline[0].index[0]
    cycle_end = pipeline[0].index[-1]
    cycle_length = cycle_end - cycle_start
    
    # add to schedule
    schedule[dwelling.current_time:dwelling.current_time + cycle_length] = pipeline[0]
        
    # update pipeline
    deltat_delay[n_pop] = dwelling.current_time - deltat_delay[n_pop]
    pipeline.pop(0)
    n_pop += 1
    # print(pipeline)
    
    return(schedule, n_pop)


def coincide_with_next_cycle(dwelling, schedule, pipeline):
    
    cycle_start = pipeline[0].index[0]
    cycle_end = pipeline[0].index[-1]
    cycle_length = cycle_end - cycle_start
    
    for i in pd.date_range(dwelling.current_time, dwelling.current_time + cycle_length, freq=dwelling.time_res, inclusive='both'):
        if schedule.loc[i] > 0:
            return(True, i)
            
    return(False, dwelling.current_time)


def exceed_threshold(P_prim, size, pipeline):
    
    if P_prim + pipeline[0][0] > 0.8*size*240/1000:
        return(True)
    else:
        return(False)


def dryer_control(sim_type, dwelling, P_prim, t, pipeline, n_delay, N, t_delay, deltat_delay, n_pop, size=0):
    
    dryer = dwelling.get_equipment_by_end_use('Clothes Dryer')
    schedule = dryer.schedule['Clothes Dryer (kW)']
    
    if (sim_type == 'circuit_sharing' and P_prim > 0) or (sim_type == 'circuit_pausing' and P_prim > 0.8*size*240/1000):
        if schedule.loc[t] > 0:
            # record and remove the cycle from schedule
            schedule = record_and_remove_cycle(dwelling, schedule, t, pipeline)
            n_delay += 1
            N += 1
            t_delay.append(dwelling.current_time)
            deltat_delay.append(dwelling.current_time)
            dryer.reset_time(start_time=dwelling.current_time)
            # print(n_delay, N, n_pop, t_delay)
    else:
        if (sim_type == 'circuit_sharing' and n_delay > 0) or (sim_type == 'circuit_pausing' and n_delay > 0 and not exceed_threshold(P_prim, size, pipeline)):
            # check if exceeds threshold if adding dryer cycle, only when circuit pausing
            # check if will coincide with next cycle
            if coincide_with_next_cycle(dwelling, schedule, pipeline)[0]: 
                schedule = record_and_remove_cycle(dwelling, schedule, coincide_with_next_cycle(dwelling, schedule, pipeline)[1], pipeline)
                n_delay += 1
                N += 1
                t_delay.append(coincide_with_next_cycle(dwelling, schedule, pipeline)[1])
                deltat_delay.append(coincide_with_next_cycle(dwelling, schedule, pipeline)[1])
                dryer.reset_time(start_time=dwelling.current_time)
                # print(n_delay, N, n_pop, t_delay)

            # add the first cycle in pipeline to schedule
            schedule, n_pop = add_first_cycle_to_schedule(dwelling, schedule, pipeline, deltat_delay, n_pop)
            n_delay -= 1
            dryer.reset_time(start_time=dwelling.current_time)
            # print(n_delay, N, n_pop, t_delay)
    
    return(n_delay, N, t_delay, deltat_delay, n_pop)
        

def circuit_pausing_control(sim_type, input_path, dwelling, tech1, size, output_path):
      
    # read panel sizes
    # panels = pd.read_csv('C:/GitHub/ochre/ResStockFiles/panels.csv', index_col=0)
    # size = panels['panel'].loc[int(input_path[-3:])]
    # print('panel size:', size)
    
    # run simulation with circuit pausing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    
    if tech1 == 'Clothes Dryer':
        # initialize an empty list for cycle recording
        pipeline = [] # for storing cycles waiting to be rescheduled
        n_delay = 0 # number of cycles waiting in the pipeline
        N = 0 # total number of delayed cycles
        t_delay = [] # for storing timestamps of delay
        deltat_delay = [] # for storing delayed timesteps of each delayed cycle
        n_pop = 0 # index for tracking number of cycles poped
        
    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)
        
        # get total power
        P_prim = house_status['Total Electric Power (kW)']
        
        # control target load
        if tech1 == 'Clothes Dryer':
            n_delay, N, t_delay, deltat_delay, n_pop = dryer_control(sim_type, dwelling, P_prim, t, pipeline, n_delay, N, t_delay, deltat_delay, n_pop, size)
        else:
            if P_prim > 0.8*size*240/1000:
                if tech1 == 'Water Heating':
                    control_signal = {tech1: {'Load Fraction': 0}}
                else: # EV
                    control_signal = {tech1: {'P Setpoint': 0}}
            else:
                # keep previous control signal (same as forward fill)
                control_signal = None
        
    if tech1 == 'Clothes Dryer':
        df = pd.DataFrame(t_delay, columns=['Timestamp'])
        df['Delayed Time (s)'] = deltat_delay
        df.to_csv(os.path.join(output_path, 'Dryer_metrics.csv'), index=True)
        # print('File saved.')

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
    # assert len(sys.argv) >= 2
    # cmd = sys.argv[1]
    # args = sys.argv[2:]
    # if cmd == 'hpc':
    #     run_multiple_hpc(*args)
    # elif cmd == 'local':
    #     run_multiple_local(*args)
    # elif cmd == 'single':
    #     run_single_building(*args)
    # else:
    #     my_print(f'Invalid command ({cmd}) for run_ochre.py. Must be "hpc", "local", or "single".')

    # # compile results from multi-run
    # if args:
    #     compile_results(args[0])
    
    # for running on HPC
    # read the spreadsheet containing all scenarios
    scenarios = pd.read_csv(os.path.join(os.getcwd(), 'bin', 'control_scenarios_processed.csv'))
    # print(scenarios)
    
    # find the controller type based on building id and upgrade number, ignore multiple controllers for now
    for l in range(len(scenarios[138:139])):
        
        k=scenarios[138:139].index[l]
        bldg_id = scenarios['building_id'].iloc[k]
        # print(l, k, bldg_id)
        
        input_path = os.path.join(os.getcwd(), 'ResStockFiles', 'upgrade'+str(scenarios['case'].iloc[k]), str(bldg_id))
        size = scenarios['panel'].iloc[k]
        
        # preprocess schedules of low-power appliances
        if scenarios['case'].iloc[k] in [6, 7, 13, 14]:
            schedule = pd.read_csv(os.path.join(input_path, 'schedules.csv'), index_col=None)
            schedule['clothes_dryer'] = schedule['clothes_dryer']/2
            schedule['cooking_range'] = schedule['cooking_range']/2
            schedule.to_csv(os.path.join(input_path, 'schedules.csv'), index=False)
        elif scenarios['case'].iloc[k] in [16, 17, 18, 19]:
            schedule = pd.read_csv(os.path.join(input_path, 'schedules.csv'), index_col=None)
            schedule['clothes_dryer'] = schedule['clothes_dryer']/2
            schedule.to_csv(os.path.join(input_path, 'schedules.csv'), index=False)
        
        if scenarios['circuit_pause'].iloc[k] > 1:
            continue
        elif (scenarios['circuit_pause'].iloc[k] == 1) and (scenarios['circuit_share'].iloc[k] == 1):
            continue
        elif scenarios['circuit_pause'].iloc[k] == 1:
            run_single_building(input_path, sim_type='circuit_pausing', tech1=scenarios['circuit_pause_name'].iloc[k], size=size)
        elif scenarios['circuit_share'].iloc[k] == 1:
            run_single_building(input_path, sim_type='circuit_sharing', tech1=scenarios['primary_cs_load_name'].iloc[k], tech2=scenarios['secondary_cs_load_name'].iloc[k], size=size)
        else:
            continue
        
        # print(k, bldg_id)
    
    
    
    
    
    
    
    
    
    
