import os
import datetime as dt
import pandas as pd

from ochre import Dwelling, Analysis

# Script to run multiple simulations. Assumes each simulation has a unique folder with all required inputs

# Download weather files from: https://data.nrel.gov/submissions/156 or https://energyplus.net/weather
weather_path = os.path.join('path', 'to', 'weather_files')


def run_single_building(input_path, size, der_type, charging_level, sim_type='ev_control', tech1='Cooking Range', tech2='Clothes Dryer', simulation_name='ochre', output_path=None):
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
        'schedule_input_file': os.path.join(input_path, 'in.schedules.csv'),
        'weather_path': weather_path,
    
        # Output parameters
        'output_path': os.path.join(input_path, 'ochre_output'),
        'output_to_parquet': False,              # saves time series files as parquet files (False saves as csv files)
        'verbosity': 7,                         # verbosity of time series files (0-9)
        
        'seed': int(input_path[-3:]),
        
        'Equipment': {

        },
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
    if sim_type == 'baseline':
        # run without controls
        dwelling.simulate()
    
    elif sim_type == 'circuit_sharing':
        circuit_sharing_control(sim_type, dwelling, tech1, tech2, output_path)
    
    elif sim_type == 'circuit_pausing':
        circuit_pausing_control(sim_type, dwelling, tech1, size, output_path)
        
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


def circuit_sharing_control(sim_type, dwelling, tech1, tech2, output_path):   
    
    # run simulation with circuit sharing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    N = 0 # total number of delayed cycles
    t_delay = [] # for storing timestamps of delay
    
    if tech2 == 'Clothes Dryer':
        # initialize an empty list for cycle recording
        pipeline = [] # for storing cycles waiting to be rescheduled
        n_delay = 0 # number of cycles waiting in the pipeline
        deltat_delay = [] # for storing delayed timesteps of each delayed cycle
        n_pop = 0 # index for tracking number of cycles poped
    
    for t in times:
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
                N = N+1
                t_delay.append(dwelling.current_time)
            else:
                # keep previous control signal (same as forward fill)
                control_signal = None

    df = pd.DataFrame(t_delay, columns=['Timestamp'])
    if tech2 == 'Clothes Dryer':
        df['Delayed Time (s)'] = deltat_delay
    df.to_csv(os.path.join(output_path, tech2+'_metrics.csv'), index=False)

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
    
    # remove cycle from schedule
    for i in pd.date_range(cycle_start, cycle_end, freq=dwelling.time_res, inclusive='both'):
        schedule.loc[i] = 0
    
    return(schedule)


def add_first_cycle_to_schedule(dwelling, schedule, pipeline, deltat_delay, n_pop):
    
    # first cycle in pipeline
    cycle_start = pipeline[0].index[0]
    cycle_end = pipeline[0].index[-1]
    cycle_length = cycle_end - cycle_start
    
    # add to schedule
    schedule[dwelling.current_time:dwelling.current_time + cycle_length] = pipeline[0]
        
    # update pipeline
    deltat_delay[n_pop] = dwelling.current_time - deltat_delay[n_pop]
    pipeline.pop(0)
    n_pop += 1
    
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

            # add the first cycle in pipeline to schedule
            schedule, n_pop = add_first_cycle_to_schedule(dwelling, schedule, pipeline, deltat_delay, n_pop)
            n_delay -= 1
            dryer.reset_time(start_time=dwelling.current_time)
    
    return(n_delay, N, t_delay, deltat_delay, n_pop)
        

def circuit_pausing_control(sim_type, dwelling, tech1, size, output_path):
          
    # run simulation with circuit pausing controls
    times = pd.date_range(dwelling.start_time, dwelling.start_time + dwelling.duration, freq=dwelling.time_res,
                          inclusive='left')

    control_signal = None
    N = 0 # total number of delayed cycles
    t_delay = [] # for storing timestamps of delay
    
    if tech1 == 'Clothes Dryer':
        # initialize an empty list for cycle recording
        pipeline = [] # for storing cycles waiting to be rescheduled
        n_delay = 0 # number of cycles waiting in the pipeline
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
                N = N+1
                t_delay.append(dwelling.current_time)
            else:
                # keep previous control signal (same as forward fill)
                control_signal = None
         
    df = pd.DataFrame(t_delay, columns=['Timestamp'])
    if tech1 == 'Clothes Dryer':
        df['Delayed Time (s)'] = deltat_delay
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
             
    input_path = os.path.join('path', 'to', 'input_files')
    size = 100 # amps, placeholder value
    der_type=None
    charging_level=None
    
    # case 1, circuit sharing with clothes dryer (primary) and WH (secondary)
    run_single_building(input_path, size, der_type, charging_level, sim_type='circuit_sharing', tech1='Clothes Dryer', tech2='Water Heating')
    
    # case 2, circuit pausing with WH
    run_single_building(input_path, size, der_type, charging_level, sim_type='circuit_pausing', tech1='Water Heating')
  
    # case 3, smart EV charging
    der_type='ev'
    charging_level='Level 2'
    run_single_building(input_path, size, der_type, charging_level, sim_type='ev_control', tech1='EV')
        
    # compile results
    compile_results(input_path)
        