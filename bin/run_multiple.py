import os
import shutil
import sys
import time
import datetime as dt
import subprocess
from multiprocessing import Pool

from ochre import Dwelling, Analysis

# Script to run multiple simulations. Assumes each simulation has a unique folder with all required inputs

# Download weather files from: https://data.nrel.gov/submissions/156 or https://energyplus.net/weather
weather_path = os.path.join('path', 'to', 'weather_files')

dwelling_args = {
    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=1),         # time resolution of the simulation
    'duration': dt.timedelta(days=365),             # duration of the simulation
    'initialization_time': dt.timedelta(days=14),  # used to create realistic starting temperature

    # Input parameters
    'hpxml_file': 'in.xml',
    'schedule_input_file': 'schedule.csv',
    'weather_path': weather_path,

    # Output parameters
    'verbosity': 3,                         # verbosity of time series files (0-9)
    # 'output_to_parquet': True,            # saves time series files as parquet files (False saves as csv files)
}


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
    if n_parallel == 1:
        for ochre_folder in ochre_folders:
            run_single_building(ochre_folder, *args)
    else:
        map_args = [(ochre_folder, *args) for ochre_folder in ochre_folders]
        with Pool(n_parallel) as p:
            p.starmap(run_single_building, map_args)

    my_print('All processes finished, exiting.')


def run_single_building(input_path, simulation_name='ochre', output_path=None):
    # run individual building case
    my_print(f'Running OCHRE for building {simulation_name} ({input_path})')
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.getcwd(), input_path)

    # determine output path, default uses same as input path
    if output_path is None:
        output_path = input_path
    os.makedirs(output_path, exist_ok=True)

    # run simulation
    dwelling = Dwelling(name=simulation_name,
                        input_path=input_path,
                        output_path=output_path,
                        **dwelling_args)
    dwelling.simulate()


def compile_results(main_folder):
    # Sample script to compile results from multiple OCHRE runs
    # assumes each run is in a different folder, and all simulation names are 'ochre'

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
