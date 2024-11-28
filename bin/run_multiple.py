import os
import sys
import shutil
import time
import datetime as dt
import subprocess
from multiprocessing import Pool

from ochre import Dwelling, Analysis
from ochre.utils import default_input_path

# Script to run multiple simulations. Assumes each simulation has a unique
# folder with all required files.

# Note: see documentation for where to download other weather files
# https://ochre-nrel.readthedocs.io/en/latest/InputsAndArguments.html#weather-file
default_weather_file = os.path.join(default_input_path, "Weather", "G0800310.epw")


def update_output_path(output_path, input_path):
    if output_path is None:
        # Use same path for input and output files
        return input_path
    elif not os.path.isabs(output_path):
        # Use relative path if output path isn't absolute
        return os.path.join(input_path, output_path)
    else:
        return output_path


def run_single_building(
    input_path,
    name="ochre",
    hpxml_file="home.xml",
    schedule_file="in.schedules.csv",
    weather_file_or_path=None,
    output_path=None,
    verbosity=3,
    start_year=2018,
    start_month=1,
    start_day=1,
    time_res=60,
    duration=365,
    initialization_time=1,
):
    # Update input file paths
    if not os.path.isabs(hpxml_file):
        hpxml_file = os.path.join(input_path, hpxml_file)
    if not os.path.isabs(schedule_file):
        schedule_file = os.path.join(input_path, schedule_file)

    output_path = update_output_path(output_path, input_path)

    if weather_file_or_path is None:
        # Assumes correct weather file is in the input_path
        weather_args = {"weather_path": input_path}
    else:
        if not os.path.isabs(weather_file_or_path):
            weather_file_or_path = os.path.join(input_path, weather_file_or_path)
        if os.path.isdir(weather_file_or_path):
            weather_args = {"weather_path": weather_file_or_path}
        elif os.path.isfile(weather_file_or_path):
            weather_args = {"weather_file": weather_file_or_path}
        else:
            raise IOError(f"Cannot parse weather input: {weather_file_or_path}")

    # Initialize
    dwelling = Dwelling(
        name=name,
        start_time=dt.datetime(start_year, start_month, start_day),
        time_res=dt.timedelta(minutes=time_res),
        duration=dt.timedelta(days=duration),
        initialization_time=dt.timedelta(days=initialization_time),
        hpxml_file=hpxml_file,
        schedule_input_file=schedule_file,
        output_path=output_path,
        verbosity=verbosity,
        **weather_args,
    )

    # Run simulation
    dwelling.simulate()


def run_single_with_dict(kwargs):
    run_single_building(**kwargs)


def find_ochre_folders(main_path, overwrite=False, **kwargs):
    # get all input folders
    main_path = os.path.abspath(main_path)
    includes_files = [kwargs.get("hpxml_file", "home.xml"),
                      kwargs.get("schedule_file", "in.schedules.csv")]

    excludes_files = None if overwrite else ["ochre_complete"]

    return Analysis.find_subfolders(main_path, includes_files, excludes_files)


def run_multiple_hpc(
    main_path,
    mem=2,
    n_max=None,
    overwrite=False,
    **kwargs,
):
    # runs multiple OCHRE simulations on HPC using Slurm
    # kwargs are passed to run_single_building
    input_paths = find_ochre_folders(main_path, overwrite, **kwargs)
    my_print(f"Found {len(input_paths)} buildings in:", main_path)

    # limit total number of runs
    if n_max is not None and len(input_paths) > n_max:
        my_print(f"Limiting number of runs to {n_max}")
        input_paths = input_paths[:n_max]

    processes = {}
    for i, input_path in enumerate(input_paths):
        # run srun command
        output_path = update_output_path(kwargs.get("output_path"), input_path)
        log_file = os.path.join(output_path, "ochre.log")
        cpu = mem // 2
        extra_args = [f"--{key}={val}" for key, val in kwargs.items()]
        cmd = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={cpu}",
            f"--mem={mem}G",
            "--exclusive",
            "-Q",
            "-o",
            log_file,
            "ochre",
            "single",
            input_path,
            *extra_args,
        ]

        # print the first few commands
        if i < 5:
            my_print(f"Running subprocess {i+1}:", " ".join(cmd))
        p = subprocess.Popen(cmd)
        processes[p] = True  # True when process is running

    my_print("Submitted all processes.")

    # wait for all processes to finish
    n_processes = len(processes)
    n_success = 0
    n_fail = 0
    n_running = n_processes
    while n_running > 0:
        time.sleep(10)
        for p, running in processes.items():
            if not running:
                # already completed
                continue
            if p.poll() is not None:
                # recently completed - print success for failure message
                processes[p] = False
                n_running -= 1
                code = p.returncode
                if code == 0:
                    n_success += 1
                    my_print(f"Process complete ({n_success}/{n_processes}):", " ".join(p.args))
                else:
                    n_fail += 1
                    my_print(f"Error in process ({n_fail}/{n_processes}):", " ".join(p.args))

    my_print("All processes finished, exiting.")


def run_multiple_local(
    main_path,
    n_parallel=1,
    n_max=None,
    overwrite=False,
    **kwargs,
):
    # runs multiple OCHRE simulations on local machine (can run in parallel or not)
    # kwargs are passed to run_single_building
    input_paths = find_ochre_folders(main_path, overwrite, **kwargs)
    my_print(f"Found {len(input_paths)} buildings in:", main_path)

    # limit total number of runs
    if n_max is not None and len(input_paths) > n_max:
        my_print(f"Limiting number of runs to {n_max}")
        input_paths = input_paths[:n_max]

    # TODO: for now, no log file. Could use subprocess.run to save logs
    # log_file = os.path.join(input_path, "ochre.log")
    if n_parallel == 1:
        # run simulations sequentially
        for input_path in input_paths:
            run_single_building(input_path, **kwargs)
    else:
        # run simulations in parallel
        ochre_data = [{"input_path": input_path, **kwargs} for input_path in input_paths]
        with Pool(n_parallel) as p:
            p.map(run_single_with_dict, ochre_data)

    my_print("All processes finished, exiting.")


def compile_results(main_path, n_max=None):
    # Sample script to compile results from multiple OCHRE runs

    # set up folder for compiled results
    output_path = os.path.join(main_path, "compiled")
    os.makedirs(output_path, exist_ok=True)
    my_print("Compiling OCHRE results for:", main_path)

    # find all building folders
    required_files = ["ochre_complete"]
    run_paths = Analysis.find_subfolders(main_path, required_files)
    n = len(run_paths)

    # ensure at least 1 run folder found
    if not n:
        my_print("No buildings found in:", main_path)
        return
    else:
        my_print(f"Found {n} completed simulations in:", main_path)

    # limit total number of runs
    if n_max is not None and n > n_max:
        my_print(f"Limiting number of runs to {n_max}")
        run_paths = run_paths[:n_max]
        n = n_max

    run_names = {os.path.relpath(path, main_path): path for path in run_paths}

    # combine input json files
    json_files = {name: os.path.join(path, "ochre.json") for name, path in run_names.items()}
    df = Analysis.combine_json_files(json_files)
    df.to_csv(os.path.join(output_path, "all_ochre_inputs.csv"))

    # combine metrics files
    metrics_files = {name: os.path.join(path, "ochre_metrics.csv") for name, path in run_names.items()}
    df = Analysis.combine_metrics_files(metrics_files)
    df.to_csv(os.path.join(output_path, "all_ochre_metrics.csv"))

    # combine single time series column for each house (e.g., total electricity consumption)
    results_files = {name: os.path.join(path, "ochre.csv") for name, path in run_names.items()}
    df = Analysis.combine_time_series_column("Total Electric Power (kW)", results_files)
    df.to_csv(os.path.join(output_path, "all_ochre_total_powers.csv"))

    # aggregate time series data across all simulations
    df = Analysis.combine_time_series_files(results_files, agg_type="House")
    df.to_csv(os.path.join(output_path, "all_ochre_results.csv"))

    my_print("Saved OCHRE results to:", output_path)


def my_print(*args):
    # prints with date and other info
    now = dt.datetime.now()
    print(now, *args)


if __name__ == "__main__":
    main_path = os.getcwd()

    # Download ResStock files to current directory
    buildings = ["bldg0112631"]
    upgrades = ["up00", "up11"]
    input_paths = []
    for upgrade in upgrades:
        for building in buildings:
            input_path = os.path.join(main_path, building, upgrade)
            os.makedirs(input_path, exist_ok=True)
            Analysis.download_resstock_model(building, upgrade, input_path, overwrite=False)
            shutil.copy(default_weather_file, input_path)
            input_paths.append(input_path)

    # Run Dwelling models sequentially
    # for input_path in input_paths:
    #     run_single_building(input_path, duration=7)

    # Run simulations in parallel
    run_multiple_local(main_path, n_parallel=2, overwrite=True, duration=7)

    # Run simulations on HPC using Slurm
    # run_multiple_hpc(main_path, overwrite=True)

    # Compile simulation data
    compile_results(main_path)

