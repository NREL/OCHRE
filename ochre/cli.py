import os
import functools
import time
import datetime as dt
import subprocess
from multiprocessing import Pool
import click

from ochre import Dwelling, Analysis

# Functions for command line interface (CLI). Uses a limited number of options
# for running OCHRE


def update_output_path(output_path, input_path):
    if output_path is None:
        # Use same path for input and output files
        return input_path
    elif not os.path.isabs(output_path):
        # Use relative path if output path isn't absolute
        return os.path.join(input_path, output_path)
    else:
        return output_path


def create_dwelling(
    input_path,
    name="ochre",
    hpxml_file="home.xml",
    hpxml_schedule_file="in.schedules.csv",
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
    if not os.path.isabs(hpxml_schedule_file):
        hpxml_schedule_file = os.path.join(input_path, hpxml_schedule_file)

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
        hpxml_schedule_file=hpxml_schedule_file,
        output_path=output_path,
        verbosity=verbosity,
        **weather_args,
    )

    return dwelling


def run_single_process(kwargs):
    input_path = kwargs.pop("input_path")
    log_file = os.path.join(input_path, "ochre.log")
    extra_args = [f"--{key}={val}" for key, val in kwargs.items()]
    cmd = [
        "ochre",
        "single",
        input_path,
        *extra_args,
    ]
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=f)


def find_ochre_folders(
    main_path,
    hpxml_file="home.xml",
    hpxml_schedule_file="in.schedules.csv",
    **kwargs,
) -> list:
    # get all input folders
    main_path = os.path.abspath(main_path)
    includes_files = [hpxml_file, hpxml_schedule_file]
    input_paths = Analysis.find_subfolders(main_path, includes_files)
    my_print(f"Found {len(input_paths)} buildings in:", main_path)

    return input_paths


def limit_input_paths(input_paths, n_max=None, overwrite=False, **kwargs):
    # limits input paths based on n_max and overwrite
    if not overwrite:
        # remove folders that already have ochre_complete
        input_paths = [
            p for p in input_paths if not os.path.exists(os.path.join(p, "ochre_complete"))
        ]

    # limit total number of runs
    if n_max is not None and len(input_paths) > n_max:
        my_print(f"Limiting number of runs to {n_max}")
        input_paths = input_paths[:n_max]

    return input_paths


def run_multiple_hpc(
    input_paths,
    mem=2,
    **kwargs,
):
    # runs multiple OCHRE simulations on HPC using Slurm
    # kwargs are passed to create_dwelling
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
            my_print(f"Running subprocess {i + 1}:", " ".join(cmd))
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
    input_paths,
    n_parallel=1,
    **kwargs,
):
    # runs multiple OCHRE simulations on local machine (can run in parallel or not)
    # kwargs are passed to create_dwelling
    ochre_data = [{"input_path": input_path, **kwargs} for input_path in input_paths]
    with Pool(n_parallel) as p:
        p.map(run_single_process, ochre_data)

    my_print("All processes finished, exiting.")


def my_print(*args):
    # prints with date and other info
    now = dt.datetime.now()
    print(now, *args)


@click.group()
def cli():
    """OCHRE commands"""
    pass


def common_options(f):
    options = [
        click.option("--name", default="ochre", help="Simulation name (for output files)"),
        click.option("--hpxml_file", default="home.xml", help="Name of HPXML file"),
        click.option(
            "--hpxml_schedule_file", default="in.schedules.csv", help="Name of HPXML schedule file"
        ),
        click.option(
            "--weather_file_or_path",
            type=click.Path(exists=True),
            help="Path to single weather file or folder of weather files",
        ),
        click.option("--output_path", help="Path to save output files"),
        click.option("--verbosity", default=3, help="Verbosity of output files"),
        click.option("--start_year", default=2018, help="Simulation start year"),
        click.option("--start_month", default=1, help="Simulation start month"),
        click.option("--start_day", default=1, help="Simulation start day"),
        click.option("--time_res", default=60, help="Time resolution, in minutes"),
        click.option("--duration", default=365, help="Simulation duration, in days"),
        click.option("--initialization_time", default=1, help="Initialization duration, in days"),
    ]
    return functools.reduce(lambda x, opt: opt(x), options[::-1], f)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@common_options
def single(**kwargs):
    """Run single OCHRE simulation"""
    dwelling = create_dwelling(**kwargs)
    dwelling.simulate()


@cli.command()
@click.argument("main_path", type=click.Path(exists=True))
@click.option("--mem", default=2, help="Memory required per run, in GB")
@click.option("--n_max", type=int, help="Limits the total number of simulations to run")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@common_options
def hpc(main_path, **kwargs):
    """Run multiple OCHRE simulations using Slurm"""
    input_paths = find_ochre_folders(main_path, **kwargs)
    input_paths = limit_input_paths(input_paths, **kwargs)
    run_multiple_hpc(**kwargs)


@cli.command()
@click.argument("main_path", type=click.Path(exists=True))
@click.option("-n", "--n_parallel", default=1, help="Number of simulations to run in parallel")
@click.option("--n_max", type=int, help="Limits the total number of simulations to run")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@common_options
def local(main_path, **kwargs):
    """Run multiple OCHRE simulations in parallel or in series"""
    input_paths = find_ochre_folders(main_path, **kwargs)
    input_paths = limit_input_paths(input_paths, **kwargs)
    run_multiple_local(input_paths, **kwargs)


cli.add_command(single)
cli.add_command(hpc)
cli.add_command(local)


if __name__ == "__main__":
    cli()
