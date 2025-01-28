import os
import shutil

from ochre import Analysis
from ochre.cli import create_dwelling, limit_input_paths, run_multiple_local, run_multiple_hpc
from ochre.utils import default_input_path

# Examples to download and run multiple Dwellings. Uses OCHRE's command line
# interface (CLI) functions.

# Note: see documentation for where to download other weather files
# https://ochre-nrel.readthedocs.io/en/latest/InputsAndArguments.html#weather-file
default_weather_file = os.path.join(default_input_path, "Weather", "G0800310.epw")


def compile_results(main_path, n_max=None):
    # Sample script to compile results from multiple OCHRE runs

    # set up folder for compiled results
    output_path = os.path.join(main_path, "compiled")
    os.makedirs(output_path, exist_ok=True)
    print("Compiling OCHRE results for:", main_path)

    # find all building folders
    required_files = ["ochre_complete"]
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
    json_files = {name: os.path.join(path, "ochre.json") for name, path in run_names.items()}
    df = Analysis.combine_json_files(json_files)
    df.to_csv(os.path.join(output_path, "all_ochre_inputs.csv"))

    # combine metrics files
    metrics_files = {
        name: os.path.join(path, "ochre_metrics.csv") for name, path in run_names.items()
    }
    df = Analysis.combine_metrics_files(metrics_files)
    df.to_csv(os.path.join(output_path, "all_ochre_metrics.csv"))

    # combine single time series column for each house (e.g., total electricity consumption)
    results_files = {name: os.path.join(path, "ochre.csv") for name, path in run_names.items()}
    df = Analysis.combine_time_series_column("Total Electric Power (kW)", results_files)
    df.to_csv(os.path.join(output_path, "all_ochre_total_powers.csv"))

    # aggregate time series data across all simulations
    df = Analysis.combine_time_series_files(results_files, agg_type="House")
    df.to_csv(os.path.join(output_path, "all_ochre_results.csv"))

    print("Saved OCHRE results to:", output_path)


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

    # Don't overwrite completed runs
    # input_paths = limit_input_paths(input_paths, overwrite=False)

    # Run Dwelling models sequentially
    # for input_path in input_paths:
    #     dwelling = create_dwelling(input_path, duration=7)
    #     dwelling.simulate()

    # Run simulations in parallel
    run_multiple_local(input_paths, n_parallel=1, duration=7)

    # Run simulations on HPC using Slurm
    # run_multiple_hpc(input_paths, duration=7)

    # Compile simulation data
    compile_results(main_path)
