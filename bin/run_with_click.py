import functools
import click

import bin.run_multiple as run_multiple


@click.group()
def cli():
    """OCHRE commands"""
    pass


def common_options(f):
    options = [
        click.option("--name", default="ochre", help="Simulation name (for output files)"),
        click.option("--hpxml_file", default="home.xml", help="Name of HPXML file"),
        click.option(
            "--schedule_file", default="in.schedules.csv", help="Name of schedule input file"
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
    return functools.reduce(lambda x, opt: opt(x), options, f)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True), help="Path to input files")
@common_options
def single(**kwargs):
    """Run single OCHRE simulation"""
    run_multiple.run_single_building(**kwargs)


@cli.command()
@click.argument("main_path", type=click.Path(exists=True))
@click.option("--mem", default=2, help="Memory required per run, in GB")
@click.option("--n_max", type=int, help="Limits the total number of simulations to run")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@common_options
def hpc(**kwargs):
    """Run multiple OCHRE simulations using Slurm"""
    run_multiple.run_multiple_hpc(**kwargs)


@cli.command()
@click.argument("main_path", type=click.Path(exists=True))
@click.option("-n", "--n_parallel", default=1, help="Number of simulations to run in parallel")
@click.option("--n_max", type=int, help="Limits the total number of simulations to run")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@common_options
def local(**kwargs):
    """Run multiple OCHRE simulations in parallel or in series"""
    run_multiple.run_multiple_local(**kwargs)


cli.add_command(single)
cli.add_command(hpc)
cli.add_command(local)


if __name__ == "__main__":
    cli()
