import os
import json
import datetime as dt
import shutil
import sys
import click
import helics
from helics.cli import run
import pandas as pd

from ochre import Dwelling, Analysis
from ochre.utils import default_input_path

# Example script to run multiple OCHRE buildings with a DER aggregator in
# co-simulation using HELICS. Uses OCHRE's command line interface (CLI)
# functions.

main_path = os.path.join(os.getcwd(), "cosimulation")
os.makedirs(main_path, exist_ok=True)

# OCHRE buildings to run in co-simulation
# see run_multiple to download files
building_ids = ["bldg0112631"]
upgrades = ["up00", "up11"]
house_paths = {}
i = 1
for building in building_ids:
    for upgrade in upgrades:
        house_paths[f"House_{i}"] = os.path.join(main_path, building, upgrade)
        i += 1
n = len(house_paths)

# Co-simulation timing parameters
start_time = dt.datetime(2018, 1, 1)
time_res = dt.timedelta(minutes=10)
duration = dt.timedelta(days=1)
sim_times = pd.date_range(
    start_time,
    start_time + duration,
    freq=time_res,
    inclusive="left",
)

# other OCHRE parameters
initialization_time = dt.timedelta(days=1)
verbosity = 3
equipment_args = {
    "PV": {"capacity": 5},
    "Battery": {"capacity": 5, "capacity_kwh": 10},
}
status_keys = [
    "Battery Electric Power (kW)",
    "Total Electric Power (kW)",
]

# Note: see documentation for where to download other weather files
# https://ochre-nrel.readthedocs.io/en/latest/InputsAndArguments.html#weather-file
default_weather_file = os.path.join(default_input_path, "Weather", "G0800310.epw")

# control parameters - keep net load within +/- 1 kW per house
min_net_load = -1 * n
max_net_load = 1 * n


def make_helics_federate(name):
    # use PyHELICS API to create federate
    # see: https://docs.helics.org/en/latest/user-guide/examples/fundamental_examples/fundamental_fedintegration.html
    fedinfo = helics.helicsCreateFederateInfo()
    helics.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    helics.helicsFederateInfoSetCoreInitString(fedinfo, "--federates=1")
    helics.helicsFederateInfoSetIntegerProperty(fedinfo, helics.HELICS_PROPERTY_INT_LOG_LEVEL, 1)
    seconds = time_res.total_seconds()
    helics.helicsFederateInfoSetTimeProperty(fedinfo, helics.HELICS_PROPERTY_TIME_PERIOD, seconds)
    helics.helicsFederateInfoSetFlagOption(fedinfo, helics.HELICS_FLAG_UNINTERRUPTIBLE, True)
    helics.helicsFederateInfoSetFlagOption(fedinfo, helics.HELICS_FLAG_TERMINATE_ON_ERROR, True)
    # helics.helicsFederateInfoSetFlagOption(fedinfo, helics.HELICS_FLAG_WAIT_FOR_CURRENT_TIME_UPDATE, True)

    fed = helics.helicsCreateValueFederate(name, fedinfo)

    # enter initialization mode, wait for all federates to enter
    fed.enter_initializing_mode()
    return fed


def register_publication(name, fed):
    return helics.helicsFederateRegisterGlobalTypePublication(fed, name, "string", "")


def register_subscription(name, fed):
    return helics.helicsFederateRegisterSubscription(fed, name, "")


def step_to(time, fed, offset=0):
    t_requested = (time - start_time).total_seconds() + offset
    while True:
        t_new = helics.helicsFederateRequestTime(fed, t_requested)
        if t_new >= t_requested:
            return
        time.sleep(0.01)
        

@click.group()
def cli():
    """OCHRE commands for co-simulation"""
    pass


@cli.command()
def setup():
    # Download ResStock files to co-simulation path
    for building in building_ids:
        for upgrade in upgrades:
            input_path = os.path.join(main_path, building, upgrade)
            os.makedirs(input_path, exist_ok=True)
            Analysis.download_resstock_model(building, upgrade, input_path, overwrite=False)
            shutil.copy(default_weather_file, input_path)


@cli.command()
@click.argument("name", type=str)
@click.argument("input_path", type=click.Path(exists=True))
def house(name, input_path):
    # create helics federate
    fed = make_helics_federate(name)

    # setup publications and subscriptions
    pub = register_publication(f"status-{name}", fed)
    sub = register_subscription("controls", fed)

    # initialize OCHRE
    dwelling = Dwelling(
        name=name,
        start_time=start_time,
        time_res=time_res,
        duration=duration,
        initialization_time=dt.timedelta(days=1),
        hpxml_file=os.path.join(input_path, "home.xml"),
        schedule_input_file=os.path.join(input_path, "in.schedules.csv"),
        weather_path=input_path,
        output_path=input_path,
        verbosity=verbosity,
        Equipment=equipment_args,
    )
    assert (dwelling.sim_times == sim_times).all()
    print(name, "initialized")

    # before simulation, publish default status
    fed.enter_executing_mode()
    dict_to_agg = {key: 0 for key in status_keys}
    str_to_agg = json.dumps(dict_to_agg)
    print(name, "sending to agg:", str_to_agg)
    pub.publish(str_to_agg)

    # begin simulation
    for t in sim_times:
        # request next time step in co-simulation
        # offset lets aggregator run first
        step_to(t, fed, offset=1)

        # get aggregator controls
        str_from_agg = sub.value
        if "-999" in str_from_agg:
            print(t, "house received bad data:", str_from_agg)
            control_signal = {}
        else:
            control_signal = json.loads(str_from_agg)
            print(t, "house received:", control_signal)

        # run 1 time step
        status = dwelling.update(control_signal)

        # publish house status - keep only relevant keys
        dict_to_agg = {key: val for key, val in status.items() if key in status_keys}
        str_to_agg = json.dumps(dict_to_agg)
        print(t, "house sending:", str_to_agg)
        pub.publish(str_to_agg)

    # finalize OCHRE
    dwelling.finalize()


@cli.command()
def aggregator():
    # create helics federate
    fed = make_helics_federate("Aggregator")

    # setup publications and subscriptions
    pub = register_publication("controls", fed)
    subs = {name: register_subscription(f"status-{name}", fed) for name in house_paths.keys()}

    # initialize Aggregator
    print("Aggregator initialized")

    # before simulation, publish default status
    fed.enter_executing_mode()
    dict_to_houses = {"Battery": {"P Setpoint": 0}}
    str_to_houses = json.dumps(dict_to_houses)
    pub.publish(str_to_houses)

    # begin simulation
    results = []
    for t in sim_times:
        # request next time step in co-simulation
        step_to(t, fed)

        # get status from all houses
        house_data = {name: json.loads(sub.value) for name, sub in subs.items()}
        print(t, "agg received:", house_data)
        total_powers = pd.DataFrame(house_data).T.sum()
        results.append(total_powers)

        # determine battery setpoints to maintain net load limits
        nonbattery_power = (
            total_powers["Total Electric Power (kW)"] - total_powers["Battery Electric Power (kW)"]
        )
        if nonbattery_power > max_net_load:
            battery_power = max_net_load - nonbattery_power
        elif nonbattery_power < min_net_load:
            battery_power = min_net_load - nonbattery_power
        else:
            battery_power = 0

        # publish house controls
        dict_to_houses = {"Battery": {"P Setpoint": battery_power / n}}
        str_to_houses = json.dumps(dict_to_houses)
        print(t, "agg sending:", str_to_houses)
        pub.publish(str_to_houses)

    # save total powers
    print("Saving Aggregator results")
    df = pd.DataFrame(results, index=sim_times)
    df.to_csv(os.path.join(main_path, "Aggregator.csv"))


def get_house_fed_config(name, input_path):
    cmd = f"{sys.executable} -u {__file__} house {name} {input_path}"
    cmd = cmd.replace("\\", "/")  # required for Windows?
    return {
        "name": name,
        "host": "localhost",
        "directory": ".",
        "exec": cmd,
    }


@cli.command()
def main():
    # create federate config information
    house_feds = [get_house_fed_config(name, path) for name, path in house_paths.items()]
    cmd = f"{sys.executable} -u {__file__} aggregator"
    cmd = cmd.replace("\\", "/")  # required for Windows?
    agg_fed = {
        "name": "Aggregator",
        "host": "localhost",
        "directory": ".",
        "exec": cmd,
    }
    # create co-sim config information
    config = {"name": "ochre_cosimulation", "broker": True, "federates": house_feds + [agg_fed]}

    # create config file
    config_file = os.path.join(main_path, "config.json")
    with open(config_file, "w+") as f:
        f.write(json.dumps(config, indent=4))

    # run
    print("Running co-simulation, files saved to:", main_path)
    run(["--path", config_file])
    pass


cli.add_command(setup)
cli.add_command(house)
cli.add_command(aggregator)
cli.add_command(main)


if __name__ == "__main__":
    cli()
