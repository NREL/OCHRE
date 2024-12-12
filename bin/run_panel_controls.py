import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from ochre import Dwelling
from bin.run_dwelling import dwelling_args
from bin.run_multiple import compile_results

# Example code for running an building with smart panel controls.

main_output_path = dwelling_args.pop("output_path", os.getcwd())
dwelling_args.update({
    "name": "ochre",
    "time_res": dt.timedelta(minutes=2),  # time resolution of the simulation
    "duration": dt.timedelta(days=10),  # duration of the simulation
    "verbosity": 6,  # verbosity of time series files (0-9)
    'seed': 1,
})


def circuit_sharing_control(dwelling, tech1, tech2):
    # run simulation with circuit sharing controls
    times = pd.date_range(
        dwelling.start_time,
        dwelling.start_time + dwelling.duration,
        freq=dwelling.time_res,
        inclusive="left",
    )

    control_signal = None
    N = 0  # total number of delayed cycles
    t_delay = []  # for storing timestamps of delay

    for t in times:
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)

        # get primary load power
        if tech1 in ["Strip Heat"]:
            P_prim = house_status["HVAC Heating ER Power (kW)"]
        else:
            P_prim = house_status[tech1 + " Electric Power (kW)"]

        # decide control for secondary load
        P_limit = 0.01 if tech1 == "Water Heating" else 0
        if P_prim > P_limit:
            P_second = house_status[tech2 + " Electric Power (kW)"]
            if tech2 == "Water Heating":
                control_signal = {tech2: {"Load Fraction": 0}}
                if P_second > 0.01:  # rule out standby WH power from number of interruptions
                    N = N + 1
                    t_delay.append(dwelling.current_time)
            else:  # EV
                control_signal = {tech2: {"P Setpoint": 0}}
            if P_second > 0:
                N = N + 1
                t_delay.append(dwelling.current_time)
        else:
            # keep previous control signal (same as forward fill)
            control_signal = None

    dwelling.finalize()

    return pd.DataFrame(t_delay, columns=["Timestamp"])


def circuit_pausing_control(dwelling, tech1, size):
    # run simulation with circuit pausing controls
    times = pd.date_range(
        dwelling.start_time,
        dwelling.start_time + dwelling.duration,
        freq=dwelling.time_res,
        inclusive="left",
    )

    control_signal = None
    N = 0  # total number of delayed cycles
    t_delay = []  # for storing timestamps of delay

    for t in times:
        # print(t, dwelling.current_time)
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)

        # get total power
        P_prim = house_status["Total Electric Power (kW)"]

        # control target load
        if P_prim > 0.8 * size * 240 / 1000:
            P_second = house_status[tech1 + " Electric Power (kW)"]
            if tech1 == "Water Heating":
                control_signal = {tech1: {"Load Fraction": 0}}
                if P_second > 0.01:  # rule out standby WH power from number of interruptions
                    N = N + 1
                    t_delay.append(dwelling.current_time)
            else:  # EV
                control_signal = {tech1: {"P Setpoint": 0}}
            if P_second > 0:
                N = N + 1
                t_delay.append(dwelling.current_time)
        else:
            # keep previous control signal (same as forward fill)
            control_signal = None

    dwelling.finalize()

    return pd.DataFrame(t_delay, columns=["Timestamp"])


def shed_ev(house_status, size, Pmin=1.44, Pmax=7.68):
    if house_status["EV Electric Power (kW)"] == 0:
        return None
    else:
        P_rest = house_status["Total Electric Power (kW)"] - house_status["EV Electric Power (kW)"]
        P_ev = 0.8 * size * 240 / 1000 - P_rest
        # Note: the Pmin and Pmax values are for L2 chargers.
        if P_ev < Pmin:
            P_ev = Pmin
        if P_ev > Pmax:
            P_ev = Pmax
        return P_ev


def ev_charger_adapter(dwelling, size):
    # run simulation with ev charger adapter controls
    times = pd.date_range(
        dwelling.start_time,
        dwelling.start_time + dwelling.duration,
        freq=dwelling.time_res,
        inclusive="left",
    )

    control_signal = None
    clock = dwelling.start_time  # record last time EV was stopped
    N = 0  # total number of delayed cycles
    t_delay = []  # for storing timestamps of delay

    for t in times:
        assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)

        if house_status["Total Electric Power (kW)"] > 0.8 * size * 240 / 1000:
            control_signal = {"EV": {"P Setpoint": shed_ev(house_status, size)}}
            clock = dwelling.current_time
            if shed_ev(house_status, size) is not None:
                N += 1
                t_delay.append(dwelling.current_time)
        elif dwelling.current_time - clock < pd.Timedelta(15, "m"):
            if clock == dwelling.start_time:  # no EV load has been shedded
                control_signal = None
            else:
                control_signal = {"EV": {"P Setpoint": shed_ev(house_status, size)}}
                if shed_ev(house_status, size) is not None:
                    N += 1
                    t_delay.append(dwelling.current_time)
        else:
            # Keep previous control signal (same as forward fill)
            control_signal = None

    dwelling.finalize()

    return pd.DataFrame(t_delay, columns=["Timestamp"])


def run_simulation(
    sim_type,
    size=None,
    tech1="Cooking Range",
    tech2="EV",
):
    # run individual building case
    my_print("Running case:", sim_type)

    # determine output path, default uses same as input path
    output_path = os.path.join(main_output_path, "ochre_output", sim_type)
    os.makedirs(output_path, exist_ok=True)

    # Initialize ResStock dwelling
    dwelling = Dwelling(output_path=output_path, **dwelling_args)

    # run simulation
    if "baseline" in sim_type:
        # run without controls
        dwelling.simulate()

    elif sim_type == "circuit_sharing":
        df = circuit_sharing_control(dwelling, tech1, tech2)
        df.to_csv(os.path.join(output_path, tech2 + "_metrics.csv"), index=False)

    elif sim_type == "circuit_pausing":
        df = circuit_pausing_control(dwelling, tech1, size)
        df.to_csv(os.path.join(output_path, tech1 + "_metrics.csv"), index=False)

    elif sim_type == "ev_control":
        df = ev_charger_adapter(dwelling, size)
        df.to_csv(os.path.join(output_path, "EV_metrics.csv"), index=False)


def my_print(*args):
    # prints with date and other info
    now = dt.datetime.now()
    print(now, *args)


if __name__ == "__main__":
    # baseline case
    run_simulation("baseline")

    # case 1, circuit sharing with cooking range (primary) and WH (secondary)
    run_simulation(
        "circuit_sharing", 
        tech1="Cooking Range",
        tech2="Water Heating",
    )

    # case 2, circuit pausing with WH
    run_simulation(
        "circuit_pausing",
        size=100,
        tech1="Water Heating",
    )

    # add EV to dwelling_args
    dwelling_args["Equipment"]["Electric Vehicle"] = {
        "vehicle_type": "BEV",
        "charging_level": "Level 2",
        "capacity": 57.5,
    }

    # baseline case - with EV
    run_simulation("baseline_ev")

    # case 3, smart EV charging
    run_simulation(
        "ev_control",
        size=100,
    )

    # compile results
    compile_results(os.path.join(main_output_path, "ochre_output"))

    # plot house powers
    powers_file = os.path.join(main_output_path, "ochre_output", "compiled", "all_ochre_total_powers.csv")
    df = pd.read_csv(powers_file, index_col="Time", parse_dates=True)
    df.plot()
    plt.show()
