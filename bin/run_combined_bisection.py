"""Combined runner for EWH/HPWH experiments

Supports three modes:
 - external: constant setpoint runs (from run_external_control_wh.py)
 - percentile: bisection using nth-percentile predicted draws (from run_percentile_bisection.py)
 - two_week: bisection using two-week rolling average (from run_two_week_bisection.py)

Runs sites in parallel (per-site parallelism). Outputs per-site CSVs to --output-dir.
"""
import os
import datetime as dt
import pandas as pd
import numpy as np
import argparse
import glob
import re
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from ochre import HeatPumpWaterHeater

# ---- Defaults ----
DEFAULT_INPUT_DIR = os.path.join("ochre", "defaults", "Input Files")
DEFAULT_RAW_DIR = os.path.join("bin", "raw_data")
DEFAULT_SITES = [90069, 90023, 90034, 90159, 22096, 13438]

# Simulation defaults (kept close to original scripts)
SETPOINT_DEFAULT = 60
DEADBAND_DEFAULT = 5.56
MAX_SETPOINT = 60
MIN_SETPOINT = 49
WATER_NODES = 12
CAPACITY = 151
TWO_WEEKS_OFFSET = 20160
SIMULATION_DAYS = 100
RUN_RANGE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _safe_load_withdraw_rate(path, required_len, offset=0):
    arr = np.loadtxt(path)
    start = int(offset)
    arr = arr[start : start + required_len]
    if len(arr) < required_len:
        pad = np.zeros(required_len - len(arr))
        arr = np.concatenate([arr, pad])
    return arr


def _find_sites_from_input(input_dir, n=None):
    pattern = os.path.join(input_dir, "net_flow_*.csv")
    files = sorted(glob.glob(pattern))
    m = re.compile(r"net_flow_(\d+)\.csv$")
    sites = []
    for f in files[:n] if n else files:
        bn = os.path.basename(f)
        mm = m.search(bn)
        if mm:
            sites.append(int(mm.group(1)))
    return sites


def predict_two_node(start_date, temp_n1, temp_n2, setpoint, draw, tank_volume=CAPACITY):
    equipment_args = {
        "start_time": start_date,
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(minutes=60 * 2 + 15),
        "verbosity": 9,
        "save_results": False,
        "Setpoint Temperature (C)": setpoint,
        "Tank Volume (L)": tank_volume,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": 2,
    }

    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )
    withdraw_rate = np.array(draw)[: len(times)]
    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": setpoint,
            "Water Heating Deadband (C)": DEADBAND_DEFAULT,
            "Zone Temperature (C)": 20,
            "Zone Wet Bulb Temperature (C)": 15,
            "Mains Temperature (C)": 7,
        },
        index=times,
    )

    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)
    hpwh.model.states[:] = np.array([temp_n1, temp_n2])
    for _ in hpwh.sim_times:
        hpwh.update(control_signal={"Setpoint": setpoint})
    df = hpwh.finalize()
    # Return outlet temperature series sampled every 15 minutes
    return df["Hot Water Outlet Temperature (C)"].iloc[14::15][:-1]


def rolling_subsets(input_list, window_size=4):
    if window_size > len(input_list) or window_size <= 0:
        raise ValueError("Window size must be positive and not greater than the list length.")
    return [input_list[i : i + window_size] for i in range(int(len(input_list)))]


def bisection_control(temp_n1, temp_n2, setpoint_initial, draw, bisection_temp=49, iterations=5):
    min_temp = MIN_SETPOINT
    max_temp = MAX_SETPOINT
    setpoint = setpoint_initial
    if len(draw) < 135:
        draw = np.append(draw, [0] * 135)
    for _ in range(iterations):
        t_out = predict_two_node(dt.datetime.utcnow(), temp_n1, temp_n2, setpoint, draw).values
        if (t_out < bisection_temp).any():
            setpoint = setpoint + (max_temp - setpoint) / 2
            setpoint = min(setpoint, max_temp)
        else:
            setpoint = setpoint - (setpoint - min_temp) / 2
            setpoint = max(setpoint, min_temp)
    return setpoint


def _write_output(df, out_path, append=True):
    write_header = True
    if append and os.path.exists(out_path):
        write_header = False
    df.to_csv(out_path, mode=("a" if append else "w"), header=write_header, index=False)


def run_external(site_number, input_dir, output_dir, start_date=None):
    """Run the constant-setpoint external control experiment for one site."""
    logging.info("External run for site %s", site_number)
    if start_date is None:
        start_date = dt.datetime(2012, 2, 11, 0, 1)

    flow_file = os.path.join(input_dir, f"net_flow_{site_number}.csv")
    if not os.path.exists(flow_file):
        raise FileNotFoundError(flow_file)

    times = pd.date_range(start_date, start_date + dt.timedelta(days=SIMULATION_DAYS), freq="1T", inclusive="left")
    withdraw_rate = _safe_load_withdraw_rate(flow_file, required_len=len(times), offset=TWO_WEEKS_OFFSET)

    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": SETPOINT_DEFAULT,
            "Water Heating Deadband (C)": DEADBAND_DEFAULT,
            "Zone Temperature (C)": 20,
            "Zone Wet Bulb Temperature (C)": 15,
            "Mains Temperature (C)": 7,
        },
        index=times,
    )

    equipment_args = {
        "start_time": start_date,
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=SIMULATION_DAYS),
        "verbosity": 9,
        "save_results": False,
        "Setpoint Temperature (C)": SETPOINT_DEFAULT,
        "Tank Volume (L)": CAPACITY,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "save_matrices": False,
        "water_nodes": WATER_NODES,
    }

    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)
    for t in hpwh.sim_times:
        hpwh.update(control_signal={"Setpoint": SETPOINT_DEFAULT})
    df = hpwh.finalize()

    cols_to_save = ["Hot Water Outlet Temperature (C)", "T_WH3", "T_WH10", "T_AMB"]
    to_save = df.loc[:, cols_to_save].copy()
    to_save["Water Heating Mode"] = df["Water Heating Mode"]
    to_save = to_save[14::15]

    kwh = df["Water Heating Electric Power (kW)"] / 60
    kwh_energy = kwh.resample("15T").sum()
    avg_withdraw_rate = np.convolve(withdraw_rate, np.ones(15), "same")[14::15]
    to_save["Water Heating Electric Power"] = pd.Series(kwh_energy.values, index=to_save.index)
    to_save["Draw Data"] = pd.Series(avg_withdraw_rate, index=to_save.index)
    to_save["Setpoint"] = SETPOINT_DEFAULT
    to_save = to_save[:-1]

    out_name = f"output_site_{site_number}_{CAPACITY}(L)_{WATER_NODES}.csv"
    out_path = os.path.join(output_dir, out_name)
    _write_output(to_save, out_path, append=True)
    logging.info("Wrote %s", out_path)
    return out_path


def run_percentile(site_number, input_dir, raw_dir, output_dir, nth_percentile=99):
    logging.info("Percentile bisection run for site %s", site_number)
    # load raw data for this site
    raw_path1 = os.path.join(raw_dir, f"{site_number}_raw.csv")
    raw_path2 = os.path.join("bin", f"{site_number}_raw.csv")
    raw_path = raw_path1 if os.path.exists(raw_path1) else raw_path2
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    raw_data = pd.read_csv(raw_path)
    raw_data["readTime"] = pd.to_datetime(raw_data["readTime"])
    raw_data = raw_data[raw_data["readTime"] > pd.Timestamp("2013-01-01")]
    raw_data["Flow"] = raw_data["Flow"] * 3.78541

    interval = int(len(raw_data) * 0.8)
    train = raw_data[0:interval].copy()
    train["isWeekend"] = train["readTime"].dt.weekday >= 5
    train = train[["Flow", "hour", "isWeekend"]]
    percentiles = train.groupby(["hour", "isWeekend"])['Flow'].quantile(nth_percentile * 0.01).reset_index()

    # Now run the full-horizon simulation similar to the original script
    start_date = dt.datetime(2013, 1, 15, 0, 1)
    times = pd.date_range(start_date, start_date + dt.timedelta(days=SIMULATION_DAYS), freq="1T", inclusive="left")
    flow_file = os.path.join(input_dir, f"net_flow_{site_number}.csv")
    withdraw_rate = _safe_load_withdraw_rate(flow_file, required_len=len(times), offset=interval)

    schedule = pd.DataFrame({
        "Water Heating (L/min)": withdraw_rate,
        "Water Heating Setpoint (C)": SETPOINT_DEFAULT,
        "Water Heating Deadband (C)": DEADBAND_DEFAULT,
        "Zone Temperature (C)": 20,
        "Zone Wet Bulb Temperature (C)": 15,
        "Mains Temperature (C)": 7,
    }, index=times)

    equipment_args = {
        "start_time": start_date,
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=SIMULATION_DAYS),
        "verbosity": 9,
        "save_results": False,
        "Setpoint Temperature (C)": SETPOINT_DEFAULT,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": WATER_NODES,
    }

    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    # closure to get nth flow using percentiles table
    def get_nth_flow(ts, horizon=2):
        hour = ts.hour
        is_weekend = ts.weekday() >= 5
        minute = ts.minute
        predicted_flow = []
        for i in range(horizon * 60):
            total_minutes = minute + i
            h_offset = total_minutes // 60
            current_hour = (hour + h_offset) % 24
            flow = percentiles.loc[(percentiles['hour'] == current_hour) & (percentiles['isWeekend'] == is_weekend), 'Flow']
            if not flow.empty:
                predicted_flow.append(flow.iloc[0] / 60)
            else:
                predicted_flow.append(0)
        return predicted_flow

    setpoint = SETPOINT_DEFAULT
    setpoints = []
    time_interval = 15
    horizon = 2
    for t in hpwh.sim_times:
        if (t.minute % time_interval) == 0:
            predict_draws = get_nth_flow(t, horizon=horizon)
            setpoint = bisection_control(hpwh.model.next_states[2], hpwh.model.next_states[9], setpoint, predict_draws)
        hpwh.update(control_signal={"Setpoint": setpoint})
        setpoints.append(setpoint)

    df = hpwh.finalize()
    cols_to_save = ["Hot Water Outlet Temperature (C)", "T_WH3", "T_WH10", "T_WH12"]
    to_save = df.loc[:, cols_to_save]
    to_save = to_save[14::15].copy()
    to_save["Water Heating Mode"] = df["Water Heating Mode"].values[14::15]
    kwh = df['Water Heating Electric Power (kW)'] / 60
    kwh_energy = kwh.resample('15T').sum()
    to_save["Water Heating Electric Power"] = kwh_energy.values
    avg_setpoints = np.convolve(setpoints, np.ones(15) / 15, 'same')[14::15]
    to_save["Setpoints"] = avg_setpoints
    to_save = to_save[:-1]

    out_name = f"output_site_{site_number}_bisectioncontrol_ud_49_{WATER_NODES}_{nth_percentile}_percentile.csv"
    out_path = os.path.join(output_dir, out_name)
    _write_output(to_save, out_path, append=False)
    logging.info("Wrote %s", out_path)
    return out_path


def run_two_week(site_number, input_dir, raw_dir, output_dir):
    logging.info("Two-week bisection run for site %s", site_number)
    raw_path = os.path.join(raw_dir, f"{site_number}_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(raw_path)
    raw_data = pd.read_csv(raw_path)
    raw_data["readTime"] = pd.to_datetime(raw_data["readTime"])
    raw_data = raw_data[raw_data["readTime"] > pd.Timestamp("2013-01-01")]
    raw_data["Flow"] = raw_data["Flow"] * 3.78541

    interval = int(len(raw_data) * 0.5)

    start_date = dt.datetime(2013, 2, 5, 0, 0)
    times = pd.date_range(start_date, start_date + dt.timedelta(days=SIMULATION_DAYS), freq="1T", inclusive="left")
    flow_file = os.path.join(input_dir, f"net_flow_{site_number}.csv")
    withdraw_rate = _safe_load_withdraw_rate(flow_file, required_len=len(times), offset=TWO_WEEKS_OFFSET)

    # prepare previous two-week slice for rolling average
    avg_interval = interval
    prev_start = max(0, avg_interval - (14 * 24 * 60))
    previous_rate = raw_data[prev_start: avg_interval + len(times) - (14 * 24 * 60)].copy()

    schedule = pd.DataFrame({
        "Water Heating (L/min)": withdraw_rate,
        "Water Heating Setpoint (C)": SETPOINT_DEFAULT,
        "Water Heating Deadband (C)": DEADBAND_DEFAULT,
        "Zone Temperature (C)": 20,
        "Zone Wet Bulb Temperature (C)": 15,
        "Mains Temperature (C)": 7,
    }, index=times)

    equipment_args = {
        "start_time": start_date,
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=SIMULATION_DAYS),
        "verbosity": 9,
        "save_results": False,
        "Setpoint Temperature (C)": SETPOINT_DEFAULT,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": WATER_NODES,
    }

    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)
    setpoint = SETPOINT_DEFAULT
    setpoints = []
    time_interval = 15

    def get_rolling_flow_avg(data, current_date, horizon=2):
        day_minutes = 60 * 24
        hour = current_date.hour
        minute = current_date.minute
        is_weekend = current_date.weekday() >= 5
        two_weeks_df = data[: day_minutes * 14].copy()
        two_weeks_df["is_weekend"] = two_weeks_df["readTime"].dt.weekday >= 5
        two_weeks_df = two_weeks_df.groupby(["hour", "is_weekend"])["Flow"].mean().reset_index()
        predicted_flow = []
        for i in range(horizon * 60):
            total_minutes = minute + i
            h_offset = total_minutes // 60
            current_hour = (hour + h_offset) % 24
            flow = two_weeks_df.loc[(two_weeks_df['hour'] == current_hour) & (two_weeks_df['is_weekend'] == is_weekend), 'Flow']
            if not flow.empty:
                predicted_flow.append(flow.iloc[0] / 60)
            else:
                predicted_flow.append(0)
        return predicted_flow

    for t in hpwh.sim_times:
        if (t.minute % time_interval) == 0:
            predict_draws = get_rolling_flow_avg(previous_rate, t)
            setpoint = bisection_control(hpwh.model.next_states[2], hpwh.model.next_states[9], setpoint, predict_draws)
            # advance previous_rate window
            previous_rate = previous_rate[(14 * 24 * 60):]
        hpwh.update(control_signal={"Setpoint": setpoint})
        setpoints.append(setpoint)

    df = hpwh.finalize()
    cols_to_save = ["Hot Water Outlet Temperature (C)", "T_WH3", "T_WH10", "T_WH12"]
    to_save = df.loc[:, cols_to_save]
    to_save = to_save[14::15].copy()
    kwh = df['Water Heating Electric Power (kW)'] / 60
    kwh_energy = kwh.resample('15T').sum()
    to_save["Water Heating Electric Power"] = kwh_energy.values
    avg_setpoints = np.convolve(setpoints, np.ones(15) / 15, 'same')[14::15]
    to_save["Setpoints"] = avg_setpoints
    to_save = to_save[:-1]

    out_name = f"output_site_{site_number}_bisectioncontrol_ud_49_{WATER_NODES}_rolling_avg.csv"
    out_path = os.path.join(output_dir, out_name)
    _write_output(to_save, out_path, append=False)
    logging.info("Wrote %s", out_path)
    return out_path


def main():
    #Manual arguments

    # parser = argparse.ArgumentParser(description="Run combined bisection/external experiments in parallel per-site")
    # parser.add_argument("--mode", choices=["external", "percentile", "two_week", "all"], default="all")
    # parser.add_argument("--sites", type=str, help="Comma-separated list of site numbers to run")
    # parser.add_argument("--n", type=int, help="Run for first N sites found in input-dir")
    # parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    # parser.add_argument("--raw-dir", type=str, default=DEFAULT_RAW_DIR)
    # parser.add_argument("--output-dir", type=str, default=".")
    # parser.add_argument("--workers", type=int, default=4)
    # args = parser.parse_args()

    args = {}
    args.sites = [90069, 90023, 90034, 90159, 22096, 13438, 11531] #All sites
    args.mode = "external"

    if args.sites:
        sites = [int(s.strip()) for s in args.sites.split(",") if s.strip()]
    elif args.n:
        sites = _find_sites_from_input(args.input_dir, n=args.n)
    else:
        sites = DEFAULT_SITES

    os.makedirs(args.output_dir, exist_ok=True)

    funcs = []
    if args.mode in ("external", "all"):
        funcs.append(partial(run_external, input_dir=args.input_dir, output_dir=args.output_dir))
    if args.mode in ("percentile", "all"):
        funcs.append(partial(run_percentile, input_dir=args.input_dir, raw_dir=args.raw_dir, output_dir=args.output_dir))
    if args.mode in ("two_week", "all"):
        funcs.append(partial(run_two_week, input_dir=args.input_dir, raw_dir=args.raw_dir, output_dir=args.output_dir))

    # Run per-site, parallel across sites; for each site run the selected functions serially
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {}
        for site in sites:
            for fn in funcs:
                fut = exe.submit(fn, site)
                futures[fut] = (site, fn.func.__name__ if hasattr(fn, 'func') else fn.__name__)

        for fut in as_completed(futures):
            site, fname = futures[fut]
            try:
                res = fut.result()
                logging.info("Site %s mode %s completed -> %s", site, fname, res)
                results.append(res)
            except Exception as e:
                logging.exception("Site %s mode %s failed: %s", site, fname, str(e))

    logging.info("All done. %d outputs produced", len(results))


if __name__ == "__main__":
    main()
