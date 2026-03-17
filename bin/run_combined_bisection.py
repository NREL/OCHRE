# ================================
# Requisites
# ================================

import os
import datetime as dt
import pandas as pd
import numpy as np
import copy

from ochre import HeatPumpWaterHeater

# ================================
# USER TOGGLE
# ================================
USE_ROLLING_FLOW_AVG = False   # False = perfect foresight

# ================================
# GLOBAL PARAMETERS
# ================================
bisection_temp = 49
setpoint_default = 51.5
deadband_default = 5.56

max_setpoint = 60
min_setpoint = 49

water_nodes = 12
simulation_days = 100
time_interval = 15
two_weeks = 20160
tank_volume = 151 # $(40g)
# ================================
# SITES
# ================================
# sites = [
#     22096, 13438, 11531, 23744, 11289, 13265, 23666,
#     90028, 90050, 90135, 10441, 90015, 90030,
#     21578, 22897, 90023, 90130, 99094, 90051,
#     90069, 90131, 90034, 99148, 99162, 99103,
#     99092, 99084
# ]
sites = [90023]
# ================================
# TWO-NODE PREDICTIVE MODEL
# ================================
def predict_two_node(temp_n1, temp_n2, setpoint, draw):
    equipment_args = {
        "start_time": start_date,
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(minutes=135),
        "verbosity": 0,
        "save_results": False,
        "Setpoint Temperature (C)": setpoint,
        "Tank Volume (L)": tank_volume,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": 2
    }

    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left"
    )

    draw = np.pad(draw, (0, max(0, len(times) - len(draw))))
    schedule = pd.DataFrame({
        "Water Heating (L/min)": draw[:len(times)],
        "Water Heating Setpoint (C)": setpoint,
        "Water Heating Deadband (C)": deadband_default,
        "Zone Temperature (C)": 20,
        "Zone Wet Bulb Temperature (C)": 15,
        "Mains Temperature (C)": 7,
    }, index=times)

    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)
    hpwh.model.states[:] = np.array([temp_n1, temp_n2])

    for _ in hpwh.sim_times:
        hpwh.update({"Setpoint": setpoint})

    df = hpwh.finalize()
    return df["Hot Water Outlet Temperature (C)"][14::15].iloc[:-1]

# ================================
# BISECTION CONTROL
# ================================
def bisection_control(temp_n1, temp_n2, setpoint, draw):
    draw = np.pad(draw, (0, max(0, 135 - len(draw))))
    for _ in range(5):
        t_out = predict_two_node(temp_n1, temp_n2, setpoint, draw).values
        if (t_out < bisection_temp).any():
            setpoint = min(max_setpoint, setpoint + (max_setpoint - setpoint) / 2)
        else:
            setpoint = max(min_setpoint, setpoint - (setpoint - min_setpoint) / 2)
    return setpoint

# ================================
# ROLLING FLOW PREDICTION
# ================================
def get_rolling_flow_avg(data, current_date, horizon=2):
    day_minutes = 1440
    two_weeks_df = data[:day_minutes * 14].copy()
    two_weeks_df["hour"] = two_weeks_df["readTime"].dt.hour
    two_weeks_df["is_weekend"] = two_weeks_df["readTime"].dt.weekday >= 5

    profile = two_weeks_df.groupby(["hour", "is_weekend"])["Flow"].mean()

    hour = current_date.hour
    minute = current_date.minute
    is_weekend = current_date.weekday() >= 5

    draws = []
    for i in range(horizon * 60):
        h = (hour + (minute + i) // 60) % 24
        draws.append(profile.get((h, is_weekend), 0) / 60)

    return np.array(draws)

# ================================
# MAIN SIMULATION LOOP
# ================================
for site_number in sites:
    flow_data = f"net_flow_{site_number}.csv"

    raw_data = pd.read_csv(f"bin/raw_data/{site_number}_raw.csv")
    raw_data["readTime"] = pd.to_datetime(raw_data["readTime"])
    raw_data["Flow"] *= 3.78541 #Convert from gallons to L

    withdraw_rate = np.loadtxt(f"ochre/defaults/Input Files/{flow_data}")
    withdraw_rate = withdraw_rate[two_weeks:]

    start_date = dt.datetime(2013, 2, 5)

    times = pd.date_range(start_date, start_date + dt.timedelta(days=simulation_days),
                          freq="1min", inclusive="left")

    schedule = pd.DataFrame({
        "Water Heating (L/min)": withdraw_rate[:len(times)],
        "Water Heating Setpoint (C)": setpoint_default,
        "Water Heating Deadband (C)": deadband_default,
        "Zone Temperature (C)": 20,
        "Zone Wet Bulb Temperature (C)": 15,
        "Mains Temperature (C)": 7,
    }, index=times)

    equipment_args = {
            "start_time": start_date,  # year, month, day, hour, minute
            "time_res": dt.timedelta(minutes=1),
            "duration": dt.timedelta(minutes = 60 * 2 + 15),
            "verbosity": 9,  # required to get setpoint and deadband in results
            "save_results": False,  # if True, must specify output_path
            # "output_path": os.getcwd(),        # Equipment parameters
            "Setpoint Temperature (C)": setpoint_default,
            "Tank Volume (L)": tank_volume,
            "Tank Height (m)": 1.22,
            "UA (W/K)": 2.17,
            "HPWH COP (-)": 4.5,
            "water_nodes": 12
    }

    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    setpoint = setpoint_default
    current_draws = copy.deepcopy(withdraw_rate)
    previous_rate = raw_data.copy()

    setpoints = []

    for t in hpwh.sim_times:
        if t.minute % time_interval == 0:
            if USE_ROLLING_FLOW_AVG:
                draws = get_rolling_flow_avg(previous_rate, t)
            else:
                draws = current_draws[:135]
                current_draws = current_draws[time_interval:]

            setpoint = bisection_control(
                hpwh.model.next_states[2],
                hpwh.model.next_states[9],
                setpoint,
                draws
            )

        setpoints.append(setpoint)
        hpwh.update({"Setpoint": setpoint})

    df = hpwh.finalize()

    suffix = "rollingavg" if USE_ROLLING_FLOW_AVG else "perfect"
    df.to_csv(
        f"output_site_{site_number}_bisection_{suffix}.csv",
        index=False
    )