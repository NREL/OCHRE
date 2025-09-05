# Requisites

import os
import datetime as dt
import pandas as pd
import numpy as np

from ochre import Dwelling
from ochre.utils import default_input_path  # for using sample files
from ochre import HeatPumpWaterHeater

#Two week rolling average
site_number = '90023' #90023 #10292#'10441'

flow_data = f'net_flow_{site_number}.csv'
nth_percentile = 50

#2 node simulation
def predict_two_node(temp_n1, temp_n2, setpoint, draw):
    setpoint_default = setpoint
    equipment_args = {
        "start_time": start_date,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(minutes = 60 * 2 + 15),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": 2
    }

    # Create water draw schedule
    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )
    #withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
    withdraw_rate = draw
    withdraw_rate = withdraw_rate[:len(times)]
    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
            "Water Heating Deadband (C)": deadband_default,  # Setting so that it can reset
            "Zone Temperature (C)": 20,
            "Zone Wet Bulb Temperature (C)": 15,  # Required for HPWH
            "Mains Temperature (C)": 7,
        },
        index=times,
    )

    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    hpwh.model.states[:] = np.array([temp_n1, temp_n2])

    # Simulate
    data = pd.DataFrame()
    data = {'draw_data' :[], 'setpoint' :[]}
    control_signal = {}
    setpoints = []

    for t in hpwh.sim_times:
        # Change setpoint based on hour of day
        setpoint = setpoint_default
        control_signal = {
            "Setpoint": setpoint
        }

        setpoints.append(setpoint)
        # Run with controls
        _ = hpwh.update(control_signal=control_signal)

    
    df = hpwh.finalize()

    cols_to_save = [
        "Hot Water Outlet Temperature (C)",
        "T_WH1",
        "T_WH2"
    ]

    to_save = df.loc[:, cols_to_save]
    to_save = to_save[14::15] #Get every 15 minutes
    to_save = to_save[:-1]

    #return to_save
    return to_save["Hot Water Outlet Temperature (C)"]


#Data transformation, Y-values are y_i+1 temperatures for the next hour
def rolling_subsets(input_list, window_size = 4):
    if window_size > len(input_list) or window_size <= 0:
        raise ValueError("Window size must be positive and not greater than the list length.")
    
    return [input_list[i:i + window_size] for i in range(int(len(input_list)))]


def csv_to_features(df, type = 1): #exports features, y = 15 minute increment predictions for 1 hour of outlet temperature
        
    y_col = df["Hot Water Outlet Temperature (C)"]
    window = 10 #2 hours
    result = rolling_subsets(y_col, window)

    power = df["Average Electric Power"]
   

    # Create feature vectors
    x2 = rolling_subsets(df["Draw Data"], window)
    x3 = df["Setpoint"]
    x4 = df["T_WH3"]
    x5 = df["T_WH10"]


    y = result[:-window]  # Remove the first index to offset, y = i+1th prediction

    x2_v = x2[:-window] #draw values
    x3_v = x3[:-window] #setpoint values
    x4_v = x4[:-window] #note temperatures should only be taken at current time interval (live readings)
    x5_v = x5[:-window]

    X = np.column_stack([x2_v, x3_v, x4_v, x5_v]) #Draw value, setpoint, upper node temp, lowe rnode temp
    return X, y

#Bisection logic 

#Input: node temperatures, setpoint, draw

def bisection_control(temp_n1, temp_n2, setpoint_initial, draw): #performs 5 bisection control iterations
    min_temp = min_setpoint
    max_temp = max_setpoint
    setpoint = setpoint_initial
    len_d = len(draw)
    if len(draw) < 135:
        draw = np.append(draw, [0] * 135) 
    for iteration in range(5):
        t_out = predict_two_node(temp_n1, temp_n2, setpoint, draw).values #returns outlet temperature
        if (t_out < 49).any():
            setpoint = setpoint + (max_temp - setpoint)/2
            if setpoint > max_temp:
                setpoint = max_temp
        else:
            setpoint = setpoint - (setpoint - min_temp)/2
            if setpoint < min_temp:
                setpoint = min_temp
    return setpoint


#get ninety-ninth percentile
raw_data = pd.read_csv("bin\90023_raw.csv")
raw_data["readTime"] = pd.to_datetime(raw_data["readTime"])
raw_data = raw_data[raw_data["readTime"] > pd.Timestamp("2013-01-01")]

# Convert flow to liters
raw_data["Flow"] = raw_data["Flow"] * 3.78541

interval = int(len(raw_data)*0.8)
train = raw_data[0:interval] 
train["isWeekend"] = train["readTime"].dt.weekday >= 5 #separate out df by weekends

#compute 99th percentile
train = train[["Flow", "hour", "isWeekend"]]
percentile = train.groupby(['hour', 'isWeekend'])['Flow'].quantile(nth_percentile*0.01).reset_index()

def get_nth_flow(timeStamp, horizon=2, percentiles = percentile):
    #Given a timeStamp, compute predicted flow data for horizon
    hour = timeStamp.hour
    is_weekend = timeStamp.weekday() >= 5
    minute = timeStamp.minute

    predicted_flow = []
    h_offset = 0
    for i in range(horizon * 60):  # loop minute-by-minute for the horizon
        # Update hour offset if minutes overflow
        total_minutes = minute + i
        h_offset = total_minutes // 60
        current_minute = total_minutes % 60

        flow = percentiles.loc[
            (percentiles['hour'] == (hour + h_offset) % 24) &
            (percentiles['isWeekend'] == is_weekend),
            'Flow'
        ]

        if not flow.empty:
            predicted_flow.append(flow.iloc[0] / 60)  # divide hourly flow into per-minute value
        else:
            predicted_flow.append(0)  # or np.nan

    return predicted_flow



train_end = interval
#run 12 node model for 99th percentile
horizon = 2
# Define equipment and simulation parameters
setpoint_default = 51  # in C #alternate b/w 60 and 49
deadband_default = 5.56  # in C

max_setpoint = 60
min_setpoint = 49

water_nodes = 12
run_range = False #runs simulation for a variety of setpoints specified in setpoint_range
simulation_days = 44#200#220 #172 #220 #remove two weeks
time_interval = 15 # adjust setpoint every 15 minutes?
avg_interval = 14 * 1440 #14 days = 



#start_date = dt.datetime(2013, 1, 17, 0, 1) #10441
start_date = dt.datetime(2013, 1, 15, 0, 1) #10292, 90023
#start_date = dt.datetime(2013, 1, 23, 0, 1) #90159
setpoint_range = [setpoint_default]

if run_range == True:
    setpoint_range = np.arange(min_setpoint, max_setpoint)

for s in setpoint_range: #run simulation for every setpoint in valid range
    setpoint_default = s
    setpoint = setpoint_default #initialize setpoint
    equipment_args = {
        "start_time": start_date,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=simulation_days),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": water_nodes
    }

    # Create water draw schedule
    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )
    water_draw_magnitude = 12  # L/min 
    withdraw_rate = np.loadtxt(f'ochre\\defaults\\Input Files\\{flow_data}')
    withdraw_rate = withdraw_rate[train_end:train_end + len(times)] #Gets end of training data
    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
            "Water Heating Deadband (C)": deadband_default,  # Setting so that it can reset
            "Zone Temperature (C)": 20,
            "Zone Wet Bulb Temperature (C)": 15,  # Required for HPWH
            "Mains Temperature (C)": 7,
        },
        index=times,
    )

    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    # Simulate
    data = pd.DataFrame()
    data = {'draw_data' :[], 'setpoint' :[]}
    control_signal = {}
    setpoints = []

    #generate noise for setpoint profile
    for t in hpwh.sim_times:
        # Change setpoint based on hour of day
        #get optimal setpoint

        #Change setpoint every 15 minutes
        current_time = t.minute
        if (current_time % time_interval == 0): #Every 15 minutes
            predict_draws = get_nth_flow(t, horizon = horizon)
            setpoint = bisection_control(hpwh.model.next_states[2], hpwh.model.next_states[9], setpoint, predict_draws) #get node temperatures from 3 and 10
        control_signal = {
            "Setpoint": setpoint
        }

        setpoints.append(setpoint)
        # Run with controls
        _ = hpwh.update(control_signal=control_signal)

    
    df = hpwh.finalize()

    cols_to_plot = [
        "Hot Water Outlet Temperature (C)",
        "Hot Water Average Temperature (C)",
        "Water Heating Deadband Upper Limit (C)",
        "Water Heating Deadband Lower Limit (C)",
        "Water Heating Electric Power (kW)",
        "Hot Water Unmet Demand (kW)",
        "Hot Water Delivered (L/min)",
    ]

    cols_to_save = [
        "Hot Water Outlet Temperature (C)",
        #"T_WH1",
        #"T_WH2"
        "T_WH3",
        #"T_WH7",
        "T_WH10",
        "T_WH12"
        #"T_AMB"
    ]


      # For the DataFrame, select columns and calculate the rolling average for each column
    to_save = df[cols_to_save].rolling(window=15).mean()
    # Calculate the rolling average for 'setpoints' with window size 15
    avg_setpoints = np.convolve(setpoints, np.ones(15)/15, 'same')
    avg_setpoints = avg_setpoints[14::15]
       # Ensure datetime index at 1-minute frequency
    df['Energy (kWh)'] = df['Water Heating Electric Power (kW)']/ 60  # energy per minute
    kwh_energy = df['Energy (kWh)'].resample('15T').sum()  # sum up 15 mins = total kWh per interval


    # For the DataFrame, select columns and calculate the rolling average for each column
    to_save = df[cols_to_save].rolling(window=15).mean()

    #electric_energy_kwh = electric_energy_kwh[14::15]

    to_save = df.loc[:, cols_to_save]
    to_save["Water Heating Mode"] = df["Water Heating Mode"]
    to_save = to_save[14::15]

    to_save["Water Heating Electric Power"] = pd.Series(kwh_energy, index=to_save.index)
   


    to_save = to_save[:-1] 
    to_save.to_csv(f'output_site_{site_number}_bisectioncontrol_{water_nodes}_{nth_percentile}_percentile.csv', header=True, index=False)