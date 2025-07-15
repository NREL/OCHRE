import os
import datetime as dt
import pandas as pd
import numpy as np

from ochre import Dwelling
from ochre.utils import default_input_path  # for using sample files
from ochre import HeatPumpWaterHeater



# Define equipment and simulation parameters
setpoint_default = 54  # in C #alternate b/w 60 and 49
deadband_default = 5.56  # in C
max_setpoint = 60
min_setpoint = 49
water_nodes = 12

run_range = False #runs simulation for a variety of setpoints specified in setpoint_range
simulation_days = 220 #172 #220

site_number = '90023' #90023 #10292#'10441'

flow_data = f'net_flow_{site_number}.csv'

start_date = dt.datetime(2013, 1, 17, 0, 1) #10441
#start_date = dt.datetime(2013, 1, 1, 0, 1) #10292, 90023
#start_date = dt.datetime(2013, 1, 23, 0, 1) #90159
setpoint_range = [setpoint_default]

if run_range == True:
    setpoint_range = np.arange(min_setpoint, max_setpoint, 0.5)

for s in setpoint_range: #run simulation for every setpoint in valid range
    setpoint_default = s
    print("Simulating Setpoint: ", setpoint_default)
    equipment_args = {
        "start_time": start_date,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=simulation_days),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        #"output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "save_matrices":False,
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
    #withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
    withdraw_rate = np.loadtxt(f'ochre\defaults\\Input Files\\{flow_data}')
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

    #hpwh.model.states[:] = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]) #manually set starting temperature

    # Simulate
    data = pd.DataFrame()

    data = {'draw_data' :[], 'setpoint' :[]}
    control_signal = {}
    setpoints = []

    #generate noise for setpoint profile
    noise = np.random.randint(0, 3, 24, dtype=np.int8)
    for t in hpwh.sim_times:
        # Change setpoint based on hour of day
        setpoint = setpoint_default
        control_signal = {
            "Setpoint": setpoint
        }

        #replace with random perturbations
       
        if t.hour in [7, 16]:
            # CTA-2045 Basic Load Add command
            control_signal = {"Deadband": deadband_default - 2.78}
        elif t.hour in [8, 17]:
            # CTA-2045 Load Shed command
            setpoint = setpoint_default - 5.56
            control_signal = {
                "Setpoint": setpoint_default - 5.56,
                "Deadband": deadband_default - 2.78,
            }
        
        if noise[t.hour] == 0:
            setpoint = setpoint_default
            control_signal = {
                "Setpoint": setpoint
            }
        elif noise[t.hour] == 1:
            setpoint = setpoint_default - 5.56
            if setpoint < min_setpoint:
                setpoint = min_setpoint
            control_signal = {
                "Setpoint": setpoint
            }
        elif noise[t.hour] == 2:
            setpoint = setpoint_default + 5.56
            if setpoint > max_setpoint:
                setpoint = max_setpoint
            control_signal = {
                "Setpoint": setpoint
            }
        else:
            control_signal = {}
        
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
        #"T_WH2",
        "T_WH3",
        #"T_WH7",
        "T_WH10",
        "T_WH12",
        "T_AMB",
        #"H_WH1",
        #"H_WH2"
    ]


    avg_withdraw_rate = np.convolve(withdraw_rate, np.ones(15), 'same')

    # Calculate the rolling average for 'setpoints' with window size 15
    avg_setpoints = np.convolve(setpoints, np.ones(15)/15, 'same')

    avg_electric = np.convolve(df['Water Heating Electric Power (kW)'], np.ones(15), 'same')


    # For the DataFrame, select columns and calculate the rolling average for each column
    to_save = df[cols_to_save].rolling(window=15).mean()


    draw_data = avg_withdraw_rate[14::15]
    avg_setpoints = avg_setpoints[14::15]
    avg_electric = avg_electric[14::15]

    to_save = df.loc[:, cols_to_save]
    to_save["Water Heating Mode"] = df["Water Heating Mode"]
    to_save = to_save[14::15]

    to_save["Water Heating Electric Power"] = pd.Series(avg_electric, index=to_save.index)
    to_save["Draw Data"] = pd.Series(draw_data, index=to_save.index)
    to_save["Setpoint"] = pd.Series(avg_setpoints, index=to_save.index)

    import matplotlib.pyplot as plt 

    #df.loc[:, cols_to_plot].plot()

    to_save = to_save[:-1]

    if run_range == True:
         to_save.to_csv(f'output_site_{site_number}_{water_nodes}.csv', mode='a', header=False, index=False)
    else:
        to_save.to_csv(f'output_site_{site_number}_{setpoint_default}_{water_nodes}.csv', mode='a', header=True, index=False)

#plt.show()

print("Simulation Copmleted: ", f'output_site_{site_number}_{setpoint_default}_{water_nodes}.csv')