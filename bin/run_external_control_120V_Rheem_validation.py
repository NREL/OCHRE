import os
import datetime as dt
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from ochre.utils import default_input_path  # for using sample files
from ochre import HeatPumpWaterHeater

#READ ME
# Input files located in ~\ochre\defaults\Input Files\Temperature Values (temp and flow data fr/ 4am for n days)
# Output is created in home directory under output_site_RHEEM_48.9_12.csv
# Output columns: Hot Water Outlet Temperature (C),Hot Water Delivered (W),Water Heating Electric Power (kW),T_AMB,Hot Water Delivered (kW),Draw Data,Setpoint
# Simulation Steps
# Compute wetbulb temperature
# Input mains temp, T_plenum
#GLOBAL PARAMS
MIN_IN_DAY = 1440
GAL_IN_L = 3.78541
UA_VALUES = {40: 2.638889,
             50: 2.375,
             65: 2.955556,
             80: 3.008333}

#Run 120V simulation for n simulation days

# Define equipment and simulation parameters
setpoint_default = 60 #Assumed 125 degree setpoing
deadband_default = 5.56  # in C
max_setpoint = 60
min_setpoint = 49
water_nodes = 12


#Testing Parameters- edit me
#------------------------------------------------------------#
gallons = 50
capacity = gallons * GAL_IN_L #Gallons to L
ambient_temp = 95



#sites = SITE_CAPACITIES[gallons]

temp_data = pd.read_csv(f'ochre\\defaults\\Input Files\\Temperature Values\\{ambient_temp}F_140_shared.csv')
start_date = dt.datetime(2022, 1, 1, 00, 00) #site_1 at 4am

simulation_duration = len(temp_data)

print("Simulating Setpoint: ", setpoint_default)
equipment_args = {
    "start_time": start_date,  # year, month, day, hour, minute
    "time_res": dt.timedelta(minutes=1),
    "duration": dt.timedelta(minutes= simulation_duration),
    "verbosity": 9,  # required to get setpoint and deadband in results
    "save_results": False,  # if True, must specify output_path
    #"output_path": os.getcwd(),        # Equipment parameters
    "Setpoint Temperature (C)": setpoint_default,
    "Tank Volume (L)": capacity,
    "Tank Height (m)": 1.22, #double check if these are accurate
    "UA (W/K)": UA_VALUES[gallons],
    "HPWH COP (-)": 4.2,
    "save_matrices":False,
    "Low Power HPWH":True,
    "water_nodes": water_nodes,
    "HPWH Capacity (W)": 1495 
}

# Create water draw schedule
times = pd.date_range(
    equipment_args["start_time"],
    equipment_args["start_time"] + equipment_args["duration"],
    freq=equipment_args["time_res"],
    inclusive="left",
)

withdraw_rate = temp_data['flow_out_gpm'].values[0: simulation_duration] * GAL_IN_L
ambient =  temp_data['T_Plenum_In'].values[0: simulation_duration]
wet = temp_data['T_wetbulb'].values[0: simulation_duration] # Required for HPWH
mains = temp_data['T_In_water'].values[0: simulation_duration]

#Initialize Schedule
#should capacity be added to our schedule?
schedule = pd.DataFrame(
    {
        "Water Heating (L/min)": withdraw_rate,
        "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
        "Water Heating Deadband (C)": deadband_default,  # Setting so that it can reset
        "Zone Temperature (C)": ambient,
        "Zone Wet Bulb Temperature (C)": wet,  # Required for HPWH
        "Mains Temperature (C)":  mains
    },
    index=times,
)

# Initialize equipment
hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)


hpwh.model.states[:] = np.array([58.6, 58.8, 59.1, 59.05, 59.0, 58.65, 58.3, 56.8, 55.3, 54.4, 53.6, 53.6]) # 50F_24hour_shared
hpwh.model.states[:] = np.array([59.7, 59.7, 59.7, 59.65, 59.6, 59.3, 59.0, 58.3, 57.7, 56.75, 55.8, 55.8]) # 50F_140_shared
hpwh.model.states[:] = np.array([59.4, 59.45, 59.5, 59.35, 59.2, 59.25, 58.3, 56.4, 54.5, 53.0, 50.6, 50.6]) # 68F_140_shared
hpwh.model.states[:] = np.array([59.4, 59.45, 	59.5, 59.36, 	59.2, 58.5,	57.8, 55.2,	52.6, 50.95,	49.3, 49.3]) # 95F_140_shared
# Simulate
data = pd.DataFrame()

data = {'draw_data' :[], 'setpoint' :[]}
control_signal = {}
setpoints = []

#Simulate at each timestep
for t in hpwh.sim_times:
    setpoints.append(setpoint_default)
    _ = hpwh.update(control_signal=control_signal)


df = hpwh.finalize()

cols = df.columns.tolist()

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
    'Hot Water Delivered (W)',
    "Water Heating Electric Power (kW)",
    "Water Heating Mode",
    "T_WH1",
    "T_WH12"
]

# Ensure datetime index at 1-minute frequency

# Convert to kW per minute
hot_water_delivered_kW = df['Hot Water Delivered (W)'] / 1000 

to_save = df.loc[:, cols_to_save]

to_save["Hot Water Delivered (kW)"] =   pd.Series(hot_water_delivered_kW, index=to_save.index)

to_save["Draw Data"] = pd.Series(withdraw_rate, index=to_save.index)
to_save["Setpoint"] = pd.Series(setpoint_default, index=to_save.index)

to_save = to_save[1:]
show_header = True
to_save.to_csv(f'output_site_RHEEM_{ambient_temp}_{setpoint_default}.csv', header=show_header, index=True)

#plt.show()

print("Simulation Completed: ", f'output_site_RHEEM_{ambient_temp}_{setpoint_default}.csv')