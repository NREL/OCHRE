import os
import datetime as dt
import pandas as pd
import numpy as np

from ochre import Dwelling
#from ochre.utils import default_input_path  # for using sample files
from ochre import HeatPumpWaterHeater

def F_to_C(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius

#GLOBAL PARAMS
# Define equipment and simulation parameters
setpoint_default = 48.9#51.5  # in C
deadband_default = 5.56  # in C
max_setpoint = 60
min_setpoint = 49
water_nodes = 12
capacity = 151#(L) #151 (40)/189(50)/227(60)/303 (80)
two_weeks = 20160

run_range = False#runs simulation for a variety of setpoints specified in setpoint_range
simulation_days = 360 #172 #220


sites = [1]
for site_number in sites: 
    flow_data = f'net_flow_{site_number}_120V.csv'

    start_date = dt.datetime(2023, 11, 3, 0, 0) #site_1
    setpoint_range = [setpoint_default]

    if run_range == True:
        setpoint_range = np.arange(min_setpoint, max_setpoint+0.5, 0.5)

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
            "Tank Volume (L)": capacity,
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

        withdraw_rate = np.loadtxt(f'ochre\\defaults\\Input Files\\{flow_data}')

        #withdraw_rate = withdraw_rate[two_weeks:two_weeks + len(times)] #Buffer by 

        #Compute Temperatures from temp data
        temp_data = pd.read_csv('..\\ochre\\defaults\\Input Files\\Temperature Values\\120V_temperatures_1.csv')
        mains_temp_C = temp_data['inlet_cold_water_temp_C'][0]
        zone_temp_C = temp_data['inlet_air_temp_C'][0]
        humidity = temp_data['inlet_RH'][0]
        wet_bulb_temp = 9.229

        schedule = pd.DataFrame(
            {
                "Water Heating (L/min)": withdraw_rate,
                "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
                "Water Heating Deadband (C)": deadband_default,  # Setting so that it can reset
                "Zone Temperature (C)": zone_temp_C,
                "Zone Wet Bulb Temperature (C)": wet_bulb_temp,  # Required for HPWH
                "Mains Temperature (C)": mains_temp_C,
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

        #Simulate at each time
        for t in hpwh.sim_times:
            setpoints.append(setpoint_default)
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
            #"T_WH12",
            "T_AMB",
            #"H_WH1",
            #"H_WH2"
        ]


        avg_withdraw_rate = np.convolve(withdraw_rate, np.ones(15), 'same')

        # Calculate the rolling average for 'setpoints' with window size 15
        avg_setpoints = np.convolve(setpoints, np.ones(15)/15, 'same')

        # Ensure datetime index at 1-minute frequency
        kwh = df['Water Heating Electric Power (kW)']/60  # energy per minute
        kwh_energy = kwh.resample('15T').sum() # sum up 15 mins = total kWh per interval

        # Convert to kWh per minute
        df['HotWaterDelivered_kWh'] = df['Hot Water Delivered (W)'] / 1000 / 60
        df['HotWaterLoss_kWh'] = df['Hot Water Heat Loss (W)'] / 1000 / 60
        df["HotWaterInjection_kWh"] = df['Hot Water Heat Injected (W)'] / 1000 / 60
        df["Hot Water Unmet Demand (kWh)"] = df['Hot Water Unmet Demand (kW)'] / 60


        # Resample to 15-min totals
        delivered_15min = df['HotWaterDelivered_kWh'].resample('15T').sum()
        loss_15min = df['HotWaterLoss_kWh'].resample('15T').sum()
        injection_15min = df['HotWaterInjection_kWh'].resample('15T').sum()
        unmet_demand_15min = df['Hot Water Unmet Demand (kWh)'].resample('15T').sum()

        # For the DataFrame, select columns and calculate the rolling average for each column
        to_save = df[cols_to_save].rolling(window=15).mean()


        draw_data = avg_withdraw_rate[14::15]
        avg_setpoints = avg_setpoints[14::15]
        #electric_energy_kwh = electric_energy_kwh[14::15]

  


        to_save = df.loc[:, cols_to_save]

        to_save["Water Heating Mode"] = df["Water Heating Mode"]
        to_save = to_save[14::15]

        to_save["Water Heating Electric Power"] = pd.Series(kwh_energy, index=to_save.index)
        to_save["Hot Water Delivered (kWh)"] = pd.Series(delivered_15min, index=to_save.index)
        #to_save["Hot Water Heat Loss (kWh)"] = pd.Series(loss_15min, index=to_save.index)
        #to_save["Hot Water Heat Injected (kWh)"] = pd.Series(injection_15min, index=to_save.index)
        #to_save["Hot Water Unmet Demand (kWh)"] = pd.Series(unmet_demand_15min, index=to_save.index)

        to_save["Draw Data"] = pd.Series(draw_data, index=to_save.index)
        to_save["Setpoint"] = pd.Series(avg_setpoints, index=to_save.index)


        import matplotlib.pyplot as plt 

        #df.loc[:, cols_to_plot].plot()

        to_save = to_save[:-1]

        if run_range == True:
            to_save.to_csv(f'output_site_{site_number}_{setpoint_default}_constant_setpoint_{water_nodes}.csv', header=True, index=False)
        else:
            to_save.to_csv(f'output_site_{site_number}_{capacity}(L)_{setpoint_default}.csv', mode='a', header=True, index=False)

    #plt.show()

    print("Simulation Copmleted: ", f'output_site_{site_number}_{setpoint_default}_{water_nodes}.csv')