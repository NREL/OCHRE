import datetime as dt
import pandas as pd
import os

df = pd.read_csv("case_5_annual/OCHRE_hourly.csv")
df["Time"] = pd.to_datetime(df["Time"])

cost = []
unmet_load = []

df.dropna(inplace=True)

for n in range(len(df)):
    date = df["Time"].iloc[n]
    TOU = True

    winter = False
    summer = False 

    winter_1_start = dt.datetime(2018, 1, 1, 0, 0) #.timestamp()  # year, month, day, hour, minute
    winter_1_end = dt.datetime(2018, 6, 1, 0, 0)  #.timestamp()  # year, month, day, hour, minute

    winter_2_start = dt.datetime(2018, 10, 1, 0, 0)#.timestamp()  # year, month, day, hour, minute
    winter_2_end = dt.datetime(2019, 1, 1, 0, 0)#.timestamp()  # year, month, day, hour, minute

    summer_start = dt.datetime(2018, 6, 1, 0, 0)#.timestamp()  # year, month, day, hour, minute
    summer_end = dt.datetime(2018, 10, 1, 0, 0)#.timestamp()  # year, month, day, hour, minute

    # figure out whether it is summer or winter rates
    if date >= winter_1_start and date < winter_1_end:
        winter = True
    elif date >= summer_start and date < summer_end:
        summer = True
    elif date >= winter_2_start and date < winter_2_end:
        winter = True
    else:
        raise Exception("invalid date input", date)

    weekno = date.weekday()
    if weekno < 5: 
        weekday = True
    else:
        weekday = False

    discount_start_1 = dt.time(1, 0, 0) #summer 
    discount_end_1 = dt.time(6, 0, 0) #summer 
    discount_start_2 = dt.time(1, 0, 0) # non summer 
    discount_end_2 = dt.time(3, 0, 0) # non summer 
    discount_start_3 = dt.time(11, 0, 0) # non summer 
    discount_end_3 = dt.time(16, 0, 0) # non summer 

    on_peak_start_1 = dt.time(18, 0, 0) #summer 
    on_peak_end_1 = dt.time(21, 0, 0) #summer 
    on_peak_start_2 = dt.time(6, 0, 0) # non summer 
    on_peak_end_2 = dt.time(9, 0, 0) # non summer 

    # figure out prices based on time of day, season, and rate
    # if TOU == True: # prices based on https://www.duke-energy.com/-/media/pdfs/for-your-home/rates/dep-nc/leaf-no-502-schedule-r-tou.pdf?rev=8f3c70ef9a5a494ea663b02eb6950abf
    if date.time() >= on_peak_start_1 and date.time() < on_peak_end_1:
        if weekday == True:
            rate = .28821
        else:
            rate = .07105
    elif date.time() >= on_peak_start_2 and date.time() < on_peak_end_2:
        if weekday == True:
            rate = .28821
        else:
            rate = .07105
    elif date.time() >= discount_start_1 and date.time() < discount_end_1:
        rate = .10911
    elif date.time() >= discount_start_2 and date.time() < discount_end_2:
        rate = .10911
    elif date.time() >= discount_start_3 and date.time() < discount_end_3:
        rate = .10911
    else:
        rate = .07105
    # else:
    #     if winter == True:
    #         rate = 0.13
    #     elif summer == True:
    #         rate = 0.16
    #     else:
    #         raise Exception("invalid input")
        
    cost += [rate*df['Total Electric Energy (kWh)'].iloc[n]]
    unmet_load += [abs(df['Unmet HVAC Load (C)'].iloc[n])]

df["Electricity Cost [$]"] = cost
print("Total Electricity Cost ($): ", sum(cost))
print("HVAC Energy Consumption (kWh)", sum(df['HVAC Heating Main Power (kW)']))
print("Backup Energy Consumption (kWh)", sum(df['HVAC Heating ER Power (kW)']))
print("Combined HVAC and Backup Energy Consumption (kWh)", sum(df['HVAC Heating Main Power (kW)'])+sum(df['HVAC Heating ER Power (kW)']))
print("Unmet Load (hr*C)", sum(unmet_load))
