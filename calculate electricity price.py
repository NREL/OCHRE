import datetime as dt
import pandas as pd
import os

df = pd.read_csv("case_1_annual/OCHRE_hourly.csv")
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

    off_peak_start_1 = dt.time(19, 0, 0)
    off_peak_end_1 = dt.time(23, 59, 59)
    off_peak_start_2 = dt.time(0, 0, 0)
    off_peak_end_2 = dt.time(13, 0, 0)

    mid_peak_start = dt.time(13, 0, 0)
    mid_peak_end = dt.time(15, 0, 0)

    on_peak_start = dt.time(15, 0, 0)
    on_peak_end = dt.time(19, 0, 0)

    # figure out prices based on time of day, season, and rate
    if TOU == True: # prices based on https://co.my.xcelenergy.com/s/billing-payment/residential-rates/time-of-use-pricing
        if date.time() >= off_peak_start_1 and date.time() <= off_peak_end_1:
            if winter == True:
                rate = 0.12
            elif summer == True:
                rate = 0.12
            else:
                raise Exception("invalid date input", date)
        elif date.time() >= off_peak_start_2 and date.time() < off_peak_end_2:
            if winter == True:
                rate = 0.12
            elif summer == True:
                rate = 0.12
            else:
                raise Exception("invalid date input", date)
        elif date.time() >= mid_peak_start and date.time() < mid_peak_end:
            if winter == True:
                rate = 0.16
            elif summer == True:
                coratest = 0.22
            else:
                raise Exception("invalid date input", date)
        elif date.time() >= on_peak_start and date.time() < on_peak_end:
            if winter == True:
                rate = 0.21
            elif summer == True:
                rate = 0.33
            else:
                raise Exception("invalid date input", date)
        else:
            raise Exception("invalid date input", date)
    else:
        if winter == True:
            rate = 0.13
        elif summer == True:
            rate = 0.16
        else:
            raise Exception("invalid input")
        
    cost += [rate*df['Total Electric Energy (kWh)'].iloc[n]]
    unmet_load += [abs(df['Unmet HVAC Load (C)'].iloc[n])]

df["Electricity Cost [$]"] = cost
print("Total Electricity Cost ($): ", sum(cost))
print("HVAC Energy Consumption (kWh)", sum(df['HVAC Heating Main Power (kW)']))
print("Backup Energy Consumption (kWh)", sum(df['HVAC Heating ER Power (kW)']))
print("Combined HVAC and Backup Energy Consumption (kWh)", sum(df['HVAC Heating Main Power (kW)'])+sum(df['HVAC Heating ER Power (kW)']))
print("Unmet Load (hr*C)", sum(unmet_load))
