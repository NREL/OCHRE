import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("case_3_annual/OCHRE_hourly.csv")
# df = pd.read_csv("test_case/OCHRE_hourly.csv")
df = pd.read_csv("ochre/defaults/OCHRE.csv")

fig, ax1 = plt.subplots(figsize=(8,8))
ax2 = ax1.twinx()

ax1.plot(df["HVAC Heating Main Power (kW)"], label="Main HVAC Power", color='red')
ax1.plot(df["HVAC Heating ER Power (kW)"], label="ER Power", color='darkred')

ax2.plot(df["Temperature - Indoor (C)"], label="Indoor Temperature", color='blue', linestyle='dashed')
ax2.plot(df["HVAC Heating Setpoint (C)"], label="Setpoint Temperature", color='darkblue', linestyle='dashed')

ax1.set_xlabel("Hours")
ax1.set_ylabel("Power [kW]")
ax2.set_ylabel("Temperature [C]")
fig.suptitle("Power of ER and Main HVAC vs Setpoint and Actual Temperatures, Case 2")

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
