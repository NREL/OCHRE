import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('case_0_annual/OCHRE.csv')
print(df)

fig, ax1 = plt.subplots(figsize=(8,8))
ax2 = ax1.twinx()

# ax1.plot(df["HVAC Heating Main Power (kW)"], label="Main HVAC Power", color='red')
ax1.plot(df["HVAC Heating ER Power (kW)"], label="ER Power", color='darkred')

ax2.plot(df["Temperature - Outdoor (C)"], label="Outdoor Temperature", color='blue', linestyle='dashed')
# ax2.plot(df["HVAC Heating Setpoint (C)"], label="Setpoint Temperature", color='darkblue', linestyle='dashed')

ax1.set_xlabel("Minutes")
ax1.set_ylabel("Power [kW]")
ax2.set_ylabel("Temperature [C]")
fig.suptitle("Power of ER vs Outdoor Temperatures")

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
