import os
import datetime as dt
import numpy as np
import pandas as pd
from multiprocessing import Pool
# import cProfile

from ochre import ElectricVehicle, ElectricResistanceWaterHeater, Battery
from ochre import CreateFigures
from ochre.utils import default_input_path

# Test script to run a fleet of equipment


def setup_ev(i) -> ElectricVehicle:
    # randomly select vehicle type, mileage, and charging level
    vehicle_type = np.random.choice(["BEV", "PHEV"])
    charging_level = np.random.choice(["Level 1", "Level 2"])
    if vehicle_type == "BEV":
        mileage = round(np.random.uniform(100, 300))
    else:
        mileage = round(np.random.uniform(20, 70))

    # Option to specify a file with EV charging events
    # Defaults to older charging event data
    # equipment_event_file = None
    lvl = charging_level.lower().replace(" ", "_")
    equipment_event_file = os.path.join(default_input_path, "EV", f"{vehicle_type}_{lvl}.csv")

    # Initialize equipment
    return ElectricVehicle(
        name=f"EV_{i}",
        seed=i,  # used to randomize charging events. Not used for randomization above
        vehicle_type=vehicle_type,
        charging_level=charging_level,
        mileage=mileage,
        start_time=dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        time_res=dt.timedelta(minutes=15),
        duration=dt.timedelta(days=5),
        verbosity=1,
        save_results=False,  # if True, must specify output_path
        # output_path=os.getcwd(),
        equipment_event_file=equipment_event_file,
    )


def run_ev(ev: ElectricVehicle):
    df = ev.simulate()
    out = df["EV Electric Power (kW)"]
    out.name = ev.name
    return out


def run_ev_fleet(n=4, n_parallel=2):
    """Runs multiple EV simulations and compiles energy consumption"""
    fleet = [setup_ev(i + 1) for i in range(n)]

    with Pool(n_parallel) as p:
        out = p.map(run_ev, fleet)

    # combine load profiles
    df = pd.concat(out, axis=1)
    print(df)
    df.plot()
    CreateFigures.plt.show()

def setup_wh(i):
    start_time = dt.datetime(2018, 1, 1, 0, 0)  # year, month, day, hour, minute
    time_res = dt.timedelta(minutes=1)
    duration = dt.timedelta(days=1)

    # Make water draw profile (constant flow rate with random water draw probability of 1%)
    times = pd.date_range(start_time, start_time + duration, freq=time_res)
    draw_rate = 12  # L/min
    water_draws = np.random.choice([0, draw_rate], p=[0.99, 0.01], size=(len(times)))

    equipment_args = {
        # Equipment parameters
        "name": f"WH_{i}",
        "start_time": start_time,
        "time_res": time_res,
        "duration": duration,
        "verbosity": 3,
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),
        # 'water_nodes': 12,
        "Setpoint Temperature (C)": np.random.randint(50, 52),
        "Capacity (W)": 4500,
        "Tank Volume (L)": np.random.uniform(227, 260),
        "Tank Height (m)": 1.22,
        "UA (W/K)": np.random.uniform(2, 2.5),
        "schedule": pd.DataFrame(
            {
                "Water Heating (L/min)": water_draws,
                "Zone Temperature (C)": np.random.uniform(18, 21),
                "Mains Temperature (C)": np.random.uniform(5, 9),
            },
            index=times,
        ),
    }

    # Initialize equipment
    return ElectricResistanceWaterHeater(**equipment_args)

def run_water_heater_fleet(n=5):
    # Initialize equipment
    fleet = [setup_wh(i + 1) for i in range(n)]

    # Simulate sequentially
    all_data = {}
    for wh in fleet:
        df = wh.simulate()
        all_data[wh.name] = df

    cols_to_plot = [
        "Water Heating Electric Power (kW)", 
        "Hot Water Outlet Temperature (C)",
        "Hot Water Delivered (L/min)",
    ]
    for col in cols_to_plot:
        data = pd.DataFrame({name: df[col] for name, df in all_data.items()})
        data.plot()
    CreateFigures.plt.show()


def setup_battery(i):
    # Initialize equipment
    capacity = np.random.randint(3, 10)
    battery = Battery(
        name=f"Battery_{i}",
        capacity=capacity,  # in kW
        capacity_kwh=capacity * 2,  # 2 hour duration
        start_time=dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        time_res=dt.timedelta(minutes=60),
        duration=dt.timedelta(days=6),
        verbosity=3,
        save_results=False,  # if True, must specify output_path
        # output_path=os.getcwd(),
    )

    # Set battery schedule
    # Note: can also be done at each time step, see run_external_control.py
    # for examples
    schedule = np.random.randint(-capacity, capacity, len(battery.sim_times))
    battery.schedule = pd.DataFrame(
        {"Battery Electric Power (kW)": schedule}, index=battery.sim_times
    )
    battery.reset_time()  # initializes the new schedule
    
    return battery


def run_battery_fleet(n=4):
    """Runs multiple EV simulations and compiles energy consumption"""
    fleet = [setup_battery(i + 1) for i in range(n)]

    # Simulate sequentially
    all_data = {}
    for battery in fleet:
        df = battery.simulate()
        all_data[battery.name] = df

    cols_to_plot = [
        "Battery Electric Power (kW)",
        "Battery SOC (-)",
    ]
    for col in cols_to_plot:
        data = pd.DataFrame({name: df[col] for name, df in all_data.items()})
        data.plot()
    CreateFigures.plt.show()


if __name__ == '__main__':
    run_ev_fleet()
    # run_water_heater_fleet()
    # run_battery_fleet()
