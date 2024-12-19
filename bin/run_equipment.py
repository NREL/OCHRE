import os
import datetime as dt
import numpy as np
import pandas as pd

from ochre import (
    Dwelling,
    ElectricVehicle,
    PV,
    Battery,
    ElectricResistanceWaterHeater,
    AirConditioner,
    ScheduledLoad,
    EventDataLoad,
)
from ochre import CreateFigures
from ochre.Models.Envelope import Envelope
from ochre.utils.schedule import import_weather
from ochre.utils import default_input_path
from bin.run_dwelling import dwelling_args


# Example scripts to run single piece of equipment, including:
#  - Equipment from a dwelling model (works for water heaters and PV)
#  - EV
#  - PV (using SAM)
#  - Battery (with daily schedule and self-consumption controls)
#  - Water Heater (with random schedule and schedule from a file)
#  - HVAC (Warning: not recommended to run outside of a dwelling model)

default_args = {
    "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    "time_res": dt.timedelta(minutes=15),
    "duration": dt.timedelta(days=10),
    "verbosity": 3,
    "save_results": False,  # if True, must specify output_path
    # "output_path": os.getcwd(),
}


def run_equipment_from_house_model(end_use):
    # Create Dwelling, see bin/run_dwelling.py
    dwelling = Dwelling(**dwelling_args)

    # Extract equipment by its name or end use
    equipment = dwelling.get_equipment_by_end_use(end_use)
    
    # Update simulation properties to save results
    equipment.main_simulator = True
    equipment.save_results = True
    equipment.set_up_results_files()

    # If necessary, update equipment schedule
    if end_use == "Water Heating":
        equipment.schedule["Zone Temperature (C)"] = 20
        equipment.schedule["Zone Wet Bulb Temperature (C)"] = 18  # only for HPWH
        equipment.reset_time()
        equipment.model.schedule["Zone Temperature (C)"] = 20
        equipment.model.reset_time()

    # Simulate equipment
    df = equipment.simulate()

    print(df.head())
    CreateFigures.plot_time_series_detailed((df[f"{end_use} Electric Power (kW)"],))
    CreateFigures.plt.show()


def run_ev():
    equipment_args = {
        # Equipment parameters
        "vehicle_type": "BEV",
        "charging_level": "Level 1",
        "mileage": 200,
        **default_args,
    }

    # Initialize equipment
    equipment = ElectricVehicle(**equipment_args)

    # Simulate equipment
    df = equipment.simulate()

    print(df.head())
    CreateFigures.plot_daily_profile(df, "EV Electric Power (kW)", plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df["EV SOC (-)"],))
    CreateFigures.plt.show()


def run_pv_with_sam():
    # load weather data
    weather, location = import_weather(dwelling_args["weather_file"], **default_args)

    equipment_args = {
        # Equipment parameters
        "capacity": 5,
        "tilt": 20,
        "azimuth": 0,
        "schedule": weather,
        "location": location,
        **default_args,
    }

    # Initialize equipment
    equipment = PV(**equipment_args)

    # Simulate equipment
    df = equipment.simulate()

    print(df.head())
    CreateFigures.plot_daily_profile(df, "PV Electric Power (kW)", plot_max=False, plot_min=False)
    CreateFigures.plt.show()


def run_battery_from_schedule():
    equipment_args = {
        # Equipment parameters
        "capacity": 5,  # in kW
        "capacity_kwh": 10,
        **default_args,
    }

    # Initialize equipment
    battery = Battery(**equipment_args)

    # Set battery schedule
    # Note: can also be done at each time step, see run_external_control.py
    # for examples
    schedule = np.random.randint(-5, 5, len(battery.sim_times))
    battery.schedule = pd.DataFrame({"Battery Electric Power (kW)": schedule},
                                    index=battery.sim_times)
    battery.reset_time()  # initializes the new schedule

    # Simulate equipment
    df = battery.simulate()

    print(df.head())
    CreateFigures.plot_daily_profile(df, "Battery Electric Power (kW)", plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df["Battery SOC (-)"],))
    CreateFigures.plt.show()


def run_battery_self_consumption():
    equipment_args = {
        # Equipment parameters
        "capacity": 5,  # in kW
        "capacity_kwh": 10,
        "self_consumption_mode": True,
        "soc_init": 0.9,
        **default_args,
    }

    # Initialize equipment
    battery = Battery(**equipment_args)

    # Set net load schedule
    # Note: can also be done at each time step, see run_external_control.py
    # for examples
    house_power = np.random.randint(-2, 3, len(battery.sim_times))
    battery.schedule = pd.DataFrame({"net_power": house_power}, index=battery.sim_times)
    battery.reset_time()  # initializes the new schedule

    # Simulate equipment
    df = battery.simulate()

    # Combine net load and results
    df["House Power (kW)"] = battery.schedule["net_power"]
    df["Total Power (kW)"] = df["House Power (kW)"] + df["Battery Electric Power (kW)"]

    print(df.head())
    CreateFigures.plot_time_series_detailed((df["Total Power (kW)"],))
    CreateFigures.plot_time_series_detailed((df["Battery SOC (-)"],))
    CreateFigures.plt.show()


def run_water_heater():
    # Create water draw schedule
    time_res = dt.timedelta(minutes=1)
    times = pd.date_range(
        default_args["start_time"],
        default_args["start_time"] + default_args["duration"],
        freq=time_res,
        inclusive="left",
    )
    water_draw_magnitude = 12  # L/min
    withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Zone Temperature (C)": 20,
            "Mains Temperature (C)": 7,
        },
        index=times,
    )

    equipment_args = {
        # Equipment parameters
        "Setpoint Temperature (C)": 51,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "schedule": schedule,
        **default_args,
        "time_res": time_res,
    }

    # Initialize equipment
    wh = ElectricResistanceWaterHeater(**equipment_args)

    # Simulate equipment
    df = wh.simulate()

    print(df.head())
    CreateFigures.plot_daily_profile(df, "Water Heating Electric Power (kW)", plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))
    CreateFigures.plt.show()


def run_water_heater_from_file():
    # Load schedule from file
    schedule_file = os.path.join(default_input_path, "Water Heating", "WH Medium UEF Schedule.csv")
    schedule = pd.read_csv(schedule_file, index_col="Time", parse_dates=True)

    equipment_args = {
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=2),
        "verbosity": 3,
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),
        # Equipment parameters
        "Setpoint Temperature (C)": 51,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "schedule": schedule,
    }

    # Initialize equipment
    wh = ElectricResistanceWaterHeater(**equipment_args)

    # Simulate equipment
    df = wh.simulate()

    print(df.head())
    CreateFigures.plot_time_series_detailed((df["Water Heating Electric Power (kW)"],))
    CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))
    CreateFigures.plt.show()


def run_hvac():
    # Note: HVAC and envelope models are difficult to create as standalone
    # models. It's recommended to run a full Dwelling model to run HVAC equipment.
    timing = {
        "start_time": dt.datetime(2018, 7, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=3),
        "verbosity": 6,  # verbosity of results file (1-9)
        "save_results": False,  # if True, must specify output_path
    }

    # create example HVAC schedule
    # TODO: add solar radiation to schedule (in H_LIV)
    times = pd.date_range(timing["start_time"], timing["start_time"] + timing["duration"], freq=timing["time_res"],
                          inclusive="left")
    deadband = (2 + 1 * np.random.randn(len(times))).clip(min=1)
    ambient_temp = 27 - np.abs(times.hour.values - 14) / 2 + 0.5 * np.random.randn(len(times))
    internal_gains = 100 + 30 * np.random.randn(len(times))
    schedule = pd.DataFrame({
        "HVAC Cooling Setpoint (C)": 22,
        "HVAC Cooling Deadband (C)": deadband,
        "Ambient Dry Bulb (C)": ambient_temp,
        "Ambient Humidity Ratio (-)": 0.001,
        # "Ambient Pressure (kPa)": 101,
        # "T_EXT": ambient_temp,
        "Internal Gains (W)": internal_gains,
    }, index=times)

    envelope_args = {
        "rc_params": {
            "R_EXT_LIV": 1e-3,  # in K/W. 1 kW of HVAC for each degree C of temperature difference
            "C_LIV": 4e6,  # in J/K. time constant = RC (seconds) ~= 1 hour
        },
        "initial_temp_setpoint": 22,
        "schedule": schedule,
        "initial_schedule": schedule.iloc[0].to_dict(),
        "enable_humidity": True,
        "external_radiation_method": None,
        "internal_radiation_method": None,
        "zones": {"Indoor": {"Volume (m^3)": 600}},  # Volume required for humidity model
        "ext_zone_labels": ["EXT"],
        "main_sim_name": "",  # For now, required when running envelope model within Equipment
        **timing,
    }
    # Initialize envelope
    envelope = Envelope(**envelope_args)

    equipment_args = {
        # Equipment parameters
        "Number of Speeds (-)": 1,
        "Capacity (W)": [20000],
        "EIR (-)": [0.25],
        "SHR (-)": 0.8,
        "Rated Auxiliary Power (W)": 100,
        "schedule": schedule,
        "initial_schedule": schedule.iloc[0].to_dict(),
        **timing,
    }
    # Initialize equipment
    equipment = AirConditioner(envelope_model=envelope, **equipment_args)

    # Simulate equipment
    df = equipment.simulate()

    print()
    # print(df.head())
    CreateFigures.plot_daily_profile(df, "HVAC Cooling Electric Power (kW)",
                                     plot_max=False, plot_min=False)
    CreateFigures.plot_hvac({"": df})
    # CreateFigures.plot_envelope({"": df})
    CreateFigures.plt.show()


def run_scheduled_load():
    # create schedule
    times = pd.date_range(
        default_args["start_time"],
        default_args["start_time"] + default_args["duration"],
        freq=default_args["time_res"],
        inclusive="left",
    )
    peak_load = 5  # kW
    dryer_power = np.random.choice([0, peak_load], p=[0.98, 0.02], size=len(times))
    schedule = pd.DataFrame({"Clothes Dryer (kW)": dryer_power}, index=times)

    equipment_args = {
        "name": "Clothes Dryer",
        "schedule": schedule,
        **default_args,
    }

    # Initialize equipment
    device = ScheduledLoad(**equipment_args)

    # Simulate equipment
    df = device.simulate()

    print(df.head())
    df.plot()
    CreateFigures.plt.show()


def run_event_based_clothes_dryer():
    # create event schedule
    s = default_args["start_time"]
    d = dt.timedelta(days=1)
    h = dt.timedelta(hours=1)
    assert default_args["duration"] >= d * 3
    event_schedule = pd.DataFrame(
        {
            "start_time": [s + h * 10, s + d + h * 14, s + d * 2 + h * 17],
            "end_time": [s + h * 11, s + d + h * 15, s + d * 2 + h * 18],
            "power": [1, 2, 0.3],  # average power, in kW
        }
    )

    equipment_args = {
        "name": "Clothes Dryer",
        "event_schedule": event_schedule,
        **default_args,
    }

    # Initialize equipment
    device = EventDataLoad(**equipment_args)

    # Simulate equipment
    df = device.simulate()

    print(df.head())
    df.plot()
    CreateFigures.plt.show()


if __name__ == "__main__":
    # Extract equipment from a Dwelling model
    # run_equipment_from_house_model("Water Heating")
    # run_equipment_from_house_model("PV")  # Must add PV in run_dwelling.py

    # Run equipment without a Dwelling model
    # run_ev()
    # run_pv_with_sam()
    # run_battery_from_schedule()
    # run_battery_self_consumption()
    # run_water_heater()
    # run_water_heater_from_file()
    # run_hvac()
    # run_scheduled_load()
    run_event_based_clothes_dryer()
    # run_equipment_from_house_model()
