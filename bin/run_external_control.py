import datetime as dt
import numpy as np
import pandas as pd

from ochre import (
    Dwelling,
    HeatPumpWaterHeater,
    ElectricVehicle,
    PV,
    Battery,
    CreateFigures,
)
from bin.run_dwelling import dwelling_args
from ochre.Equipment.EV import EV_EFFICIENCY

# Example scripts to run Dwelling or equipment models with external control
# signals, including:
#  - Run Dwelling/HVAC with modified schedule
#  - Run Dwelling/HVAC with dynamic occupancy control
#  - Run HPWH with CTA-2045 control
#  - Run EV with no TOU peak charging
#  - Run EV with perfectly managed charging
#  - Run PV with Volt-VAR control (using co-optimization)
#  - Run Battery with dynamic (random) controls


def run_hvac_modify_schedule():
    # Initialize
    dwelling = Dwelling(**dwelling_args)

    # Get HVAC heater schedule
    heater = dwelling.get_equipment_by_end_use("HVAC Heating")
    schedule = heater.schedule

    # Reduce heating setpoint by 1C from 5-9PM (setpoint is already in the schedule)
    peak_times = (schedule.index.hour >= 17) & (schedule.index.hour < 21)
    schedule.loc[peak_times, "HVAC Heating Setpoint (C)"] -= 1

    # Adjust the HVAC deadband temperature (not in the schedule yet)
    schedule["HVAC Heating Deadband (C)"] = 1
    schedule.loc[peak_times, "HVAC Heating Deadband (C)"] = 2

    # Reset the schedule to implement the changes
    heater.reset_time()

    # Simulate
    dwelling.simulate()


def run_hvac_dynamic_control():
    # Update verbosity to get setpoint and temperature results
    if dwelling_args.get("verbosity", 0) < 6:
        dwelling_args["verbosity"] = 6

    # Initialize
    dwelling = Dwelling(**dwelling_args)

    # Get HVAC heater setpoints and occupancy schedule
    heater = dwelling.get_equipment_by_end_use("HVAC Heating")
    setpoints = heater.schedule["HVAC Heating Setpoint (C)"]
    ambient_temps = dwelling.envelope.schedule["Ambient Dry Bulb (C)"]

    # Simulate
    control_signal = {}
    for t in dwelling.sim_times:
        # Get setpoint and ambient temperature at current time
        setpoint = setpoints[t]
        ambient_temp = ambient_temps[t]

        # Change setpoint based on ambient temperature
        if ambient_temp < 0:
            control_signal = {"HVAC Heating": {"Setpoint": setpoint - 1}}
        else:
            control_signal = {}

        # Run with controls
        house_status = dwelling.update(control_signal=control_signal)

        # Get setpoint and ambient temperature from house status (optional)
        setpoint = house_status["HVAC Heating Setpoint (C)"]
        ambient_temp = house_status["Temperature - Outdoor (C)"]

    df, _, _ = dwelling.finalize()

    df.loc[:, ["HVAC Heating Setpoint (C)", "Temperature - Outdoor (C)"]].plot()
    CreateFigures.plt.show()


def run_hpwh_cta_2045():
    # Define equipment and simulation parameters
    setpoint_default = 51  # in C
    deadband_default = 5.56  # in C
    equipment_args = {
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=1),
        "verbosity": 6,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
    }

    # Create water draw schedule
    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )
    water_draw_magnitude = 12  # L/min
    withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
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
    control_signal = {}
    for t in hpwh.sim_times:
        # Change setpoint based on hour of day
        if t.hour in [7, 16]:
            # CTA-2045 Basic Load Add command
            control_signal = {"Deadband": deadband_default - 2.78}
        elif t.hour in [8, 17]:
            # CTA-2045 Load Shed command
            control_signal = {"Setpoint": setpoint_default - 5.56, "Deadband": deadband_default - 2.78}
        else:
            control_signal = {}

        # Run with controls
        _ = hpwh.update(control_signal=control_signal)

    df = hpwh.finalize()

    # print(df.head())
    cols_to_plot = [
        "Hot Water Outlet Temperature (C)",
        "Hot Water Average Temperature (C)",
        "Water Heating Deadband Upper Limit (C)",
        "Water Heating Deadband Lower Limit (C)",
        "Water Heating Electric Power (kW)",
        "Hot Water Delivered (L/min)",
    ]
    df.loc[:, cols_to_plot].plot()
    CreateFigures.plt.show()


def run_ev_tou(equipment):
    if equipment.schedule is None:
        equipment.schedule = pd.DataFrame(index=equipment.sim_times)
    equipment.schedule["EV Max Power (kW)"] = equipment.max_power
    hours = equipment.schedule.index.hour
    equipment.schedule.loc[
        (hours >= 17) & (hours < 21), "EV Max Power (kW)"
    ] = 0
    equipment.reset_time()

    return equipment.simulate()


def run_ev_perfect(equipment):
    # slow charge from start to end of parking
    for _ in equipment.sim_times:
        parking_time = equipment.event_end - equipment.event_start
        remaining_kwh = (1 - equipment.soc) * equipment.capacity / EV_EFFICIENCY
        if parking_time:
            power = remaining_kwh / parking_time.total_seconds() * 3600
            equipment.update({"Max Power": power})
        else:
            equipment.update()

    return equipment.finalize()


def run_pv_voltvar():
    # Initialization
    dwelling = Dwelling(name="OCHRE with Controller", **dwelling_args)

    # Simulation
    for t in dwelling.sim_times:
        # set up simulation time step (only run once)
        dwelling.update_inputs()

        # simulate 1 time step with different controls
        control_signal = {}
        converged = False
        while not converged:
            # run model and get results without advancing time
            dwelling.update_model(control_signal)
            _ = dwelling.generate_results()

            # check house_status for convergence
            if True:
                converged = True
            else:
                # change control if necessary
                control_signal = {}

        # complete time step and get results
        _ = dwelling.update_results()

    # Finalize simulation
    return dwelling.finalize()


def run_battery_dynamic_control():
    pass


if __name__ == '__main__':
    # Run HVAC with modified schedule
    # run_hvac_modify_schedule()
    
    # Run HVAC with dynamic occupancy control
    # run_hvac_dynamic_control()
    
    # # Run HPWH with CTA-2045 control
    run_hpwh_cta_2045()
    
    # # Run EV with no TOU peak charging
    # run_ev_tou()

    # # Run EV with perfectly managed charging
    # run_ev_perfect()

    # # Run PV with Volt-VAR control (using co-optimization)
    # run_pv_voltvar()

    # # Run Battery with dynamic (random) controls
    # run_battery_dynamic_control()
