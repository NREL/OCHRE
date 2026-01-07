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
from ochre.utils.schedule import import_weather

# Example scripts to run Dwelling or equipment models with external control
# signals, including:
#  - Run Dwelling/HVAC with modified schedule
#  - Run Dwelling/HVAC with dynamic control
#  - Run Dwelling/HVAC using external HVAC model
#  - Run HPWH with CTA-2045 control
#  - Run EV with no TOU peak charging
#  - Run EV with perfectly managed charging
#  - Run PV with Volt-VAR control (using co-optimization)
#  - Run Battery with dynamic (random) controls


# update dwelling_args for HVAC control examples
dwelling_args.update(
    {
        "time_res": dt.timedelta(minutes=1),  # time resolution of the simulation
        "duration": dt.timedelta(days=1),  # duration of the simulation
        "verbosity": 6,  # verbosity of time series files (0-9)
    }
)


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
    df, _, _ = dwelling.simulate()

    cols_to_plot = [
        "HVAC Heating Setpoint (C)",
        "Temperature - Indoor (C)",
        "Temperature - Outdoor (C)",
        "Unmet HVAC Load (C)",
        "HVAC Heating Electric Power (kW)",
    ]
    df.loc[:, cols_to_plot].plot()
    CreateFigures.plt.show()


def run_hvac_dynamic_control():
    # Update verbosity to get setpoint and temperature results
    if dwelling_args.get("verbosity", 0) < 6:
        dwelling_args["verbosity"] = 6

    # Initialize
    dwelling = Dwelling(**dwelling_args)

    # Get HVAC heater setpoint schedule and ambient temperature
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

    cols_to_plot = [
        "HVAC Heating Setpoint (C)",
        "Temperature - Indoor (C)",
        "Temperature - Outdoor (C)",
        "Unmet HVAC Load (C)",
        "HVAC Heating Electric Power (kW)",
    ]
    df.loc[:, cols_to_plot].plot()
    CreateFigures.plt.show()

def run_exteral_hvac_model():
    #Example of an external HVAC model, intended to be used with some other HVAC model in an external python script.

    # Update verbosity to get FULL results
    if dwelling_args.get("verbosity", 0) < 9:
        dwelling_args["verbosity"] = 9

    # Initialize
    dwelling = Dwelling(**dwelling_args)

    # Get HVAC heater
    heater = dwelling.get_equipment_by_end_use("HVAC Heating")
    
    
    #OPTIONAL: external capacity of HVAC
    #capacity = heater.capacity  # Get original capacity

    cooling = dwelling.get_equipment_by_end_use("HVAC Cooling")


    heater.duct_dse = 1.0 #FIXME: This removes ducts for now
    heater.c_d = 0.0  # Remove cycling losses from example, handled by external model
    cooling.duct_dse = 1.0 #FIXME: This removes ducts for now
    cooling.c_d = 0.0  # Remove cycling losses from example, handled by external model

    ambient_temps = dwelling.envelope.schedule["Ambient Dry Bulb (C)"]
    ambient_w = dwelling.envelope.schedule["Ambient Humidity Ratio (-)"]

    
    ext_capacity = 0.0 #Disable backup element for this example
    heater.er_ext_capacity = ext_capacity #Disable backup ER if you're purely controlling HP
    heater.capacity_min = -heater.capacity_ideal #Allow for reverse cycle defrost up to full capacity

    load = heater.capacity_ideal #The actual capacity to meet the load 100%
    control_signal = {}
    for t in dwelling.sim_times:
        if (t - dwelling.start_time).seconds > 0: #Don't overwrite at t=0
            heater.use_ideal_capacity = True
            cooling.use_ideal_capacity = True
            heater.ext_ignore_thermostat = True  #Set to true to ignore thermostat setpoint and deadband
            # Change capacity based on hour of day
            capacity_fixed = 20 # * t.hour #W
            control_signal = {'HVAC Heating': {'Capacity': capacity_fixed, 'ER Capacity': ext_capacity}, 'HVAC Cooling': {'Capacity': ext_capacity}} #An arbitrary example, run at 20W
            #heater.ext_capacity = load * 0.25   #An arbitrary example, run at 25% of max capacity

        # Run with controls
        house_status = dwelling.update(control_signal=control_signal)
    #house_status = dwelling.update(heater.ext_capacity=capacity_fixed)
    
    # Simulate
    df, _, _ = dwelling.simulate()

    cols_to_plot = [
        "HVAC Heating Setpoint (C)",
        "Temperature - Indoor (C)",
        "Temperature - Outdoor (C)",
        "Unmet HVAC Load (C)",
        "HVAC Heating Electric Power (kW)",
    ]
    df.loc[:, cols_to_plot].plot()
    CreateFigures.plt.show()


def run_hpwh_cta_2045():
    # Define equipment and simulation parameters
    setpoint_default = 51.67  # in C
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
            control_signal = {
                "Setpoint": setpoint_default - 5.56,
                "Deadband": deadband_default - 2.78,
            }
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
        "Hot Water Unmet Demand (kW)",
        "Hot Water Delivered (L/min)",
    ]
    df.loc[:, cols_to_plot].plot()
    CreateFigures.plt.show()


def run_ev_tou():
    equipment_args = {
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=60),
        "duration": dt.timedelta(days=20),
        "verbosity": 3,
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),
        # Equipment parameters
        "vehicle_type": "BEV",
        "charging_level": "Level 1",
        "range": 150,
    }

    # Initialize
    ev = ElectricVehicle(**equipment_args)

    # Set max power to zero during peak period
    ev.schedule = pd.DataFrame(index=ev.sim_times)  # create schedule
    ev.schedule["EV Max Power (kW)"] = ev.max_power
    # Using a long peak period to show unmet loads
    peak_times = (ev.sim_times.hour >= 15) & (ev.sim_times.hour < 24)
    ev.schedule.loc[peak_times, "EV Max Power (kW)"] = 0
    ev.reset_time()

    df = ev.simulate()

    CreateFigures.plot_daily_profile(df, "EV Electric Power (kW)", plot_max=False, plot_min=False)
    df.loc[:, ["EV Electric Power (kW)", "EV Unmet Load (kWh)", "EV SOC (-)"]].plot()
    CreateFigures.plt.show()


def run_ev_perfect():
    equipment_args = {
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=60),
        "duration": dt.timedelta(days=20),
        "verbosity": 3,
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),
        # Equipment parameters
        "vehicle_type": "BEV",
        "charging_level": "Level 1",
        "range": 150,
    }

    # Initialize
    ev = ElectricVehicle(**equipment_args)

    # slow charge from start to end of parking
    for t in ev.sim_times:
        remaining_hours = (ev.event_end - t).total_seconds() / 3600
        remaining_kwh = (1 - ev.soc) * ev.capacity
        if t >= ev.event_start and remaining_hours:
            power = remaining_kwh / remaining_hours / EV_EFFICIENCY
            ev.update({"Max Power": power})
        else:
            ev.update()

    df = ev.finalize()

    CreateFigures.plot_daily_profile(df, "EV Electric Power (kW)", plot_max=False, plot_min=False)
    df.loc[:, ["EV Electric Power (kW)", "EV Unmet Load (kWh)", "EV SOC (-)"]].plot()
    CreateFigures.plt.show()


def run_pv_voltvar():
    # load weather data (based on parameters in run_dwelling)
    weather, location = import_weather(**dwelling_args)

    equipment_args = {
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=15),
        "duration": dt.timedelta(days=10),
        "verbosity": 1,
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),
        # Equipment parameters
        "capacity": 5,
        "tilt": 20,
        "azimuth": 0,
        "schedule": weather,
        "location": location,
    }

    # Initialize
    pv = PV(**equipment_args)

    # Simulate
    voltage_results = []
    for _ in pv.sim_times:
        # set up simulation time step (only run once)
        pv.update_inputs()

        # Run 1 time step with different controls under converged
        q_setpoint = 0
        converged = False
        while not converged:
            # run model and get results without advancing time
            pv.update_model(control_signal={"Q Setpoint": q_setpoint})
            results = pv.generate_results()
            p = results["PV Electric Power (kW)"]
            q = results["PV Reactive Power (kVAR)"]

            # very simple grid model to determine voltage
            voltage = 1 - p / 50 - q / 50

            # check voltage for convergence
            if 0.95 <= voltage <= 1.05:
                converged = True
            else:
                # change reactive power if necessary
                q_setpoint += (voltage - 1) / 10

        # complete time step and get results
        _ = pv.update_results()
        voltage_results.append(voltage)

    # Finalize simulation
    df = pv.finalize()

    df["Voltage (-)"] = voltage_results
    # print(df.head())
    df.plot()
    CreateFigures.plt.show()


def run_battery_dynamic_control():
    equipment_args = {
        "start_time": dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=15),
        "duration": dt.timedelta(days=3),
        "verbosity": 6,
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),
        # Equipment parameters
        "capacity": 5,  # in kW
        "capacity_kwh": 10,
    }

    # Initialize equipment
    battery = Battery(**equipment_args)

    power = 0
    for _ in battery.sim_times:
        # Set the battery power randomly
        power += np.random.randint(-1, 2)
        power = min(max(power, -5), 5)
        battery.update({"P Setpoint": power})

    df = battery.finalize()

    # print(df.head())
    df.loc[:, ["Battery Electric Power (kW)", "Battery Setpoint (kW)", "Battery SOC (-)"]].plot()
    CreateFigures.plt.show()


if __name__ == "__main__":
    # Run HVAC with modified schedule
    # run_hvac_modify_schedule()

    # Run HVAC with dynamic control
    #run_hvac_dynamic_control()

    # Run HVAC using external HVAC model
    run_exteral_hvac_model()

    # # Run HPWH with CTA-2045 control
    # run_hpwh_cta_2045()

    # # Run EV with no TOU peak charging
    #run_ev_tou()

    # # Run EV with perfectly managed charging
    # run_ev_perfect()

    # # Run PV with Volt-VAR control (using co-optimization)
    # run_pv_voltvar()

    # # Run Battery with dynamic (random) controls
    # run_battery_dynamic_control()
