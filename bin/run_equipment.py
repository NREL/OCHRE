import os
import datetime as dt
import numpy as np
import pandas as pd

from ochre import Dwelling, Battery, ElectricResistanceWaterHeater, AirConditioner, ElectricVehicle
from ochre import CreateFigures
from ochre.Models.Envelope import Envelope
from bin.run_dwelling import dwelling_args


# Test scripts to run single piece of equipment, examples include:
#  - Battery (with daily schedule and external control)
#  - Water Heater
#  - EV

default_args = {
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=15),
    'duration': dt.timedelta(days=365),
    'verbosity': 7,  # verbosity of results (1-9)
    'save_results': True,  # if True, must specify output_path
    # 'output_path': os.path.join(os.getcwd(),'EVProfiles','Level2')
}


def run_battery():
    equipment_args = {
        # Equipment parameters
        # See defaults/Battery/default_parameters.csv for more options
        'capacity_kwh': 10,
        'control_type': 'Schedule',
        'charge_start_hour': 10,  # 10AM
        'discharge_start_hour': 17,  # 5PM
        **default_args,
    }

    # Initialize equipment
    equipment = Battery(**equipment_args)

    # Simulate equipment
    df = equipment.simulate()

    print(df.head())
    CreateFigures.plot_daily_profile(df, 'Battery Electric Power (kW)', plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df['Battery SOC (-)'],))
    CreateFigures.plt.show()


def run_battery_controlled():
    equipment_args = {
        # Equipment parameters
        # See defaults/Battery/default_parameters.csv for more options
        'capacity_kwh': 10,
        'control_type': 'Off',
        **default_args,
    }

    # Initialize equipment
    equipment = Battery(**equipment_args)

    # Simulate equipment
    for t in equipment.sim_times:
        assert equipment.current_time == t
        control_signal = {'P Setpoint': np.random.randint(-5, 5)}  # in kW
        equipment.update(control_signal, {})

    df = equipment.finalize()
    print(df.head())
    CreateFigures.plot_daily_profile(df, 'Battery Electric Power (kW)', plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df['Battery SOC (-)'],))
    CreateFigures.plt.show()


def run_water_heater():
    time_res = dt.timedelta(minutes=1)

    # create example water draw schedule
    times = pd.date_range(default_args['start_time'], default_args['start_time'] + default_args['duration'], 
                          freq=time_res)
    water_draw_magnitude = 12  # L/min
    withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
    schedule = pd.DataFrame({
        'Water Heating (L/min)': withdraw_rate,
        'Zone Temperature (C)': 20,
        'Mains Temperature (C)': 7,
    }, index=times)

    equipment_args = {
        # Equipment parameters
        # 'water_nodes': 1,
        'Initial Temperature (C)': 49,
        'Setpoint Temperature (C)': 51,
        'Deadband Temperature (C)': 5,
        'Capacity (W)': 4800,
        'Efficiency (-)': 1,
        'Tank Volume (L)': 250,
        'Tank Height (m)': 1.22,
        'UA (W/K)': 2.17,
        'schedule': schedule,
        **default_args,
        'time_res': time_res,
    }

    # Initialize equipment
    equipment = ElectricResistanceWaterHeater(**equipment_args)

    # Simulate equipment
    df = equipment.simulate()

    # print(df.head())
    CreateFigures.plot_daily_profile(df, 'Water Heating Electric Power (kW)', plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df['Hot Water Outlet Temperature (C)'],))
    CreateFigures.plt.show()


def run_hvac():
    timing = {
        'start_time': dt.datetime(2018, 7, 1, 0, 0),  # year, month, day, hour, minute
        'time_res': dt.timedelta(minutes=1),
        'duration': dt.timedelta(days=3),
        'verbosity': 6,  # verbosity of results file (1-9)
        'save_results': False,  # if True, must specify output_path
    }

    # create example HVAC schedule
    # TODO: add solar radiation to schedule (in H_LIV)
    times = pd.date_range(timing['start_time'], timing['start_time'] + timing['duration'], freq=timing['time_res'],
                          inclusive='left')
    deadband = (2 + 1 * np.random.randn(len(times))).clip(min=1)
    ambient_temp = 27 - np.abs(times.hour.values - 14) / 2 + 0.5 * np.random.randn(len(times))
    internal_gains = 100 + 30 * np.random.randn(len(times))
    schedule = pd.DataFrame({
        'HVAC Cooling Setpoint (C)': 22,
        'HVAC Cooling Deadband (C)': deadband,
        'Ambient Dry Bulb (C)': ambient_temp,
        'Ambient Humidity Ratio (-)': 0.001,
        # 'Ambient Pressure (kPa)': 101,
        # 'T_EXT': ambient_temp,
        'Internal Gains (W)': internal_gains,
    }, index=times)

    envelope_args = {
        'rc_params': {
            'R_EXT_LIV': 1e-3,  # in K/W. 1 kW of HVAC for each degree C of temperature difference
            'C_LIV': 4e6,  # in J/K. time constant = RC (seconds) ~= 1 hour
        },
        'initial_temp_setpoint': 22,
        'schedule': schedule,
        'initial_schedule': schedule.iloc[0].to_dict(),
        'enable_humidity': True,
        'external_radiation_method': None,
        'internal_radiation_method': None,
        'zones': {'Indoor': {'Volume (m^3)': 600}},  # Volume required for humidity model
        'ext_zone_labels': ['EXT'],
        'main_sim_name': '',  # For now, required when running envelope model within Equipment
        **timing,
    }
    # Initialize envelope
    envelope = Envelope(**envelope_args)

    equipment_args = {
        # Equipment parameters
        'Number of Speeds (-)': 1,
        'Capacity (W)': [20000],
        'EIR (-)': [0.25],
        'SHR (-)': 0.8,
        'Rated Auxiliary Power (W)': 100,
        'schedule': schedule,
        'initial_schedule': schedule.iloc[0].to_dict(),
        **timing,
    }
    # Initialize equipment
    equipment = AirConditioner(envelope_model=envelope, **equipment_args)

    # Simulate equipment
    df = equipment.simulate()

    print()
    # print(df.head())
    CreateFigures.plot_daily_profile(df, 'HVAC Cooling Electric Power (kW)',
                                     plot_max=False, plot_min=False)
    CreateFigures.plot_hvac({'': df})
    # CreateFigures.plot_envelope({'': df})
    CreateFigures.plt.show()


def run_ev(seed):
    equipment_args = {
        # Equipment parameters
        "vehicle_type": "BEV",
        "charging_level": "Level 1",
        "equipment_event_file": "pdf_Veh4_Level2.csv",
        "capacity": 57.5,
        "seed": seed,
        "output_path": os.path.join(os.getcwd(), "EVProfiles", "Level2", str(seed)),
        **default_args,
    }

    # Initialize equipment
    equipment = ElectricVehicle(**equipment_args)

    # Simulate equipment
    equipment.main_simulator = True
    df = equipment.simulate()

    print(df.head())
    # CreateFigures.plot_daily_profile(df, 'EV Electric Power (kW)', plot_max=False, plot_min=False)
    # CreateFigures.plot_time_series_detailed((df['EV SOC (-)'],))
    # CreateFigures.plt.show()


def run_equipment_from_house_model():
    # Create Dwelling from input files, see bin/run_dwelling.py
    dwelling = Dwelling(name='OCHRE House', **dwelling_args)

    # Extract equipment by its end use and update simulation properties
    equipment = dwelling.equipment_by_end_use['Water Heating'][0]
    equipment.main_simulator = True
    equipment.save_results = dwelling.save_results
    equipment.export_res = dwelling.export_res
    equipment.results_file = dwelling.results_file

    # If necessary, update equipment schedule
    equipment.model.schedule['Zone Temperature (C)'] = 20
    equipment.reset_time()

    # Simulate equipment
    equipment.simulate()


if __name__ == '__main__':
    # Choose a scenario to run:
    
    # run_battery()
    # run_battery_controlled()
    # run_water_heater()
    # run_hvac()
    run_ev()
    # run_equipment_from_house_model()
