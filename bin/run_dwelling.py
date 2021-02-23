import os
import datetime as dt
import cProfile
import time

from ochre import Dwelling
# from ochre.FileIO import default_input_path, default_output_path

# Test script to run single Dwelling

simulation_name = 'REopt-HPWH'
properties_name = 'test_case'
weather_name = 'CO_FORT-COLLINS-LOVELAND-AP_724769S_18'

ochre_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
docs_path = os.path.join(os.path.expanduser('~'), 'Documents', 'OCHRE')

dwelling_args = {
    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=60),
    'duration': dt.timedelta(days=365),
    'initialization_time': dt.timedelta(days=7),

    # Input parameters
    # 'input_path': default_input_path,
    # 'output_path': default_output_path,
    'properties_file': properties_name + '_rc_model.properties',
    'schedule_file': properties_name + '_schedule.properties',
    'weather_file': weather_name + '.epw',
    # 'water_draw_file': 'DHW_2bed_unit5_1min.csv',

    # Output parameters
    'save_results': True,
    'export_res': dt.timedelta(days=61),
    'verbosity': 9,  # verbosity of results file (1-9)

    # Other parameters
    'assume_equipment': True,
    'uncontrolled_equipment': ['Lighting', 'Exterior Lighting', 'Range', 'Dishwasher', 'Refrigerator',
                               'Clothes Washer', 'Clothes Dryer', 'MELs'],
    # 'ext_time_res': dt.timedelta(minutes=15),
    # 'save_matrices': True,
    # 'show_eir_shr': True,
}

# Note: most equipment assumed from properties file
equipment = {
    'Air Source Heat Pump': {
        # 'use_ideal_capacity': True,
    },
    # 'Air Conditioner': {
    #     'speed_type': 'Double'
    # },
    # 'Gas Furnace': {
    #     'heating capacity (W)': 6000,
    #     # 'supplemental heating capacity (W)': 6000,
    # },
    'Heat Pump Water Heater': {
        'EF': 2.35,
        # 'hp_only_mode': True
    },
    # 'Electric Resistance Water Heater': {
    #     'use_ideal_capacity': True,
    # },
    # 'EV': {
    #     'vehicle_type': 'PHEV',
    #     'charging_level': 'Level 1',
    #     'mileage': 50,
    # },
    # 'PV': {
    #     'capacity': 5,
    #     # 'tilt': 20,
    #     # 'azimuth': 180,
    # },
    # 'Battery': {
    #     'capacity_kwh': 6,
    #     'capacity_kw': 3,
    #     'control_type': 'Self-Consumption'
    # },
}

if __name__ == '__main__':
    # Initialization
    t_start = time.time()
    dwelling = Dwelling(simulation_name, equipment, **dwelling_args)
    # cProfile.run("dwelling = Dwelling('Test House', equipment, **default_args)", sort='cumulative')
    t_1 = time.time()

    # Simulation
    # df, metrics = dwelling.simulate()
    cProfile.run('dwelling.simulate()', sort='cumulative')
    t_2 = time.time()

    print('time to initialize: {}'.format(t_1 - t_start))
    print('time to simulate: {}'.format(t_2 - t_1))
    print('time to for both: {}'.format(t_2 - t_start))
