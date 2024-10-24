import os
import datetime as dt
import pandas as pd

from ochre import Dwelling, Analysis, CreateFigures
from ochre.utils import default_input_path

# Test script to run single Dwelling

pd.set_option('display.precision', 3)      # precision in print statements
pd.set_option('expand_frame_repr', False)  # Keeps results on 1 line
pd.set_option('display.max_rows', 30)      # Shows up to 30 rows of data
# pd.set_option('max_columns', None)       # Prints all columns

dwelling_args = {
    # 'name': 'OCHRE_Test_House'  # simulation name

    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=10),         # time resolution of the simulation
    'duration': dt.timedelta(days=3),             # duration of the simulation
    'initialization_time': dt.timedelta(days=1),  # used to create realistic starting temperature
    'time_zone': None,                            # option to specify daylight savings, in development

    # Input parameters - Sample building (uses HPXML file and time series schedule file)
    'hpxml_file': os.path.join(default_input_path, 'Input Files', 'sample_resstock_properties.xml'),
    'schedule_input_file': os.path.join(default_input_path, 'Input Files', 'sample_resstock_schedule.csv'),

    # Input parameters - weather (note weather_path can be used when Weather Station is specified in HPXML file)
    # 'weather_path': weather_path,
    'weather_file': os.path.join(default_input_path, 'Weather', 'USA_CO_Denver.Intl.AP.725650_TMY3.epw'),

    # Output parameters
    'verbosity': 6,                         # verbosity of time series files (0-9)
    # 'metrics_verbosity': 6,               # verbosity of metrics file (0-9), default=6
    # 'save_results': False,                # saves results to files. Defaults to True if verbosity > 0
    # 'output_path': os.getcwd(),           # defaults to hpxml_file path
    # 'save_args_to_json': True,            # includes data from this dictionary in the json file
    # 'output_to_parquet': True,            # saves time series files as parquet files (False saves as csv files)
    # 'save_schedule_columns': [],          # list of time series inputs to save to schedule file
    # 'export_res': dt.timedelta(days=61),  # time resolution for saving files, to reduce memory requirements

    # Envelope parameters
    # 'Envelope': {
    #     'save_results': True,  # Saves detailed envelope inputs and states
    #     'linearize_infiltration': True,
    #     'external_radiation_method': 'linear',
    #     'internal_radiation_method': 'linear',
    #     'reduced_states': 7,
    #     'save_matrices': True,
    #     'zones': {'Indoor': {
    #         'enable_humidity': False,
    #     }},
    # },

    # Equipment parameters
    'Equipment': {
        # HVAC equipment
        # Note: dictionary key can be end use (e.g., HVAC Heating) or specific equipment name (e.g., Gas Furnace)
        # 'HVAC Heating': {
        #     # 'use_ideal_capacity': True,
        #     # 'show_eir_shr': True,
        # },
        # 'Air Conditioner': {
        #     'speed_type': 'Double',
        # },
        # 'Gas Furnace': {
        #     'heating capacity (W)': 6000,
        #     # 'supplemental heating capacity (W)': 6000,
        # },

        # Water heating equipment
        # Note: dictionary key can be end use (Water Heating) or specific equipment name (e.g., Gas Water Heater)
        # 'Water Heating': {
        #     'water_nodes': 1,
        #     'rc_params': {'R_WH1_AMB': 1,
        #                 'C_WH1': 1e6},
        #     'Water Tank': {
        #         'save_results': True,
        #     },
        #     'save_ebm_results': True,
        # },
        # 'Heat Pump Water Heater': {
        #     'HPWH COP': 4.5,
        #     # 'hp_only_mode': True
        # },
        # 'Electric Resistance Water Heater': {
        #     'use_ideal_capacity': True,
        # },

        # Other equipment
        # 'EV': {
        #     'vehicle_type': 'PHEV',
        #     'charging_level': 'Level 1',
        #     'mileage': 50,
        # },
        # 'PV': {
        #     'capacity': 5,
        #     'tilt': 20,
        #     'azimuth': 180,
        # },
        # 'Battery': {
        #     'capacity_kwh': 6,
        #     'capacity': 3,
        #     'soc_init': 0.5,
        #     'zone': 'Indoor',
        #     # 'control_type': 'Schedule',
        #     'verbosity': 6,
        # },
    },

    # 'modify_hpxml_dict': {},  # Directly modifies values from HPXML input file
    # 'schedule': {},  # Directly modifies columns from OCHRE schedule file (dict or pandas.DataFrame)
}

if __name__ == '__main__':
    # Initialization
    dwelling = Dwelling(**dwelling_args)

    # Simulation
    df, metrics, hourly = dwelling.simulate()

    # Load results from previous run
    # output_path = dwelling_args.get('output_path', os.path.dirname(dwelling_args['hpxml_file']))
    # df, metrics, hourly = Analysis.load_ochre(output_path, simulation_name)

    # Plot results
    data = {'': df}
    CreateFigures.plot_all_powers(data)
    CreateFigures.plot_power_stack(df)
    # CreateFigures.plot_envelope(data)
    # CreateFigures.plot_hvac(data)
    CreateFigures.plt.show()
