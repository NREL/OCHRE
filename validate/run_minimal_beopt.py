from multiprocessing import Pool
import itertools

from ochre import Dwelling, Analysis, CreateFigures
from utils import *

# Script to run minimal building tests

# List of tests with minimal equipment (ideal HVAC only)
minimal_test_names = [
    'Fully_minimal',
    'Minimal_Intgain',

    # 'Minimal_walls_unins',
    # 'Minimal_walls_R7',
    # 'Minimal_walls_R13',
    # 'Minimal_walls_R21',
    'Minimal_walls_unins_OSB',
    'Minimal_walls_unins_R5XPS',
    'Minimal_walls_R7_OSB',
    'Minimal_walls_R7_R5XPS',
    'Minimal_walls_R13_OSB',
    'Minimal_walls_R13_R5XPS',
    'Minimal_walls_R15_OSB',
    'Minimal_walls_R15_R5XPS',
    'Minimal_walls_R19_OSB',
    'Minimal_walls_R19_R5XPS',
    'Minimal_walls_R21_OSB',
    'Minimal_walls_R21_R5XPS',
    'Minimal_walls_R23_OSB',
    'Minimal_walls_R23_R5XPS',
    'Minimal_walls_CMU_6in_hollow',
    'Minimal_walls_CMU_6in_R7',
    'Minimal_walls_CMU_6in_R10',
    'Minimal_walls_CMU_6in_R11',
    'Minimal_walls_CMU_6in_R19',
    'Minimal_walls_CMU_12in_hollow',
    'Minimal_walls_CMU_12in_R10',
    # 'Minimal_walls_unins_OSB_no_rad',
    # 'Minimal_walls_R7_OSB_noRadiation',
    # 'Minimal_walls_R23_OSB_no_rad',
    # 'Minimal_walls_unins_no_rad',
    'Minimal_walls_unins_OSB_stucco',
    'Minimal_walls_R21_OSB_stucco',

    'Minimal_ceiling_unins',
    'Minimal_ceiling_R13',
    'Minimal_ceiling_R19',
    'Minimal_ceiling_R30',
    'Minimal_ceiling_R38',
    'Minimal_ceiling_R49',
    # 'Minimal_ceiling_unins_no_rad',
    # 'Minimal_ceiling_R13_no_rad',
    # 'Minimal_ceiling_R19_no_rad',
    # 'Minimal_ceiling_R30_no_rad',
    # 'Minimal_ceiling_R38_no_rad',
    # 'Minimal_ceiling_R49_no_rad',

    'Minimal_slab_unins',
    'Minimal_slab_2ftR5_ext',
    'Minimal_slab_2ftR5_R5G',
    'Minimal_slab_2ftR10_ext',
    'Minimal_slab_2ftR10_R5G',
    'Minimal_slab_4ftR5_R5G',
    'Minimal_slab_4ftR10_R5G',
    'Minimal_slab_wholeR10_R5G',
    # 'Minimal_slab_unins_attached_garage',
    # 'Minimal_slab_2ftR10_attached_garage',

    'Minimal_windows_U0.27_SHGC0.26',
    'Minimal_windows_U0.32_SHGC0.56',
    'Minimal_windows_U0.37_SHGC0.3',
    'Minimal_windows_U0.49_SHGC0.56',
    'Minimal_windows_U0.84_SHGC0.63',
    'Minimal_windows_front_50ft2',
    'Minimal_windows_back_50ft2',
    'Minimal_windows_left_50ft2',
    'Minimal_windows_right_50ft2',

    'Minimal_inf_1ACH',
    'Minimal_inf_3ACH',
    'Minimal_inf_7ACH',
    'Minimal_inf_10ACH',
    'Minimal_inf_25ACH',

    'Minimal_ducts_none',
    'Minimal_ducts_30leak_unins',
    'Minimal_ducts_30leak_R6',
    'Minimal_ducts_15leak_unins',
    'Minimal_ducts_15leak_R6',
    'Minimal_ducts_8cfm25p100_unins',
    'Minimal_ducts_8cfm25p100_R6',
    'Minimal_ducts_4cfm25p100_unins',
    'Minimal_ducts_4cfm25p100_R6',

    'Minimal_basement_unins',
    # 'Minimal_basement_unins_no_inf',
    'Minimal_basement_halfwall_R5',
    'Minimal_basement_halfwall_R10',
    'Minimal_basement_wholewall_R5',
    'Minimal_basement_wholewall_R10',
    'Minimal_basement_wholewall_R20',
    'Minimal_basement_ceiling_R13',
    'Minimal_basement_ceiling_R19',
    'Minimal_basement_ceiling_R30',

    'Minimal_crawlspace_uninsul_unvent',
    'Minimal_crawlspace_uninsul_vent',
    'Minimal_crawlspace_R10_unvent',
    'Minimal_crawlspace_R13_vent',
    'Minimal_crawlspace_R19_vent',
    'Minimal_crawlspace_R30_vent',

    'Minimal_fin_basement_unins',
    'Minimal_fin_basement_R5',
    'Minimal_fin_basement_R13',
    'Minimal_fin_basement_R13_R5XPS',
    'Minimal_fin_basement_R15',

    'Minimal_attic_unins',
    'Minimal_attic_R7',
    'Minimal_attic_R19',
    'Minimal_attic_R30',
    'Minimal_attic_R38',
    'Minimal_attic_R49',
    'Minimal_attic_R60',

    'Minimal_garage_unins',
    'Minimal_garage_R7',
    'Minimal_garage_R13',
    'Minimal_garage_R13_R5XPS',
    'Minimal_garage_R21',
]

# Lists of tests for equipment (FortCollins only, for now)
misc_load = {'properties_name': 'misc electric'}
ideal = {'use_ideal_capacity': True, 'Disable HVAC Part Load Factor': True}
ashp_equipment = {'ASHP Heater': ideal,
                  'ASHP Cooler': ideal,
                  'MELs': misc_load}
mshp_equipment = {'MSHP Heater': ideal,
                  'MSHP Cooler': ideal,
                  'MELs': misc_load}

minimal_equipment = {
    'Electric Baseboard': {'use_ideal_capacity': True, 'Disable HVAC Biquadratics': True},
    'Air Conditioner': {'use_ideal_capacity': True, 'Disable HVAC Biquadratics': True},
}
minimal_locations = [
    # 'ArtificialTestWeather',
    # 'Atlanta',
    'FortCollins',
    # 'Miami',
    # 'Phoenix',
    # 'Portland',
]
equipment_test_names = {
    # Ventilation Fan Tests
    'Minimal_inf_vent_1ACH': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_inf_vent_3ACH': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_inf_vent_7ACH': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_inf_vent_10ACH': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_inf_vent_25ACH': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_vent_exhaust': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_vent_supply': {'Ventilation Fan': {}, **minimal_equipment},
    'Minimal_vent_HRV': {'Ventilation Fan': {}, 'cooling SHR': None, **minimal_equipment},
    'Minimal_vent_ERV': {'Ventilation Fan': {}, 'cooling SHR': None, **minimal_equipment},

    # HVAC Equipment (10 min resolution default)
    'Minimal_CenAC_13SEER': {'Generic Heater': {}, 'Air Conditioner': ideal, 'MELs': misc_load},
    'Minimal_CenAC_15SEER': {'Generic Heater': {}, 'Air Conditioner': ideal, 'MELs': misc_load},
    'Minimal_CenAC_16SEER': {'Generic Heater': {}, 'Air Conditioner': ideal, 'MELs': misc_load},
    'Minimal_CenAC_18SEER': {'Generic Heater': {}, 'Air Conditioner': ideal, 'MELs': misc_load},
    'Minimal_CenAC_24.5SEER': {'Generic Heater': {}, 'Air Conditioner': ideal, 'MELs': misc_load},

    'Minimal_RoomAC_8.5EER_30cond': {'Generic Heater': {}, 'Room AC': {}, 'MELs': misc_load},
    'Minimal_RoomAC_8.5EER': {'Generic Heater': {}, 'Room AC': {}, 'MELs': misc_load},
    'Minimal_RoomAC_10.7EER_20cond': {'Generic Heater': {}, 'Room AC': {}, 'MELs': misc_load},
    'Minimal_RoomAC_10.7EER': {'Generic Heater': {}, 'Room AC': {}, 'MELs': misc_load},

    'Minimal_GasFurnace_60AFUE': {'Gas Furnace': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_GasFurnace_68AFUE': {'Gas Furnace': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_GasFurnace_78AFUE': {'Gas Furnace': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_GasFurnace_95AFUE': {'Gas Furnace': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_GasFurnace_90AFUE': {'Gas Furnace': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_GasFurnace_98AFUE': {'Gas Furnace': {}, 'Generic Cooler': {}, 'MELs': misc_load},

    'Minimal_boiler_elec_100AFUE': {'Electric Boiler': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_boiler_gas_FD_72AFUE': {'Gas Boiler': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_boiler_gas_FD_80AFUE': {'Gas Boiler': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_boiler_gas_FD_85AFUE': {'Gas Boiler': {}, 'Generic Cooler': {}, 'MELs': misc_load},
    'Minimal_boiler_gas_cond_96AFUE': {'Gas Boiler': {}, 'Generic Cooler': {}, 'MELs': misc_load},

    'Minimal_ASHP_13SEER_7.7HSPF': ashp_equipment,
    'Minimal_ASHP_14SEER_8.2HSPF': ashp_equipment,
    'Minimal_ASHP_15SEER_8.5HSPF': ashp_equipment,
    'Minimal_ASHP_16SEER_8.6HSPF': ashp_equipment,
    'Minimal_ASHP_18SEER_9.3HSPF': ashp_equipment,
    'Minimal_ASHP_18SEER_12.5HSPF': ashp_equipment,
    'Minimal_ASHP_22SEER_10HSPF': ashp_equipment,

    'Minimal_MSHP_B15_17SEER_9.4HSPF': mshp_equipment,
    'Minimal_MSHP_D12_26SEER_12.5HSPF': mshp_equipment,
    'Minimal_MSHP_E9_33SEER_14.2HSPF': mshp_equipment,

    # Water Heaters
    'Minimal_WH_Elec_Tankless': {'Tankless Water Heater': {}, **minimal_equipment},
    'Minimal_WH_Gas_Tankless': {'Gas Tankless Water Heater': {}, **minimal_equipment},
    'Minimal_WH_Elec_Std': {'Electric Resistance Water Heater': {}, **minimal_equipment},
    # 'Minimal_WH_HPWH_50gal': {'Heat Pump Water Heater': {'hp_only_mode': True}, **minimal_equipment},
    'Minimal_HPWH_65gal': {'Heat Pump Water Heater': {'hp_only_mode': True}, **minimal_equipment},
    'Minimal_WH_Gas_Std': {'Gas Water Heater': {}, **minimal_equipment},
    'Minimal_WH_Gas_Prem_Cond': {'Gas Water Heater': {}, **minimal_equipment},

    # Other equipment
    'Minimal_elec_appliances': {**minimal_equipment},
    'Minimal_gas_appliances': {**minimal_equipment},
    'Minimal_PV': {'PV': {'capacity': 5}, **minimal_equipment},
}
minimal_locations = [
    # 'ArtificialTestWeather',
    # 'Atlanta',
    # 'FortCollins',
    # 'Miami',
    # 'Phoenix',
    # 'Portland',
    'DC',
]



def run_minimal_building(location, name, run_simulation=True, **kwargs):
    # Create a dwelling and simulate for 1 year
    simulation_name = f'{location}_{name}'

    weather_file = os.path.join(test_suite_path, location, '{}.epw'.format(location))
    dwelling_args = {
        # Timing parameters
        'start_time': dt.datetime(2019, 1, 1),
        'time_res': dt.timedelta(minutes=60),
        'duration': dt.timedelta(days=365),
        'initialization_time': dt.timedelta(days=7),

        # Input and Output Files
        'output_path': test_suite_output_path,
        'hpxml_file': os.path.join(test_suite_path, location, '{}_rc_model.properties'.format(name)),
        'schedule_input_file': os.path.join(test_suite_path, location, '{}_schedule.properties'.format(name)),
        'weather_file': weather_file,
        'verbosity': 9,  # verbosity of results file (0-9); 8: include envelope; 9: include water heater
        'metrics_verbosity': 9,  # verbosity of metrics file (0-9)
    }
    dwelling_args.update(kwargs)

    # Load validation file - 8760 rows
    eplus_file = os.path.join(test_suite_path, location, '{}_Hourly.csv'.format(name))
    eplus_hourly = Analysis.load_eplus_file(eplus_file, eplus_format='BEopt', year=2019)

    # keep only comparable days
    duration = dwelling_args['duration']
    if duration.days != 365:
        start = dwelling_args['start_time']
        end = start + dwelling_args['duration'] - dwelling_args['time_res']  # Note: datetime slicing is inclusive!
        eplus_hourly = eplus_hourly.loc[start: end]

    # create comparison metrics
    eplus_metrics = Analysis.calculate_metrics(eplus_hourly, metrics_verbosity=6)

    # run simulation, or load from existing files
    if run_simulation:

        d = Dwelling(name=simulation_name, **dwelling_args)
        d.simulate()
    else:
        # Recalculate metrics and update metrics file
        metrics = Analysis.calculate_metrics(
            results_file=os.path.join(test_suite_output_path, simulation_name + '.csv'),
            metrics_verbosity=dwelling_args['metrics_verbosity'])
        df_metrics = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        df_metrics.to_csv(os.path.join(test_suite_output_path, simulation_name + '_metrics.csv'), index=False)

    # load OCHRE results
    results, results_metrics, results_hourly = Analysis.load_ochre(test_suite_output_path, simulation_name,
                                                                   combine_schedule=True)

    # Update power/energy if MELs in equipment
    if 'MELs Electric Power (kW)' in results:
        results['Total Electric Power (kW)'] -= results['MELs Electric Power (kW)']
        results_hourly['Total Electric Power (kW)'] -= results_hourly['MELs Electric Power (kW)']
        results_metrics['Total Electric Energy (kWh)'] -= results_metrics['MELs Electric Energy (kWh)']

    # Compare the actual and expected metrics for the minimal building test
    compare_metrics = Analysis.create_comparison_metrics(results_hourly, eplus_hourly, results_metrics, eplus_metrics)

    # save test metrics
    metrics_file = os.path.join(test_suite_output_path, simulation_name + '_comparison.csv')
    compare_metrics.to_csv(metrics_file)

    pass_fail_metrics(f'{location}_{name}')

    return {'df': results, 'df_hourly': results_hourly, 'eplus_hourly': eplus_hourly,
            'df_metrics': results_metrics, 'eplus_metrics': eplus_metrics, 'compare_metrics': compare_metrics}


def run_single(test_info, fail_on_error=False, **kwargs):
    # add equipment if necessary
    location, test_name = test_info
    if test_name in minimal_test_names:
        new_kwargs = minimal_equipment
        if test_name == 'Minimal_Intgain' or '_ducts_' in test_name:
            new_kwargs['MELs'] = misc_load
    elif test_name in equipment_test_names:
        new_kwargs = equipment_test_names[test_name]
    else:
        raise Exception('Bad test name: {}'.format(test_name))
    kwargs.update(new_kwargs)

    # run the simulation
    try:
        return run_minimal_building(location, test_name, **kwargs)
    except Exception as e:
        print('ERROR ({} {}):'.format(location, test_name), str(e))
        if fail_on_error:
            raise e
        else:
            return


def run_parallel(tests, num_processors=6):
    # run multiple simulations in parallel using multiprocessing.Pool
    with Pool(num_processors) as p:
        p.map(run_single, tests)


def run_all_minimal():
    # run all tests for all locations
    tests = itertools.product(minimal_locations, minimal_test_names)
    run_parallel(tests)


def run_all_equipment():
    # run all equipment tests in FortCollins
    tests = itertools.product(minimal_locations, equipment_test_names)
    run_parallel(tests)


def run_all_debug():
    # run tests in series with debug parameters: don't save results, run for 1 day
    debug_args = {
        # 'verbosity': 0,
        # 'time_res': dt.timedelta(minutes=60),
        # 'start_time': dt.datetime(2019, 1, 1),
        'duration': dt.timedelta(days=1),
        'initialization_time': None,
        'run_simulation': True,
    }
    location = 'DC'
    tests = [test for test in minimal_test_names + list(equipment_test_names.keys())]
    # tests = [test for test in list(equipment_test_names.keys())]
    for test in tests:
        run_single((location, test), fail_on_error=True, **debug_args)


def compile_all():
    # load all metrics, combine to 1 file
    all_metrics = []
    for location in minimal_locations:
        for name in minimal_test_names:
            metrics = pass_fail_metrics(f'{location}_{name}', show_metrics=False)
            all_metrics.append(metrics)

        for name in equipment_test_names:
            metrics = pass_fail_metrics(f'{location}_{name}', show_metrics=False)
            all_metrics.append(metrics)

    # combine metrics and save to files on Box and Teams
    df = pd.DataFrame(all_metrics)
    if len(minimal_locations) == 1:
        all_metrics_file = 'all_metrics_{}.csv'.format(minimal_locations[0])
    else:
        all_metrics_file = 'all_metrics.csv'
    df.to_csv(os.path.join(test_suite_path, all_metrics_file))

    # table of pass/fail
    table = pd.pivot(df, 'Test', 'Location', 'Passed')
    print('Table showing tests that passed:')
    print(table)
    print('Total failures:', (~ table).sum().sum())


if __name__ == '__main__':
    single_test_name = 'Fully_minimal'
    single_test_location = 'DC'

    # For running single case:
    data = run_single((single_test_location, single_test_name), fail_on_error=True,
                      # start_time=dt.datetime(2019, 5, 15),
                      # duration=dt.timedelta(days=10),
                      # time_res=dt.timedelta(minutes=1),
                      # initialization_time=dt.timedelta(days=1),
                      # verbosity=7,
                      run_simulation=False,
                      # linearize=True,
                      )

    # show plots
    plot_data = {
        'OCHRE': data['df_hourly'],
        'E+': data['eplus_hourly'],
        'OCHRE, exact': data['df'],
    }
    # CreateFigures.plot_external(plot_data)
    # CreateFigures.plot_envelope(plot_data)
    # CreateFigures.plot_hvac(plot_data)
    # CreateFigures.plot_wh(plot_data)
    # CreateFigures.plot_all_powers(plot_data)
    # CreateFigures.plot_end_use_powers(plot_data)
    # CreateFigures.plot_monthly_powers(plot_data)
    CreateFigures.plt.show()

    # For multiple runs:
    # run_all_debug()
    run_all_minimal()
    # run_all_equipment()
    # compile_all()
