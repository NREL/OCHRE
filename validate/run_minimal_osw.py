import os
import datetime as dt
import pandas as pd
import json

from ochre import Dwelling, Analysis, CreateFigures
from validate.utils import create_inputs_from_osw, pass_fail_metrics

# import matplotlib.pyplot as plt
# from ochre import CreateFigures

ochre_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
teams_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Documents',
                          'ResStock Integration with HPXML', 'Minimal Buildings')
# TODO: Change to 'Validation' folder from 'ResStock Integration with HPXML'?
# TODO: Download OpenStudio-HPXML from https://github.com/NREL/OpenStudio-HPXML and add path to local directory for os_hpxml_path
os_hpxml_path = os.path.join(os.path.expanduser('~'), 'VSCodeProjects', 'OpenStudio-HPXML') 
all_locations = ['Denver', 'Phoenix']

dwelling_args = {
    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=60),
    'duration': dt.timedelta(days=365),
    'initialization_time': dt.timedelta(days=7),

    # Output parameters
    'save_results': True,
    'export_res': dt.timedelta(days=61),
    'verbosity': 7,  # verbosity of results file (1-9)
}


def run_ochre(simulation_name, location='Denver'):
    simulation_path = os.path.join(teams_path, location, simulation_name)

    # Initialization
    dwelling = Dwelling(name=simulation_name,
                        hpxml_file=os.path.join(simulation_path, 'in.xml'),
                        schedule_input_file=os.path.join(simulation_path, 'schedules.csv'),
                        weather_path=os.path.join(teams_path, location),
                        output_path=simulation_path,
                        **dwelling_args)

    # Simulation
    return dwelling.simulate()


def run_comparison(simulation_name, location='Denver', show_metrics=True, show_plots=True):
    simulation_path = os.path.join(teams_path, location, simulation_name)

    # Load OCHRE files
    _, ochre_metrics, ochre = Analysis.load_ochre(simulation_path, simulation_name, load_main=False,
                                                  combine_schedule=True)
    if ochre is None:
        return

    start = ochre.index[0]
    end = ochre.index[-1] + (ochre.index[1] - start)

    # Load E+ files
    eplus_file = os.path.join(simulation_path, 'results_timeseries.csv')
    eplus = Analysis.load_eplus_file(eplus_file, year=start.year)
    eplus_file_detailed = os.path.join(simulation_path, 'output-in', 'inout.csv')
    if os.path.exists(eplus_file_detailed):
        eplus_detailed = Analysis.load_eplus_file(eplus_file_detailed, eplus_format='Eplus Detailed', year=start.year)
        if 'WALL1:Surface Inside Face Convection Heat Transfer Coefficient [W/m2-K](Hourly)' in eplus_detailed.columns:
            # load json file to get metadata parameters (e.g. boundary areas)
            with open(os.path.join(simulation_path, 'in.json')) as f:
                properties = json.load(f)
            eplus_detailed = Analysis.add_eplus_detailed_results(eplus_detailed, ochre, properties)
        eplus = eplus.join(eplus_detailed.loc[:, [col for col in eplus_detailed if col not in eplus.columns]])

    # only keep simulation days - note, datetime slicing is inclusive
    eplus = eplus.loc[start: end]
    eplus_metrics = Analysis.calculate_metrics(eplus, metrics_verbosity=6)

    # Compare metrics and save to file
    compare_metrics = Analysis.create_comparison_metrics(ochre, eplus, ochre_metrics, eplus_metrics)
    metrics_file = os.path.join(simulation_path, simulation_name + '_comparison.csv')
    compare_metrics.to_csv(metrics_file)

    if show_metrics:
        show = compare_metrics.loc[[metric for metric in compare_metrics.index
                                    if ' RMSE' not in metric and ' MEAN' not in metric]]
        print(f'Comparison Metrics for {simulation_name}:')
        print(show)

    # show plots
    if show_plots:
        # data = {'OCHRE (exact)': ochre_exact, 'OCHRE': ochre, 'E+': eplus}
        data = {'OCHRE': ochre, 'E+': eplus}
        # CreateFigures.plot_external(data)
        # CreateFigures.plot_envelope(data)
        # CreateFigures.plot_hvac(data)
        CreateFigures.plot_wh(data)
        CreateFigures.plot_end_use_powers(data)
        # CreateFigures.plot_all_powers(data)


def compile_metrics(*simulations, location=None):
    all_metrics = []
    for sim_name in simulations:
        if isinstance(sim_name, tuple):
            sim_name, sim_location = sim_name
        else:
            sim_location = location

        simulation_path = os.path.join(teams_path, sim_location, sim_name)
        metrics = pass_fail_metrics(sim_name, path=simulation_path, show_metrics=False,
                                    keep_comparisons=['Total HVAC Heating Delivered (kWh)', 'Total HVAC Cooling Delivered (kWh)'])
        metrics['Location'] = sim_location
        all_metrics.append(metrics)

    if not len(all_metrics):
        print('No scenarios to compile.')
        return

    # combine metrics and save to file - folder depends on number of locations/simulations
    df = pd.DataFrame(all_metrics)
    if len(df['Location'].unique()) == 1:
        location = df['Location'].unique()[0]
        df = df.set_index('Name').drop(columns=['Location'])
        df.to_csv(os.path.join(teams_path, location, 'all_metrics.csv'))
    else:
        df = df.set_index(['Location', 'Name'])
        df.to_csv(os.path.join(teams_path, 'all_metrics.csv'))

    # table of pass/fail
    n = len(df)
    n_passed = df['Passed'].sum()
    print()
    print(f'Tests passed: {n_passed} out of {n}')
    print()

    if n <= 5:
        print(df['Error Message'])
        print()
        # Show all non-RMSE metrics for small number of cases
        drop_cols = ['Last Update', 'Error Message'] + [col for col in df.columns if 'RMSE' in col or 'MEAN' in col]
        df_show = df.drop(columns=drop_cols)
        print(df_show.T)
    else:
        # show only total electricity and HVAC delivered for many cases
        show_cols = [
            # 'Total Electric Energy (kWh)',
            'Total HVAC Heating Delivered (kWh), OCHRE', 'Total HVAC Cooling Delivered (kWh), OCHRE',
            'Total HVAC Heating Delivered (kWh), EnergyPlus', 'Total HVAC Cooling Delivered (kWh), EnergyPlus',
            'Total HVAC Heating Delivered (kWh)', 'Total HVAC Cooling Delivered (kWh)',
            'HVAC Heating Delivered (kW) RMSE', 'HVAC Cooling Delivered (kW) RMSE'
        ]
        df_show = df.loc[:, [col for col in show_cols if col in df.columns]]
        print(df_show)


def run_all(*locations, name_matches=None, run_ochre=True, show_metrics=False, show_plots=False):
    # run run_comparison for all cases in given locations
    if not locations:
        locations = all_locations
    if name_matches is None:
        name_matches = ['']  # accepts all simulation names

    simulations = []
    for location in locations:
        location_path = os.path.join(teams_path, location)
        for simulation_name in os.listdir(location_path):
            if (any([match in simulation_name for match in name_matches])
                    and os.path.isdir(os.path.join(location_path, simulation_name))
                    and simulation_name != 'Archive'):
                run_comparison(simulation_name, location,
                               run_ochre=run_ochre, show_metrics=show_metrics, show_plots=show_plots)
                simulations.append((simulation_name, location))

    compile_metrics(*simulations)


if __name__ == '__main__':
    # TODO:
    #  - phoenix ceiling uninsulated: Attic Floor R value (2.10) is far from closest match: None, R=1.47
    #  - walls_R19_woodstud: missing R19 materials
    # TODO schema 4.0 updates:
    #  - AirInfiltrationMeasurement.InfiltrationHeight
    #  - Envelope.extension.PartitionWallMass and FurnitureMass
    #  - flip frame floors (floor=1, ceiling=2)

    simulation_name = 'MSHP_19SEER_10HSPF'
    location = 'Denver'
    simulation_path = os.path.join(teams_path, location, simulation_name)
    os_hpxml_simulation_path = os.path.join(os_hpxml_path, 'workflow')
    create_inputs_from_osw(simulation_path, os_hpxml_simulation_path)
    # run_ochre(simulation_name, location)
    # run_comparison(simulation_name, location, show_metrics=True, show_plots=True)

    # run all cases in location folder (or all locations if not specified)
    # run_all('Denver', name_matches=['WH'], run_ochre=False, show_metrics=False, show_plots=False)
    # run_all(name_matches=['fully', 'walls', 'infiltration'], run_ochre=False, show_metrics=False, show_plots=False)
    # run_all(run_ochre=False, show_metrics=False, show_plots=False)

    CreateFigures.plt.show()
