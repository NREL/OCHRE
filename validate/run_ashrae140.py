import os
import datetime as dt
import shutil

import pandas as pd

from ochre import Dwelling, Analysis, CreateFigures
from validate.compare_with_eplus import pass_fail_metrics

# import matplotlib.pyplot as plt
# from ochre import CreateFigures

ochre_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
teams_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Documents',
                          'ResStock Integration with HPXML', 'ASHRAE_Standard_140')

dwelling_args = {
    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=60),
    'duration': dt.timedelta(days=365),
    'initialization_time': dt.timedelta(days=7),

    # Output parameters
    'save_results': True,
    'export_res': dt.timedelta(days=61),
    'verbosity': 9,  # verbosity of results file (1-9)
}


def run_single(simulation_name):
    simulation_path = os.path.join(teams_path, simulation_name)

    # Create folder and move xml file into folder, if necessary
    if not os.path.exists(simulation_path):
        os.mkdir(simulation_path)
        shutil.move(os.path.join(teams_path, f'{simulation_name}.xml'),
                    os.path.join(simulation_path, f'{simulation_name}.xml'))

    # Initialization
    dwelling = Dwelling(name=simulation_name, output_path=simulation_path,
                        hpxml_file=os.path.join(simulation_path,  f'{simulation_name}.xml'),
                        # schedule_input_file=os.path.join(simulation_path, 'stochastic.csv'),
                        # weather_path=teams_path,
                        **dwelling_args)

    # Simulation
    return dwelling.simulate()


def run_comparison(simulation_name, run_ochre=True, show_plots=True):
    simulation_path = os.path.join(teams_path, simulation_name)

    # Load E+ files
    eplus_file = os.path.join(simulation_path, 'results_timeseries.csv')
    eplus = Analysis.load_eplus_file(eplus_file, year=dwelling_args['start_time'].year)
    eplus_file_detailed = os.path.join(simulation_path, 'in_with_outputs.csv')
    if os.path.exists(eplus_file_detailed):
        eplus_detailed = Analysis.load_eplus_file(eplus_file_detailed, eplus_format='Eplus Detailed',
                                                  year=dwelling_args['start_time'].year)
        eplus = eplus.join(eplus_detailed.loc[:, [col for col in eplus_detailed if col not in eplus.columns]])

    # only keep a few days - note, datetime slicing is inclusive
    start = dwelling_args['start_time']
    end = start + dwelling_args['duration'] - dwelling_args['time_res']
    eplus = eplus.loc[start: end]
    eplus_metrics = Analysis.calculate_metrics(eplus, metrics_verbosity=6)

    # Run OCHRE if necessary
    if run_ochre:
        run_single(simulation_name)

    # Load OCHRE files
    ochre_exact, ochre_metrics, ochre = Analysis.load_ochre(simulation_path, simulation_name,
                                                            load_main=False, combine_schedule=True)

    # Compare metrics and save to file
    compare_metrics = Analysis.create_comparison_metrics(ochre, eplus, ochre_metrics, eplus_metrics)
    metrics_file = os.path.join(simulation_path, simulation_name + '_comparison.csv')
    compare_metrics.to_csv(metrics_file)

    show_metrics = compare_metrics.loc[[metric for metric in compare_metrics.index if ('RMSE' not in metric and
                                                                                       'MEAN' not in metric)]]
    print(f'Comparison Metrics for {simulation_name}:')
    print(show_metrics)

    # show plots
    if show_plots:
        # data = {'OCHRE (exact)': ochre_exact, 'OCHRE': ochre, 'E+': eplus}
        data = {'OCHRE': ochre, 'E+': eplus}
        # CreateFigures.plot_external(data)
        CreateFigures.plot_envelope(data)
        # CreateFigures.plot_hvac(data)
        # CreateFigures.plot_wh(data)
        # CreateFigures.plot_end_use_powers(data)
        # CreateFigures.plot_all_powers(data)


def compile_metrics(simulation_names):
    # load all metrics, combine to 1 file
    all_metrics = []
    for simulation_name in simulation_names:
        simulation_path = os.path.join(teams_path, simulation_name)
        metrics = pass_fail_metrics(simulation_name, path=simulation_path, show_metrics=False)
        all_metrics.append(metrics)

    # combine metrics and save to file
    df = pd.DataFrame(all_metrics).set_index('Name')
    if len(simulation_names) == 1:
        all_metrics_file = f'all_metrics_{simulation_names[0]}.csv'
    else:
        all_metrics_file = 'all_metrics.csv'
    df.to_csv(os.path.join(teams_path, all_metrics_file))

    # table of pass/fail
    n = len(df)
    n_passed = df['Passed'].sum()
    print()
    print(f'Tests passed: {n_passed} out of {n}')
    print()

    if n <= 5:
        # Show all non-RMSE metrics for small number of cases
        df_show = df.drop(columns=['Last Update', 'Error Message'] + [col for col in df.columns if 'RMSE' in col])
        print(df_show.T)
    else:
        # show only total electricity and HVAC delivered for many cases
        df_show = df.loc[:, ['Total Electric Energy (kWh)', 'Total HVAC Heating Delivered (kWh)',
                             'Total HVAC Cooling Delivered (kWh)']]
        print(df_show)


if __name__ == '__main__':
    run_single('L100AC')
    # run_comparison('L100AC', run_ochre=True, show_plots=True)
    # CreateFigures.plt.show()

    # run all cases in Teams folder
    # sim_names = []
    # for sim_name in os.listdir(teams_path):
    #     if 'ceiling' in sim_name:
    #         continue
    #     if os.path.isdir(os.path.join(teams_path, sim_name)):
    #         # run_single(sim_name)
    #         # run_comparison(sim_name, run_ochre=False, show_plots=False)
    #         sim_names.append(sim_name)
    #         # if 'infiltration' in sim_name:
    #         #     sim_names.append(sim_name)
    # compile_metrics(sim_names)
