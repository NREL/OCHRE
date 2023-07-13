#!/home/mblonsky/.conda/envs/nova/bin/python -u
#SBATCH --account=novametrics
#SBATCH --time=4:15:00
#xSBATCH --mail-user=mblonsky@nrel.gov
#xSBATCH --mail-type=ALL
#SBATCH --output=eio_%j.out
#SBATCH --qos=high
#xSBATCH --partition=debug

# import sys
# sys.path.append('/projects/novametrics/ochre')

import os
import datetime as dt
import pandas as pd

from ochre import Dwelling, Analysis, CreateFigures
from utils import run_eplus, pass_fail_metrics

upgrade_name = 'up{:02d}'  # for ResStock
bldg_name = 'bldg{:07d}'  # for ResStock

ochre_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
teams_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Validation')
default_weather_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - General', 'Weather', 
                                    'BuildStock_TMY3_FIPS')

dwelling_args = {
    # Timing parameters
    'start_time': dt.datetime(2018, 1, 1, 0, 0),  # year, month, day, hour, minute
    'time_res': dt.timedelta(minutes=60),
    'duration': dt.timedelta(days=365),
    'initialization_time': dt.timedelta(days=1),

    # Input parameters
    # 'hpxml_file': os.path.join(resstock_path, hpxml_name + '.xml'),
    # 'schedule_input_file': os.path.join(resstock_path, hpxml_name + '_existing_schedules.csv'),
    # 'weather_path': default_weather_path,

    # 'boundary_types_file': os.path.join(teams_path, 'ResStock Integration with HPXML', 'Envelope Properties', 'Envelope Boundary Types.csv'),
    # 'materials_file': os.path.join(teams_path, 'ResStock Integration with HPXML', 'Envelope Properties', 'Envelope Materials.csv'),

    # Output parameters
    'save_results': True,
    # 'output_path': os.path.join(resstock_path, 'OCHRE outputs'),
    'verbosity': 7,  # verbosity of results file (1-9)
    # 'export_res': dt.timedelta(days=61),
}


def run_ochre(folder, sim_name='ochre', xml_name='in.xml', fail_on_error=True):
    # should have IDF file and OCHRE input files (<xml_name>.xml and schedules.csv)
    schedule_input_file = os.path.join(folder, 'schedules.csv')
    if not os.path.exists(schedule_input_file):
        schedule_input_file = None

    # Get weather path
    if 'BEopt' in folder:
        weather_path = os.path.abspath(os.path.join(folder, os.pardir, os.pardir))
    else:
        weather_path = default_weather_path

    try:
        # Initialization
        dwelling = Dwelling(name=sim_name,
                            hpxml_file=os.path.join(folder, xml_name),
                            schedule_input_file=schedule_input_file,
                            weather_path=weather_path,
                            output_path=folder,
                            **dwelling_args)

        # Simulation
        out = dwelling.simulate()
    except Exception as e:
        if fail_on_error:
            raise e
        else:
            print(f'Error when running {folder}: {e}')
            out = None

    return out


def run_comparison(folder, sim_name='ochre', show_metrics=True, show_plots=True):
    # Load E+ files
    eplus_file = os.path.join(folder, 'results_timeseries.csv')
    eplus_format = 'ResStock' if 'BEopt' not in folder else 'BEopt'
    eplus = Analysis.load_eplus_file(eplus_file, eplus_format, year=dwelling_args['start_time'].year)
    # TODO: add detailed E+ file results, need to accept variable time resolution
    # eplus_file_detailed = os.path.join(folder, 'eplusout.csv')
    # if os.path.exists(eplus_file_detailed):
    #     eplus_detailed = Analysis.load_eplus_file(eplus_file_detailed, eplus_format='Eplus Detailed',
    #                                               year=dwelling_args['start_time'].year)
    #     eplus = eplus.join(eplus_detailed.loc[:, [col for col in eplus_detailed if col not in eplus.columns]])

    # only keep simulation days - note, datetime slicing is inclusive
    start = dwelling_args['start_time']
    end = start + dwelling_args['duration'] - dwelling_args['time_res']
    eplus = eplus.loc[start: end]
    eplus_metrics = Analysis.calculate_metrics(eplus, metrics_verbosity=6)

    # Load OCHRE files
    _, ochre_metrics, ochre = Analysis.load_ochre(folder, sim_name, load_main=False, combine_schedule=True)
    if ochre is None:
        print('Missing OCHRE hourly file, cannot compare:', folder)
        return

    # Compare metrics and save to file
    compare_metrics = Analysis.create_comparison_metrics(ochre, eplus, ochre_metrics, eplus_metrics)
    metrics_file = os.path.join(folder, f'{sim_name}_comparison.csv')
    compare_metrics.to_csv(metrics_file)

    if show_metrics:
        metrics = [metric for metric in compare_metrics.index if ('RMSE' not in metric and 'MEAN' not in metric)]
        metrics = compare_metrics.loc[metrics]
        print(f'Comparison Metrics for:', folder)
        print(metrics)

    if show_plots:
        # data = {'OCHRE (exact)': ochre_exact, 'OCHRE': ochre, 'E+': eplus}
        data = {'OCHRE': ochre, 'E+': eplus}
        # CreateFigures.plot_external(data)
        CreateFigures.plot_envelope(data)
        # CreateFigures.plot_hvac(data)
        # CreateFigures.plot_wh(data)
        # CreateFigures.plot_end_use_powers(data)
        # CreateFigures.plot_all_powers(data)
        CreateFigures.plt.show()


def compare_batch(batch_path, rerun_eplus=None, rerun_ochre=None, rerun_comparison=None):
    # find all simulation folders
    folders = []
    for root, _, file_names in os.walk(batch_path):
        if 'in.xml' in file_names:
            folders.append(root)


    # get metrics for each simulation
    all_metrics = []
    all_folders = {}
    for folder in folders:
        running = False
        files = os.listdir(folder)

        # run eplus if necessary (if None, do not overwrite)
        if rerun_eplus or (rerun_eplus is None and 'eplusout.csv' not in files):
            run_eplus(folder)
            running = True
        
        # run ochre if necessary (if None, do not overwrite)
        if rerun_ochre or (rerun_ochre is None and 'ochre_complete' not in files):
            run_ochre(folder, fail_on_error=False)
            running = True

        # run comparison if necessary (if None, do not overwrite)
        compare_ready = 'eplusout.csv' in files and 'ochre_complete' in files
        if compare_ready and (rerun_comparison or (rerun_comparison is None and 'ochre_comparison.csv' not in files)):
            run_comparison(folder, show_metrics=False, show_plots=False)

        if running:
            # add space between runs
            print()

        # get pass/fail metrics
        metrics = pass_fail_metrics(folder, show_metrics=False)
        all_metrics.append(metrics)
        all_folders[metrics['Name']] = folder

    # combine metrics and save to file
    df = pd.DataFrame(all_metrics).set_index('Name')
    df.to_csv(os.path.join(batch_path, 'all_metrics.csv'))

    # aggregate inputs to csv file
    json_files = {name: os.path.join(folder, 'ochre.json') for name, folder in all_folders.items()}
    df_inputs = Analysis.combine_json_files(json_files)
    df_inputs.to_csv(os.path.join(batch_path, 'all_inputs.csv'))

    # table of pass/fail
    n = len(df)
    n_passed = df['Passed'].sum()
    print()
    print(f'Tests passed: {n_passed} out of {n}')
    print()

    if n <= 5:
        df_show = df.drop(columns=['Last Update', 'Error Message'] + [col for col in df.columns if 'RMSE' in col])
        print(df_show.T)


if __name__ == '__main__':
    # validate_folder = '/projects/novametrics/national_scenarios/national_500/ResStock/up00'
    # validate_folder = os.path.join(teams_path, 'Multifamily', 'national_100')
    # validate_folder = os.path.join(teams_path, 'Multifamily', 'testing_500')
    validate_folder = os.path.join(teams_path, 'Multifamily', 'BEopt Test Cases')
    
    # run comparison for single case
    # single_folder = os.path.join(validate_folder, 'up00', 'bldg0000022')
    single_folder = os.path.join(validate_folder, 'Duplex_L', '1', 'run')
    # run_eplus(single_folder)
    run_ochre(single_folder)
    run_comparison(single_folder)
    # pass_fail_metrics(single_folder)

    # run comparison for a set of cases (in subfolders). Does not overwrite completed runs by default
    # compare_batch(validate_folder, 
    #               rerun_eplus=False, 
    #               rerun_ochre=False, 
    #             #   rerun_comparison=True,
    #               )

    # for b in range(1, 301):
    # for b in [39, 48]:
    #     single_folder = os.path.join(validate_folder, 'up00', bldg_name.format(b))
    #     # run_eplus(single_folder)
    #     run_ochre(single_folder)
    #     # run_ochre(single_folder, fail_on_error=False)
    #     run_comparison(single_folder)
