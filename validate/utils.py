import os
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import datetime as dt

from ochre.utils import default_input_path, import_hpxml

# test_suite_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Documents', 'Test Suite')
# test_suite_output_path = os.path.join(test_suite_path, 'outputs')
metrics_criteria_file = os.path.join(os.path.dirname(__file__), 'metrics_criteria.csv')
df_criteria = pd.read_csv(metrics_criteria_file, index_col='Metric')
df_criteria = df_criteria.drop(columns=['Metric Category', 'Metric Verbosity', 'Description'])

if os.environ.get('NREL_CLUSTER') == 'eagle':
    weather_path = '/projects/novametrics/weather'
    os_exec_path = None  # TODO: get executables on eagle
    ep_exec_path = '/shared-projects/EnergyPlus/v22.1.0/build/Products/energyplus'
elif os.name == 'nt':
    weather_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - General', 'Weather', 
                                'BuildStock_TMY3_FIPS')
    os_exec_path = r'C:\openstudio-3.5.0\bin\openstudio.exe'
    ep_exec_path = r'C:\openstudio-3.5.0\EnergyPlus\energyplus.exe'
else:
    weather_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Documents', 'General', 'Weather',
                                'BuildStock_TMY3_FIPS')
    os_exec_path = '/Applications/OpenStudio-3.4.0/bin/openstudio'
    ep_exec_path = '/Applications/EnergyPlus/bin'

# os_workflow_path = '/OpenStudio-HPXML/workflow'

idf_output_files_to_include = ['Output EIO', 'Output RDD', 'Output CSV']
idf_outputs_to_add = [
    'Electric Equipment Electricity Rate',
    'Electric Equipment Latent Gain Rate',
    'Electric Equipment Total Heating Rate',
    'Other Equipment Latent Gain Rate',
    'Other Equipment Total Heating Rate',
    'People Sensible Heating Rate',
    'People Latent Gain Rate',
    'Zone Infiltration Current Density Volume Flow Rate',
    'Cooling Coil Total Cooling Energy',
    'Cooling Coil Sensible Cooling Energy',
    'Baseboard Total Heating Energy',
    'Heating Coil Heating Energy',
    'Surface Outside Face Convection Heat Transfer Coefficient',
    'Surface Inside Face Convection Heat Transfer Coefficient',
    'Surface Inside Face Net Surface Thermal Radiation Heat Gain Rate per Area',
    'Surface Outside Face Net Thermal Radiation Heat Gain Rate per Area',
]


def add_idf_detailed_outputs(idf_file_in, idf_file_out='modified.idf', schedule_input_file='schedules.csv'):
    # load idf file
    with open(idf_file_in, 'r') as f:
        idf = f.readlines()

    # get absolute file paths, default uses idf file folder
    if not os.path.isabs(idf_file_out):
        idf_file_out = os.path.join(os.path.dirname(idf_file_in), idf_file_out)
    if not os.path.isabs(schedule_input_file):
        schedule_input_file = os.path.join(os.path.dirname(idf_file_in), schedule_input_file)

    for i, line in enumerate(idf):
        if any([s in line for s in idf_output_files_to_include]):
            # Add eio, rdd, and csv files as outputs
            idf[i] = line.replace('No', 'Yes')
        if '/var/simdata/openstudio/generated_files/schedules' in line:
            # Replace with schedule file location
            idf[i] = f'  {schedule_input_file}, !- File Name\n'

    # add info for eio file
    idf.extend([
        "\n",
        "Output:Constructions,\n",
        "  Constructions,\n",
        "  Materials;\n",
    ])

    # add detailed time series outputs
    for output in idf_outputs_to_add:
        idf.extend([
            "\n",
            "Output:Variable,\n",
            "  *,                       !- Key Value\n",
            f"  {output}, !- Variable Name\n",
            "  hourly;                  !- Reporting Frequency\n",
        ])

    # overwrite idf file
    with open(idf_file_out, 'w') as f:
        f.writelines(idf)


def create_osws_from_template(scenario_names=None, input_path=None, template_name='Minimal_building', output_path=None,
                              arguments=None):
    # Load osw template file, create a copy with updated arguments for each scenario name specified
    if input_path is None:
        input_path = os.path.join(default_input_path, 'OSW Templates')
    if arguments is None:
        arguments = {}

    # Load template
    template_file = os.path.join(input_path, template_name + '.osw')
    with open(template_file, 'r') as f:
        template = json.load(f)

    # Load template arguments file, if it exists
    arguments_file = os.path.join(input_path, template_name + '.csv')
    if os.path.exists(arguments_file):
        df = pd.read_csv(arguments_file, index_col='Argument Name')
        if scenario_names is None:
            scenario_names = df['Scenario Name'].unique().tolist()
    else:
        df = None

    for scenario_name in scenario_names:
        # get full set of updated arguments
        new_args = df.loc[df['Scenario Name'] == scenario_name, 'Argument Value'].to_dict() if df is not None else {}
        if not new_args:
            print(f'Warning: Cannot find "{scenario_name}" in {template_name}.csv file.')
        arguments = {**new_args, **arguments}
        
        # Update values in OSW file
        osw = template.copy()
        bad_args = [arg for arg in arguments if arg not in osw['steps'][0]['arguments']]
        if bad_args:
            raise Exception(f'Arguments not allowed in osw: {bad_args}')
        osw['steps'][0]['arguments'].update(arguments)

        # save to new file
        if output_path is None:
            output_path = input_path
        osw_file = os.path.join(output_path, scenario_name + '.osw')
        with open(osw_file, 'w') as f:
            json.dump(osw, f, indent=2)
    

def create_inputs_from_osw(input_path, os_hpxml_simulation_path=None, output_path=None, osw_name='Minimal_building.osw'):
    # Check OS version
    version = subprocess.run([os_exec_path, 'openstudio_version'], capture_output=True).stdout.decode()
    print('OpenStudio Version:', version)

    # Run OS
    osw_file = os.path.join(os_hpxml_simulation_path, osw_name)
    os_results = subprocess.run([os_exec_path, 'run', '-w', osw_file], capture_output=True).stdout.decode()
    os_results = os_results.replace('\\n', '\n')
    if 'ERROR' in os_results:
        print(os_results)
        raise Exception('OS Error')
    elif not os.path.exists(os.path.join(input_path, 'in.idf')):
        raise Exception('OS Error: No IDF file created')

    # Copy all files to output path
    if output_path:
        shutil.copy(input_path, output_path)


def run_eplus(input_path, output_path=None, fail_on_error=False):
    # update idf file - add outputs and output files
    idf_file_in = os.path.join(input_path, 'in.idf')
    idf_file_modified = os.path.join(input_path, 'modified.idf')
    add_idf_detailed_outputs(idf_file_in, idf_file_modified)

    # Get weather file from HPXML file
    hpxml_data = import_hpxml(os.path.join(input_path, 'in.xml'))
    weather_name = hpxml_data['ClimateandRiskZones']['WeatherStation']['Name']
    weather_name = weather_name.strip('./')

    # Get weather file location
    if 'BEopt' in input_path:
        input_weather_path = os.path.abspath(os.path.join(input_path, os.pardir, os.pardir))
    else:
        input_weather_path = weather_path
    weather_file = os.path.join(input_weather_path, weather_name + '.epw')

    # Run EnergyPlus - need to check eplus executable, error outputs, etc.
    out = subprocess.run([ep_exec_path, '-w', weather_file, idf_file_modified], cwd=input_path, capture_output=True)
    ep_output_file = os.path.join(input_path, 'eplusout.log')
    with open(ep_output_file, 'w') as f:
        f.writelines(out.stdout.decode())
    if 'EnergyPlus Completed Successfully' in out.stderr.decode():
        print('EnergyPlus completed for:', input_path)
    else:
        msg = f'EnergyPlus error ({out.stderr.decode().strip()}) for: {input_path}'
        if fail_on_error:
            raise Exception(msg)
        else:
            print(msg)

    # Copy all files to output path
    if output_path:
        shutil.copy(input_path, output_path)


def pass_fail_metrics(folder, simulation_name=None, show_metrics=True, keep_comparisons=None):
    # load comparison metrics and check if the test passes
    if keep_comparisons is None:
        keep_comparisons = []
    if simulation_name is None:
        if 'BEopt' in folder:
            simulation_name = folder.split(os.sep)[-3]
        else:
            simulation_name = os.path.basename(folder)

    comparison_file = os.path.join(folder, 'ochre_comparison.csv')
    if not os.path.exists(comparison_file):
        return {'Name': simulation_name, 'Passed': False, 'Error Message': 'No metrics comparison file'}

    compare_metrics = pd.read_csv(comparison_file, index_col=0)
    compare_metrics.index.name = 'Metric'

    # Check that all metrics exist
    missing = ~ compare_metrics.index.isin(df_criteria.index)
    if missing.any():
        missing = compare_metrics.index[missing].to_list()
        # print(f'Metrics missing in metrics criteria file: {missing}')
        raise Exception(f'Metrics missing in metrics criteria file: {missing}')

    # Get all metrics to include in pass/fail test
    # Keep original order of metrics
    compare_metrics = compare_metrics.reset_index().join(df_criteria, on='Metric').set_index('Metric')
    compare_metrics = compare_metrics.loc[compare_metrics['Include']].drop(columns=['Include'])

    # Check that all metrics pass the given test criteria
    # For annual metrics, can pass either the absolute or percent criteria to pass
    # For RMSE metrics, must pass the absolution criteria
    compare_metrics['Passed'] = (
        (compare_metrics['Absolute Error'].abs() <= compare_metrics['Absolute Threshold']) &
        (compare_metrics['Percent Error (%)'].abs() <= compare_metrics['Percent Threshold'].fillna(np.inf))
    )
    passed = compare_metrics['Passed'].all()
    if passed:
        msg = 'Pass'
    else:
        # get first metric to fail and print thresholds
        first_fail = compare_metrics.loc[~ compare_metrics['Passed']].iloc[0]
        if abs(first_fail['Absolute Error']) <= first_fail['Absolute Threshold']:
            error = first_fail['Percent Error (%)']
            threshold = first_fail['Percent Threshold']
        else:
            error = first_fail['Absolute Error']
            threshold = first_fail['Absolute Threshold']
        msg = f'Failed on {first_fail.name}: |{error}| > {threshold}'

    if show_metrics:
        print('')
        print(f'{simulation_name} Test: {msg}')
        print(compare_metrics)
        print('')

    mod_time = dt.datetime.fromtimestamp(os.path.getmtime(comparison_file))
    errors = compare_metrics.loc[:, ['Absolute Error', 'Percent Error (%)']].T.reset_index().melt(id_vars='index')
    errors.index = errors['Metric'] + ': ' + errors['index']
    metrics = {
        'Name': simulation_name, 'Passed': passed, 'Last Update': mod_time, 'Error Message': msg,
        **errors['value'].to_dict(),
    }

    for m in keep_comparisons:
        if m in compare_metrics.index:
            metrics[f'{m}, OCHRE'] = compare_metrics.loc[m, 'OCHRE']
            metrics[f'{m}, EnergyPlus'] = compare_metrics.loc[m, 'EnergyPlus']

    return metrics


if __name__ == '__main__':
    teams_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Documents',
                              'ResStock Integration with HPXML')
    # Make osw files from osw template
    # create_osws_from_template(['Minimal_building_test'],
    #                           arguments={'air_leakage_house_pressure': 99},
    #                           output_path=teams_path
    #                           )

    # Make inputs for single run using an osw file
    create_inputs_from_osw(teams_path, osw_name='Minimal_building_test.osw')
    run_eplus(teams_path, output_path=None)
