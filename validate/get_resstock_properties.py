# Script to parse set of .eio and .xml files to extract envelope RC coefficients

import os
import pandas as pd
# import datetime as dt
# import numpy as np

from ochre import Analysis
from ochre.utils import default_input_path, convert, import_hpxml
import ochre.utils.hpxml as utils_hpxml

pd.set_option('display.precision', 3)  # precision in print statements
pd.set_option('expand_frame_repr', False)  # Keeps results on 1 line
pd.set_option('display.max_rows', 30)  # Shows up to 30 rows of data
pd.set_option('display.max_columns', None)  # Prints all columns

teams_path = os.path.join(os.path.expanduser('~'), 'NREL', 'Team OCHRE - Documents')
envelope_path = os.path.join(teams_path, 'ResStock Integration with HPXML', 'Envelope Properties')
equipment_path = os.path.join(teams_path, 'ResStock Integration with HPXML', 'Equipment Properties')
github_path = os.path.join(default_input_path, 'Envelope')

HVAC_TYPE_MAP = {
    ('central air conditioner', 'Cooling'): 'Air Conditioner',
    ('room air conditioner', 'Cooling'): 'Room AC',
    ('air-to-air', 'Cooling'): 'ASHP Cooler',
    ('mini-split', 'Cooling'): 'MSHP Cooler',
    ('air-to-air', 'Heating'): 'ASHP Heater',
    ('mini-split', 'Heating'): 'MSHP Heater',
}

def load_eio(file_name):
    # read in file
    df = pd.read_csv(file_name, header=None, names=range(30))

    # collect boundary rows, rename columns
    boundaries = df.loc[df[0] == ' Construction CTF', [1, 6]]
    boundaries.columns = ['Surface Name', 'Assembly R Value']

    # calculate Assembly Effective R Value from Conductance (column 2)
    boundaries['Assembly R Value'] = 1 / boundaries['Assembly R Value'].astype(float) * 5.678

    # collect material rows, add capacitance
    materials = df.loc[df[0] == ' Material CTF Summary', 1:6]
    materials.iloc[:, 1:] = materials.iloc[:, 1:].astype(float)
    materials.columns = ['Material Name', 'Thickness (m)', 'Conductivity (W/m-K)',
                         'Density (kg/m^3)', 'Specific Heat (J/kg-K)', 'Resistance (m^2-K/W)']
    materials['Capacitance (kJ/m^2-K)'] = (materials['Specific Heat (J/kg-K)'] * materials['Density (kg/m^3)'] *
                                           materials['Thickness (m)'] / 1000)

    # Remove number from end of Material Name, e.g. 'Name 1' -> 'Name'
    materials['Material Name'] = materials['Material Name'].replace(' ([1-9])$', '', regex=True)

    # add boundary name to materials
    bd_location = boundaries.index.searchsorted(materials.index) - 1
    materials['Surface Name'] = boundaries.reset_index(drop=True).iloc[bd_location]['Surface Name'].values

    # Remove unused boundaries
    surfaces = boundaries['Surface Name']
    remove_strings = ['ADIABATIC', 'REVERSED']
    remove_bd = surfaces.str.contains('|'.join(remove_strings))
    duplicate_bd = surfaces.str.replace(' [1-9]$', '', regex=True).duplicated()

    boundaries = boundaries.loc[~ {**remove_bd, **duplicate_bd}]
    materials = materials.loc[materials['Surface Name'].isin(boundaries['Surface Name'])]

    # rename to remove numbers at end of surface name
    boundaries['Surface Name'] = boundaries['Surface Name'].str.replace(' [1-9]$', '', regex=True)
    materials['Surface Name'] = materials['Surface Name'].str.replace(' [1-9]$', '', regex=True)

    return boundaries, materials


def load_hpxml_surfaces(file_name):
    # load HPXML file, get boundary names and surface info
    hpxml = import_hpxml(file_name)

    # Parse envelope properties
    hpxml_bd_data, construction = utils_hpxml.parse_hpxml_boundaries(hpxml, return_boundary_dicts=True)

    # reformat/rename surfaces and boundaries
    surfaces = {s: bd_name for bd_name, bd_dict in hpxml_bd_data.items() for s in bd_dict.keys()}
    surfaces = {s.upper().replace(' ', '_') + ' CONSTRUCTION': val for s, val in surfaces.items()}
    surfaces.update({
        'FURNITURE CONSTRUCTION LIVING SPACE': 'Indoor Furniture',
        'FURNITURE CONSTRUCTION GARAGE': 'Garage Furniture',
        'FURNITURE CONSTRUCTION BASEMENT - UNCONDITIONED': 'Foundation Furniture',
        'PARTITIONWALLCONSTRUCTION': 'Interior Wall',
        'DOOR': 'Door',
    })

    # get OCHRE boundary properties
    new_data = {bd_name: utils_hpxml.parse_hpxml_boundary(bd_name, bd) for bd_name, bd in hpxml_bd_data.items()}

    for bd_name in new_data:
        # add foundation type to Foundation boundaries
        if 'Foundation' in bd_name:
            new_data[bd_name]['Foundation Type'] = construction['Foundation Type']

        # add siding to garage attached wall
        if bd_name == 'Garage Attached Wall':
            new_data[bd_name]['Finish Type'] = new_data['Exterior Wall']['Finish Type']


    return surfaces, new_data


def load_idf(filename):
    # read lines from IDF file
    with open(filename, 'r+') as idf_reader:
        idf_lines = idf_reader.read().strip()

    # parse IDF code blocks, convert to dict of dicts like {(category: {name: {param_name: param_value}}}
    def parse_line(line):
        key = line.split('!- ')[1]
        val = line.split('!- ')[0].strip()[:-1]
        # convert string to float or list if possible
        try:
            val = eval(val)
        except (NameError, SyntaxError):
            pass
        return key, val

    idf_blocks = idf_lines.split('\n\n')
    idf_data = {}
    for block in idf_blocks:
        all_lines = block.split('\n')
        category = all_lines[0][:-1]
        block_dict = dict([parse_line(line) for line in all_lines[1:] if '!-' in line])
        # name = block_dict.get('Name')
        if category in idf_data:
            # turn into a list if multiple blocks have the same category
            if isinstance(idf_data[category], dict):
                idf_data[category] = [idf_data[category]]
            idf_data[category].append(block_dict)
        else:
            idf_data[category] = block_dict

    return idf_data


def load_schedule(filename):
    df = pd.read_csv(filename)
    return df


def get_boundary_info(name, r_value, hpxml_data, resstock_data):
    # returns dict with keys: ['Boundary Type', 'Construction Type', 'Finish Type', 'Insulation Details']
    if name in ['Exterior Wall', 'Garage Attached Wall']:
        const = hpxml_data['Construction Type']
        finish = hpxml_data['Finish Type']
        # color = hpxml_data['Color']
        insulation = resstock_data['Insulation Wall']
        insulation = ', '.join(insulation.split(', ')[1:])
        return {
            'Boundary Type': f'{const}, {finish}, {insulation}',
            'Construction Type': const,
            'Finish Type': finish,
            # 'Color': color,
            'Insulation Details': insulation,
        }
    elif name in ['Roof', 'Attic Roof', 'Garage Roof']:
        finish = hpxml_data['Finish Type']
        insulation = resstock_data.get('Insulation Roof', f'Est. R-{r_value:.1f}')
        const = hpxml_data['Construction Type']
        # color = hpxml_data['RoofColor']
        return {
            'Boundary Type': f'{finish}, {insulation}, {const}',
            'Construction Type': const,
            'Finish Type': finish,
            # 'Color': color,
            'Insulation Details': insulation,
        }
    elif name in ['Floor', 'Foundation Floor', 'Garage Floor']:
        return {
            'Boundary Type': resstock_data.get('Insulation Slab', f'Est. R-{r_value:.1f}'),
            'Insulation Details': hpxml_data.get('Insulation Details'),
        }
    elif name in ['Adjacent Foundation Wall']:
        const = hpxml_data['Foundation Type']
        return {
            'Boundary Type': const,
            'Construction Type': const,
        }
    # elif name in ['Foundation Wall']:
    #     const = hpxml_data['Type']
    #     finish = ', '.join([hpxml_data['Siding'], hpxml_data['Color']])
    #     insulation = hpxml_data.get('Insulation Details')
    #     # insulation = resstock_data.get('Insulation Foundation Wall', f'Est. R-{r_value:.1f}')
    #     return {
    #         'Boundary Type': f'{const}, {finish}, {insulation}',
    #         'Construction Type': const,
    #         'Finish Type': finish,
    #         'Insulation Details': insulation,
    #     }
    elif name in ['Foundation Ceiling', 'Raised Floor', 'Garage Interior Ceiling']:
        insulation = resstock_data.get('Insulation Floor', f'Est. R-{r_value:.1f}')
        return {
            'Boundary Type': insulation,
            'Insulation Details': insulation,
        }
    elif name in ['Adjacent Wall']:
        # Adjacent walls depend on wall type
        const = hpxml_data['Construction Type']
        return {
            'Boundary Type': const,
            'Construction Type': const,
        }
    elif name in ['Garage Wall']:
        # Garage walls don't include insulation by default in ResStock
        const = hpxml_data['Construction Type']
        finish = hpxml_data['Finish Type']
        # color = hpxml_data['Color']
        return {
            'Boundary Type': f'{const}, {finish}',
            'Construction Type': const,
            'Finish Type': finish,
            # 'Color': color,
        }
    elif name in ['Attic Wall']:
        # TODO: Some attic walls have insulation, not sure why
        const = hpxml_data['Construction Type']
        finish = hpxml_data['Finish Type']
        insulation = f'Est. R-{r_value:.1f}'
        # color = hpxml_data['Color']
        return {
            'Boundary Type': f'{const}, {finish}, {insulation}',
            'Construction Type': const,
            'Finish Type': finish,
            # 'Color': color,
            'Insulation Details': insulation,
        }
    elif name in ['Attic Floor']:
        insulation = resstock_data.get('Insulation Ceiling', f'Est. R-{r_value:.1f}')
        return {
            'Boundary Type': insulation,
            'Insulation Details': insulation,
        }
    elif name in ['Rim Joist']:
        finish = hpxml_data['Finish Type']
        # color = hpxml_data['Color']
        insulation = resstock_data.get('Insulation Rim Joist', f'Est. R-{r_value:.1f}')
        return {
            'Boundary Type': f'{finish}, {insulation}',
            'Finish Type': finish,
            # 'Color': color,
            'Insulation Details': insulation,
        }
    elif name in ['Door']:
        # door_type = resstock_data.get('Doors')  # Door type info not in HPXML, do not use
        insulation = hpxml_data['Boundary R Value']
        return {
            'Boundary Type': f'R-{insulation:.1f}',
            # 'Construction Type': door_type,
            'Insulation Details': f'R-{insulation:.1f}',
        }
    elif name in ['Interior Wall', 'Garage Ceiling'] or 'Furniture' in name or 'Adjacent' in name:
        return {
            'Boundary Type': 'Standard',
        }
    else:
        raise Exception(f'Unknown boundary name: {name}')


def parse_max_values(house_name, schedule, idf, hpxml, resstock_data):
    schedule_mean = schedule.mean().to_dict()

    # refactor idf data by schedule name
    idf_by_schedule_name = {
        **{d['Number of People Schedule Name']: d for d in idf['People'].values()},
        **{d['Schedule Name']: d for d in idf['ElectricEquipment'].values()},
        **{d['Schedule Name']: d for d in idf['OtherEquipment'].values()},
        **{d['Schedule Name']: d for d in idf['Lights'].values()},
        **{d['Schedule Name']: d for d in idf['Exterior:Lights'].values()},
        **{d['Flow Rate Fraction Schedule Name']: d for d in idf['WaterUse:Equipment'].values()},
    }

    # TODO: this is outdated
    resstock_type_map = {
        'occupants': 'build_existing_model.occupants',

        'cooking_range': 'build_existing_model.cooking_range',
        'refrigerator': 'build_existing_model.refrigerator',
        'extra_refrigerator': 'build_existing_model.misc_extra_refrigerator',
        'freezer': 'build_existing_model.misc_freezer',
        'dishwasher_power': 'build_existing_model.dishwasher',
        'clothes_washer_power': 'build_existing_model.clothes_washer',
        'clothes_dryer': 'build_existing_model.clothes_dryer',
        'pool_pump': 'build_existing_model.misc_pool_pump',
        'hot_tub_heater': 'build_existing_model.misc_hot_tub_spa',
        'hot_tub_pump': 'build_existing_model.misc_hot_tub_spa',
        'ceiling_fan': 'build_existing_model.ceiling_fan',
        'plug_loads_other': 'build_existing_model.plug_loads',
        'plug_loads_tv': 'build_existing_model.plug_loads',
        'plug_loads_well_pump': 'build_existing_model.plug_loads',
        'fuel_loads_grill': 'build_existing_model.misc_gas_grill',
        'fuel_loads_fireplace': 'build_existing_model.misc_gas_fireplace',
        'fuel_loads_lighting': 'build_existing_model.misc_gas_lighting',

        'lighting_interior': 'build_existing_model.lighting',
        'lighting_exterior': 'build_existing_model.lighting',
        'lighting_garage': 'build_existing_model.lighting',
        'lighting_exterior_holiday': 'build_existing_model.holiday_lighting',

        # water fixtures
        'fixtures': 'build_existing_model.hot_water_fixtures',
        'dishwasher': 'build_existing_model.dishwasher',
        'clothes_washer': 'build_existing_model.clothes_washer',
    }

    # collect max value data
    max_values = {}
    for name, data in idf_by_schedule_name.items():
        # idf_name = data['Name']  # different from schedule name
        mean_val = schedule_mean.get(name)
        if mean_val is None:
            if name not in ['mech vent bath fan 0 schedule', 'mech vent range fan 0 schedule', 'Always On Discrete']:
                print(f'Skipping {name} for house {house_name}')
            continue

        if 'Number of People' in data:
            max_value = data['Number of People']
            annual_value = None
            max_units, annual_units = 'People', None
        elif 'Design Level {W}' in data or 'Lighting Level {W}' in data:
            max_value = data.get('Design Level {W}', data.get('Lighting Level {W}')) / 1000
            annual_value = mean_val * max_value * 8760
            max_units, annual_units = 'kW', 'kWh/year'
        elif 'Peak Flow Rate {m3/s}' in data:
            max_value = data['Peak Flow Rate {m3/s}'] * 1000 * 60
            annual_value = mean_val * max_value * 60 * 8760
            max_units, annual_units = 'L/min', 'L/year'
        else:
            raise Exception(f'IDF file for house {house_name} missing data for {name}')
            # print(f'Skipping {name} for house {house_name}')
            # continue

        max_values[name] = {
            'Type (ResStock)': resstock_data[resstock_type_map[name]],
            'Type (HPXML)': None,
            'Max Value': max_value,
            'Max Value Unit': max_units,
            'Annual Value': annual_value,
            'Annual Value Unit': annual_units,
        }

    max_values = pd.DataFrame(max_values).T
    max_values.index.name = 'Schedule Name'

    # TODO
    # max_values['Occupancy HPXML (-)'] = hpxml['Occupancy']['Number of Occupants (-)']
    max_values['Occupancy ResStock'] = resstock_data['build_existing_model.occupants']
    max_values['Bedrooms ResStock'] = resstock_data['build_existing_model.bedrooms']
    max_values['Sqft ResStock'] = resstock_data['build_existing_model.geometry_floor_area']
    # max_values['Square Footage HPXML (m^2)'] = hpxml['Indoor']['Zone Area (m^2)']
    max_values = max_values.reset_index()

    # max_values = {
    #     **{name: idf_dict.get('Number of People') for name, idf_dict in idf_data['People'].items()},
    #     **{name: idf_dict.get('Design Level {W}') for name, idf_dict in idf_data['ElectricEquipment'].items()},
    #     **{name: idf_dict.get('Design Level {W}') for name, idf_dict in idf_data['OtherEquipment'].items()},
    #     **{name: idf_dict.get('Peak Flow Rate {m3/s}') for name, idf_dict in idf_data['WaterUse:Equipment'].items()},
    #     **{name: idf_dict.get('Lighting Level {W}') for name, idf_dict in idf_data['Lights'].items()},
    #     **{name: idf_dict.get('Design Level {W}') for name, idf_dict in idf_data['Exterior:Lights'].items()},
    # }
    # return max_values

    return max_values


def load_single_home(house_name, resstock_data, input_path):
    # load eio and hpxml
    boundaries, materials = load_eio(os.path.join(input_path, 'eplusout.eio'))
    surfaces, hpxml_bd_data = load_hpxml_surfaces(os.path.join(input_path, 'in.xml'))

    # map surface to boundary name
    boundaries['Boundary Name'] = boundaries['Surface Name'].replace(surfaces)
    materials['Boundary Name'] = materials['Surface Name'].replace(surfaces)
    bad_names = boundaries['Boundary Name'] == boundaries['Surface Name']
    if bad_names.any():
        raise Exception(f'Unknown surface names: {boundaries.loc[bad_names]}')

    # rename 2nd side of interior wall to prevent flagging for duplicates
    interior_index = materials['Boundary Name'] == 'Interior Wall'
    if interior_index.any():
        idx = interior_index[::-1].idxmax()
        materials.loc[idx, 'Material Name'] += ' REV'

    # remove duplicate boundaries - all columns should be identical for each boundary
    boundaries = boundaries.loc[~ boundaries.drop(columns=['Surface Name']).duplicated()]
    bad_rows = boundaries['Boundary Name'].duplicated(keep=False)
    if bad_rows.any():
        raise Exception(f'Multiple boundaries defined for house {house_name}: \n{boundaries.loc[bad_rows]}')
    boundaries = boundaries.drop(columns=['Surface Name']).set_index('Boundary Name')

    # remove duplicate materials
    materials = materials.loc[~ materials.drop(columns=['Surface Name']).duplicated()]
    bad_rows = materials.loc[:, ['Boundary Name', 'Material Name']].duplicated(keep=False)
    if bad_rows.any():
        raise Exception(f'Multiple materials defined for house {house_name}: \n{materials.loc[bad_rows]}')
    materials = materials.drop(columns=['Surface Name'])

    # check sum of material R values against boundary Assembly R Value
    bd_r_values = boundaries['Assembly R Value'].to_dict()
    for bd_name, r_assembly in bd_r_values.items():
        r_sum = materials.loc[materials['Boundary Name'] == bd_name, 'Resistance (m^2-K/W)'].sum() * 5.678
        if abs(r_sum - r_assembly) > 0.1:
            print(
                f'WARNING: Assembly R Value ({r_assembly:.2f}) different from sum of material R values ({r_sum:.2f})'
                f' for {bd_name} in house {house_name}.')

    # get boundary type from ResStock data
    boundary_type_info = {name: get_boundary_info(name, r_value, hpxml_bd_data.get(name, {}), resstock_data)
                          for name, r_value in bd_r_values.items()}
    boundary_type_info = pd.DataFrame(boundary_type_info).T

    # merge type info with boundaries. Only merge Boundary Type with materials
    boundaries = boundary_type_info.merge(boundaries, how='right', right_index=True, left_index=True)
    materials = materials.join(boundary_type_info['Boundary Type'], on='Boundary Name')

    # reorder columns
    boundaries = boundaries.reset_index()
    materials = materials.loc[:, ['Boundary Name', 'Boundary Type'] + materials.columns.to_list()[:-2]]

    # remove 'None' types
    boundaries = boundaries.loc[boundaries['Boundary Type'] != 'None']
    materials = materials.loc[materials['Boundary Type'] != 'None']

    # load idf and schedule
    # idf = load_idf(os.path.join(input_path, house_name + '.idf'))
    # schedule = pd.read_csv(os.path.join(input_path, house_name + '_existing_schedules.csv'))
    #
    # max_values = parse_max_values(house_name, schedule, idf, hpxml, resstock_data)
    max_values = None

    return boundaries, materials, max_values


def remove_duplicates(df, index_cols, ignore_cols=None):
    # remove duplicate entries from dataframe
    # check that count of each material is the same
    if 'Material Name' in index_cols:
        def check_material_counts(s):
            counts = s.value_counts()
            if not counts.eq(counts.iloc[0]).all():
                raise Exception(f'Unequal number of materials listed for {s.name}:\n{counts}')

        cols = [col for col in index_cols if col != 'Material Name']
        df.groupby(cols)['Material Name'].apply(check_material_counts)

    # ignore columns in ignore_cols - takes first value
    if ignore_cols is not None:
        df_drop = df.drop(columns=ignore_cols)
    else:
        df_drop = df

    # Checks to ensure unique values in index_cols. Raises error if there are duplicates in the index
    df = df.loc[~ df_drop.duplicated()]
    bad_rows = df.loc[:, index_cols].duplicated(keep=False)
    if bad_rows.any():
        raise Exception(f'Duplicate entries in index:\n{df.loc[bad_rows].sort_values(index_cols)}')

    return df


def get_single_hvac_properties(house_path, resstock_data):
    hpxml_full = import_hpxml(os.path.join(house_path, 'in.xml'))

    # load idf
    idf_full = load_idf(os.path.join(house_path, 'in.idf'))

    hvacs = []
    for hvac_type in ['Heating', 'Cooling']:
        # get relevant hpxml data, see utils_equipment.parse_hvac
        system = hpxml_full['Systems'].get('HVAC', {}).get('HVACPlant', {}).get(f'{hvac_type}System')
        heat_pump = hpxml_full['Systems'].get('HVAC', {}).get('HVACPlant', {}).get('HeatPump')
        if system and heat_pump:
            raise IOError(f'HVAC {hvac_type} system and heat pump cannot both be specified.')
        elif not system and not heat_pump:
            return {}
        hpxml = heat_pump if heat_pump else system

        resstock_type = resstock_data[f'HVAC {hvac_type} Efficiency']  # not used due to bad organization
        resstock_name = hpxml['HeatPumpType'] if heat_pump else hpxml[f'{hvac_type}SystemType']
        ochre_name = HVAC_TYPE_MAP.get((resstock_name, hvac_type))
        if ochre_name is None:
            print(f'Skipping HVAC {hvac_type} type: {resstock_name}')
            continue
        if f'Coil:{hvac_type}:DX:MultiSpeed' not in idf_full:
            print(f'Skipping HVAC {hvac_type} {resstock_name}: No multispeed coil in idf')
            continue
        if resstock_data['HVAC System Is Faulted'] == 'Yes':
            print(f'Skipping HVAC {hvac_type} {resstock_name}: System is faulted')
            continue

        capacity_rated = convert(hpxml[f'{hvac_type}Capacity'], 'Btu/hour', 'W')
        airflow_rated = convert(hpxml.get('extension', {}).get(f'{hvac_type}AirflowCFM', 0), 'ft^3/min', 'm^3/s')
        efficiency = hpxml[f'Annual{hvac_type}Efficiency']
        efficiency = f"{efficiency['Value']} {efficiency['Units']}"

        # get relevant idf data
        idf_coil = idf_full[f'Coil:{hvac_type}:DX:MultiSpeed']
        idf_sys = idf_full[f'UnitarySystemPerformance:Multispeed']
        # n_speeds = idf_coil['Number of Speeds']
        n_speeds = idf_sys[f'Number of Speeds for {hvac_type}']
        hvac = {
            'HVAC Name': HVAC_TYPE_MAP[(resstock_name, hvac_type)],
            'HVAC Efficiency': efficiency,
            'Number of Speeds': n_speeds
        }

        for speed in range(1, n_speeds + 1):
            t = 'Total ' if hvac_type == 'Cooling' else ''
            capacity = idf_coil[f'Speed Gross Rated {t}{hvac_type} Capacity {speed} {{W}}']
            # airflow_ratio = idf_coil[f'Speed Rated Air Flow Rate {speed} {{m3/s}}'] / airflow_rated
            airflow_ratio = idf_sys[f'{hvac_type} Speed Supply Air Flow Ratio {speed}']
            hvac.update({
                f'Capacity Ratio {speed}': capacity / capacity_rated,
                f'Air Flow Ratio {speed}': airflow_ratio,
                f'COP {speed}': idf_coil[f'Speed Gross Rated {hvac_type} COP {speed} {{W/W}}'],
            })
            if hvac_type == 'Cooling':
                hvac[f'SHR {speed}'] = idf_coil[f'Speed Gross Rated Sensible Heat Ratio {speed}']

            # Round to 5 digits
            hvac = {key: round(val, 5) if isinstance(val, float) else val for key, val in hvac.items()}
        
        hvacs.append(hvac)

    return hvacs


def collect_hvac_properties(resstock_path, results_file_name='buildstock.csv', project_name=None, output_path=equipment_path):
    if project_name is None:
        project_name = os.path.basename(resstock_path)

    # load BuildStock batch file
    resstock_results_file = os.path.join(resstock_path, results_file_name)
    resstock_results = pd.read_csv(resstock_results_file, index_col='Building')

    # find subfolders with valid files (hpxml, idf)
    required_file_patterns = ['in.xml', 'in.idf']
    folders = Analysis.find_subfolders(resstock_path, required_file_patterns)

    # collect all data
    all_hvac = []
    for folder in folders:
        house_name = os.path.basename(folder)
        assert house_name[:4] == 'bldg'
        house_idx = int(house_name[4:])
        resstock_data = resstock_results.loc[house_idx].to_dict()
        hvacs = get_single_hvac_properties(folder, resstock_data)
        for hvac in hvacs:
            hvac['Received From'] = f'{project_name} - {house_name}'
            all_hvac.append(hvac)
    
    # merge, remove duplicates, and sort
    all_hvac = pd.DataFrame(all_hvac)
    all_hvac = remove_duplicates(all_hvac, ['HVAC Name', 'HVAC Efficiency'], ['Received From'])
    all_hvac = all_hvac.sort_values(['HVAC Name', 'HVAC Efficiency'])

    # save to files
    all_hvac.to_csv(os.path.join(output_path, f'HVAC Multispeed Parameters - {project_name}.csv'), index=False)

    # print number of hvac types
    print(f'Finished compiling HVAC data for {len(folders)} houses.')
    print(f'Found {len(all_hvac)} HVAC types:')
    print(all_hvac[['HVAC Name', 'HVAC Efficiency', 'Number of Speeds']])


def collect_materials(resstock_path, results_file_name='buildstock.csv', project_name=None, output_path=envelope_path):
    if project_name is None:
        project_name = os.path.basename(resstock_path)

    # load BuildStock batch file
    resstock_results_file = os.path.join(resstock_path, results_file_name)
    resstock_results = pd.read_csv(resstock_results_file, index_col='Building')

    # find subfolders with valid files (eio, hpxml)
    required_file_patterns = ['eplusout.eio', 'in.xml']
    folders = Analysis.find_subfolders(resstock_path, required_file_patterns)

    # collect all data
    all_boundaries, all_materials = [], []
    for folder in folders:
        house_name = os.path.basename(folder)
        assert house_name[:4] == 'bldg'
        house_idx = int(house_name[4:])
        resstock_data = resstock_results.loc[house_idx].to_dict()
        bd, mat, _ = load_single_home(house_name, resstock_data, folder)
        bd['Received From'] = f'{project_name} - {house_name}'
        # mat['Received From'] = f'{project_name} - {house_name}'
        all_boundaries.append(bd)
        all_materials.append(mat)
    
    # merge, remove duplicates
    all_boundaries = pd.concat(all_boundaries, sort=False)
    all_materials = pd.concat(all_materials, sort=False)
    all_boundaries = remove_duplicates(all_boundaries, ['Boundary Name', 'Boundary Type'], ['Received From'])
    all_boundaries = all_boundaries.sort_values(['Boundary Name', 'Boundary Type'])
    all_materials = remove_duplicates(all_materials, ['Boundary Name', 'Boundary Type', 'Material Name'])
    all_materials = all_materials.sort_values(['Boundary Name', 'Boundary Type'])

    # save to files
    all_boundaries.to_csv(os.path.join(output_path, f'Envelope Boundary Types - {project_name}.csv'), index=False)
    all_materials.to_csv(os.path.join(output_path, f'Envelope Materials - {project_name}.csv'), index=False)

    # print number of boundaries
    print(f'Finished compiling data for {len(folders)} houses.')
    print(f'Found {len(all_boundaries)} boundaries and {len(all_materials)} materials total.')
    print(f'Number of boundaries by name:')
    print(all_boundaries['Boundary Name'].value_counts())


def merge_projects(*project_names, output_path=envelope_path):
    # open selected materials file and aggregate into boundaries
    new_materials = pd.read_csv(os.path.join(output_path, 'Selected Materials to Add.csv')).fillna('')

    def all_equal(items):
        assert all([item == items.iloc[0] for item in items])
        return items.iloc[0]

    # aggregate new materials into boundary types
    new_boundaries = new_materials.groupby(['Boundary Name', 'Boundary Type']).agg({
        'Construction Type': all_equal,
        'Insulation Details': all_equal,
        'Resistance (m^2-K/W)': sum,
        'Received From': all_equal,
    }).reset_index()
    new_boundaries['Assembly R Value'] = new_boundaries['Resistance (m^2-K/W)'] * 5.678
    # new_boundaries['Exterior Emissivity (-)'] = 0.9
    # new_boundaries['Interior Emissivity (-)'] = 0.9
    # new_boundaries['Exterior Solar Absorptivity (-)'] = 0.6
    # new_boundaries['Interior Solar Absorptivity (-)'] = 0.6
    del new_boundaries['Resistance (m^2-K/W)']

    # load project boundary and material files
    boundaries = [new_boundaries]
    materials = [new_materials]
    for project_name in project_names:
        boundaries.append(pd.read_csv(os.path.join(output_path, f'Envelope Boundary Types - {project_name}.csv')))
        materials.append(pd.read_csv(os.path.join(output_path, f'Envelope Materials - {project_name}.csv')))

    # merge, remove duplicates, and sort
    boundaries = pd.concat(boundaries)
    materials = pd.concat(materials)
    boundaries = remove_duplicates(boundaries, ['Boundary Name', 'Boundary Type'], ['Received From'])
    boundaries = boundaries.sort_values(['Boundary Name', 'Boundary Type'])
    materials = remove_duplicates(materials, ['Boundary Name', 'Boundary Type', 'Material Name'])
    materials = materials.sort_values(['Boundary Name', 'Boundary Type'])

    # save to files
    boundaries.to_csv(os.path.join(output_path, 'Envelope Boundary Types.csv'), index=False)
    materials.to_csv(os.path.join(output_path, 'Envelope Materials.csv'), index=False)

    # save directly to github defaults files
    # WARNING: this will overwrite the defaults!
    boundaries.to_csv(os.path.join(github_path, 'Envelope Boundary Types.csv'), index=False)
    materials.to_csv(os.path.join(github_path, 'Envelope Materials.csv'), index=False)


if __name__ == '__main__':
    # Envelope properties
    project_path = os.path.join(teams_path, 'Validation', 'Multifamily', 'national_100')
    collect_materials(project_path, project_name='MF_100')
    project_path = os.path.join(teams_path, 'Validation', 'Multifamily', 'testing_300')
    collect_materials(project_path, project_name='MF_300')
    
    projects = ['MF_100', 'MF_300']
    merge_projects(*projects)

    # HVAC properties
    # project_path = os.path.join(teams_path, 'Validation', 'testing_500')
    # collect_hvac_properties(project_path)
    
