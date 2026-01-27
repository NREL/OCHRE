"""
ResStock output format utilities for OCHRE.

This module provides functions to convert OCHRE simulation output to 
ResStock-compatible CSV format (results_timeseries.csv and results_annual.csv).
"""

import os
import pandas as pd

from ochre.utils.base import default_input_path


# Conversion constants (matching Ruby implementation)
KWH_TO_MBTU = 0.003412141633
KW_TO_KBTU_HR = 3.412141633
C_TO_F_MULT = 9.0 / 5.0
C_TO_F_ADD = 32.0
M3S_TO_CFM = 2118.88


def load_crosswalk(crosswalk_file=None):
    """
    Load the ResStock-OCHRE crosswalk CSV.
    
    Parameters
    ----------
    crosswalk_file : str, optional
        Path to crosswalk CSV file. If None, uses the default file in 
        ochre/defaults/resstock_ochre_crosswalk.csv
    
    Returns
    -------
    pd.DataFrame
        Crosswalk DataFrame with columns: 'ResStock Annual', 'ResStock Timeseries', 'OCHRE'
    """
    if crosswalk_file is None:
        crosswalk_file = os.path.join(default_input_path, 'resstock_ochre_crosswalk.csv')
    return pd.read_csv(crosswalk_file)


def get_unit_type(ochre_col):
    """
    Determine unit type from OCHRE column name.
    
    Parameters
    ----------
    ochre_col : str
        OCHRE column name (e.g., 'Total Electric Power (kW)')
    
    Returns
    -------
    str
        Unit type: 'kw', 'w', 'celsius', 'm3s', or 'passthrough'
    """
    if '(kW)' in ochre_col:
        return 'kw'
    if '(W)' in ochre_col:
        return 'w'
    if '(C)' in ochre_col:
        return 'celsius'
    if '(m^3/s)' in ochre_col:
        return 'm3s'
    return 'passthrough'


def get_timeseries_unit(unit_type):
    """
    Get the ResStock timeseries unit string for a given unit type.
    
    Parameters
    ----------
    unit_type : str
        Unit type from get_unit_type()
    
    Returns
    -------
    str
        Unit string for ResStock timeseries output
    """
    unit_map = {
        'kw': 'kWh',
        'w': 'kBtu',
        'celsius': 'F',
        'm3s': 'cfm',
        'passthrough': '',
    }
    return unit_map.get(unit_type, '')


def convert_timeseries_value(value, unit_type, hours_per_step):
    """
    Convert a single value for timeseries output.
    
    Parameters
    ----------
    value : float
        Raw value from OCHRE output
    unit_type : str
        Unit type from get_unit_type()
    hours_per_step : float
        Hours per simulation timestep (e.g., 1.0 for hourly, 0.0167 for 1-minute)
    
    Returns
    -------
    float
        Converted value for ResStock timeseries
    """
    if pd.isna(value):
        return value
    
    if unit_type == 'kw':
        # kW -> kWh: multiply by hours per step
        return value * hours_per_step
    elif unit_type == 'w':
        # W -> kBtu: W -> kW -> kWh -> kBtu
        return value * 0.001 * hours_per_step * KW_TO_KBTU_HR
    elif unit_type == 'celsius':
        # °C -> °F
        return value * C_TO_F_MULT + C_TO_F_ADD
    elif unit_type == 'm3s':
        # m³/s -> cfm
        return value * M3S_TO_CFM
    return value


def build_resstock_timeseries(df, crosswalk, time_res):
    """
    Convert OCHRE DataFrame to ResStock timeseries format.
    
    Parameters
    ----------
    df : pd.DataFrame
        OCHRE results DataFrame with DatetimeIndex
    crosswalk : pd.DataFrame
        Crosswalk DataFrame from load_crosswalk()
    time_res : datetime.timedelta
        Simulation time resolution
    
    Returns
    -------
    tuple
        (result_df, units_dict) where result_df is the converted DataFrame
        and units_dict maps column names to unit strings
    """
    hours_per_step = time_res.total_seconds() / 3600
    
    # Filter crosswalk to rows with both OCHRE and Timeseries columns
    valid_mappings = crosswalk[
        crosswalk['OCHRE'].notna() & 
        (crosswalk['OCHRE'] != '') &
        crosswalk['ResStock Timeseries'].notna() &
        (crosswalk['ResStock Timeseries'] != '')
    ].copy()
    
    result = pd.DataFrame(index=df.index)
    units_dict = {}
    
    for _, row in valid_mappings.iterrows():
        ochre_col = row['OCHRE']
        resstock_col = row['ResStock Timeseries']
        
        if ochre_col not in df.columns:
            continue
        
        unit_type = get_unit_type(ochre_col)
        
        # Vectorized conversion for performance
        if unit_type == 'kw':
            result[resstock_col] = df[ochre_col] * hours_per_step
        elif unit_type == 'w':
            result[resstock_col] = df[ochre_col] * 0.001 * hours_per_step * KW_TO_KBTU_HR
        elif unit_type == 'celsius':
            result[resstock_col] = df[ochre_col] * C_TO_F_MULT + C_TO_F_ADD
        elif unit_type == 'm3s':
            result[resstock_col] = df[ochre_col] * M3S_TO_CFM
        else:
            result[resstock_col] = df[ochre_col]
        
        units_dict[resstock_col] = get_timeseries_unit(unit_type)
    
    return result, units_dict


def accumulate_annual_sums(df, crosswalk, time_res, existing_sums=None):
    """
    Accumulate kWh sums for annual totals (used for incremental export).
    
    Parameters
    ----------
    df : pd.DataFrame
        OCHRE results DataFrame with DatetimeIndex
    crosswalk : pd.DataFrame
        Crosswalk DataFrame from load_crosswalk()
    time_res : datetime.timedelta
        Simulation time resolution
    existing_sums : dict, optional
        Existing accumulated sums to add to
    
    Returns
    -------
    dict
        Mapping of OCHRE column names to accumulated kWh values
    """
    hours_per_step = time_res.total_seconds() / 3600
    sums = existing_sums.copy() if existing_sums else {}
    
    # Filter crosswalk to rows with both OCHRE and Annual columns
    valid_mappings = crosswalk[
        crosswalk['OCHRE'].notna() & 
        (crosswalk['OCHRE'] != '') &
        crosswalk['ResStock Annual'].notna() &
        (crosswalk['ResStock Annual'] != '')
    ].copy()
    
    for _, row in valid_mappings.iterrows():
        ochre_col = row['OCHRE']
        
        if ochre_col not in df.columns:
            continue
        
        unit_type = get_unit_type(ochre_col)
        
        # Only accumulate for energy columns (kW and W)
        if unit_type not in ('kw', 'w'):
            continue
        
        # Convert to kWh for accumulation
        if unit_type == 'kw':
            kwh_sum = df[ochre_col].sum() * hours_per_step
        else:  # 'w'
            kwh_sum = df[ochre_col].sum() * 0.001 * hours_per_step
        
        # Accumulate
        if ochre_col in sums:
            sums[ochre_col] += kwh_sum
        else:
            sums[ochre_col] = kwh_sum
    
    return sums


def convert_accumulated_sums_to_annual(sums, crosswalk):
    """
    Convert accumulated kWh sums to final annual MBtu values.
    
    Parameters
    ----------
    sums : dict
        Mapping of OCHRE column names to accumulated kWh values
    crosswalk : pd.DataFrame
        Crosswalk DataFrame from load_crosswalk()
    
    Returns
    -------
    dict
        Mapping of ResStock annual column names to values in MBtu
    """
    annual = {}
    
    # Build mapping from OCHRE column to annual column
    valid_mappings = crosswalk[
        crosswalk['OCHRE'].notna() & 
        (crosswalk['OCHRE'] != '') &
        crosswalk['ResStock Annual'].notna() &
        (crosswalk['ResStock Annual'] != '')
    ]
    
    ochre_to_annual = dict(zip(valid_mappings['OCHRE'], valid_mappings['ResStock Annual']))
    
    for ochre_col, kwh_sum in sums.items():
        if ochre_col in ochre_to_annual:
            annual_col = ochre_to_annual[ochre_col]
            annual[annual_col] = round(kwh_sum * KWH_TO_MBTU, 3)
    
    return annual


def calculate_annual_totals(df, crosswalk, time_res):
    """
    Calculate annual totals in MBtu for energy columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        OCHRE results DataFrame with DatetimeIndex
    crosswalk : pd.DataFrame
        Crosswalk DataFrame from load_crosswalk()
    time_res : datetime.timedelta
        Simulation time resolution
    
    Returns
    -------
    dict
        Mapping of ResStock annual column names to values in MBtu
    """
    hours_per_step = time_res.total_seconds() / 3600
    annual = {}
    
    # Filter crosswalk to rows with both OCHRE and Annual columns
    valid_mappings = crosswalk[
        crosswalk['OCHRE'].notna() & 
        (crosswalk['OCHRE'] != '') &
        crosswalk['ResStock Annual'].notna() &
        (crosswalk['ResStock Annual'] != '')
    ].copy()
    
    for _, row in valid_mappings.iterrows():
        ochre_col = row['OCHRE']
        annual_col = row['ResStock Annual']
        
        if ochre_col not in df.columns:
            continue
        
        unit_type = get_unit_type(ochre_col)
        
        # Only calculate annual sums for energy columns (kW and W)
        if unit_type not in ('kw', 'w'):
            continue
        
        # Sum to kWh, then convert to MBtu
        if unit_type == 'kw':
            kwh_sum = df[ochre_col].sum() * hours_per_step
        else:  # 'w'
            kwh_sum = df[ochre_col].sum() * 0.001 * hours_per_step
        
        annual[annual_col] = round(kwh_sum * KWH_TO_MBTU, 3)
    
    return annual


def write_resstock_timeseries(df, units_dict, file_path, append=False):
    """
    Write ResStock timeseries CSV with header + units rows.
    
    The ResStock timeseries format has:
    - Row 1: Column headers
    - Row 2: Unit strings
    - Row 3+: Data rows
    
    Parameters
    ----------
    df : pd.DataFrame
        Converted timeseries DataFrame from build_resstock_timeseries()
    units_dict : dict
        Mapping of column names to unit strings
    file_path : str
        Output file path
    append : bool, optional
        If True, append data without headers (for incremental export)
    """
    if df is None or df.empty:
        return
    
    # Reset index to include Time column
    df_out = df.reset_index()
    df_out = df_out.rename(columns={'index': 'Time'})
    
    if append and os.path.exists(file_path):
        # Append data rows only (no header, no units)
        df_out.to_csv(file_path, index=False, header=False, mode='a')
    else:
        # Build units row (Time column has empty unit)
        units_row = [''] + [units_dict.get(col, '') for col in df.columns]
        
        # Write header, units row, then data
        with open(file_path, 'w') as f:
            # Write header
            f.write(','.join(['Time'] + list(df.columns)) + '\n')
            # Write units row
            f.write(','.join(units_row) + '\n')
        
        # Append data rows
        df_out.to_csv(file_path, index=False, header=False, mode='a')


def update_resstock_annual(annual_dict, file_path):
    """
    Update results_annual.csv with calculated values.
    
    Reads existing file if present, updates/adds values from annual_dict,
    and writes back.
    
    Parameters
    ----------
    annual_dict : dict
        Mapping of metric names to values
    file_path : str
        Output file path for results_annual.csv
    """
    if not annual_dict:
        return
    
    # Read existing content if file exists
    existing = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        key, value = parts
                        existing[key.strip()] = value.strip()
    
    # Update with new values
    existing.update({k: str(v) for k, v in annual_dict.items()})
    
    # Write back
    with open(file_path, 'w') as f:
        for key, value in existing.items():
            f.write(f'{key},{value}\n')
