import math
import numpy as np
import psychrolib

from ochre.utils import OCHREException, load_csv, convert

psychrolib.SetUnitSystem(psychrolib.SI)

# List of utility functions for OCHRE Equipment


EQUIPMENT_NAMES_BY_TYPE = {
    'HVAC Heating': {
        ('ElectricResistance', 'Electricity'): 'Electric Baseboard',
        ('Furnace', 'Electricity'): 'Electric Furnace',
        ('WallFurnace', 'Electricity'): 'Electric Furnace',
        ('FloorFurnace', 'Electricity'): 'Electric Furnace',
        ('Boiler', 'Electricity'): 'Electric Boiler',
        ('air-to-air', 'Electricity'): 'ASHP Heater',
        ('mini-split', 'Electricity'): 'MSHP Heater',
        ('Ideal Heater', 'Electricity'): 'Generic Heater',
        (None, 'Electricity'): 'Generic Heater',
        ('Furnace', 'Natural gas'): 'Gas Furnace',
        ('WallFurnace', 'Natural gas'): 'Gas Furnace',
        ('FloorFurnace', 'Natural gas'): 'Gas Furnace',
        ('Boiler', 'Natural gas'): 'Gas Boiler',
    },
    'HVAC Cooling': {
        ('central air conditioner', 'Electricity'): 'Air Conditioner',
        ('room air conditioner', 'Electricity'): 'Room AC',
        ('air-to-air', 'Electricity'): 'ASHP Cooler',
        ('mini-split', 'Electricity'): 'MSHP Cooler',
        ('Ideal Cooler', 'Electricity'): 'Generic Cooler',
        (None, 'Electricity'): 'Generic Cooler',
    },
    'Water Heating': {
        ('storage water heater', 'Electricity'): 'Electric Resistance Water Heater',
        ('instantaneous water heater', 'Electricity'): 'Tankless Water Heater',
        ('heat pump water heater', 'Electricity'): 'Heat Pump Water Heater',
        ('storage water heater', 'Natural gas'): 'Gas Water Heater',
        ('instantaneous water heater', 'Natural gas'): 'Gas Tankless Water Heater',
    },
}


def get_duct_info(ducts, zones, boundaries, construction, location, **kwargs):
    # Get zone type from duct_zone and zone info
    duct_zone = ducts['Zone']
    fnd_type = zones.get('Foundation', {}).get('Zone Type')
    fnd_wall_ins = boundaries.get('Foundation Wall', {}).get('Insulation Details', 'Uninsulated')
    fnd_ceil_ins = boundaries.get('Foundation Ceiling', {}).get('Boundary R Value', 0)

    if duct_zone == 'Attic':
        vented = 'vented' if zones['Attic']['Vented'] else 'unvented'
        radiant_barrier = '_radiant_barrier' if boundaries.get('Attic Roof', {}).get('Radiant Barrier', False) else ''
        zone_type = f'attic_{vented}{radiant_barrier}'
    elif duct_zone == 'Garage':
        zone_type = 'garage'
    elif duct_zone == 'Foundation' and fnd_type == 'Crawlspace':
        vented = 'vent' if zones['Foundation']['Vented'] else 'unvent'
        if fnd_wall_ins != 'Uninsulated' and fnd_ceil_ins > 5.3:
            insulation = 'crawlspace_ins_floor_wall'
        elif fnd_ceil_ins > 5.3:
            insulation = 'crawlspace_ins_floor'
        else:
            insulation = 'unins_crawlspace'
        zone_type = f'{vented}_{insulation}'
    elif duct_zone == 'Foundation' and 'Basement' in fnd_type:
        if fnd_wall_ins != 'Uninsulated':
            zone_type = 'basement_ins_walls'
        elif fnd_ceil_ins > 5.3:
            zone_type = 'basement_ins_ceiling'
        else:
            zone_type = 'unins_basement'
    else:
        raise OCHREException('Unknown duct location: {duct_location}')

    return {
        'Zone Type': zone_type,
        'House Volume (ft^3)': convert(construction['Conditioned Volume (m^3)'], 'm^3', 'ft^3'),
        'Latitude': location['latitude'],
        'Longitude': location['longitude'],

    }


def update_equipment_properties(properties, schedule, zip_parameters_file='ZIP Parameters.csv', **kwargs):
    all_equipment = properties['equipment']

    # add location properties to PV if it exists
    if 'PV' in all_equipment:
        all_equipment['PV']['location'] = properties['location']
    
    # split heat pump equipment into heater and cooler
    for heat_pump_name, short_name in [('Air Source Heat Pump', 'ASHP'),
                                       ('Minisplit Heat Pump', 'MSHP')]:
        if heat_pump_name in all_equipment:
            hvac_dict = all_equipment.pop(heat_pump_name)
            all_equipment[f'{short_name} Heater'] = hvac_dict
            all_equipment[f'{short_name} Cooler'] = hvac_dict

    # Update HVAC/Water Heating equipment names based on type and fuel
    for end_use, names_by_type in EQUIPMENT_NAMES_BY_TYPE.items():
        # Get equipment properties using either generic or named key or both
        generic = all_equipment.pop(end_use, {})
        named = {key: val for key, val in all_equipment.items() if key in names_by_type.values()}
        if len(named) > 1:
            eq_names = list(named.keys())
            raise OCHREException(f'Only 1 {end_use} equipment is allowed, but multiple were included in inputs: {eq_names}')
        
        # Get equipment name from named dict and from generic (using name and fuel)
        eq_name, eq_data = list(named.items())[0] if named else (None, {})
        eq_type = generic.get('Equipment Name')
        eq_fuel = generic.get('Fuel')
        if eq_fuel not in ['Electricity', 'Natural gas', None]:
            print(f'WARNING: Converting {eq_fuel} to natural gas for {end_use}.')
            eq_fuel = 'Natural gas'
        generic_name = names_by_type.get((eq_type, eq_fuel))
        if generic_name is None and eq_type is not None:
            raise OCHREException(f'Unknown {end_use} type ({eq_type}) and fuel ({eq_fuel}) combo.')

        # compare names from generic and named dicts
        if generic_name is None and eq_name is None:
            # no equipment in the end use
            continue
        elif eq_name is None:
            eq_name = generic_name
        elif generic_name is None:
            print(f'Using a {eq_name} for {end_use}, but no equipment specified in HPXML file')
        elif generic_name != eq_name:
            # if generic and named names are different, print a note
            print(f'Using a {eq_name} for {end_use} instead of a {generic_name}')

        # Merge generic parameters and equipment-specific parameters (if they exist)
        equipment = {**generic, **eq_data}

        # Add setpoint parameters based on schedule
        setpoints = schedule.get(f'{end_use} Setpoint (C)')
        if setpoints is not None:
            equipment['Max Setpoint (C)'] = setpoints.max()
            equipment['Min Setpoint (C)'] = setpoints.min()

        # Get additional parameters for ducts from envelope
        if 'Supply Leakage (-)' in equipment.get('Ducts', {}):
            equipment['Ducts'].update(get_duct_info(equipment['Ducts'], **properties))

        all_equipment[eq_name] = equipment

    # Load ZIP properties file and add default equipment properties for scheduled equipment
    df_zip = load_csv(zip_parameters_file, index_col='Equipment Name')
    zip_columns = ['Zp', 'Ip', 'Pp', 'Zq', 'Iq', 'Pq', 'pf']
    zip_data = df_zip.loc[df_zip['Included in OCHRE'], zip_columns].to_dict('index')
    for eq_name, eq_dict in all_equipment.items():
        all_equipment[eq_name] = {**zip_data.get(eq_name, {}), **eq_dict}

    return all_equipment


def calculate_duct_dse(hvac, ducts, climate_file='ASHRAE152_climate_data.csv',
                       zone_temp_file='ASHRAE152_zone_temperatures.csv', **kwargs):
    # Inputs from HPXML
    zone_type = ducts['Zone Type']
    # zone_type_fractions = {zone_type: 1}  # FUTURE: allow for multiple zone types
    latitude = ducts['Latitude']
    longitude = ducts['Longitude']
    house_volume = ducts['House Volume (ft^3)']
    # num_returns = hvac_distribution['DistributionSystemType']['AirDistribution']['NumberofReturnRegisters']

    supply_nom_leakage = ducts['Supply Leakage (-)']
    supply_area = ducts['Supply Area (ft^2)']
    supply_nom_r = ducts['Supply R Value']
    if supply_nom_r <= 0:
        supply_r = 1.7
    else:
        supply_r = 2.2438 + 0.5619 * supply_nom_r

    return_nom_leakage = ducts['Return Leakage (-)']
    return_area = ducts['Return Area (ft^2)']
    return_nom_r = ducts['Return R Value']
    if return_nom_r <= 0:
        return_r = 1.7
    else:
        return_r = 2.0388 + 0.7053 * return_nom_r

    # Inputs from HVAC
    hvac_type = 'Heating' if hvac.is_heater else 'Cooling'
    if hvac.n_speeds == 1:
        capacity_low = None
        fan_flow_low = None
        capacity = convert(hvac.capacity_list[1], 'W', 'Btu/hour')
        fan_flow = convert(hvac.flow_rate_list[1], 'm^3/s', 'cubic_feet/min')
    elif hvac.n_speeds == 2:
        capacity_low = convert(hvac.capacity_list[1], 'W', 'Btu/hour')
        fan_flow_low = convert(hvac.flow_rate_list[1], 'm^3/s', 'cubic_feet/min')
        capacity = convert(hvac.capacity_list[2], 'W', 'Btu/hour')
        fan_flow = convert(hvac.flow_rate_list[2], 'm^3/s', 'cubic_feet/min')
    elif hvac.n_speeds == 4:
        capacity_low = convert(hvac.capacity_list[2], 'W', 'Btu/hour')
        fan_flow_low = convert(hvac.flow_rate_list[2], 'm^3/s', 'cubic_feet/min')
        capacity = convert(hvac.capacity_list[4], 'W', 'Btu/hour')
        fan_flow = convert(hvac.flow_rate_list[4], 'm^3/s', 'cubic_feet/min')
    else:
        raise OCHREException(f'Unknown number of speeds for {hvac.name}: {hvac.n_speeds}')

    # Other inputs
    ambient_temp = 68 if hvac.is_heater else 78
    duct_thermal_mass_corr = 'Sheet Metal'  # Options: Sheet Metal or Flex Duct
    cooling_control = 'TXV'  # Options for cooling systems control: TXV or Other

    # Load climate file
    df_climate = load_csv(climate_file, index_col='Index')

    # Get data from ASHRAE152_climate_data.csv file
    # Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    lat = df_climate['Latitude'].values
    longit = df_climate['Longitude'].values
    dlat = np.radians(lat) - np.radians(latitude)
    dlon = np.radians(longit) - np.radians(longitude)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(latitude)) * np.cos(np.radians(lat)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df_climate['Distance'] = 6373.0 * c
    location_index = df_climate['Distance'].argmin() + 1
    climate_data = df_climate.loc[location_index].to_dict()

    heating_des_init = float(climate_data['Heating Design Temp'])  # required for evaluating zone temp file
    cooling_des_init = float(climate_data['Cooling Design Temp'])  # required for evaluating zone temp file
    heating_seas_init = float(climate_data['Heating Seasonal Temp'])  # required for evaluating zone temp file
    cooling_seas_init = float(climate_data['Cooling Seasonal Temp'])  # required for evaluating zone temp file
    # des_HR = float(climate_data['Wdesign'])
    # des_in_HR = float(climate_data['Windesign'])
    seas_HR = float(climate_data['Wseasonal'])
    # seas_in_HR = float(climate_data['Winseasonal'])
    # des_enthalpy = float(climate_data['Design hout'])
    # des_in_enthalpy = float(climate_data['Design hin'])
    seas_enthalpy = float(climate_data['Seasonal hout'])
    seas_in_enthalpy = float(climate_data['Seasonal hin'])
    ground_temp = (heating_des_init + cooling_des_init) / 2

    # Load zone temperature file
    df_zone_temps = load_csv(zone_temp_file, index_col='Zone Type')
    zone_type_data = df_zone_temps.loc[zone_type].to_dict()
    for key, val in zone_type_data.items():
         if isinstance(val, str):
            zone_type_data[key] = eval(val)

    # Calculate supply/return zone temp and enthalpy
    # des_temp = zone_type_data[f'{hvac_type.lower()}_des_temp']
    # des_supply_zone_temp = des_temp
    seas_temp = zone_type_data[f'{hvac_type.lower()}_seas_temp']
    seas_supply_zone_temp = seas_temp
    # des_supply_zone_enthalpy = des_supply_zone_temp * 0.24 + des_HR * (1061 + 0.444 * des_supply_zone_temp)
    seas_supply_zone_enthalpy = seas_supply_zone_temp * 0.24 + seas_HR * (1061 + 0.444 * seas_supply_zone_temp)
    supply_regain = zone_type_data['supply_duct_thermal_regain']

    # Note: hvac_mult = -1 for cooling. Sign will flip for cooling
    # if des_supply_zone_temp * hvac.hvac_mult > ambient_temp * hvac.hvac_mult:
    #     des_return_zone_temp = (heating_des_temp + des_supply_zone_temp) / 2
    # else:
    #     des_return_zone_temp = des_supply_zone_temp
    if hvac.is_heater:
        seas_return_zone_temp =  (heating_seas_init + seas_supply_zone_temp)/2 if seas_temp > ambient_temp else seas_supply_zone_temp
    else:
        seas_return_zone_temp = (cooling_seas_init + seas_supply_zone_temp)/2 if seas_temp < ambient_temp else seas_supply_zone_temp
    # else:
    # TODO: not using the 3 lines above, since (heating_seas_temp == seas_supply_zone_temp)
    # seas_return_zone_temp = seas_supply_zone_temp
    # if des_supply_zone_enthalpy * hvac.hvac_mult > des_in_enthalpy * hvac.hvac_mult:
    #     des_return_zone_enthalpy = (des_enthalpy + des_supply_zone_enthalpy) / 2
    # else:
    #     des_return_zone_enthalpy = des_supply_zone_enthalpy
    if seas_supply_zone_enthalpy * hvac.hvac_mult > seas_in_enthalpy * hvac.hvac_mult:
        seas_return_zone_enthalpy = (seas_enthalpy + seas_supply_zone_enthalpy) / 2
    else:
        seas_return_zone_enthalpy = seas_supply_zone_enthalpy
    return_regain = zone_type_data['return_duct_thermal_regain']

    if duct_thermal_mass_corr == 'Flex Duct':
        fcycloss = 0.02
    else:
        fcycloss = 0.05
    infil_fan_off = 0.35 * house_volume / 60
    manu_fan_flow = 0.0333 * capacity if not hvac.is_heater else None

    # des_supply_temp_diff = ambient_temp - des_supply_zone_temp
    seas_supply_temp_diff = ambient_temp - seas_supply_zone_temp
    # des_return_temp_diff = ambient_temp - des_return_zone_temp
    seas_return_temp_diff = ambient_temp - seas_return_zone_temp

    supply_duct_leakage = fan_flow * supply_nom_leakage  # [cfm]
    return_duct_leakage = fan_flow * return_nom_leakage  # [cfm]
    if hvac.n_speeds > 1:
        supply_duct_leakage_low = fan_flow_low * supply_nom_leakage  # [cfm]
        return_duct_leakage_low = fan_flow_low * return_nom_leakage  # [cfm]

    # ---------- High Speed ----------
    as_high = (fan_flow - supply_duct_leakage) / fan_flow
    ar_high = (fan_flow - return_duct_leakage) / fan_flow
    dTe_high = capacity * hvac.hvac_mult / (60 * fan_flow * 0.075 * 0.24)
    Bs_high = math.exp(-supply_area / (60 * fan_flow * 0.075 * 0.24 * supply_r))
    Br_high = math.exp(-return_area / (60 * fan_flow * 0.075 * 0.24 * return_r))
    imb_flow = abs(supply_duct_leakage - return_duct_leakage)
    if supply_duct_leakage > return_duct_leakage:
        infil = (infil_fan_off ** 1.5 + imb_flow ** 1.5) ** 0.67
    elif imb_flow > infil_fan_off:
        infil = 0
    else:
        infil = (infil_fan_off ** 1.5 - imb_flow ** 1.5) ** 0.67

    # ---------- Low Speed ----------
    if hvac.n_speeds > 1:
        as_low = (fan_flow_low - supply_duct_leakage_low) / fan_flow_low
        ar_low = (fan_flow_low - return_duct_leakage_low) / fan_flow_low
        dTe_low = capacity_low * hvac.hvac_mult / (60 * fan_flow_low * 0.0775 * 0.24)
        Bs_low = np.exp(-supply_area / (60 * fan_flow_low * 0.075 * 0.24 * supply_r))
        Br_low = math.exp(-return_area / (60 * fan_flow_low * 0.075 * 0.24 * return_r))
        # not used
        # imb_flow_low = abs(supply_duct_leakage_low - return_duct_leakage_low)
        # if supply_duct_leakage_low > return_duct_leakage_low:
        #     infil_low = (infil_fan_off ** 1.5 + imb_flow_low ** 1.5) ** 0.67
        # elif imb_flow_low > infil_fan_off:
        #     infil_low = 0
        # else:
        #     infil_low = (infil_fan_off ** 1.5 - imb_flow_low ** 1.5) ** 0.67

    # ---------- Uncorrected DE ----------
    if hvac.is_heater:
        # des_uncorr_de = as_high * Bs_high - as_high * Bs_high * (
        #     1 - Br_high * ar_high) * des_return_temp_diff / dTe_high - as_high * (
        #     1 - Bs_high) * des_supply_temp_diff / dTe_high
        if hvac.n_speeds == 1:
            seas_uncorr_de = (as_high * Bs_high - 
                              as_high * Bs_high * (1 - Br_high * ar_high) * seas_return_temp_diff / dTe_high -
                              as_high * (1 - Bs_high) * seas_supply_temp_diff / dTe_high)
        else:
            seas_uncorr_de = as_low * Bs_low - as_low * Bs_low * (
                1 - Br_low * ar_low) * seas_return_temp_diff / dTe_low - as_low * (
                1 - Bs_low) * seas_supply_temp_diff / dTe_low
    else:
        # des_uncorr_de = as_high * fan_flow * 60 * 0.075 / -capacity * (
        #     -capacity / fan_flow / (60 * 0.075) + (1 - ar_high) * (
        #         des_return_zone_enthalpy - des_in_enthalpy) + 0.24 * ar_high * (
        #         Br_high - 1) * (ambient_temp - des_return_zone_temp) + 0.24 * (
        #         Bs_high - 1) * (55 - des_supply_zone_temp))
        if hvac.n_speeds == 1:
            seas_uncorr_de = as_high * fan_flow * 60 * 0.075 / -capacity * (
                -capacity / fan_flow / (0.075 * 60) + (1 - ar_high) * (
                    seas_return_zone_enthalpy - seas_in_enthalpy) + 0.24 * ar_high * (
                    Br_high - 1) * (ambient_temp - seas_return_zone_temp) + 0.24 * (
                    Bs_high - 1) * (55 - seas_supply_zone_temp))
        else:
            seas_uncorr_de = as_low * fan_flow_low * 60 * 0.075 / -capacity_low * (
                -capacity_low / fan_flow_low / (60 * 0.075) + (1 - ar_low) * (
                    seas_return_zone_enthalpy - seas_in_enthalpy) + 0.24 * ar_low * (
                    Br_low - 1) * (ambient_temp - seas_return_zone_temp) + 0.24 * (
                    Bs_low - 1) * (55 - seas_supply_zone_temp))

    # ---------- High Speed ----------
    if hvac.is_heater:
        # des_load_factor = 1 - (60 * 0.075 * 0.24 * (ambient_temp - heating_des_temp) * (
        #     infil - infil_fan_off)) / des_uncorr_de / capacity
        seas_load_factor = 1 - (60 * 0.075 * 0.24 * (ambient_temp - heating_seas_init) * (
            infil - infil_fan_off)) / seas_uncorr_de / capacity
    else:
        # des_load_factor = 1 - (60 * 0.075 * (infil - infil_fan_off) * (
        #     des_in_enthalpy - des_enthalpy)) / -capacity / des_uncorr_de
        seas_load_factor = 1 - (60 * 0.075 * (infil - infil_fan_off) * (
            seas_in_enthalpy - seas_enthalpy)) / -capacity / seas_uncorr_de

    if hvac.is_heater:
        # des_equip_factor = 1
        if hvac.n_speeds == 1:
            seas_equip_factor = 1
        elif hvac.name in ['ASHP Heater', 'MSHP Heater']:
            seas_equip_factor = 0.44 + 0.56 * seas_uncorr_de
        else:
            seas_equip_factor = 0.91 + 0.09 * seas_uncorr_de
    else:
        # if cooling_control == 'TXV':
        #     des_equip_factor = 1.62 - 0.62 * fan_flow / manu_fan_flow + 0.647 * math.log(fan_flow / manu_fan_flow)
        # else:
        #     des_equip_factor = 0.65 + 0.35 * fan_flow / manu_fan_flow
        if hvac.n_speeds == 1:
            if cooling_control == 'TXV':
                seas_equip_factor = 1.62 - 0.62 * fan_flow / manu_fan_flow + 0.647 * math.log(fan_flow / manu_fan_flow)
            else:
                seas_equip_factor = 0.65 + 0.35 * fan_flow / manu_fan_flow
        else:
            if cooling_control == 'TXV':
                seas_equip_factor = (0.82 + 0.18 * seas_uncorr_de) * (
                    1.62 - 0.62 * fan_flow / manu_fan_flow) + 0.647 * math.log(fan_flow / manu_fan_flow)
            else:
                seas_equip_factor = (0.82 + 0.18 * seas_uncorr_de) * (
                    0.65 + 0.35 * fan_flow / manu_fan_flow)

    # ---------- Low Speed ---------- (not used)
    # if hvac.n_speeds > 1:
    #     if hvac.is_heater:
    #         des_load_factor_low = 1 - (60 * 0.075 * 0.24 * (ambient_temp - heating_des_temp) * (
    #             infil_low - infil_fan_off)) / des_uncorr_de / capacity_low
    #         seas_load_factor_low = 1 - (60 * 0.075 * 0.24 * (ambient_temp - heating_seas_temp) * (
    #             infil_low - infil_fan_off)) / seas_uncorr_de / capacity_low
    #     else:
    #         des_load_factor_low = 1 - (60 * 0.075 * (infil - infil_fan_off) * (
    #             des_in_enthalpy - des_enthalpy)) / -capacity_low / des_uncorr_de
    #         seas_load_factor_low = 1 - (60 * 0.075 * (infil - infil_fan_off) * (
    #             seas_in_enthalpy - seas_enthalpy)) / -capacity_low / seas_uncorr_de

    # ---------- Delivery Effectiveness ----------
    # des_de = (des_uncorr_de + supply_regain * (1 - des_uncorr_de) -
    #           (supply_regain - return_regain -
    #            Br_high * (ar_high * supply_regain - return_regain)) * des_return_temp_diff / dTe_high)
    seas_de = (seas_uncorr_de + supply_regain * (1 - seas_uncorr_de) -
               (supply_regain - return_regain -
                Br_high * (ar_high * supply_regain - return_regain)) * seas_return_temp_diff / dTe_high)

    # ---------- Distribution System Efficiency ----------
    # des_dse = des_de * des_equip_factor * des_load_factor * (1 - fcycloss)
    seas_dse = seas_de * seas_equip_factor * seas_load_factor * (1 - fcycloss)

    # Using seasonal DSE, not design DSE
    dse = seas_dse
    if 1 < dse <= 1.1:
        print(f'WARNING: {hvac_type} DSE slightly above 1.0 ({dse}). Setting to 1.0')
    elif not (0 < dse <= 1):
        raise OCHREException(f'{hvac_type} DSE out of bounds: {dse}')
    elif dse < 0.4:
        print(f'WARNING: Low {hvac_type} DSE: {dse}')

    return dse


# Psychrometric functions for HVAC
# Originally taken from BEopt python code, author: shorowit
# see: https://cbr.nrel.gov/BEopt2/svn/trunk/Modeling/util.py
def iterate(x0, f0, x1, f1, x2, f2, icount, TolRel=1e-5, small=1e-9):
    """
    Description:
    ------------
        Determine if a guess is within tolerance for convergence
        if not, output a new guess using the Newton-Raphson method

    Source:
    -------
        Based on XITERATE f77 code in ResAC (Brandemuehl)

    Inputs:
    -------
        x0      float    current guess value
        f0      float    value of function f(x) at current guess value

        x1,x2   floats   previous two guess values, used to create quadratic
                         (or linear fit)
        f1,f2   floats   previous two values of f(x)

        icount  int      iteration count
        cvg     bool     Has the iteration reached convergence?

    Outputs:
    --------
        x_new   float    new guess value
        cvg     bool     Has the iteration reached convergence?

        x1,x2   floats   updated previous two guess values, used to create quadratic
                         (or linear fit)
        f1,f2   floats   updated previous two values of f(x)

    Example:
    --------

        # Find a value of x that makes f(x) equal to some specific value f:

        # initial guess (all values of x)
        x = 1.0
        x1 = x
        x2 = x

        # initial error
        error = f - f(x)
        error1 = error
        error2 = error

        itmax = 50  # maximum iterations
        cvg = False # initialize convergence to 'False'

        for i in range(1,itmax+1):
            error = f - f(x)
            x,cvg,x1,error1,x2,error2 = \
                                     Iterate(x,error,x1,error1,x2,error2,i,cvg)

            if cvg == True:
                break
        if cvg == True:
            print 'x converged after', i, 'iterations'
        else:
            print 'x did NOT converge after', i, 'iterations'

        print 'x, when f(x) is', f,'is', x
    """

    dx = 0.1

    # Test for convergence
    if (abs(x0 - x1) < TolRel * max(abs(x0), small) and icount != 1) or f0 == 0:
        x_new = x0
        cvg = True
    else:
        x_new = None
        cvg = False

        if icount == 1:  # Perturbation
            mode = 1
        elif icount == 2:  # Linear fit
            mode = 2
        else:  # Quadratic fit
            mode = 3

        if mode == 3:
            # Quadratic fit
            if x0 == x1:  # If two xi are equal, use a linear fit
                x1 = x2
                f1 = f2
                mode = 2
            elif x0 == x2:  # If two xi are equal, use a linear fit
                mode = 2
            else:
                # Set up quadratic coefficients
                c = ((f2 - f0) / (x2 - x0) - (f1 - f0) / (x1 - x0)) / (x2 - x1)
                b = (f1 - f0) / (x1 - x0) - (x1 + x0) * c
                a = f0 - (b + c * x0) * x0

                if abs(c) < small:  # If points are co-linear, use linear fit
                    mode = 2
                elif abs((a + (b + c * x1) * x1 - f1) / f1) > small:
                    # If coefficients do not accurately predict data points due to
                    # round-off, use linear fit
                    mode = 2
                else:
                    D = b ** 2 - 4.0 * a * c  # calculate discriminant to check for real roots
                    if D < 0.0:  # if no real roots, use linear fit
                        mode = 2
                    else:
                        if D > 0.0:  # if real unequal roots, use nearest root to recent guess
                            x_new = (-b + math.sqrt(D)) / (2 * c)
                            x_other = -x_new - b / c
                            if abs(x_new - x0) > abs(x_other - x0):
                                x_new = x_other
                        else:  # If real equal roots, use that root
                            x_new = -b / (2 * c)

                        if f1 * f0 > 0 and f2 * f0 > 0:  # If the previous two f(x) were the same sign as the new
                            if abs(f2) > abs(f1):
                                x2 = x1
                                f2 = f1
                        else:
                            if f2 * f0 > 0:
                                x2 = x1
                                f2 = f1
                        x1 = x0
                        f1 = f0

        if mode == 2:
            # Linear Fit
            m = (f1 - f0) / (x1 - x0)
            if m == 0:  # If slope is zero, use perturbation
                mode = 1
            else:
                x_new = x0 - f0 / m
                x2 = x1
                f2 = f1
                x1 = x0
                f1 = f0

        if mode == 1:
            # Perturbation
            if abs(x0) > small:
                x_new = x0 * (1 + dx)
            else:
                x_new = dx
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0

    return x_new, cvg, x1, f1, x2, f2


def calculate_mass_flow_rate(DBin, Win, P, flow):
    """
   Description:
    ------------
        Calculate the mass flow rate at the given incoming air state (entering drybubl and wetbulb) and CFM

    Source:
    -------


    Inputs:
    -------
        Tdb    float    Entering Dry Bulb (degC)
        Win    float    Entering Humidity Ratio (kg/kg)
        P      float    Barometric pressure (kPa)
        flow   float    Volumetric flow rate of unit (m^3/s)
    Outputs:
    --------
        mfr    float    mass flow rate (kg/s)
    """
    rho_in = psychrolib.GetMoistAirDensity(DBin, Win, P * 1000)
    mfr = flow * rho_in
    return mfr


def calculate_shr(DBin, Win, P, Q, flow, Ao):
    """
           Description:
            ------------
                Calculate the coil SHR at the given incoming air state, CFM, total capacity, and coil
                Ao factor

            Source:
            -------
                EnergyPlus source code

            Inputs:
            -------
                Tdb    float    Entering Dry Bulb (degC)
                Win    float    Entering Humidity Ratio (kg/kg)
                P      float    Barometric pressure (kPa)
                Q      float    Total capacity of unit (kW)
                flow   float    Volumetric flow rate of unit (m^3/s)
                Ao     float    Coil Ao factor (=UA/Cp - IN SI UNITS)
            Outputs:
            --------
                SHR    float    Sensible Heat Ratio
            """
    mfr = calculate_mass_flow_rate(DBin, Win, P, flow)
    bf = math.exp(-1.0 * Ao / mfr) if mfr > 0 else 0.0

    # P = Units.psi2kPa(P)
    # DBin = convert(DBin, 'degF', 'degC')
    # Hin = h_fT_w_SI(Tin, Win)
    Hin = psychrolib.GetMoistAirEnthalpy(DBin, Win)  # in J/kg
    dH = Q * 1000 / mfr if mfr > 0 else 0.0
    H_ADP = Hin - dH / (1 - bf)

    # T_ADP = Tsat_fh_P_SI(H_ADP, P)
    # W_ADP = w_fT_h_SI(T_ADP, H_ADP)

    # Initialize
    T_ADP = psychrolib.GetTDewPointFromHumRatio(DBin, Win, P * 1000)
    T_ADP_1 = T_ADP  # (degC)
    T_ADP_2 = T_ADP  # (degC)
    W_ADP = psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, P * 1000)
    # error = H_ADP - h_fT_w_SI(T_ADP, W_ADP)
    error = H_ADP - psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)
    error1 = error
    error2 = error

    itmax = 50  # maximum iterations
    cvg = False

    for i in range(1, itmax + 1):

        W_ADP = psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, P * 1000)
        # error = H_ADP - h_fT_w_SI(T_ADP, W_ADP)
        error = H_ADP - psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)

        T_ADP, cvg, T_ADP_1, error1, T_ADP_2, error2 = \
            iterate(T_ADP, error, T_ADP_1, error1, T_ADP_2, error2, i)

        if cvg:
            break

    if not cvg:
        print('Warning: Tsat_fh_P failed to converge')

    # h_Tin_Wadp = h_fT_w_SI(Tin, W_ADP)
    h_Tin_Wadp = psychrolib.GetMoistAirEnthalpy(DBin, W_ADP)

    if Hin - H_ADP != 0:
        shr = min((h_Tin_Wadp - H_ADP) / (Hin - H_ADP), 1.0)
    else:
        shr = 1

    return shr


def coil_ao_factor(DBin, Win, P, Qdot, flow, shr):
    """
   Description:
    ------------
        Find the coil Ao factor at the given incoming air state (entering drybubl and wetbulb) and CFM,
        total capacity and SHR


    Source:
    -------
        EnergyPlus source code

    Inputs:
    -------
        Tdb    float    Entering Dry Bulb (degC)
        Win    float    Entering Humidity Ratio (kg/kg)
        P      float    Barometric pressure (kPa)
        Qdot   float    Total capacity of unit (kW)
        cfm    float    Volumetric flow rate of unit (m^3/s)
        shr    float    Sensible heat ratio
    Outputs:
    --------
        Ao    float    Coil Ao Factor
    """
    bf = coil_bypass_factor(DBin, Win, P, Qdot, flow, shr)
    mfr = calculate_mass_flow_rate(DBin, Win, P, flow)

    ntu = -1.0 * math.log(bf)
    Ao = ntu * mfr
    return Ao


def coil_bypass_factor(DBin, Win, P, Qdot, flow, shr):
    """
   Description:
    ------------
        Find the coil bypass factor at the given incoming air state (entering drybubl and wetbulb) and CFM,
        total capacity and SHR


    Source:
    -------
        EnergyPlus source code

    Inputs:
    -------
        Tdb    float    Entering Dry Bulb (degC)
        Win    float    Entering Humidity Ratio (kg/kg)
        P      float    Barometric pressure (kPa)
        Qdot   float    Total capacity of unit (kW)
        flow   float    Volumetric flow rate of unit (m^3/s)
        shr    float    Sensible heat ratio
    Outputs:
    --------
        CBF    float    Coil Bypass Factor
    """

    mfr = calculate_mass_flow_rate(DBin, Win, P, flow)

    dH = Qdot * 1000 / mfr  # W / kg/s == J/kg
    Hin = psychrolib.GetMoistAirEnthalpy(DBin, Win)
    h_Tin_Wout = Hin - (1 - shr) * dH
    Wout = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h_Tin_Wout, DBin)
    dW = Win - Wout
    Hout = Hin - dH
    Tout = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(Hout, Wout)
    RH_out = psychrolib.GetRelHumFromHumRatio(Tout, Wout, P * 1000)

    T_ADP = psychrolib.GetTDewPointFromHumRatio(Tout, Wout, P * 1000)  # Initial guess for iteration

    if shr == 1:
        W_ADP = psychrolib.GetHumRatioFromTWetBulb(T_ADP, T_ADP, P * 1000)
        H_ADP = psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)
        BF = (Hout - H_ADP) / (Hin - H_ADP)
        return max(BF, 0.01)

    if RH_out > 1:
        print('Error: Conditions passed to CoilBypassFactor result in outlet RH > 100%')

    dT = DBin - Tout
    M_c = dW / dT

    cnt = 0
    tol = 1.0
    errorLast = 100
    d_T_ADP = 5.0

    W_ADP = None
    while cnt < 100 and tol > 0.001:
        # for i in range(1,itmax+1):

        if cnt > 0:
            T_ADP = T_ADP + d_T_ADP

        W_ADP = psychrolib.GetHumRatioFromTWetBulb(T_ADP, T_ADP, P * 1000)

        M = (Win - W_ADP) / (DBin - T_ADP)
        error = (M - M_c) / M_c

        if error > 0 and errorLast < 0:
            d_T_ADP = -1.0 * d_T_ADP / 2.0

        if error < 0 and errorLast > 0:
            d_T_ADP = -1.0 * d_T_ADP / 2.0

        errorLast = error
        tol = math.fabs(error)
        cnt = cnt + 1

    H_ADP = psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)

    BF = (Hout - H_ADP) / (Hin - H_ADP)
    return max(BF, 0.01)
