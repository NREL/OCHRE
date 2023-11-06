import math
import numpy as np
import pandas as pd
import pvlib

from ochre.utils import OCHREException, load_csv, convert

# List of utility functions for OCHRE Envelope

ZONES = {
    'LIV': 'Indoor',
    'FND': 'Foundation',
    'GAR': 'Garage',
    'ATC': 'Attic'}
EXT_ZONES = {'EXT': 'Outdoor',
             'GND': 'Ground'}

CARDINAL_DIRECTIONS = {
    0: 'North',
    90: 'East',
    180: 'South',
    270: 'West',
}


def get_boundary_tilt(name):
    # get boundary tilt (i.e. orientation) based on boundary name (0-90 degrees)
    if any([x in name for x in ['Floor', 'Ceiling']]):
        tilt = 0  # horizontal
    elif any([x in name for x in ['Wall', 'Rim Joist', 'Window', 'Door', 'Furniture']]):
        tilt = 90  # vertical
    elif 'Roof' in name:
        tilt = None  # tilt should already be defined
    else:
        raise OCHREException('Unknown boundary name:', name)

    return tilt


def calculate_plane_irradiance(df, tilt, panel_azimuth, window_data=None, albedo=0.2, separate=False):
    panel_azimuth = panel_azimuth % 360

    # https://pvlib-python.readthedocs.io/en/latest/api.html#irradiance
    incidence_angle = pvlib.irradiance.aoi(tilt, panel_azimuth, df['zenith'], df['azimuth'])
    incidence_cosine = np.cos(np.radians(incidence_angle))

    # https://pvlib-python.readthedocs.io/en/stable/generated/pvlib.irradiance.get_total_irradiance.html
    irr = pvlib.irradiance.get_total_irradiance(tilt, panel_azimuth,
                                                df['zenith'], df['azimuth'],
                                                df['DNI (W/m^2)'], df['GHI (W/m^2)'], df['DHI (W/m^2)'],
                                                dni_extra=df['dni_extraterrestrial'], airmass=df['airmass'],
                                                albedo=albedo, model='perez')

    # Diffuse irradiance can be NA if GHI=0 and zenith < 90, force irradiance to 0
    if irr.isna().any().any():
        na_times = irr.isna().any(axis=1)
        if na_times.any():
            large_ghi = df.loc[na_times, 'GHI (W/m^2)'] > 10
            if large_ghi.any():
                raise OCHREException(f'Solar calculation is returning NA. First instance is: {large_ghi.idxmax()}')
            irr = irr.fillna(0)

    # Limit sky diffuse to DHI+GHI, based on simple Sandia model:
    # https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-sky-diffuse/simple-sandia-sky-diffuse-model/
    max_diffuse = df['DHI (W/m^2)'] + df['GHI (W/m^2)']
    high_diffuse = irr['poa_sky_diffuse'] > max_diffuse
    if high_diffuse.any():
        delta = (irr['poa_sky_diffuse'] - max_diffuse).clip(lower=0)
        irr['poa_sky_diffuse'] -= delta
        irr['poa_diffuse'] -= delta
        irr['poa_global'] -= delta
        # print(f'WARNING: Limiting sky diffuse irradiance based on DHI for {high_diffuse.sum()} time steps. '
        #       f'See: {(delta > 0).idxmax()}')

    if window_data is not None:
        window_shgc = window_data.get('SHGC (-)') * window_data.get('Shading Fraction (-)')
        window_u = window_data.get('U Factor (W/m^2-K)')

        # Note: moving transmittance/absorptance factors to Envelope.py

        # from https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/window-calculation-module.html
        # see step-7.-determine-angular-performance (not interpolating from figure)
        if window_u > 3.98:
            if window_shgc > 0.625:
                t_params = [0.0147, 1.486, -3.852, 3.355, -0.001474][::-1]  # transmittance curve A
            elif window_shgc > 0.3:
                t_params = [0.504475, 0.0474825, -2.289, 2.74225, -0.00116][::-1]  # transmittance curve BDCD
            else:
                t_params = [0.3462, 0.3963, -2.582, 2.845, -0.0002804][::-1]  # transmittance curve D
        elif window_u > 1.56:
            if window_shgc > 0.525:
                t_params = [2.883, -5.873, 2.489, 1.51, -0.002577][::-1]  # transmittance curve E
            else:
                t_params = [3.025, -6.366, 3.137, 1.213, -0.001367][::-1]  # transmittance curve F
        else:
            if window_shgc > 0.4:
                t_params = [2.883, -5.873, 2.489, 1.51, -0.002577][::-1]  # transmittance curve E
            else:
                t_params = [3.744, -8.836, 6.018, 0.08407, 0.0004825][::-1]  # transmittance curve J

        irr['poa_direct'] *= np.dot(t_params, [incidence_cosine.clip(lower=0) ** i for i in range(len(t_params))])

        # TODO: Fudge factor for diffuse irradiance: EPlus transmitted diffuse solar is lower than expected
        irr['poa_diffuse'] *= 0.854

        irr['poa_global'] = irr['poa_direct'] + irr['poa_diffuse']

    if separate:
        # return data frame with separate columns for each irradiance type
        return irr
    else:
        return irr['poa_global']


def calculate_solar_irradiance(weather, weather_timezone, location, boundaries, **house_args):
    # calculate solar angles, irradiance, other variables for solar calculations, using weather timezone
    # df.index = df.index.tz_localize(weather_timezone)
    time_index = weather.index.tz_localize(weather_timezone)
    df = pvlib.solarposition.get_solarposition(time_index,
                                               latitude=location['latitude'],
                                               longitude=location['longitude'])
    df['dni_extraterrestrial'] = pvlib.irradiance.get_extra_radiation(time_index)
    df['airmass'] = pvlib.atmosphere.get_relative_airmass(df['apparent_zenith'].values)
    df.index = df.index.tz_localize(None)
    weather = weather.join(df)

    # add solar irradiance for all external boundaries (except raised floors)
    irradiance_data = []
    for bd_name, bd in boundaries.items():
        if bd['Exterior Zone'] != 'Outdoor':
            continue

        areas = bd['Area (m^2)']
        tilt = bd.get('Tilt (deg)', get_boundary_tilt(bd_name))
        default_azimuth = [0] if tilt == 0 else list(CARDINAL_DIRECTIONS.keys())
        azimuths = bd.get('Azimuth (deg)', default_azimuth)
        window_data = bd if bd_name == 'Window' else None
        if len(areas) != len(azimuths):
            raise OCHREException(f'Number of areas and azimuths for {bd_name} are not equal.'
                            f' Areas: {areas}, Azimuths: {azimuths}')

        irr = sum([calculate_plane_irradiance(weather, tilt, az, window_data) * area
                   for area, az in zip(areas, azimuths)])
        irr.name = f'{bd_name} Irradiance (W)'
        irradiance_data.append(irr)

        if house_args.get('verbosity', 1) >= 8:
            # add detailed irradiance data
            for az in azimuths:
                irr = calculate_plane_irradiance(weather, tilt, az, window_data, separate=True)
                orientation = CARDINAL_DIRECTIONS[az] if az in CARDINAL_DIRECTIONS else f'{az} deg'
                irr.columns = [f'{bd_name} Irradiance - {orientation}, {col} (W/m^2)' for col in irr.columns]
                irradiance_data.append(irr)

    if house_args.get('verbosity', 1) >= 8:
        irr_horizontal = calculate_plane_irradiance(weather, 0, 0)
        irr_horizontal.name = 'Horizontal Irradiance (W/m^2)'
        irradiance_data.append(irr_horizontal)

    weather = pd.concat([weather] + irradiance_data, axis=1)
    return weather


def get_boundary_rc_values(all_bd_properties, raise_error=False, **house_args):
    # load property files for envelope boundaries and materials
    boundaries = load_csv(house_args.get('boundaries_file', 'Envelope Boundaries.csv'),
                          sub_folder='Envelope', index_col='Boundary Name')
    boundary_types = load_csv(house_args.get('boundary_types_file',
                                             'Envelope Boundary Types.csv'), sub_folder='Envelope')
    boundary_types = boundary_types.fillna('')
    materials = load_csv(house_args.get('materials_file', 'Envelope Materials.csv'), sub_folder='Envelope')

    # Add boundary RC data
    generic_args = {key: val for key, val in house_args.items() if key.lower() == key}
    for bd_name in list(all_bd_properties.keys()):
        bd_properties = {**boundaries.loc[bd_name].to_dict(), **all_bd_properties[bd_name]}
        if sum(bd_properties['Area (m^2)']) == 0:
            # remove boundary
            del all_bd_properties[bd_name]
            continue
        if bd_name == 'Window':
            # Note: Skips Window RC lookup, parameters are calculated based on U and SHGC later
            bd = boundaries.loc[bd_name].to_dict()
            all_bd_properties[bd_name] = {**generic_args, **bd, **bd_properties}
            continue

        construction = bd_properties.get('Construction Type')
        finish = bd_properties.get('Finish Type')
        insulation = bd_properties.get('Insulation Details')
        r_value = bd_properties.get('Boundary R Value')
        if r_value and r_value >= 100:
            # set to minimal building insulation at R500, ignore other properties
            r_value = 500
            construction = None
            finish = None
            insulation = None

        # Use lookup file to match R Value, Construction Type, and Insulation Details to Boundary Type
        # Add film resistances to R value in lookup file
        bd_choices = boundary_types.loc[boundary_types['Boundary Name'] == bd_name].copy()
        if bd_name in ['Attic Floor', 'Garage Interior Ceiling', 'Garage Ceiling', 'Foundation Ceiling',
                       'Adjacent Ceiling', 'Adjacent Floor']:
            film_r = 1.5
        else:
            film_r = 0.9
        bd_choices['Boundary R Value'] = bd_choices['Assembly R Value'] + film_r

        if construction is not None:
            bd_choices = bd_choices.loc[bd_choices['Construction Type'] == construction]
        if finish is not None:
            bd_choices = bd_choices.loc[bd_choices['Finish Type'] == finish]
        if insulation is not None:
            bd_choices = bd_choices.loc[bd_choices['Insulation Details'] == insulation]

        if len(bd_choices) == 0:
            keys = ['Construction Type', 'Finish Type', 'Insulation Details']
            p = {key: val for key, val in bd_properties.items() if key in keys}
            raise OCHREException(f'Cannot find material properties for {bd_name} with properties: {p}')
        elif len(bd_choices) == 1:
            bd_data = bd_choices.iloc[0].to_dict()
        elif r_value is None:
            raise OCHREException(f'Boundary R Value must be specified for {bd_name} with properties: {bd_properties}')
        else:
            # Find row with closest Boundary R Value
            r_options = bd_choices['Boundary R Value']
            closest_idx = (r_options - r_value).abs().idxmin()
            bd_data = bd_choices.loc[closest_idx].to_dict()
        bd_type = bd_data['Boundary Type']
        r_closest = bd_data['Boundary R Value']

        # Check for reasonable difference in R value (print warning if above R0.5)
        if r_value is not None:
            if r_value == 0:
                raise OCHREException(f'{bd_name} R value is zero, double check inputs')
            error_abs = abs(r_closest - r_value)
            error_pct = error_abs / r_value
            if raise_error and error_abs > 1 and error_pct > 0.15:
                raise OCHREException(f'{bd_name} R value ({r_value:0.2f}) does not match closest option: {bd_type},'
                                f' R={r_closest:0.2f}')
            elif error_abs > 0.5 and error_pct > 0.10:
                print(f'WARNING: {bd_name} R value ({r_value:0.2f}) is far from closest match: {bd_type},'
                      f' R={r_closest:0.2f}')

        # Extract rows from materials file with corresponding Boundary Type
        df = materials.loc[(materials['Boundary Name'] == bd_name) &
                           (materials['Boundary Type'] == bd_type)]
        if not len(df):
            raise OCHREException(f'Cannot find boundary properties for {bd_name} with type: {bd_type}')

        # Create RC parameters
        label = bd_properties['Boundary Label']
        for i, (_, row) in enumerate(df.iterrows()):
            bd_properties[f'R_{label}{i + 1}'] = row['Resistance (m^2-K/W)']
            bd_properties[f'C_{label}{i + 1}'] = row['Capacitance (kJ/m^2-K)']

        # update boundary properties with generic properties and with defaults
        all_bd_properties[bd_name] = {**generic_args, **bd_data, **bd_properties}

    return all_bd_properties


def create_rc_data(cap_list, res_list, same_zones=False, u_window=None):
    if u_window is not None:
        # update resistance values based on E+ equations, see step-1.-determine-glass-to-glass-resistance:
        # https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/window-calculation-module.html
        if u_window < 5.85:
            res_int_w = 1 / (0.359073 * np.log(u_window) + 6.949915)
        else:
            res_int_w = 1 / (1.788041 * u_window - 2.886625)
        res_ext_w = 0  # 1 / (0.025342 * u + 29.163853)
        r_window = 1 / u_window - res_int_w - res_ext_w
        return [], [r_window]

    nodes = len(cap_list)

    if same_zones:
        # if start and end zones are equal, cut boundary in half
        new_nodes = nodes // 2
        # self.area *= 2  # Not doubling the area - assuming it includes both sides of the boundary
        if nodes % 2 == 0:
            cap_list = cap_list[:new_nodes]
            res_list = res_list[:new_nodes]
        else:
            cap_list = cap_list[:new_nodes] + [cap_list[new_nodes] / 2]  # cut middle node in half
            res_list = res_list[:new_nodes + 1]  # middle resistance is split later
        nodes = len(cap_list)

    # R is split before and after given node - leads to 1 more R than C
    # exterior/interior film resistances included in first/last R
    res_list = [0] + res_list + [0]
    res_list = [(res_list[i] + res_list[i + 1]) / 2 for i in range(nodes + 1)]

    # consolidate nodes with 0 capacitance
    while any([c == 0 for c in cap_list]):
        # remove capacitance of 0, remove 1 R before, add it to the following R
        idx = cap_list.index(0)
        cap_list.pop(idx)
        r_to_move = res_list.pop(idx)
        res_list[idx] += r_to_move
        nodes -= 1
    assert len(cap_list) == nodes

    # remove last resistor if start and end zones are equal
    if same_zones:
        res_list = res_list[:-1]

    return cap_list, res_list


def calculate_film_resistances(name, boundary, location):
    # Using DOE-2 model for exterior surfaces, constant for wind speed:
    # https://bigladdersoftware.com/epx/docs/9-6/engineering-reference/outside-surface-heat-balance.html#doe-2-model
    # Using TARP model for interior surfaces, constant deltaT:
    # https://bigladdersoftware.com/epx/docs/9-6/engineering-reference/inside-heat-balance.html#tarp-algorithm
    ext_zone_label = boundary['Exterior Zone Label']
    int_zone_label = boundary['Interior Zone Label']
    wind_speed = location.get('Average Wind Speed (m/s)', 2)
    tilt = boundary.get('Tilt (deg)', get_boundary_tilt(name))

    # Determine orientation of zones
    zone_order = ['GND', 'FND', 'LIV', 'GAR', 'ATC', 'EXT']  # order for height and for interpolating temperatures
    ext_above = zone_order.index(ext_zone_label) > zone_order.index(int_zone_label)

    # Get typical zone temperatures, in C
    t_zones = pd.Series([location.get('Average Ground Temperature (C)', 10),
                         np.nan,
                         20,  # assumed average temperature for indoor air, in C
                         np.nan,
                         np.nan,
                         location.get('Average Ambient Temperature (C)', 10) + 5  # adding 5 C for solar gains
                         ], index=zone_order).interpolate()
    t_ext_zone = t_zones[ext_zone_label]
    t_int_zone = t_zones[int_zone_label]
    t_ext_hotter = t_ext_zone >= t_int_zone
    above_hotter = not (ext_above ^ t_ext_hotter)

    # Interior vertical boundaries default to h = 3.076 W/m^2-K, based on Simple Natural Convection Algorithm
    delta_t = max(12.9, abs(t_ext_zone - t_int_zone))

    # get interior heat transfer coefficient (TARP Algorithm)
    # TODO: option to use ASHRAE140 method for film coefficients
    if tilt == 90:
        # vertical boundary
        h_natural = 1.31 * delta_t ** (1 / 3)  # in W/m^2-K
    else:
        cos_tilt = abs(np.cos(convert(tilt, 'deg', 'rad')))
        if above_hotter:
            # use "enhanced" convection coefficient
            h_natural = 9.482 * delta_t ** (1 / 3) / (7.238 - cos_tilt)
        else:
            # use "reduced" convection coefficient
            h_natural = 1.810 * delta_t ** (1 / 3) / (1.382 + cos_tilt)

    if ext_zone_label == 'EXT':
        # add forced convection factor due to wind using DOE-2
        # using average values for windward/leeward coefficients
        r_f = 1.67  # ROUGHNESS_BY_FINISH_TYPE.get(boundary.get('Finish Type'), 1.67)
        h_glass = (h_natural ** 2 + (3.40 * wind_speed ** 0.75) ** 2) ** 0.5
        h_forced = r_f * (h_glass - h_natural)
    else:
        h_forced = 0

    return {
        'Exterior Film Resistance (m^2-K/W)': 1 / (h_natural + h_forced),
        'Interior Film Resistance (m^2-K/W)': 1 / h_natural,
    }


def calculate_window_parameters(window_shgc, window_u, res_material):
    # Calculate window transmittance, see step-4.-determine-layer-solar-transmittance
    # https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/window-calculation-module.html
    # Note: Interpolation not yet implemented
    if window_u > 3.95:
        if window_shgc < 0.7206:
            transmittance = 0.939998 * window_shgc ** 2 + 0.20332 * window_shgc
        else:
            transmittance = 1.30415 * window_shgc - 0.30515
    else:
        if window_shgc < 0.15:
            transmittance = 0.41040 * window_shgc
        else:
            transmittance = 0.085775 * window_shgc ** 2 + 0.963954 * window_shgc - 0.084958

    # Calculate window reflectance/absorptance, see step-5.-determine-layer-solar-reflectance
    x = window_shgc - transmittance
    if window_u > 3.95:
        res_int_s = 1 / (29.436546 * x ** 3 - 21.943415 * x ** 2 + 9.945872 * x + 7.426151)
        res_ext_s = 1 / (2.225824 * x + 20.57708)
    else:
        res_int_s = 1 / (199.8208128 * x ** 3 - 90.639733 * x ** 2 + 19.737055 * x + 6.766575)
        res_ext_s = 1 / (5.763355 * x + 20.541528)
    radiation_frac = (res_ext_s + res_material / 2) / (res_ext_s + res_material + res_int_s)
    absorptivity = x / radiation_frac

    return transmittance, absorptivity


def get_fnd_wall_insulation(wall):
    # Reduce area based on depth below grade
    height = wall.get('Height', 1)
    fnd_height = wall.get('DepthBelowGrade', 1)
    if fnd_height != height:
        wall['Area'] *= fnd_height / height

    # Add insulation details - take sum of interior and exterior layer R value
    insulation_layers = wall.get('Insulation', {}).get('Layer', [])
    r_value = sum([layer.get('NominalRValue', 0) for layer in insulation_layers])
    if r_value:
        # Add half wall or whole wall in insulation details
        insulation_height = min(
            [layer.get('DistanceToBottomOfInsulation', height) - layer.get('DistanceToTopOfInsulation', 0)
             for layer in insulation_layers]
        )
        if 0 < insulation_height <= height / 2:
            insulation_details = f'Half R{r_value:n}'
        else:
            insulation_details = f'R{r_value:n}'
    else:
        insulation_details = 'Uninsulated'

    return insulation_details


def get_slab_insulation(floor, name=None):
    # TODO: Not incorporating carpets (some carpet is assumed for floor and fnd_floor)
    r_perimeter = floor.get('PerimeterInsulation', {}).get('Layer', {}).get('NominalRValue', 0)
    r_under = floor.get('UnderSlabInsulation', {}).get('Layer', {}).get('NominalRValue', 0)
    if r_perimeter >= 100 and r_under >= 100:
        insulation = 'Minimal'
    elif r_perimeter and not r_under:
        # Perimeter = vertical insulation, usually on outside?
        depth = floor.get('PerimeterInsulation', {}).get('Layer', {}).get('InsulationDepth', 0)
        insulation = f'{depth:n}ft R{r_perimeter:n} Perimeter'
    elif not r_perimeter and r_under:
        # Under = horizontal insulation underneath slab, option for specific width or whole slab
        full_width = floor.get('UnderSlabInsulation', {}).get('Layer', {}).get('InsulationSpansEntireSlab', False)
        if full_width:
            insulation = f'R{r_under:n} Whole Slab'
        else:
            width = floor.get('UnderSlabInsulation', {}).get('Layer', {}).get('InsulationWidth', 0)
            insulation = f'{width:n}ft R{r_under:n} Exterior'
    elif not r_perimeter and not r_under:
        insulation = 'Uninsulated'
    else:
        raise OCHREException(f'Unknown slab insulation parameters for {name}: {floor}')

    return insulation


def calculate_ashrae_infiltration_params(indoor_inf, construction, site, has_flue_or_chimney=None):
    assert indoor_inf['HousePressure'] == 50
    assert indoor_inf['BuildingAirLeakage']['UnitofMeasure'] in ['ACH']  # only allow ACH50, not ACHnatural
    ach = indoor_inf['BuildingAirLeakage']['AirLeakage']
    infiltration_height_ft = indoor_inf['InfiltrationHeight']
    infiltration_height = convert(infiltration_height_ft, 'ft', 'm')

    # get shelter coefficient based on site shielding
    # see ResStock airflow.get_aim2_shelter_coefficient
    shelter_coeff_defaults = {
        'normal': 0.5,
        'exposed': 0.9,
        'well-shielded': 0.3,
    }
    shelter_coeff = shelter_coeff_defaults[site.get('ShieldingofHome', 'normal')]

    if has_flue_or_chimney:
        # 0.2 is a "typical" value according to THE ALBERTA AIR INFIL1RATION MODEL, Walker and Wilson, 1990
        y_i = 0.2  # Fraction of leakage through the flue
        s_wflue = 1  # Flue Shelter Coefficient
    else:
        y_i = 0  # Fraction of leakage through the flue
        s_wflue = 0  # Flue Shelter Coefficient


    # TODO: base on site['SiteType']. Or... move to ELA function??
    ashrae_terrain_exp = 0.14  # assumed rural at weather station
    ashrae_terrain_thick = 270
    ashrae_site_terrain_exp = 0.22  # assumed suburban on site
    ashrae_site_terrain_thick = 370
    weather_station_height = 10  # in m
    f_t = (((ashrae_terrain_thick / weather_station_height) ** ashrae_terrain_exp) * 
           ((infiltration_height / ashrae_site_terrain_thick) ** ashrae_site_terrain_exp))
    inf_sft = f_t * (shelter_coeff * (1 - y_i) + s_wflue * 1.5 * y_i)
    
    # Stack and wind coef calculations
    # Based on "Field Validation of Algebraic Equations for Stack and Wind Driven Air Infiltration Calculations"
    # by Walker and Wilson (1998)
    # see also ResStock airflow.apply_infiltration_to_conditioned
    inf_conv_factor = 776.25  # [ft/min]/[inH2O^(1/2)*ft^(3/2)/lbm^(1/2)]
    delta_pref = 0.016  # inH2O
    outside_air_density = 0.0765  # TODO: calculate based on average outdoor air temperature and altitude
    assumed_indoor_temp = 73.5  # F
    g = 32.174  # gravity (ft/s2)

    # Pressure Exponent
    n_i = 0.65

    # Calculate SLA for above-grade portion of the building (in IP, ft2)
    indoor_volume = construction['Conditioned Volume (m^3)']
    indoor_floor_area = construction['Indoor Floor Area (m^2)']
    building_height = construction['Ceiling Height (m)'] * construction['Indoor Floors']
    building_height_ft = convert(building_height, 'm', 'ft')  # in ft
    house_pressure = 50
    living_sla = ((ach * 0.2835 * 4 ** n_i * convert(indoor_volume, 'm^3', 'ft^3')) / (
        convert(indoor_floor_area, 'm^2', 'in^2') * house_pressure ** n_i * 60))

    # Effective Leakage Area (ft2); FUTURE: apportion to unit based on exposed wall area
    a_o = living_sla * convert(indoor_floor_area, 'm^2', 'ft^2')

    # Flow Coefficient (cfm/inH2O^n) (based on ASHRAE HoF)
    c_i = a_o * (2 / outside_air_density) ** 0.5 * delta_pref ** (0.5 - n_i) * inf_conv_factor
    inf_c = convert(c_i, 'cubic_feet/min', 'm^3/s') / (convert(1, 'inch_H2O_39F', 'Pa') ** n_i)

    # Leakage distributions per Iain Walker (LBL) recommendations
    if construction['Foundation Type'] == 'Crawlspace':
        # 15% ceiling, 35% walls, 50% floor leakage distribution for vented crawl
        leakage_ceiling = 0.15
        leakage_walls = 0.35
        leakage_floor = 0.50
    else:
        # 25% ceiling, 50% walls, 25% floor leakage distribution for slab/basement/unvented crawl
        leakage_ceiling = 0.25
        leakage_walls = 0.50
        leakage_floor = 0.25
    assert leakage_ceiling + leakage_walls + leakage_floor == 1

    r_i = leakage_ceiling + leakage_floor
    x_i = leakage_ceiling - leakage_floor
    r_i = r_i * (1 - y_i)
    x_i = x_i * (1 - y_i)

    # Calculate Stack Coefficient
    m_o = (x_i + (2 * n_i + 1) * y_i) ** 2 / (2 - r_i)
    if m_o <= 1:
        m_i = m_o  # eq. 10
    else:
        m_i = 1  # eq. 11
    if has_flue_or_chimney:
        ncfl_ag = construction['Indoor Floors']  # conditioned floors above grade
        z_f = (ncfl_ag + 0.5) / ncfl_ag if ncfl_ag > 0 else 1
        # Critical value of ceiling-floor leakage difference where the neutral level is located at the ceiling (eq. 13)
        x_c = r_i + (2 * (1 - r_i - y_i)) / (n_i + 1) - 2 * y_i * (z_f - 1) ** n_i
        # Additive flue function, Eq. 12
        f_i = n_i * y_i * (z_f - 1) ** ((3 * n_i - 1) / 3) * (
            1 - (3 * (x_c - x_i) ** 2 * r_i ** (1 - n_i)) / (2 * (z_f + 1)))
    else:
        f_i = 0  # Additive flue function (eq. 12)
    f_s = ((1 + n_i * r_i) / (n_i + 1)) * (0.5 - 0.5 * m_i ** 1.2) ** (n_i + 1) + f_i
    stack_coef = f_s * (convert(outside_air_density * g * infiltration_height_ft, 'pounds/ft/s^2', 'inch_H2O_39F') /
                        (assumed_indoor_temp + 460)) ** n_i  # inH2O^n/R^n
    inf_Cs = stack_coef * convert(1, 'inch_H2O_39F/degR', 'Pa/K') ** n_i

    # Calculate wind coefficient
    if construction['Foundation Type'] == 'Crawlspace':
        if x_i > 1 - 2 * y_i:
            # Critical floor to ceiling difference above which
            # f_w does not change (eq. 25)
            x_i = 1 - 2 * y_i

        # Redefined R for wind calculations for houses with crawlspaces (eq. 21)
        R_x = 1 - r_i * (n_i / 2 + 0.2)
        # Redefined Y for wind calculations for houses with crawlspaces (eq. 22)
        Y_x = 1 - y_i / 4
        # Used to calculate X_x (eq. 24)
        X_s = (1 - r_i) / 5 - 1.5 * y_i
        # Redefined X for wind calculations for houses with crawlspaces (eq. 23)
        X_x = 1 - (((x_i - X_s) / (2 - r_i)) ** 2) ** 0.75
        # Wind factor (eq. 20)
        f_w = 0.19 * (2 - n_i) * X_x * R_x * Y_x
    else:
        J_i = (x_i + r_i + 2 * y_i) / 2
        f_w = 0.19 * (2 - n_i) * (1 - ((x_i + r_i) / 2) ** (1.5 - y_i)) - y_i / 4 * (J_i - 2 * y_i * J_i ** 4)
    wind_coef = f_w * convert(outside_air_density / 2, 'pound/ft^3', 'inch_H2O_39F/mph^2') ** n_i  # inH2O^n/mph^2n
    inf_Cw = wind_coef * convert(1, 'inch_H2O_39F/mph^2', 'Pa/(m/s)^2') ** n_i

    return {
        'Infiltration Method': 'ASHRAE',
        'inf_f_t': f_t,      # can compare directly to f_t in infil_program in idf file
        'inf_sft': inf_sft,  # can compare directly to sft in infil_program in idf file
        'inf_n_i': n_i,      # can compare directly to n in infil_program in idf file
        'inf_c': inf_c,      # can compare directly to c in infil_program in idf file
        'inf_Cs': inf_Cs,    # can compare directly to Cs in infil_program in idf file
        'inf_Cw': inf_Cw,    # can compare directly to Cw in infil_program in idf file
    }


def calculate_ela_coefficients(zone_name, zone_height, zone_height_above_ground=0):
    # ELA/SLA parameters taken from ResStock
    # see airflow.rb, calc_wind_stack_coeffs

    # defaults and weather station constants
    g = 32.174  # in ft / s^2
    default_indoor_temp = 73.5  # degF
    terrain_multiplier = 1.0
    terrain_exponent = 0.15
    height = 32.8

    # site constants (TODO: get from HPXML)
    site_terrain_multiplier = 0.85  # assumed rural for now
    site_terrain_exponent = 0.20  # assumed rural for now
    shielding_coeff = 0.5 / 3  # assumed 'normal' for now. Using s_g_shielding_coef from ResStock
    neutral_level = 0.5  # same for all zones
    if zone_name == 'Attic':
        hor_lk_frac = 0.75
    elif zone_name == 'Garage':
        hor_lk_frac = 0.4
    else:
        # assert zone_name == 'Indoor'
        hor_lk_frac = 0.0
    zone_height = convert(zone_height, 'm', 'ft')
    zone_height_above_ground = convert(zone_height_above_ground, 'm', 'ft')

    # TODO: compare f_t calc with ASHRAE formula, merge if possible
    f_t = site_terrain_multiplier * ((zone_height + zone_height_above_ground) / height) ** site_terrain_exponent / (
        terrain_multiplier * (height / height) ** terrain_exponent)
    f_stack = 2.0 / 3.0 * (1 + hor_lk_frac / 2.0) * (2.0 * neutral_level * (1.0 - neutral_level)
                                                     )**0.5 / (neutral_level**0.5 + (1.0 - neutral_level)**0.5)
    stack_coeff = f_stack ** 2.0 * g * zone_height / (default_indoor_temp + 460.0)
    stack_coeff = convert(stack_coeff, 'ft^2/(s^2*degR)', 'L^2/(s^2*cm^4*K)')

    f_wind = shielding_coeff * (1.0 - hor_lk_frac)**(1.0 / 3.0) * f_t
    wind_coeff = f_wind ** 2.0
    wind_coeff = wind_coeff / 100  # unit conversion?

    # TODO: note: stack coeff units are wrong
    return {
        'ELA stack coefficient (L/s/cm^4/K)': stack_coeff,  # 0.000105911,
        'ELA wind coefficient (L/s/cm^4/(m/s))': wind_coeff,  # 0.000142748,
    }
