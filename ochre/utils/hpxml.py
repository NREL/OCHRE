import math
import pandas as pd

from ochre.utils import OCHREException, convert, nested_update, import_hpxml
from ochre.utils.units import pitch2deg
import ochre.utils.envelope as utils_envelope

# List of variables and functions for loading and parsing HPXML files


ZONE_NAME_OPTIONS = {
    'Indoor': ['conditioned space'],
    'Foundation': ['crawlspace', 'basement', 'finishedbasement', 'basement - conditioned', 'basement - unconditioned',
                   'crawlspace - vented', 'crawlspace - unvented'],
    'Garage': ['garage'],
    'Attic': ['unfinishedattic', 'attic - vented', 'attic - unvented'],
    'Outdoor': ['exterior', 'outside', 'other exterior'],
    'Ground': ['ground'],
    'Adjacent': ['other'],
}

# fraction of conditioned floor area used for furniture area
ZONE_FURNITURE_AREA_FRACTIONS = {
    'Indoor': 0.4,
    'Foundation': 0.4,
    'Garage': 0.1,
    'Attic': 0,  # Not adding Attic Furniture
}

MEL_NAMES = {
    'TV other': 'TV',
    'other': 'MELs',
    'well pump': 'Well Pump',
    'electric vehicle charging': 'Electric Vehicle',  # gets converted to EV equipment
    'grill': 'Gas Grill',
    'fireplace': 'Gas Fireplace',
    'lighting': 'Gas Lighting',
}

# FUTURE: update roughness by material
# ROUGHNESS_BY_FINISH_TYPE = {
#     'asphalt or fiberglass shingles': 1.67,
#     'metal surfacing': 1.13,
#     'slate or tile shingles': 1.67,
#     'wood shingles or shakes': 1.67,
#     'aluminum siding': 1.13,
#     'asbestos siding': 1.67,
#     'brick veneer': 1.67,
#     'composite shingle siding': 1.67,
#     'fiber cement siding': 1.52,
#     'stucco': 2.17,
#     'vinyl siding': 1.67,
#     'wood siding': 1.13,
# }


def parse_zone_name(hpxml_name):
    if hpxml_name is None:
        return None

    # Look through all zone name options, return OCHRE name
    for ochre_name, old_name_options in ZONE_NAME_OPTIONS.items():
        if hpxml_name == ochre_name:
            return ochre_name
        for option in old_name_options:
            if hpxml_name == option:
                return ochre_name

    # Note, multifamily zones (e.g. 'other housing unit') all include 'other' in the name
    if 'other' in hpxml_name:
        return 'Adjacent'

    if hpxml_name is not None:
        print(f'WARNING: Cannot parse zone name "{hpxml_name}". Setting zone to None.')

    return None


def get_boundaries_by_zones(boundaries, default_int_zone='Indoor', default_ext_zone='Outdoor'):
    bd_by_zones = {}
    for name, boundary in boundaries.items():
        interior = parse_zone_name(boundary.get('InteriorAdjacentTo'))
        exterior = parse_zone_name(boundary.get('ExteriorAdjacentTo'))
        if interior is None:
            interior = default_int_zone
        if exterior is None:
            exterior = default_ext_zone
        elif exterior == 'Adjacent':
            exterior = interior

        zones = (interior, exterior)
        if zones not in bd_by_zones:
            bd_by_zones[zones] = {}
        boundary['Interior Zone'] = interior
        boundary['Exterior Zone'] = exterior
        bd_by_zones[zones][name] = boundary

    return bd_by_zones


def get_boundaries_by_wall(boundaries, ext_walls, gar_walls, attic_walls, adj_walls):
    # for windows and doors, determine interior zone based on attached wall zone
    ext_bd, gar_bd, attic_bd = {}, {}, {}
    for name, boundary in boundaries.items():
        boundary['Exterior Zone'] = 'Outdoor'
        wall = boundary['AttachedToWall']['@idref']
        if wall in ext_walls:
            boundary['Interior Zone'] = 'Indoor'
            ext_bd[name] = boundary
            ext_walls[wall]['Area'] -= boundary['Area']
        elif wall in gar_walls:
            boundary['Interior Zone'] = 'Garage'
            gar_bd[name] = boundary
            gar_walls[wall]['Area'] -= boundary['Area']
        elif wall in attic_walls:
            boundary['Interior Zone'] = 'Attic'
            attic_bd[name] = boundary
            attic_walls[wall]['Area'] -= boundary['Area']
        elif wall in adj_walls:
            # skip doors on adjacent (adiabatic) walls
            pass
        else:
            raise OCHREException(f'Unknown attached wall for {name}: {wall}')
    
    return ext_bd, gar_bd, attic_bd

def parse_hpxml_surface(bd_name, bd_data):
    out = {
        'Interior Zone': bd_data['Interior Zone'],
        'Exterior Zone': bd_data['Exterior Zone'],
        'Area (m^2)': convert(bd_data['Area'], 'ft^2', 'm^2'),
    }

    # Add azimuth if it exists
    azimuth = bd_data.get('Azimuth')
    if azimuth is not None:
        out['Azimuth (deg)'] = azimuth

    # Add R value if it exists
    r_value = bd_data.get('Insulation', {}).get('AssemblyEffectiveRValue', bd_data.get('RValue'))
    if r_value:
        out['Boundary R Value'] = r_value

    # Exterior boundary properties (roofs, walls, and rim joists). Not used for raised floors
    if bd_data.get('Exterior Zone') == 'Outdoor':
        if 'SolarAbsorptance' in bd_data:
            out['Exterior Solar Absorptivity (-)'] = bd_data['SolarAbsorptance']
        if 'Emittance' in bd_data:
            out['Exterior Emissivity (-)'] = bd_data['Emittance']

    # Boundary-specific properties
    if 'Roof' in bd_name:
        construction_type = 'Pitched' if bd_data.get('Pitch') > 0 else 'Flat'
        if bd_data.get('RadiantBarrier'):
            construction_type += ', Radiant Barrier'
        out.update({
            'Finish Type': bd_data.get('RoofType'),
            'Construction Type': construction_type,
            'Tilt (deg)': pitch2deg(bd_data.get('Pitch')),
            'Radiant Barrier': bd_data.get('RadiantBarrier', False),
        })
    elif 'Floor' in bd_name and bd_data.get('Exterior Zone') == 'Ground':
        out.update({
            'Insulation Details': utils_envelope.get_slab_insulation(bd_data, bd_name),
        })
    elif bd_name == 'Foundation Wall':
        out.update({
            'Insulation Details': utils_envelope.get_fnd_wall_insulation(bd_data),
            'Height (m)': convert(bd_data.get('Height'), 'ft', 'm'),
        })
    elif 'Wall' in bd_name and bd_data.get('Exterior Zone') == 'Outdoor':
        # not used for all walls
        out.update({
            'Finish Type': bd_data.get('Siding'),
            'Construction Type': bd_data.get('WallType'),
        })
    elif bd_name in ['Adjacent Wall', 'Garage Attached Wall']:
        out.update({
            'Construction Type': bd_data.get('WallType'),
        })
    elif bd_name == 'Rim Joist':
        out.update({
            'Finish Type': bd_data.get('Siding'),
        })
    elif bd_name == 'Window':
        out.update({
            'U Factor (W/m^2-K)': bd_data.get('UFactor',),
            'SHGC (-)': bd_data.get('SHGC'),
            'Shading Fraction (-)': bd_data.get('InteriorShading', {}).get('SummerShadingCoefficient'),
        })

    return out


def parse_hpxml_boundary(bd_name, bd_data):
    # transform HPXML boundary info into OCHRE names and units
    bd_data = [parse_hpxml_surface(bd_name, bd_dict) for bd_dict in bd_data.values()]

    # combine boundary data. keep area and azimuth lists, and check that all other parameters are equal
    out = bd_data[0]
    for key, val in out.items():
        if key in ['Area (m^2)', 'Azimuth (deg)'] and val is not None:
            out[key] = [bd[key] for bd in bd_data]
        else:
            if not all([bd[key] == val for bd in bd_data]):
                raise OCHREException(f'{bd_name} {key} values in HPXML are not all equal: {[bd[key] for bd in bd_data]}')

    # Consolidate areas and azimuths (e.g. if multiple windows per wall)
    if 'Area (m^2)' in out and out.get('Azimuth (deg)') is not None:
        areas, azs = out['Area (m^2)'], out['Azimuth (deg)']
        az_unique = set(azs)
        if len(az_unique) < len(azs):
            out['Area (m^2)'] = [sum([area for area, az in zip(areas, azs) if az == az2]) for az2 in az_unique]
            out['Azimuth (deg)'] = list(az_unique)

    return out


def parse_hpxml_boundaries(hpxml, return_boundary_dicts=False, **kwargs):
    enclosure = hpxml['Enclosure']
    construction = hpxml['BuildingSummary']['BuildingConstruction']

    # Get variables for calculating zone volumes, check geometry assumptions
    conditioned_floor_area = convert(construction['ConditionedFloorArea'], 'ft^2', 'm^2')  # indoor + foundation
    conditioned_volume = convert(construction['ConditionedBuildingVolume'], 'ft^3', 'm^3')
    # aspect_ratio = kwargs.get('aspect ratio', 1.8)  # assumes length (sides of house) is shorter than width
    # house_length = (floor_area / aspect_ratio) ** 0.5
    # house_width = house_length * aspect_ratio
    ceiling_height = conditioned_volume / conditioned_floor_area
    if 'AverageCeilingHeight' in construction:
        assert abs(convert(construction['AverageCeilingHeight'], 'ft', 'm') - ceiling_height) < 0.1

    # Get number of bedrooms and bathrooms
    n_beds = construction['NumberofBedrooms']
    n_baths = construction.get('NumberofBathrooms', n_beds / 2 + 0.5)

    # Get foundation type
    # TODO: if fnd wall insulation doesn't depend on foundation type, move this to parse_hpxml_zones
    total_floors = construction['NumberofConditionedFloors']
    indoor_floors = construction.get('NumberofConditionedFloorsAboveGrade', total_floors)
    assert indoor_floors >= 1
    foundations = list(enclosure.get('Foundations', {}).values())
    if not foundations:
        foundation = None
        foundation_name = None
    else:
        assert len(foundations) == 1
        foundation = foundations[0]
        foundation_type = foundation.get('FoundationType')
        if isinstance(foundation_type, dict):
            foundation_name = list(foundation_type.keys())[0]  # Crawlspace or Basement
            if foundation_name == 'Basement':
                if total_floors > indoor_floors:
                    foundation_name = 'Finished Basement'
                else:
                    foundation_name = 'Unfinished Basement'
        elif foundation_type in ['SlabOnGrade', 'Ambient', 'AboveApartment', None]:
            # no foundation zone for slab or pier+beam foundations
            foundation_name = None
        else:
            raise OCHREException(f'Unknown foundation type: {foundation_type}')

    construction_dict = {
        'Number of Bedrooms (-)': n_beds,
        'Number of Bathrooms (-)': n_baths,
        'House Type': construction['ResidentialFacilityType'],
        'Total Floors': total_floors,
        'Indoor Floors': indoor_floors,
        'Conditioned Floor Area (m^2)': conditioned_floor_area,
        'Conditioned Volume (m^3)': conditioned_volume,
        'Ceiling Height (m)': ceiling_height,
        'Foundation Type': foundation_name,
    }

    # Get all walls
    all_walls = get_boundaries_by_zones(enclosure.get('Walls', {}))
    ext_walls = all_walls.pop(('Indoor', 'Outdoor'), {})
    attic_walls = all_walls.pop(('Attic', 'Outdoor'), {})
    gar_walls = all_walls.pop(('Garage', 'Outdoor'), {})
    attached_walls = all_walls.pop(('Indoor', 'Garage'), {})
    adj_walls = all_walls.pop(('Indoor', 'Indoor'), {})
    adj_attic_walls = all_walls.pop(('Attic', 'Attic'), {})
    adj_gar_walls = all_walls.pop(('Garage', 'Garage'), {})
    assert not all_walls  # verifies that all boundaries are accounted for

    # Get foundation walls
    all_fnd_walls = get_boundaries_by_zones(enclosure.get('FoundationWalls', {}))
    fnd_walls = all_fnd_walls.pop(('Foundation', 'Ground'), {})
    # fnd_walls_above = all_fnd_walls.pop(('Foundation', 'Outdoor'), {})  # not used, see rim joist
    adj_fnd_wall = all_fnd_walls.pop(('Foundation', 'Foundation'), {})
    assert not all_fnd_walls  # verifies that all boundaries are accounted for

    # Get ceilings and non-slab floors, i.e. FrameFloors
    all_ceilings = get_boundaries_by_zones({**enclosure.get('Floors', {}), **enclosure.get('FrameFloors', {})})
    ceilings = all_ceilings.pop(('Indoor', 'Attic'), {})
    fnd_ceilings = all_ceilings.pop(('Indoor', 'Foundation'), {})
    raised_floors = all_ceilings.pop(('Indoor', 'Outdoor'), {})
    garage_ceilings = all_ceilings.pop(('Garage', 'Attic'), {})
    garage_ceilings_int = all_ceilings.pop(('Indoor', 'Garage'), {})
    adj_ceilings = all_ceilings.pop(('Indoor', 'Indoor'), {})
    assert not all_ceilings  # verifies that all boundaries are accounted for

    # Separate adjacent floors and ceilings (may have different insulation levels)
    adj_floors = {key: val for key, val in adj_ceilings.items() if val.get('FloorOrCeiling') == 'floor'}
    for key in adj_floors:
        adj_ceilings.pop(key)

    # Get floors (slabs in HPXML)
    all_slabs = get_boundaries_by_zones(enclosure.get('Slabs', {}), default_ext_zone='Ground')
    floors = all_slabs.pop(('Indoor', 'Ground'), {})
    fnd_floors = all_slabs.pop(('Foundation', 'Ground'), {})
    garage_floors = all_slabs.pop(('Garage', 'Ground'), {})
    assert not all_slabs  # verifies that all boundaries are accounted for

    # Get roofs
    all_roofs = get_boundaries_by_zones(enclosure.get('Roofs', {}))
    roofs = all_roofs.pop(('Indoor', 'Outdoor'), {})  # Note: not necessarily flat
    attic_roofs = all_roofs.pop(('Attic', 'Outdoor'), {})
    garage_roofs = all_roofs.pop(('Garage', 'Outdoor'), {})
    assert not all_roofs  # verifies that all boundaries are accounted for

    # Get rim joists (outdoor to foundation only)
    all_rim_joists = get_boundaries_by_zones(enclosure.get('RimJoists', {}))
    rim_joists = all_rim_joists.pop(('Foundation', 'Outdoor'), {})
    adj_rim_joists = all_rim_joists.pop(('Foundation', 'Foundation'), {})
    assert not all_rim_joists  # verifies that all boundaries are accounted for

    # Get windows (only accepts windows to indoor zone for now), and subtract wall area
    all_windows = enclosure.get('Windows', {})
    windows, gar_windows, attic_windows = get_boundaries_by_wall(
        all_windows, ext_walls, gar_walls, attic_walls, adj_walls
    )
    assert not gar_windows
    assert not attic_windows
    
    # Get doors (only accepts doors to indoor zone and garage), and subtract wall area
    all_doors = enclosure.get('Doors', {})
    doors, gar_doors, attic_doors = get_boundaries_by_wall(
        all_doors, ext_walls, gar_walls, attic_walls, adj_walls
    )
    assert not attic_doors

    boundaries = {
        'Exterior Wall': ext_walls,
        'Attic Wall': attic_walls,
        'Garage Attached Wall': attached_walls,
        'Garage Wall': gar_walls,
        'Adjacent Wall': adj_walls,
        'Foundation Wall': fnd_walls,
        # 'Foundation Above-ground Wall': fnd_walls_above,
        'Adjacent Attic Wall': adj_attic_walls,
        'Adjacent Garage Wall': adj_gar_walls,
        'Adjacent Foundation Wall': adj_fnd_wall,
        'Attic Floor': ceilings,
        'Foundation Ceiling': fnd_ceilings,
        'Garage Ceiling': garage_ceilings,
        'Garage Interior Ceiling': garage_ceilings_int,
        'Adjacent Ceiling': adj_ceilings,
        'Adjacent Floor': adj_floors,
        'Floor': floors,
        'Foundation Floor': fnd_floors,
        'Garage Floor': garage_floors,
        'Raised Floor': raised_floors,
        'Attic Roof': attic_roofs,
        'Roof': roofs,
        'Garage Roof': garage_roofs,
        'Rim Joist': rim_joists,
        'Adjacent Rim Joist': adj_rim_joists,
        'Window': windows,
        'Door': doors,
        'Garage Door': gar_doors,

    }
    boundaries = {key: val for key, val in boundaries.items() if len(val)}  # remove empty boundaries
    if return_boundary_dicts:
        return boundaries, construction_dict

    # update boundary properties
    for bd_name, bd_data in boundaries.items():
        boundaries[bd_name] = parse_hpxml_boundary(bd_name, bd_data)

        # Add Construction Type to foundation walls - used for getting insulation
        if 'Foundation Wall' in bd_name:
            boundaries[bd_name]['Construction Type'] = foundation_name

        # Add siding to garage attached walls - taken from exterior walls
        if bd_name == 'Garage Attached Wall':
            boundaries[bd_name]['Finish Type'] = boundaries['Exterior Wall']['Finish Type']

    # Get main floor area - should have only 1 main floor boundary option
    main_floor_options = ['Floor', 'Foundation Floor', 'Raised Floor', 'Adjacent Floor']
    main_floor_areas = [area for floor_option in main_floor_options
                        for area in boundaries.get(floor_option, {}).get('Area (m^2)', [])]
    if len(main_floor_areas) == 1:
        first_floor_area = main_floor_areas[0]  # area of first (lowest above grade) floor. Excludes garage
    else:
        raise OCHREException(f'Unable to parse multiple floor areas: {main_floor_areas}')

    # Get attic and top floor area - should have only 1 attic floor boundary option (plus maybe 'Garage Ceiling')
    top_floor_options = ['Attic Floor', 'Roof', 'Adjacent Ceiling']
    top_floor_areas = [area for floor_option in top_floor_options
                        for area in boundaries.get(floor_option, {}).get('Area (m^2)', [])]
    if len(top_floor_areas) == 1:
        top_floor_area = top_floor_areas[0]  # area of first (lowest above grade) floor. Excludes garage
    else:
        raise OCHREException(f'Unable to parse multiple attic floor areas: {top_floor_areas}')
    attic_floor_area = top_floor_area + sum(boundaries.get('Garage Ceiling', {}).get('Area (m^2)', []))

    if 'Garage Floor' in boundaries:
        # get garage area and wall height
        garage_floor_area = boundaries['Garage Floor']['Area (m^2)'][0]
        garage_wall_ar = (boundaries['Garage Wall']['Area (m^2)'] + 
                          boundaries.get('Adjacent Garage Wall', {}).get('Area (m^2)', []))
        garage_wall_az = (boundaries['Garage Wall']['Azimuth (deg)'] + 
                          boundaries.get('Adjacent Garage Wall', {}).get('Azimuth (deg)', []))
        garage_wall_az = [az % 180 for az in garage_wall_az]
        a1, a2 = tuple([max([ar for ar, az in zip(garage_wall_ar, garage_wall_az)
                             if az == azimuth]) for azimuth in set(garage_wall_az)])
        garage_wall_height = (a1 * a2 / garage_floor_area) ** 0.5

        attached_wall_areas = boundaries['Garage Attached Wall']['Area (m^2)']
        n_walls = len(attached_wall_areas)
        if n_walls == 1:
            garage_area_in_main = 0
        elif n_walls == 2:
            # usually for 1-story home or garage that is fully under the 2nd story
            a1, a2 = tuple(attached_wall_areas)
            garage_area_in_main = a1 * a2 / garage_wall_height ** 2
        elif n_walls == 3:
            # usually for 2-story home with protruding garage. 2 regular walls + 1 gable wall
            attached_wall_azimuths = [az % 180 for az in boundaries['Garage Attached Wall']['Azimuth (deg)']]
            a1, a2 = tuple([max([ar for ar, az in zip(attached_wall_areas, attached_wall_azimuths)
                                    if az == azimuth]) for azimuth in set(attached_wall_azimuths)])
            garage_area_in_main = a1 * a2 / garage_wall_height ** 2
        else:
            raise OCHREException('Invalid geometry. Cannot parse more than 3 garage walls.')
        assert 0 <= garage_area_in_main / garage_floor_area < 1.001  # should be close to 50% for ResStock cases
    else:
        garage_floor_area = 0
        garage_area_in_main = 0

    indoor_floor_area = conditioned_floor_area - first_floor_area * (total_floors - indoor_floors)
    indoor_floor_check = first_floor_area + top_floor_area * (indoor_floors - 1)
    if abs(indoor_floor_check - indoor_floor_area) > 10:
        print(f'WARNING: Indoor floor area calculations do not agree: '
              f'{indoor_floor_area} m^2 and {indoor_floor_check} m^2')
    construction_dict.update({
        'First Floor Area (m^2)': first_floor_area,
        'Attic Floor Area (m^2)': attic_floor_area,
        'Garage Floor Area (m^2)': garage_floor_area,
        'Garage Protruded Area (m^2)': garage_floor_area - garage_area_in_main,  # area not in main rectangle
        'Indoor Floor Area (m^2)': indoor_floor_area,
    })

    return boundaries, construction_dict


def parse_indoor_infiltration(hpxml, construction, equipment):
    # get infiltration data from HPXML
    enclosure = hpxml['Enclosure']
    indoor_infiltration = enclosure['AirInfiltration']
    inf = indoor_infiltration['AirInfiltrationMeasurement']
    site = hpxml['BuildingSummary']['Site']
    
    # Check if house has a flue or chimney
    has_flue_or_chimney = indoor_infiltration.get('extension', {}).get('HasFlueOrChimneyInConditionedSpace')
    if has_flue_or_chimney is None:
        # TODO: equipment has to be in conditioned space (indoor or conditioned basement)
        heater = equipment.get('HVAC Heating', {})
        gas_heater = heater.get('Fuel', 'Electricity') != 'Electricity' and (1 / heater.get('EIR (-)', 1)) < 0.89
        wh = equipment.get('Water Heating', {})
        gas_wh = wh.get('Fuel', 'Electricity') != 'Electricity' and wh.get('Energy Factor (-)', 1) < 0.63
        has_flue_or_chimney = gas_heater or gas_wh

    return utils_envelope.calculate_ashrae_infiltration_params(inf, construction, site, has_flue_or_chimney)


def parse_hpxml_zones(hpxml, boundaries, construction):
    enclosure = hpxml['Enclosure']
    first_floor_area = construction['First Floor Area (m^2)']
    attic_floor_area = construction['Attic Floor Area (m^2)']
    garage_floor_area = construction['Garage Floor Area (m^2)']
    garage_protruded_area = construction['Garage Protruded Area (m^2)']
    indoor_floor_area = construction['Indoor Floor Area (m^2)']
    ceiling_height = construction['Ceiling Height (m)']
    building_height = ceiling_height * construction['Indoor Floors']
    has_garage = garage_floor_area > 0

    if 'Attic Roof' in boundaries:
        roof_tilt = convert(boundaries['Attic Roof']['Tilt (deg)'], 'deg', 'rad')
    else:
        roof_tilt = None
    if 'Garage Roof' in boundaries:
        garage_tilt = convert(boundaries['Garage Roof']['Tilt (deg)'], 'deg', 'rad')
    else:
        garage_tilt = roof_tilt

    # Indoor ventilation parameters - Whole ventilation fans only
    # Note: Indoor infiltration parameters depend on equipment, added in add_indoor_infiltration
    nat_ventilation_params = utils_envelope.calculate_ela_coefficients('Indoor', building_height)
    zones = {
        'Indoor': {
            'Zone Area (m^2)': indoor_floor_area,
            'Volume (m^3)': indoor_floor_area * ceiling_height,
            **nat_ventilation_params,
        }}
    vent_fans = hpxml['Systems'].get('MechanicalVentilation', {}).get('VentilationFans', {})
    vent_fans = {key: val for key, val in vent_fans.items() if val.get('UsedForWholeBuildingVentilation', False)}
    if vent_fans:
        assert len(vent_fans) == 1
        vent_fan = list(vent_fans.values())[0]
        fan_type = vent_fan['FanType']
        balanced = fan_type in ['energy recovery ventilator', 'heat recovery ventilator', 'balanced']
        if 'recovery ventilator' in fan_type:
            assert 'SensibleRecoveryEfficiency' in vent_fan
        sensible_recovery = vent_fan.get('SensibleRecoveryEfficiency', 0)
        latent_recovery = vent_fan.get('TotalRecoveryEfficiency', 0) - sensible_recovery
        zones['Indoor'].update({
            'Ventilation Rate (cfm)': vent_fan['RatedFlowRate'],
            'Ventilation Type': fan_type,
            'Balanced Ventilation': balanced,
            'Sensible Recovery Efficiency (-)': sensible_recovery,
            'Latent Recovery Efficiency (-)': latent_recovery,
        })

    # Add Attic Zone
    # Note: pitched roofs connected to Indoor zone have a conditioned attic in HPXML, not using 'Attics'
    # attics = [a for a in enclosure.get('Attics', {}).values()
    #         if a.get('AtticType') not in ['FlatRoof', 'BelowApartment', None]]
    if 'Attic Roof' in boundaries:
        attics = list(enclosure.get('Attics', {}).values())
        assert len(attics) == 1
        attic = attics[0]

        # Get gable wall areas for attic and (possibly) garage
        attic_wall_areas = (boundaries.get('Attic Wall', {}).get('Area (m^2)', []) +
                            boundaries.get('Adjacent Attic Wall', {}).get('Area (m^2)', []))
        if len(attic_wall_areas) == 2:
            # standard gable roof with attic
            assert abs(attic_wall_areas[1] - attic_wall_areas[0]) < 0.2  # computational errors possible
            attic_gable_area = attic_wall_areas[0]
            third_gable_area = 0
        elif has_garage and len(attic_wall_areas) == 3:
            # 2 attic gables plus 1 garage gable, garage gable has area that is 'more different'
            attic_gable_area = attic_wall_areas[1]
            low, med, high = tuple(sorted(attic_wall_areas))
            third_gable_area = low if med - low > high - med else high
        else:
            raise OCHREException('Unable to calculate attic area, likely an issue with gable walls.')

        # Get attic properties
        # tan(roof_tilt) = height / (width / 2)
        attic_height = (attic_gable_area * math.tan(roof_tilt)) ** 0.5
        if third_gable_area > 0:
            # assumes a combined attic, add volume over garage and in between
            square_area = attic_floor_area - garage_protruded_area
            garage_height = (third_gable_area * math.tan(garage_tilt)) ** 0.5
            garage_width = 2 * third_gable_area / garage_height
            # garage_length = garage_floor_area / garage_width
            garage_depth_in_house = garage_height * math.tan(roof_tilt)  # length from attached wall to roof connection point
            attic_volume = (1 / 2 * square_area * attic_height +                           # prism over square
                            1 / 2 * garage_protruded_area * garage_height +                # prism over garage
                            1 / 6 * garage_width * garage_depth_in_house * garage_height)  # pyramid between prisms
        else:
            attic_volume = 1 / 2 * attic_floor_area * attic_height  # volume = 1/2 l*w*h

        vented = attic.get('AtticType', {}).get('Attic', {}).get('Vented', True)
        zones['Attic'] = {
            'Zone Area (m^2)': attic_floor_area,
            'Volume (m^3)': attic_volume,
            'Vented': vented,
        }

        # Add attic infiltration
        inf_units = attic.get('VentilationRate', {}).get('UnitofMeasure')
        if inf_units == 'ACHnatural':
            attic_ach = attic['VentilationRate']['Value']
            zones['Attic'].update({
                'Infiltration Method': 'ACH',
                'Air Changes (1/hour)': attic_ach,
            })
        elif inf_units == 'SLA':
            # Update ELA based on attic area
            attic_ela = attic['VentilationRate']['Value'] * attic_floor_area * 1e4  # m^2 to cm^2
            zones['Attic'].update({
                'Infiltration Method': 'ELA',
                'ELA (cm^2)': attic_ela,
                **utils_envelope.calculate_ela_coefficients('Attic', attic_height, building_height)
            })
        elif not vented and inf_units is None:
            zones['Attic'].update({
                'Infiltration Method': 'ACH',
                'Air Changes (1/hour)': 0.1,
            })
        else:
            raise OCHREException(f'Cannot parse Attic infiltration rate from properties: {attic}')

    # Add foundation zone
    if construction['Foundation Type'] is not None:
        foundations = list(enclosure.get('Foundations', {}).values())
        assert len(foundations) == 1
        foundation = foundations[0]

        # Get foundation volume
        foundation_height = boundaries['Foundation Wall']['Height (m)']
        foundation_volume = first_floor_area * foundation_height
        zones['Foundation'] = {
            'Zone Type': construction['Foundation Type'],
            'Zone Area (m^2)': first_floor_area,
            'Volume (m^3)': foundation_volume,
        }

        # Get foundation infiltration parameters (takes ACH or SLA)
        if construction['Foundation Type'] == 'Crawlspace':
            zones['Foundation']['Vented'] = foundation.get('FoundationType')['Crawlspace'].get('Vented', True)
        else:
            zones['Foundation']['Vented'] = False

        inf_units = foundation.get('VentilationRate', {}).get('UnitofMeasure')
        inf_value = foundation.get('VentilationRate', {}).get('Value')
        if inf_units == 'ACHnatural':
            zones['Foundation'].update({
                'Infiltration Method': 'ACH',
                'Air Changes (1/hour)': inf_value,
            })
        elif inf_units == 'SLA':
            fnd_ela = inf_value * convert(first_floor_area, 'm^2', 'cm^2')
            zones['Foundation'].update({
                'Infiltration Method': 'ELA',
                'ELA (cm^2)': fnd_ela,
                **utils_envelope.calculate_ela_coefficients('Foundation', foundation_height)
            })
        elif zones['Foundation']['Vented']:
            inf_value = 2  # taken from ResStock, options_lookup.tsv file
            zones['Foundation'].update({
                'Infiltration Method': 'ACH',
                'Air Changes (1/hour)': inf_value,
            })
        else:
            zones['Foundation'].update({
                'Infiltration Method': None,
            })

    # Add garage zone
    if has_garage:
        # Note: garage roof space is included in attic for 1-story homes
        # FUTURE: convert to ELA? Can use 1sqft at 4.9 SLA = 1.2854145 CFM50, will need to convert ACH50 to ACH
        garage_volume = garage_floor_area * ceiling_height  # excluding garage attic space
        garage_roof_areas = boundaries.get('Garage Roof', {}).get('Area (m^2)', [])
        if len(garage_roof_areas) > 0:
            # add garage roof space - should work for triangle roof or gable roof
            garage_volume += 1/2 * math.atan(garage_tilt) * garage_protruded_area

        indoor_infiltration = list(enclosure['AirInfiltration'].values())[0]
        indoor_ach = indoor_infiltration['BuildingAirLeakage']['AirLeakage']  # ACH50
        zones['Garage'] = {
            'Zone Area (m^2)': garage_floor_area,
            'Volume (m^3)': garage_volume,
            'Infiltration Method': 'ACH',
            'Air Changes (1/hour)': indoor_ach,
        }

    return zones


def add_interior_boundaries(hpxml, boundaries, zones):
    enclosure_extn = hpxml['Enclosure'].get('extension', {})

    # Add interior wall boundary, assumes interior wall area is equal to floor area
    int_walls = enclosure_extn.get('PartitionWallMass')
    if int_walls:
        # For now, require the defaults
        assert int_walls.get('AreaFraction', 1.0) == 1.0
        assert int_walls.get('InteriorFinish', {}).get('Type', 'gypsum board') == 'gypsum board'
        assert int_walls.get('InteriorFinish', {}).get('Thickness', 0.5) == 0.5
    boundaries['Interior Wall'] = {
        'Area (m^2)': [zones['Indoor']['Zone Area (m^2)']],
        'Interior Zone': 'Indoor',
        'Exterior Zone': 'Indoor',
        'Insulation Level': 'Standard'
    }

    # Add zone furniture - automatically added if the zone exists
    indoor_fraction = enclosure_extn.get('FurnitureMass', {}).get('AreaFraction')
    for zone_name, zone in zones.items():
        if zone_name == 'Indoor' and indoor_fraction is not None:
            fraction = indoor_fraction
        else:
            fraction = ZONE_FURNITURE_AREA_FRACTIONS[zone_name]
        area = zone.get('Zone Area (m^2)', 0)
        if area * fraction > 0:
            boundaries[f'{zone_name} Furniture'] = {
                'Area (m^2)': [area * fraction],
                'Interior Zone': zone_name,
                'Exterior Zone': zone_name,
                'Insulation Level': 'Standard',
            }

    return boundaries


def parse_hpxml_envelope(hpxml, occupancy, **house_args):
    # Parse envelope properties
    # TODO: add option to modify envelope propoerties from house_args? Similar to Equipment
    boundaries, construction = parse_hpxml_boundaries(hpxml, **house_args)
    zones = parse_hpxml_zones(hpxml, boundaries, construction)
    boundaries = add_interior_boundaries(hpxml, boundaries, zones)

    # Get adjusted number of bedrooms based on bedrooms and occupants
    house_type = construction['House Type']
    n_occupants = occupancy['Number of Occupants (-)']
    if house_type in ['single-family detached', 'manufactured home']:
        n_bedrooms_adj = max(-0.68 + 1.09 * n_occupants, 0)
    elif house_type in ['single-family attached', 'apartment unit']:
        n_bedrooms_adj = max(-1.47 + 1.69 * n_occupants, 0)
    else:
        raise OCHREException(f'Unknown house type: {house_type}')
    construction['Number of Bedrooms, Adjusted (-)'] = n_bedrooms_adj

    return boundaries, zones, construction


def add_simple_schedule_params(extension, prefix=''):
    # Get HPXML weekday and weekend hourly schedule fractions and month multipliers
    # for details, see https://openstudio-hpxml.readthedocs.io/en/latest/workflow_inputs.html
    if f'{prefix}WeekdayScheduleFractions' not in extension:
        return {}
    else:
        return {
            'weekday_fractions': extension.get(f'{prefix}WeekdayScheduleFractions'),
            'weekend_fractions': extension.get(f'{prefix}WeekendScheduleFractions'),
            'month_multipliers': extension.get(f'{prefix}MonthlyScheduleMultipliers'),
        }


def parse_hpxml_occupancy(hpxml):
    occupants = hpxml['BuildingSummary']['BuildingOccupancy']
    extension = occupants.get('extension', {})
    return {
        'Number of Occupants (-)': occupants['NumberofResidents'],
        **add_simple_schedule_params(extension),
    }


def parse_hvac(hvac_type, hvac_all):
    # Get HVAC HPXML parameters from HVAC Plant or Heat Pump
    system = hvac_all.get('HVACPlant', {}).get(f'{hvac_type}System')
    heat_pump = hvac_all.get('HVACPlant', {}).get('HeatPump')
    if system and heat_pump:
        raise OCHREException(f'HVAC {hvac_type} system and heat pump cannot both be specified.')
    elif not system and not heat_pump:
        return None
    has_heat_pump = bool(heat_pump)
    hvac = heat_pump if has_heat_pump else system

    # Main HVAC parameters
    name = hvac['HeatPumpType'] if has_heat_pump else hvac[f'{hvac_type}SystemType']
    if isinstance(name, dict):
        # HVAC heating has different structure, ignroring pilot light for now
        assert len(name) == 1
        name, data = list(name.items())[0]
        # pilot = data.get('PilotLight', False)
        # pilot_rate = data.get('extension', {}).get('PilotLightBtuh', 500)
    fuel = hvac['HeatPumpFuel'] if has_heat_pump else hvac.get(f'{hvac_type}SystemFuel')
    capacity = convert(hvac[f'{hvac_type}Capacity'], 'Btu/hour', 'W')
    space_fraction = hvac.get(f'Fraction{hvac_type[:-3]}LoadServed', 1.0)
    efficiency = hvac[f'Annual{hvac_type}Efficiency']
    if efficiency['Units'] in ['Percent', 'AFUE']:
        eir = 1 / efficiency['Value']
        if efficiency['Units'] == 'Percent':
            # for reporting only
            efficiency['Value'] *= 100
    elif efficiency['Units'] in ['EER', 'SEER', 'HSPF']:
        eir = 1 / convert(efficiency['Value'], 'Btu/hour', 'W')
    else:
        raise OCHREException(f'Unknown inputs for HVAC {hvac_type} efficiency: {efficiency}')
    efficiency_string = f"{efficiency['Value']} {efficiency['Units']}"

    # Get number of speeds
    speed_options = {
        'single stage': 1,
        'two stage': 2,
        'variable speed': 4,
    }
    if name == 'mini-split':
        number_of_speeds = 4  # MSHP always variable speed
    elif hvac.get('CompressorType') in speed_options:
        number_of_speeds = speed_options[hvac.get('CompressorType')]
    elif convert(1 / eir, 'W', 'Btu/hour') <= 15:
        number_of_speeds = 1  # Single-speed for SEER <= 15
    elif convert(1 / eir, 'W', 'Btu/hour') <= 21:
        number_of_speeds = 2  # Two-speed for 15 < SEER <= 21
    else:
        number_of_speeds = 4  # Variable speed for SEER > 21

    # Get SHR
    is_heater = hvac_type == 'Heating'
    if is_heater:
        shr = None
    elif has_heat_pump:
        shr = hvac.get('CoolingSensibleHeatFraction')
    else:
        shr = hvac.get('SensibleHeatFraction')

    # Get auxiliary power (fans, pumps, etc.) air flow rate
    hvac_ext = hvac.get('extension', {})
    if name == 'Boiler':
        # Note: ResStock assumes 2080 hours/year, see hvac.rb line 1754 (get_default_boiler_eae)
        # see also ANSI/RESNET/ICC 301-2019 Equation 4.4-5
        aux_power = hvac.get('ElectricAuxiliaryEnergy', 0) / 2080 * 1000  # converts kWh/year to W
    elif 'FanPowerWattsPerCFM' in hvac_ext:
        # Note: air flow rate is only used for non-dymanic HVAC models with fans, e.g., furnaces
        # airflow_cfm = hvac_ext.get(f'{hvac_type}AirflowCFM', 0)
        cfm_per_ton = 350 if is_heater else 312
        power_per_cfm = hvac_ext.get('FanPowerWattsPerCFM', 0)
        aux_power = power_per_cfm * cfm_per_ton * convert(capacity, 'W', 'refrigeration_ton')
    else:
        aux_power = hvac_ext.get('FanPowerWatts', 0)

    out = {
        'Equipment Name': name,
        'Fuel': fuel.capitalize(),
        'Capacity (W)': capacity,
        'EIR (-)': eir,
        'Rated Efficiency': efficiency_string,
        'SHR (-)': shr,
        'Conditioned Space Fraction (-)': space_fraction,
        'Number of Speeds (-)': number_of_speeds,
        'Rated Auxiliary Power (W)': aux_power,
    }

    # Get HVAC setpoints, optional
    controls = hvac_all['HVACControl']
    extension = controls.get('extension', {})
    if f'WeekdaySetpointTemps{hvac_type}Season' in extension:
        weekday_setpoints = extension[f'WeekdaySetpointTemps{hvac_type}Season']
        weekend_setpoints = extension[f'WeekendSetpointTemps{hvac_type}Season']
        out.update({
            'Weekday Setpoints (C)': convert(weekday_setpoints, 'degF', 'degC').tolist(),
            'Weekend Setpoints (C)': convert(weekend_setpoints, 'degF', 'degC').tolist(),
        })
    elif f'SetpointTemp{hvac_type}Season' in controls:
        weekday_setpoint = controls[f'SetpointTemp{hvac_type}Season']
        weekend_setpoint = controls[f'SetpointTemp{hvac_type}Season']
        out.update({
            'Weekday Setpoints (C)': [convert(weekday_setpoint, 'degF', 'degC')] * 24,
            'Weekend Setpoints (C)': [convert(weekend_setpoint, 'degF', 'degC')] * 24,
        })

    if has_heat_pump and hvac_type == 'Heating':
        backup_capacity = heat_pump.get('BackupHeatingCapacity', 0)
        backup_fuel = heat_pump.get('BackupSystemFuel')

        if backup_capacity and backup_fuel == 'electricity':
            # assumes efficiency units are in Percent or AFUE
            out.update({
                'Supplemental Heater EIR (-)': 1 / heat_pump.get('BackupAnnualHeatingEfficiency', {}).get('Value'),
                'Supplemental Heater Capacity (W)': convert(backup_capacity, 'Btu/hour', 'W'),
                'Supplemental Heater Cut-in Temperature (C)':
                    convert(heat_pump.get('BackupHeatingSwitchoverTemperature'), 'degF', 'degC'),
            })
        else:
            if backup_capacity:
                print(f'WARNING: Using electric backup heater for ASHP instead of {backup_fuel} equipment')
            out.update({
                'Supplemental Heater Capacity (W)': backup_capacity,
            })

    # Get duct info for calculating DSE
    distribution = hvac_all.get('HVACDistribution', {})
    distribution_type = distribution.get('DistributionSystemType', {})
    air_distribution = distribution_type.get('AirDistribution', {})
    duct_leakage = air_distribution.get('DuctLeakageMeasurement')
    ducts = [d for d in air_distribution.get('Ducts', {}).values()
             if parse_zone_name(d.get('DuctLocation')) not in ['Indoor', None]]

    if f'Annual{hvac_type}DistributionSystemEfficiency' in distribution:
        # Note, ducts are assumed to be in ambient space, DSE losses aren't added to another zone
        out['Ducts'] = {
            'DSE (-)': distribution[f'Annual{hvac_type}DistributionSystemEfficiency'],
            'Zone': None,
        }
    elif duct_leakage is not None and len(ducts):
        # Get parameters to calculate DSE using ASHRAE 152
        # Must be called within HVAC.__init__, as it requires multi-speed parameters
        assert len(ducts) == 2
        assert len(duct_leakage) == 2
        duct_location = ducts[0]['DuctLocation']
        duct_zone = parse_zone_name(duct_location)
        duct_info = {}
        for duct, duct_leakage, duct_type in zip(ducts, duct_leakage, ['supply', 'return']):
            assert duct['DuctType'] == duct_type
            assert duct_leakage['DuctType'] == duct_type
            assert duct_leakage['DuctLeakage']['Units'] == 'Percent' or duct_leakage['DuctLeakage']['Value'] == 0
            duct_info.update({
                f'{duct_type.capitalize()} Leakage (-)': duct_leakage['DuctLeakage']['Value'],
                f'{duct_type.capitalize()} Area (ft^2)': duct['DuctSurfaceArea'],  # * duct['FractionDuctArea'],
                f'{duct_type.capitalize()} R Value': duct['DuctInsulationRValue'],
            })
        out['Ducts'] = {
            'Zone': duct_zone,
            **duct_info,
        }

    return out


def parse_water_heater(water_heater, water, construction, solar_fraction=0):
    # Returns a dictionary of calculated water heater/water tank properties
    # If using EF:
    #   Calculates the U value, UA of the tank and conversion efficiency (eta_c)
    #   based on the Energy Factor and recovery efficiency of the tank
    #   Source: Burch and Erickson 2004 - http://www.nrel.gov/docs/gen/fy04/36035.pdf
    # IF using UEF:
    #   Calculates the U value, UA of the tank and conversion efficiency (eta_c)
    #   based on the Uniform Energy Factor, First Hour Rating, and Recovery Efficiency of the tank
    #   Source: Maguire and Roberts 2020 -
    #   https://www.ashrae.org/file%20library/conferences/specialty%20conferences/2020%20building%20performance/papers/d-bsc20-c039.pdf

    # Inputs from HPXML
    water_heater_type = water_heater['WaterHeaterType']
    is_electric = water_heater['FuelType'] == 'electricity'
    t_set = convert(water_heater.get('HotWaterTemperature', 125), 'degF', 'degC')
    energy_factor = water_heater.get('EnergyFactor')
    uniform_energy_factor = water_heater.get('UniformEnergyFactor')
    n_beds = construction['Number of Bedrooms (-)']
    n_beds_adj = construction['Number of Bedrooms, Adjusted (-)']

    # For tank water heaters only (HPWH does not include some of these)
    volume_gal = water_heater.get('TankVolume')  # in gallons
    height = convert(water_heater.get('TankHeight', 4), 'ft', 'm')  # assumed to be 4 ft
    heating_capacity = water_heater.get('HeatingCapacity')
    first_hour_rating = water_heater.get('FirstHourRating')
    recovery_efficiency = water_heater.get('RecoveryEfficiency')
    tank_jacket_r = water_heater.get('WaterHeaterInsulation', {}).get('Jacket', {}).get('JacketRValue', 0)

    # calculate actual volume from rated volume
    if volume_gal is not None:
        if is_electric:
            volume_gal *= 0.9
        else:
            volume_gal *= 0.95
        volume = convert(volume_gal, 'gallon', 'L')  # in L
    else:
        volume = None

    if energy_factor is None and uniform_energy_factor is None:
        raise OCHREException('Energy Factor or Uniform Energy Factor input required for Water Heater.')

    # calculate UA and eta_c for each water heater type
    if water_heater_type == 'instantaneous water heater':
        if energy_factor is None:
            energy_factor = uniform_energy_factor
            performance_adjustment = water_heater.get('PerformanceAdjustment', 0.94)
        else:
            performance_adjustment = water_heater.get('PerformanceAdjustment', 0.92)
        eta_c = energy_factor * performance_adjustment
        ua = 0.0

    elif water_heater_type == 'heat pump water heater':
        assert is_electric
        eta_c = 1
        # HPWH UA calculation taken from ResStock:
        # https://github.com/NREL/resstock/blob/run/restructure-v3/resources/hpxml-measures/HPXMLtoOpenStudio/resources/waterheater.rb#L765
        if volume_gal <= 58.0:
            ua = 3.6
        elif volume_gal <= 73.0:
            ua = 4.0
        else:
            ua = 4.7

    elif water_heater_type == 'storage water heater':
        density = 8.2938  # lb/gal
        cp = 1.0007  # Btu/lb-F
        t_in = 58.0  # F
        t_env = 67.5  # F

        if energy_factor is not None:
            t = 135.0  # F
            volume_drawn = 64.3  # gal/day
        else:
            t = 125.0  # F
            if first_hour_rating < 18.0:
                volume_drawn = 10.0  # gal
            elif first_hour_rating < 51.0:  # Includes 18 gal up to (but not including) 51
                volume_drawn = 38.0  # gal
            elif first_hour_rating < 75.0:
                volume_drawn = 55.0  # gal
            else:
                volume_drawn = 84.0  # gal
        draw_mass = volume_drawn * density  # lb
        q_load = draw_mass * cp * (t - t_in)  # Btu/day

        # Note: UEF to EF calculation taken from https://www.resnet.us/wp-content/uploads/RESNET-EF-Calculator-2017.xlsx
        if not is_electric:
            if energy_factor is not None:
                ua = (recovery_efficiency / energy_factor - 1.0) / (
                    (t - t_env) * (24.0 / q_load - 1.0 / (heating_capacity * energy_factor)))  # Btu/hr-F
                eta_c = (recovery_efficiency + ua * (
                    t - t_env) / heating_capacity)  # conversion efficiency is supposed to be calculated with initial tank ua
            else:
                ua = ((recovery_efficiency / uniform_energy_factor) - 1.0) / (
                    (t - t_env) * (24.0 / q_load) - ((t - t_env) / (heating_capacity * uniform_energy_factor)))  # Btu/hr-F
                eta_c = recovery_efficiency + ((ua * (
                    t - t_env)) / heating_capacity)  # conversion efficiency is slightly larger than recovery efficiency
                energy_factor = 0.9066 * uniform_energy_factor + 0.0711
        else:
            if energy_factor is not None:
                ua = q_load * (1.0 / energy_factor - 1.0) / ((t - t_env) * 24.0)
            else:
                ua = q_load * (1.0 / uniform_energy_factor - 1.0) / (
                    (24.0 * (t - t_env)) * (0.8 + 0.2 * ((t_in - t_env) / (t - t_env))))
                energy_factor = 2.4029 * uniform_energy_factor - 1.2844
            eta_c = 1.0

    else:
        raise OCHREException(f'Unknown water heater type: {water_heater_type}')

    # Increase insulation from tank jacket (reduces UA)
    if tank_jacket_r:
        jacket_insulation = 5.0  # R5, in F-ft2-hr/Btu
        jacket_thickness = 1 if is_electric and energy_factor < 0.7 else 2  # in inches
        diameter = 2 * convert((volume / 1000 / height / math.pi) ** 0.5, 'm', 'ft')  # in ft
        a_side = math.pi * diameter * convert(height, 'm', 'ft')
        u_pre_skin = 1.0 / (jacket_thickness * jacket_insulation + 1.0 / 1.3 + 1.0 / 52.8)
        ua -= tank_jacket_r / (1.0 / u_pre_skin + tank_jacket_r) * u_pre_skin * a_side
    ua *= (1.0 - solar_fraction)

    if ua < 0.0:
        raise OCHREException('A negative water heater standby loss coefficient (UA) was calculated.'
                        ' Double check water heater inputs.')
    if eta_c > 1.0:
        raise OCHREException('A water heater heat source (either burner or element) efficiency of > 1 has been calculated.'
                        ' Double check water heater inputs.')

    wh = {
        "Equipment Name": water_heater_type,
        "Fuel": water_heater["FuelType"].capitalize(),
        "Zone": parse_zone_name(water_heater["Location"]),
        "Setpoint Temperature (C)": t_set,
        "Tempering Valve Setpoint (C)": None,
        # 'Heat Transfer Coefficient (W/m^2/K)': u,
        "UA (W/K)": convert(ua, "Btu/hour/degR", "W/K"),
        "Efficiency (-)": eta_c,
        "Energy Factor (-)": energy_factor,
        "Tank Volume (L)": volume,
        "Tank Height (m)": height,
    }
    if heating_capacity is not None:
        wh['Capacity (W)'] = convert(heating_capacity, 'Btu/hour', 'W')

    if water_heater_type == 'heat pump water heater':
        # add HPWH COP, from ResStock, defaults to using UEF
        if uniform_energy_factor is None:
            uniform_energy_factor = (0.60522 + energy_factor) / 1.2101

        # Add/update parameters for low power HPWH
        # FIXME: temporary flag for designating 120V HPWHs in panels branch of ResStock
        if uniform_energy_factor == 4.9:
            wh.update({
                "Low Power HPWH": True,
                "HPWH COP (-)": 4.2,
                "HPWH Capacity (W)": 1499.4,
                "Setpoint Temperature (C)": convert(140, "degF", "degC"),
                "Tempering Valve Setpoint (C)": convert(125, "degF", "degC"),
                "hp_only_mode": True,
            })
        else:
            # Based on simulation of the UEF test procedure at varying COPs
            wh["HPWH COP (-)"] = 1.174536058 * uniform_energy_factor

    if water_heater_type == 'instantaneous water heater' and wh['Fuel'] != 'Electricity':
        on_time_frac = [0.0269, 0.0333, 0.0397, 0.0462, 0.0529][n_beds - 1]
        wh['Parasitic Power (W)'] = 5 + 60 * on_time_frac

    # Add water draw parameters (clothes washer and dishwasher added later)
    # From ResStock, ANSI/RESNET 301-2014 Addendum A-2015, Amendment on Domestic Hot Water (DHW) Systems
    assert water_heater['FractionDHWLoadServed'] == 1
    extension = water.get('extension', {})
    fixture_usage_ref = 14.6 + 10.0 * n_beds_adj  # in gal/day
    fixture_eff = 0.95 if list(water.get('WaterFixture', {}).values())[0].get('LowFlow', False) else 1.0
    fixture_multiplier = extension.get('WaterFixturesUsageMultiplier')
    fixture_gal_per_day = fixture_eff * fixture_usage_ref * fixture_multiplier

    # Add simple schedule for water fixtures
    wh.update(add_simple_schedule_params(extension, prefix='WaterFixtures'))

    # Get water distribution parameters
    # From ResStock, ANSI/RESNET 301-2014 Addendum A-2015, Amendment on Domestic Hot Water (DHW) Systems
    # 4.2.2.5.2.11 Service Hot Water Use
    distribution = water.get('HotWaterDistribution', {})
    distribution_type = distribution.get('SystemType', {})
    distribution_r = distribution.get('PipeInsulation', {}).get('PipeRValue', 0)
    if len(distribution_type) != 1:
        raise OCHREException(f'Cannot handle multiple water distribution types: {distribution_type}')
    elif 'Standard' in distribution_type:
        distribution_factor = 0.9 if distribution_r >= 3 else 1.0
        total_sqft = convert(construction['Conditioned Floor Area (m^2)'], 'm^2', 'ft^2')
        total_floors = construction['Total Floors']
        has_unfin_bsmt = construction['Foundation Type'] == 'Unfinished Basement'
        default_length = 2.0 * (total_sqft / total_floors) ** 0.5 + 10.0 * total_floors + 5.0 * has_unfin_bsmt
        p_ratio = distribution_type['Standard'].get('PipingLength', default_length) / default_length
        wd_eff = 1.0
    elif 'Recirculation' in distribution_type:
        distribution_factor = 1.0 if distribution_r >= 3 else 1.11
        p_ratio = distribution_type['Recirculation'].get('BranchPipingLoopLength', 10) / 10
        wd_eff = 0.1
    else:
        print(f'Warning: Unknown water distribution type: {distribution_type}')
        distribution_factor = 1.0
        p_ratio = 1.0
        wd_eff = 1.0
    ref_w_gpd = 9.8 * (n_beds_adj ** 0.43)
    o_frac = 0.25
    o_cd_eff = 0.0
    o_w_gpd = ref_w_gpd * o_frac * (1.0 - o_cd_eff)
    s_w_gpd = (ref_w_gpd - ref_w_gpd * o_frac) * p_ratio * distribution_factor
    mw_gpd = fixture_eff * (o_w_gpd + s_w_gpd * wd_eff)
    distribution_gal_per_day = mw_gpd * fixture_multiplier

    # Combine fixture and distribution water draws in schedule
    wh['Average Water Draw (L/day)'] = convert(fixture_gal_per_day + distribution_gal_per_day, 'gallon/day', 'L/day')

    return wh


def parse_clothes_washer(clothes_washer, n_bedrooms):
    # From ResStock, using ERI Version >= '2019A'
    rated_annual_kwh = clothes_washer.get('RatedAnnualkWh', 400.0)
    capacity = clothes_washer.get('Capacity', 3.0)  # in ft^3
    usage = clothes_washer.get('LabelUsage', 6.0)  # in cycles/week
    gas_rate = clothes_washer.get('LabelGasRate', 1.09)  # in $/therm
    gas_cost = clothes_washer.get('LabelAnnualGasCost', 27.0)  # in $/year
    electric_rate = clothes_washer.get('LabelElectricRate', 0.12)  # in $/kWh
    multiplier = clothes_washer.get('extension', {}).get('UsageMultiplier', 1)

    gas_h20 = 0.3914  # (gal/cyc) per (therm/y)
    elec_h20 = 0.0178  # (gal/cyc) per (kWh/y)
    lcy = usage * 52.0  # label cycles per year
    scy = 164.0 + n_bedrooms * 46.5
    acy = scy * ((3.0 * 2.08 + 1.59) / (capacity * 2.08 + 1.59))  # Annual Cycles per Year
    cw_appl = (gas_cost * gas_h20 / gas_rate - (rated_annual_kwh * electric_rate) * elec_h20 / electric_rate) / (
        electric_rate * gas_h20 / gas_rate - elec_h20)
    annual_kwh = cw_appl / lcy * acy
    annual_kwh *= multiplier

    water_draw = (rated_annual_kwh - cw_appl) * elec_h20 * acy / 365.0  # in gal/day
    water_draw *= multiplier

    frac_lost = 0.70
    frac_sens = (1.0 - frac_lost) * 0.90
    frac_lat = 1.0 - frac_sens - frac_lost

    return {
        'Annual Electric Energy (kWh)': annual_kwh,
        'Average Water Draw (L/day)': convert(water_draw, 'gallon/day', 'L/day'),
        'Convective Gain Fraction (-)': frac_sens,
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': frac_lat,
        **add_simple_schedule_params(clothes_washer.get('extension', {})),
    }


def parse_clothes_dryer(clothes_dryer, clothes_washer, n_bedrooms):
    # From ResStock, using ERI Version >= '2019A'
    if 'CombinedEnergyFactor' not in clothes_dryer and 'EnergyFactor' in clothes_dryer:
        combined_energy_factor = clothes_dryer['EnergyFactor'] / 1.15
    else:
        combined_energy_factor = clothes_dryer.get('CombinedEnergyFactor', 3.01)
    # energy_factor = clothes_dryer.get('EnergyFactor', combined_energy_factor * 1.15)
    fuel_type = clothes_dryer.get('FuelType', 'electricity')
    if fuel_type in ['electricity', 'natural gas']:
        pass
    elif fuel_type in ['propane', 'fuel oil']:
        print(f'WARNING: Converting clothes dryer fuel from {fuel_type} to natural gas.')
    else:
        raise OCHREException(f'Invalid fuel type for clothes dryer: {fuel_type}')
    is_electric = fuel_type == 'electricity'
    is_vented = clothes_dryer.get('Vented', True)
    multiplier = clothes_dryer.get('extension', {}).get('UsageMultiplier', 1)

    washer_rated_annual_kwh = clothes_washer.get('RatedAnnualkWh', 400.0)
    washer_imef = clothes_washer.get('IntegratedModifiedEnergyFactor', 1.0)  # in ft^3 / (kWh/cyc)
    # modified_energy_factor = clothes_washer.get('ModifiedEnergyFactor', 0.503 + 0.95 * washer_imef)
    washer_capacity = clothes_washer.get('Capacity', 3.0)  # in ft^3

    rmc = (0.97 * (washer_capacity / washer_imef) - washer_rated_annual_kwh / 312.0) / (
        (2.0104 * washer_capacity + 1.4242) * 0.455) + 0.04
    acy = (164.0 + 46.5 * n_bedrooms) * ((3.0 * 2.08 + 1.59) / (washer_capacity * 2.08 + 1.59))
    annual_kwh = (((rmc - 0.04) * 100) / 55.5) * (8.45 / combined_energy_factor) * acy
    if is_electric:
        annual_therm = 0.0
    else:
        annual_therm = annual_kwh * 3412.0 * (1.0 - 0.07) * (3.73 / 3.30) / 100000
        annual_kwh = annual_kwh * 0.07 * (3.73 / 3.30)
    annual_kwh *= multiplier
    annual_therm *= multiplier

    frac_lost = 0.85 if is_vented else 0.0
    if is_electric:
        frac_sens = (1.0 - frac_lost) * 0.90
    else:
        elec_btu = convert(annual_kwh, 'kWh', 'Btu')
        gas_btu = annual_therm * 1e5
        frac_sens = (1.0 - frac_lost) * ((0.90 * elec_btu + 0.8894 * gas_btu) / (elec_btu + gas_btu))
    frac_lat = 1.0 - frac_sens - frac_lost

    return {
        'Annual Electric Energy (kWh)': annual_kwh,
        'Annual Gas Energy (therms)': annual_therm,
        'Convective Gain Fraction (-)': frac_sens,
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': frac_lat,
        **add_simple_schedule_params(clothes_dryer.get('extension', {})),
    }


def parse_dishwasher(dishwasher, n_bedrooms):
    # From ResStock, using ERI Version >= '2019A'
    rated_annual_kwh = dishwasher.get('RatedAnnualkWh', 467.0)
    # energy_factor = dishwasher.get('EnergyFactor', 215.0 / rated_annual_kwh)
    capacity = dishwasher.get('PlaceSettingCapacity', 12)
    usage = dishwasher.get('LabelUsage', 4.0)  # in cycles/week
    gas_rate = dishwasher.get('LabelGasRate', 1.09)  # in $/therm
    gas_cost = dishwasher.get('LabelAnnualGasCost', 33.12)  # in $/year
    electric_rate = dishwasher.get('LabelElectricRate', 0.12)  # in $/kWh
    multiplier = dishwasher.get('extension', {}).get('UsageMultiplier', 1)

    usage = usage * 52.0  # in cycles/year?
    kwh_per_cyc = ((gas_cost * 0.5497 / gas_rate - rated_annual_kwh * electric_rate * 0.02504 / electric_rate) /
                   (electric_rate * 0.5497 / gas_rate - 0.02504)) / usage
    dwcpy = (88.4 + 34.9 * n_bedrooms) * (12.0 / capacity)
    annual_kwh = kwh_per_cyc * dwcpy
    annual_kwh *= multiplier

    water_draw = (rated_annual_kwh - kwh_per_cyc * usage) * 0.02504 * dwcpy / 365.0  # in gal/day
    water_draw *= multiplier

    frac_lost = 0.40
    frac_sens = (1.0 - frac_lost) * 0.50
    frac_lat = 1.0 - frac_sens - frac_lost

    return {
        'Annual Electric Energy (kWh)': annual_kwh,
        'Average Water Draw (L/day)': convert(water_draw, 'gallon/day', 'L/day'),
        'Convective Gain Fraction (-)': frac_sens,
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': frac_lat,
        **add_simple_schedule_params(dishwasher.get('extension', {})),
    }


def parse_refrigerator(refrigerator, n_bedrooms):
    # TODO: Only taking first refrigerator for now
    if isinstance(refrigerator, list):
        assert len(refrigerator) == 2
        print("WARNING: Combining 2 refrigerators into 1 piece of equipment, ignoring 2nd fridge heat gains")
        main_fridge = [r for r in refrigerator if r.get('PrimaryIndicator', True)][0]
        extra_fridge = [r for r in refrigerator if r.get('PrimaryIndicator', True)][0]
    else:
        main_fridge = refrigerator
        extra_fridge = {}

    # Get main refrigerator inputs from HPXML
    extension = main_fridge.get('extension', {})
    multiplier = extension.get('UsageMultiplier', 1)
    if 'AdjustedAnnualkWh' in extension:
        main_annual_kwh = extension['AdjustedAnnualkWh'] * multiplier
    elif 'RatedAnnualkWh' in main_fridge:
        main_annual_kwh = main_fridge['RatedAnnualkWh'] * multiplier
    else:
        main_annual_kwh = (637.0 + 18.0 * n_bedrooms) * multiplier

    # Get extra refrigerator inputs from HPXML
    extension2 = extra_fridge.get('extension', {})
    multiplier2 = extension2.get('UsageMultiplier', 1)
    if 'AdjustedAnnualkWh' in extension2:
        second_annual_kwh = extension2['AdjustedAnnualkWh'] * multiplier2
    elif 'RatedAnnualkWh' in extra_fridge:
        second_annual_kwh = extra_fridge['RatedAnnualkWh'] * multiplier2
    else:
        second_annual_kwh = 0

    out = {
        'Annual Electric Energy (kWh)': main_annual_kwh + second_annual_kwh,
        'Convective Gain Fraction (-)': (main_annual_kwh + second_annual_kwh) / main_annual_kwh,
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': 0,
    }

    if 'WeekdayScheduleFractions' in extension2:
        assert extension['WeekdayScheduleFractions'] == extension2['WeekdayScheduleFractions']
        assert extension['MonthlyScheduleMultipliers'] == extension2['MonthlyScheduleMultipliers']

    out.update(add_simple_schedule_params(extension))

    return out


def parse_cooking_range(range_dict, oven_dict, n_bedrooms):
    # Get range inputs from HPXML
    # TODO: Check that booleans are passed as strings
    fuel_type = range_dict.get('FuelType', 'electricity')
    if fuel_type in ['electricity', 'natural gas']:
        pass
    elif fuel_type in ['propane', 'fuel oil']:
        print(f'WARNING: Converting cooking range fuel from {fuel_type} to natural gas.')
    else:
        raise OCHREException(f'Invalid fuel type for cooking range: {fuel_type}')
    is_electric = fuel_type == 'electricity'
    is_induction = range_dict.get('IsInduction', False)
    is_convection = oven_dict.get('IsConvection', False)
    multiplier = range_dict.get('extension', {}).get('UsageMultiplier', 1)
    assert parse_zone_name(range_dict.get('Location')) in ['Indoor', None]

    # get total energy usage
    burner_ef = 0.91 if is_induction else 1
    oven_ef = 0.95 if is_convection else 1
    if is_electric:
        annual_kwh = burner_ef * oven_ef * (331 + 39.0 * n_bedrooms)
        annual_therm = 0
    else:
        annual_kwh = 22.6 + 2.7 * n_bedrooms
        annual_therm = oven_ef * (22.6 + 2.7 * n_bedrooms)
    annual_kwh *= multiplier
    annual_therm *= multiplier

    # get sensible/latent gains
    frac_lost = 0.20
    if is_electric:
        frac_sens = (1.0 - frac_lost) * 0.90
    else:
        annual_gas_kwh = convert(annual_therm, 'therm', 'kWh')
        annual_total_kwh = annual_kwh + annual_gas_kwh
        if annual_total_kwh != 0:
            frac_sens = (1.0 - frac_lost) * (0.90 * annual_kwh + 0.7942 * annual_gas_kwh) / annual_total_kwh
        else:
            frac_sens = 0
    frac_lat = 1.0 - frac_sens - frac_lost

    return {
        'Annual Electric Energy (kWh)': annual_kwh,
        'Annual Gas Energy (therms)': annual_therm,
        'Convective Gain Fraction (-)': frac_sens,
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': frac_lat,
        **add_simple_schedule_params(range_dict.get('extension', {}))
    }


def parse_lighting(location, df_lights, floor_area, extension=None):
    # df_lights is a dictionary with keys for ['CompactFluorescent', 'FluorescentTube', 'LightEmittingDiode']
    # if fractions sum to <1, remainder is incandescent lighting
    if extension is None:
        extension = {}

    if 'LightingType' in df_lights:
        # Fractions of each lighting type specified
        fractions = df_lights.set_index('LightingType')['FractionofUnitsInLocation'].to_dict()

        f_led = fractions['LightEmittingDiode']
        f_flr = fractions['CompactFluorescent'] + fractions['FluorescentTube']
        f_inc = 1 - f_led - f_flr
        area_ft2 = convert(floor_area, 'm^2', 'ft^2')
        e_led = 15 / 90
        e_flr = 15 / 60
        e_inc = 15 / 15

        if location == 'interior':
            int_adj = f_inc * e_inc + f_flr * e_flr + f_led * e_led
            annual_kwh = (0.9 / 0.925 * (455.0 + 0.8 * area_ft2) * int_adj) + (0.1 * (455.0 + 0.8 * area_ft2))
        elif location == 'exterior':
            ext_adj = f_inc * e_inc + f_flr * e_flr + f_led * e_led
            annual_kwh = (100.0 + 0.05 * area_ft2) * ext_adj
        elif location == 'garage':
            grg_adj = f_inc * e_inc + f_flr * e_flr + f_led * e_led
            annual_kwh = 100.0 * grg_adj
        else:
            raise OCHREException(f'Unknown lighting location: {location}')
        
    else:
        # Annual kWh specified
        load = df_lights['Load']
        assert load['Units'] == 'kWh/year'
        annual_kwh = load['Value']

    # TODO: get default fractions/multipliers for lighting
    lights = {
        'Annual Electric Energy (kWh)': annual_kwh * extension.get(f'{location.capitalize()}UsageMultiplier'),
        'Convective Gain Fraction (-)': 1,
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': 0,
        **add_simple_schedule_params(extension, prefix=location.capitalize())
    }

    return lights


def parse_mel(mel, load_name, is_gas=None):
    if is_gas is None:
        is_gas = mel['Load']['Units'] == 'therm/year'

    fuel_type = 'Gas' if is_gas else 'Electric'
    load_units = 'therm' if is_gas else 'kWh'
    if mel['Load']['Units'] != f'{load_units}/year':
        raise OCHREException(f'Invalid load units for {load_name}:', mel['Load']['Units'])
    if is_gas and mel.get('FuelType', 'natural gas') != 'natural gas':
        raise OCHREException(f'Invalid fuel type for MGL:', mel['FuelLoadType'])

    # TODO: use default annual load values from OS-HPXML
    mel_load = mel['Load']['Value'] * mel.get('extension', {}).get('UsageMultiplier', 1)

    # TODO: use FracSensible from OS-HPXML
    extension = mel.get('extension', {})
    if load_name == 'other':
        sensible_gain_default = 0.855
    elif load_name == 'fireplace':
        sensible_gain_default = 0.5
    else:
        sensible_gain_default = 0
    if load_name == 'other':
        latent_gain_default = 0.045
    elif load_name == 'fireplace':
        latent_gain_default = 0.1
    else:
        latent_gain_default = 0

    load_units = 'therms' if is_gas else 'kWh'
    out = {
        f'Annual {fuel_type} Energy ({load_units})': mel_load,
        'Convective Gain Fraction (-)': extension.get('FracSensible', sensible_gain_default),
        'Radiative Gain Fraction (-)': 0,
        'Latent Gain Fraction (-)': extension.get('FracLatent', latent_gain_default),
        **add_simple_schedule_params(extension),
    }

    return out


def parse_mels(mel_dict, is_gas=False):
    mels = {}
    for mel in mel_dict.values():
        load_name = mel['FuelLoadType'] if is_gas else mel['PlugLoadType']
        ochre_name = MEL_NAMES[load_name]
        mels[ochre_name] = parse_mel(mel, load_name, is_gas=is_gas)

    return mels


def parse_ev(ev):
    # create EV equipment from MEL info
    print('Creating EV equipment with a Level 2 charger from HPXML ')
    ev_load = ev['Annual Electric Energy (kWh)']
    return {
        'vehicle_type': 'BEV',
        'charging_level': 'Level 2',
        'mileage': 100 if ev_load < 1500 else 250  # Splits the two EV size options from ResStock
    }


def parse_pool_equipment(hpxml):
    # Get pool and spa equipment
    pool_equipment = {}
    for hpxml_name in ['Pool', 'Spa']:
        pool_list = list(hpxml.get(f'{hpxml_name}s', {}).values())
        if not pool_list or pool_list[0].get('Type', 'none') == 'none':
            continue

        pump = list(pool_list[0]['Pumps'].values())[0]
        heater = pool_list[0]['Heater']
        assert len(pool_list) == 1 and isinstance(pump, dict) and isinstance(heater, dict)
        if 'Load' in pump and pump.get('Type', 'none') != 'none':
            pool_equipment[f'{hpxml_name} Pump'] = parse_mel(pump, f'{hpxml_name} Pump')
        if 'Load' in heater and heater.get('Type', 'none') != 'none':
            pool_equipment[f'{hpxml_name} Heater'] = parse_mel(heater, f'{hpxml_name} Heater')

    return pool_equipment


def parse_vent_fan(vent_fan):
    # Note: ventilation fan is not included in the schedule. Assumes a constant power schedule
    # TODO: Add local ventilation fan, garage fan
    # TODO: need to update indoor zone parameters too
    whole_building_fan = vent_fan.get('UsedForWholeBuildingVentilation', False)
    whole_house_fan = vent_fan.get('UsedForSeasonalCoolingLoadReduction', False)
    assert whole_building_fan or whole_house_fan
    assert vent_fan.get('HoursInOperation', 24) == 24

    fan_type = 'whole house fan' if whole_house_fan else vent_fan['FanType']
    flow_rate = vent_fan['RatedFlowRate']  # in cfm
    default_powers = {
        'energy recovery ventilator': 1,
        'heat recovery ventilator': 1,
        'balanced': 0.7,
        'exhaust only': 0.35,
        'supply only': 0.35,
        'whole house fan': 0.1   # in W/cfm
    }
    power = vent_fan.get('FanPower', flow_rate * default_powers[fan_type])  # in W
    return {
        'Power (W)': power,
        'Ventilation Rate (cfm)': flow_rate,
        # 'Fan Type': fan_type,
        # 'Convective Gain Fraction (-)': extension.get('FracSensible', sensible_gain_default),
        # 'Radiative Gain Fraction (-)': 0,
        # 'Latent Gain Fraction (-)': extension.get('FracLatent', latent_gain_default),
    }


def parse_hpxml_equipment(hpxml, occupancy, construction):
    # Add HVAC equipment
    equipment = {}
    hvac_all = hpxml['Systems'].get('HVAC', {})
    for hvac_type in ['Heating', 'Cooling']:
        hvac = parse_hvac(hvac_type, hvac_all)
        if hvac is not None:
            equipment[f'HVAC {hvac_type}'] = hvac

    # Add water heater
    water = hpxml['Systems'].get('WaterHeating', {})
    water_heater = water.get('WaterHeatingSystem')
    if water_heater is not None:
        # Add water heater parameters
        wh = parse_water_heater(water_heater, water, construction)
        equipment['Water Heating'] = wh

    # Add appliances
    appliances = hpxml.get('Appliances', {})
    n_bedrooms = construction['Number of Bedrooms, Adjusted (-)']
    # appliances = {re.sub(r"(\w)([A-Z])", r"\1 \2", name): val for name, val in appliances.items()}
    if 'ClothesWasher' in appliances:
        equipment['Clothes Washer'] = parse_clothes_washer(appliances['ClothesWasher'],
                                                           n_bedrooms)
    if 'ClothesDryer' in appliances:
        equipment['Clothes Dryer'] = parse_clothes_dryer(appliances['ClothesDryer'],
                                                         appliances['ClothesWasher'],
                                                         n_bedrooms)
    if 'Dishwasher' in appliances:
        equipment['Dishwasher'] = parse_dishwasher(appliances['Dishwasher'], n_bedrooms)
    if 'Refrigerator' in appliances:
        equipment['Refrigerator'] = parse_refrigerator(appliances['Refrigerator'], n_bedrooms)
    # TODO: add freezer and dehumidifier
    if 'CookingRange' in appliances:
        equipment['Cooking Range'] = parse_cooking_range(appliances['CookingRange'],
                                                         appliances.get('Oven', {}),
                                                         n_bedrooms)

    # Add lighting
    lighting = hpxml.get('Lighting', {})
    lighting_group = lighting.get('LightingGroup')
    if lighting_group is not None:
        df_lights_all = pd.DataFrame(lighting_group).T
        extension = lighting.get('extension', {})
        for loc, df_lights in df_lights_all.groupby('Location'):
            if loc == 'interior':
                equipment['Indoor Lighting'] = parse_lighting(loc, df_lights,
                                                              construction['Indoor Floor Area (m^2)'], extension)
                if construction['Foundation Type'] == 'Finished Basement':
                    equipment['Basement Lighting'] = parse_lighting(loc, df_lights, 
                                                                    construction['First Floor Area (m^2)'], extension)
            elif loc == 'exterior':
                equipment['Exterior Lighting'] = parse_lighting(loc, df_lights,
                                                                construction['Indoor Floor Area (m^2)'], extension)
            elif loc == 'garage':
                if construction['Garage Floor Area (m^2)'] > 0:
                    equipment['Garage Lighting'] = parse_lighting(loc, df_lights, 
                                                                construction['Garage Floor Area (m^2)'], extension)
                else:
                    print('WARNING: Skipping garage lighting, since no garage is modeled.')
            else:
                raise OCHREException(f'Unknown lighting location: {loc}')

    # Get plug loads and fuel loads, some strange behavior depending on number of MELs/MGLs
    misc_loads = hpxml.get('MiscLoads', {})
    if 'PlugLoad' in misc_loads:
        mel_dict = misc_loads.get('PlugLoad', {})
        if 'Load' in mel_dict:
            mel_dict = {'PlugLoad1': mel_dict}
        mgl_dict = misc_loads.get('FuelLoad', {})
        if 'Load' in mgl_dict:
            mgl_dict = {'FuelLoad1': mgl_dict}
    else:
        mel_dict = {key: val for key, val in misc_loads.items() if 'PlugLoad' in key}
        mgl_dict = {key: val for key, val in misc_loads.items() if 'FuelLoad' in key}

    # Add MELs: TV, other MELs, well pump, EV
    mels = parse_mels(mel_dict)
    if 'Electric Vehicle' in mels:
        ev = mels.pop('Electric Vehicle')
        equipment['Electric Vehicle'] = parse_ev(ev)
    equipment.update(mels)

    # Add MGLs: Grill, Fireplace, and Lighting
    # Assumes fuel is gas, raises an error if not
    mgls = parse_mels(mgl_dict, is_gas=True)
    equipment.update(mgls)

    # Add pool/spa equipment: pumps and heaters
    pool_equipment = parse_pool_equipment(hpxml)
    equipment.update(pool_equipment)

    # Add ceiling fan
    ceiling_fan = lighting.get('CeilingFan')
    if ceiling_fan:
        # From ResStock (ANSI 301-2019), flow rate = 3000 cfm, operation time = 10.5 hours per day
        n_fans = ceiling_fan.get('Count', n_bedrooms + 1)
        efficiency = ceiling_fan.get('Airflow', {}).get('Efficiency', 3000 / 42.6)  # in cfm/W
        fan_annual_kwh = n_fans * 3000 / efficiency * 10.5 * 365.0 / 1000  # in kWh/year (assumes 10.5 hr/day)
        ceiling_fan['Load'] = {'Units': 'kWh/year',
                               'Value': fan_annual_kwh}
        equipment['Ceiling Fan'] = parse_mel(ceiling_fan, 'CeilingFan')

    # Add Ventilation Fan
    vent_fans = hpxml['Systems'].get('MechanicalVentilation', {}).get('VentilationFans', {})
    vent_fans = {key: val for key, val in vent_fans.items()
                 if val.get('UsedForWholeBuildingVentilation', False) or val.get('UsedForSeasonalCoolingLoadReduction', False)}
    if vent_fans:
        assert len(vent_fans) == 1
        equipment['Ventilation Fan'] = parse_vent_fan(list(vent_fans.values())[0])

    return equipment


def load_hpxml(modify_hpxml_dict=None, **house_args):
    hpxml = import_hpxml(**house_args)

    # modify HPXML properties from house_args
    if modify_hpxml_dict is not None:
        hpxml = nested_update(hpxml, modify_hpxml_dict)

    # Parse occupancy
    occupancy = parse_hpxml_occupancy(hpxml)
    if 'Occupancy' in house_args:
        occupancy = nested_update(occupancy, house_args.pop('Occupancy'))

    # Parse envelope properties and merge with house_args
    boundaries, zones, construction = parse_hpxml_envelope(hpxml, occupancy, **house_args)
    envelope = house_args.get('Envelope', {})
    if 'boundaries' in envelope:
        boundaries = nested_update(boundaries, house_args['Envelope'].pop('boundaries'))
    if 'zones' in envelope:
        zones = nested_update(zones, house_args['Envelope'].pop('zones'))

    # Parse equipment properties and merge with house_args
    equipment_dict = parse_hpxml_equipment(hpxml, occupancy, construction)
    if 'Equipment' in house_args:
        equipment_dict = nested_update(equipment_dict, house_args.pop('Equipment'))

    # update indoor zone infiltration (depends on equipment)
    # TODO: move to Envelope.init to get weather information (for air density)
    zones['Indoor'].update(parse_indoor_infiltration(hpxml, construction, equipment_dict))

    # combine all HPXML properties
    properties = {
        'occupancy': occupancy,
        'construction': construction,
        'boundaries': boundaries,
        'zones': zones,
        'equipment': equipment_dict,
        # 'location': location,
    }

    # Get weather station
    weather_station = hpxml.get('ClimateandRiskZones', {}).get('WeatherStation', {}).get('Name')
    weather_station = weather_station.strip('./')

    return properties, weather_station
