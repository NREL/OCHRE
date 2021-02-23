import numpy as np
import datetime as dt

from ochre import Units, FileIO
from ochre.Models import RCModel, HumidityModel, ModelException

ZONES = {
    'LIV': 'Indoor',
    'FND': 'Foundation',
    'GAR': 'Garage',
    'ATC': 'Attic'}
EXT_ZONES = {'EXT': 'Outdoor',
             'GND': 'Ground'}

# Each boundary has 4 parameters: [Name, Exterior node, Interior node, Alternate interior node]
BOUNDARIES = {
    'CL': ('Ceiling roof', 'ATC', 'LIV', None),
    'FL': ('Floor', 'FND', 'LIV', None),
    'WD': ('Windows', 'EXT', 'LIV', None),
    'IW': ('Interior Wall', 'LIV', 'LIV', None),
    'IM': ('Living Furniture', 'LIV', 'LIV', None),
    'EW': ('Exterior wall', 'EXT', 'LIV', None),
    'AW': ('Attached wall', 'GAR', 'LIV', None),
    'GW': ('Garage wall', 'EXT', 'GAR', None),
    'GM': ('Garage Furniture', 'GAR', 'GAR', None),
    'FW': ('Foundation wall', 'GND', 'FND', None),  # below ground
    'CW': ('Crawlspace wall', 'EXT', 'FND', None),  # above ground
    'BM': ('Basement Furniture', 'FND', 'FND', None),
    'RF': ('Roof', 'EXT', 'ATC', 'LIV'),
    'RG': ('Gable wall', 'EXT', 'ATC', 'LIV'),
    'AM': ('Attic Furniture', 'ATC', 'ATC', None),
    'GR': ('Garage roof', 'EXT', 'GAR', None),
    'GF': ('Garage floor', 'GND', 'GAR', None),
    'FF': ('Foundation floor', 'GND', 'FND', 'LIV'),
}
EXT_BOUNDARIES = {key: val for key, val in BOUNDARIES.items() if val[1] == 'EXT'}
cp_air = 1.006  # kJ/kg/K


class BoundarySurface:
    """
    Class for a surface of a boundary, e.g. the exterior surface of the roof.
    """

    def __init__(self, boundary, is_ext_surface, zone='', node_name='', res_film=0, res_total=1, **kwargs):
        self.boundary = boundary.label
        self.boundary_name = boundary.name
        self.zone = zone
        self.is_exterior = self.zone == 'EXT'
        self.area = boundary.area
        self.node = node_name
        self.resistance = res_total
        self.t_idx = None
        self.h_idx = None

        # Radiation variables
        self.solar_gain = 0  # solar gain to surface, in W
        self.lwr_gain = 0  # LWR gain to surface, in W
        self.temperature = None  # in deg C
        self.t_prev = None
        self.iterations = kwargs['time_res'] // dt.timedelta(minutes=5) + 1  # number of iterations to calculate LWR

        # Calculate radiation gain fraction and thermal resistance for surface temp calculation
        if res_film != 0:
            res_material = res_total - res_film
            assert res_material >= 0
            self.radiation_frac = res_film / res_total
            self.radiation_res = res_film * res_material / res_total
        else:
            if not (self.zone in ['GND'] or self.boundary_name == 'Windows' or self.node == ''):
                print('WARNING: No film resistance for {}. May impact radiation.'.format(self.boundary_name))
            self.radiation_frac = 1
            self.radiation_res = 0

        # Calculate exterior view factors: sky/ground for exterior. Interior done in Zone
        if self.is_exterior and self.boundary_name != 'Windows':
            # Note: removing sky view factor for all external boundaries
            self.view_factors = {'EXT': 1, 'SKY': 0}

            # calculate fraction to sky, air, and ground. Note ground + air are combined since both use ambient temp
            # https://bigladdersoftware.com/epx/docs/8-0/engineering-reference/page-020.html#external-longwave-radiation
            # if 'wall' in self.boundary_name.lower():
            #     tilt = 0  # all walls are vertical, tilt = cos(phi)
            # elif 'roof' in self.boundary_name.lower():
            #     tilt = np.cos(Units.deg2rad(kwargs.get('roof pitch', 0)))  # roof tilt defaults to horizontal
            # else:
            #     raise ModelException('Unknown exterior boundary name: {}.'.format(self.boundary_name))
            # sky_frac = ((1 + tilt) / 2) ** 1.5  # incorporates F and Beta from reference
            # self.view_factors = {'EXT': 1 - sky_frac, 'SKY': sky_frac}
        else:
            self.view_factors = None

        # TODO: remove this once properties files are updated for Foundation floor
        if self.boundary_name == 'Foundation floor' and self.boundary_name + ' properties' not in kwargs:
            boundary_properties = kwargs['Slab' + ' properties']
        else:
            boundary_properties = kwargs[self.boundary_name + ' properties']
        if is_ext_surface:
            self.emissivity = boundary_properties.get('Exterior Emissivity')  # unitless
            self.absorptivity = boundary_properties.get('Exterior Solar Absorptivity')  # unitless
        else:
            self.emissivity = boundary_properties.get('Interior Thermal Absorptivity')  # unitless
            self.absorptivity = boundary_properties.get('Interior Solar Absorptivity')  # unitless

        # Check that all surface not connected to ground have emissivity and absorptivity defined
        if self.zone not in ['', 'GND']:
            if self.emissivity is None:
                raise ModelException('{} missing Exterior Emissivity.'.format(self.boundary_name))
            if self.absorptivity is None:
                raise ModelException('{} missing Exterior Solar Absorptivity.'.format(self.boundary_name))

        if self.emissivity is not None:
            self.e_factor = self.emissivity * 5.6704e-8 * self.area  # in W/K^4
        else:
            self.e_factor = None

    def calculate_exterior_radiation(self, t_boundary, t_ext, t_sky, solar_gain):
        # calculate exterior surface temperature and radiation
        self.solar_gain = solar_gain * self.absorptivity
        # TODO: fudge factor to increase solar gain
        self.solar_gain *= 1.32

        # get inputs for LWR calculation: injected radiation, zone temperature, and surface temperature from conduction
        v_ext = self.view_factors['EXT']
        v_sky = self.view_factors['SKY']
        h_lwr_inj = self.e_factor * (v_ext * Units.C2K(t_ext) ** 4 + v_sky * Units.C2K(t_sky) ** 4)
        t_surf_init = self.radiation_frac * t_boundary + (1 - self.radiation_frac) * t_ext  # excludes irradiance

        # Option 1: Use previous time step surface temperature to calculate LWR, then recalculate surface temperature
        #  - running multiple times per time step for longer time steps
        #  - adding heavy ball convergence to reduce instability in determining the temperature
        for _ in range(self.iterations):
            self.lwr_gain = h_lwr_inj - self.e_factor * Units.C2K(self.temperature) ** 4
            t_surf_new = t_surf_init + (self.solar_gain + self.lwr_gain) * self.radiation_res
            t_surf_new = np.clip(t_surf_new, self.temperature - 2, self.temperature + 2)
            t_surf = self.temperature + 0.5 * (t_surf_new - self.temperature) + 0.1 * (self.temperature - self.t_prev)
            self.t_prev = self.temperature
            self.temperature = t_surf
        self.lwr_gain = h_lwr_inj - self.e_factor * Units.C2K(self.temperature) ** 4

        # # Option 2: Solve system of equations for surface temp and LWR gains:
        # #  * h_lwr = e_factor * (t_sky ** 4 * sky_frac + t_amb ** 4 * (1 - sky_frac) - t_surf ** 4)
        # #  * t_surf = self.radiation_frac * t_boundary + (1 - self.radiation_frac) * t_amb + (h_solar + h_lwr) * self.radiation_res
        # # If speed up is necessary, see:
        # # https://stackoverflow.com/questions/35795663/fastest-way-to-find-the-smallest-positive-real-root-of-quartic-polynomial-4-degr
        # polynomial = (e_factor * self.radiation_res, 0, 0, 1,
        #               - t_surf_init - (self.solar_gain + h_lwr_inj) * self.radiation_res)
        # roots = np.roots(polynomial)
        # valid = np.isreal(roots) & (roots.real > 100)
        # if len(roots[valid]) == 1:
        #     t_surf = roots[valid].real[0]
        # elif len(roots[valid]) > 1:
        #     # very unlikely - choose smallest option
        #     print('WARNING: Multiple solutions for {} surface temperature calculation.'.format(self.boundary_name))
        #     t_surf = roots[valid].real.min()
        # else:
        #     raise ModelException('No solution for {} surface temperature calculation.'.format(self.boundary_name))
        # self.lwr_gain = h_lwr_inj - e_factor * t_surf ** 4
        # self.temperature = Units.K2C(t_surf)

        # if abs(self.lwr_gain + self.solar_gain) > 10000:
        #     print('radiation issues...')
        return self.lwr_gain + self.solar_gain


class Zone:
    """
    Class for interior air zone, e.g. main indoor zone, garage, attic, foundation.
    """

    def __init__(self, label, capacitance, **kwargs):
        self.label = label
        self.name = ZONES[label]
        self.surfaces = []  # used for internal radiation
        self.view_factors = []  # used for internal radiation
        self.t_idx = None
        self.h_idx = None
        self.iterations = kwargs['time_res'] // dt.timedelta(minutes=5) + 1  # number of iterations to calculate LWR

        # get zone capacitance from RC parameters
        self.capacitance = capacitance

        # Infiltration parameters
        self.inf_heat = 0  # in W
        self.inf_flow = 0  # in m^3/s

        if self.name == 'Indoor':
            self.volume = kwargs['building volume (m^3)']
            assert kwargs['infiltration method'] == 'ASHRAE'
            inf_sft = kwargs['inf_f_t'] * (
                    kwargs['ws_S_wo'] * (1 - kwargs['inf_Y_i']) + kwargs['inf_S_wflue'] * 1.5 * kwargs['inf_Y_i'])
            self.infiltration_parameters = {'method': kwargs['infiltration method'],
                                            'inf_C_i': kwargs['inf_C_i'],
                                            'inf_n_i': kwargs['inf_n_i'],
                                            'inf_stack_coef': kwargs['inf_stack_coef'],
                                            'inf_wind_coef': kwargs['inf_wind_coef'],
                                            'inf_sft': inf_sft}
        elif self.name == 'Foundation':
            self.volume = kwargs['basement volume (m^3)'] if 'basement volume (m^3)' in kwargs \
                else kwargs['crawlspace volume (m^3)']
            assert kwargs['foundation infiltration method'] == 'ConstantACH'
            self.infiltration_parameters = {'method': kwargs['foundation infiltration method'],
                                            'ConstantACH': kwargs['infiltration ACH']}
        elif self.name == 'Garage':
            self.volume = kwargs['garage volume (m^3)']
            assert kwargs['garage infiltration method'] == 'ELA'
            self.infiltration_parameters = {
                'method': kwargs['garage infiltration method'],
                'ELA': kwargs['garage ELA (cm^2)'],
                'inf_stack_coef': kwargs['garage stack coefficient {(L/s)/(cm^4-K)}'],
                'inf_wind_coef': kwargs['garage wind coefficient {(L/s)/(cm^4-(m/s))}']
            }
        elif self.name == 'Attic':
            self.volume = kwargs['attic volume (m^3)']
            assert kwargs['attic infiltration method'] == 'ELA'
            self.infiltration_parameters = {
                'method': kwargs['attic infiltration method'],
                'ELA': kwargs['attic ELA (cm^2)'],
                'inf_stack_coef': kwargs['attic stack coefficient {(L/s)/(cm^4-K)}'],
                'inf_wind_coef': kwargs['attic wind coefficient {(L/s)/(cm^4-(m/s))}']
            }
        else:
            raise ModelException('Unknown zone: {}'.format(self.name))

        self.max_flow_rate = self.volume / kwargs['time_res'].total_seconds()  # in m^3/s

        # Ventilation parameters - Indoor zone only for now
        if self.name == 'Indoor':
            self.ventilation_type = kwargs.get('ventilation type', 'exhaust')
            self.ventilation_max_flow_rate = kwargs['ventilation cfm']
            self.ventilation_sens_eff = kwargs.get('erv sensible effectiveness', 0)
            self.ventilation_lat_eff = kwargs.get('erv latent effectiveness', 0)
            self.vent_flow = 0  # m^3/s, only for LIV node
            self.air_changes = 0  # ACH, only for LIV node
        else:
            self.ventilation_type = None
            self.ventilation_max_flow_rate = None
            self.ventilation_sens_eff = None
            self.ventilation_lat_eff = None
            self.vent_flow = None
            self.air_changes = None

    def create_surfaces(self, boundaries):
        # find surfaces connected to this zone
        self.surfaces.extend([b.int_surface for b in boundaries if b.int_surface.zone == self.label])
        self.surfaces.extend([b.ext_surface for b in boundaries if b.ext_surface.zone == self.label])

        # Check that all surfaces have emissivity defined
        for surface in self.surfaces:
            if surface.boundary_name != 'Windows' and surface.emissivity is None:
                raise ModelException('{} missing Interior Thermal Absorptivity.'.format(surface.boundary_name))

        # calculate view factors for each surface
        total_area = sum([s.area for s in self.surfaces])

        # Option 1: no self-viewing surfaces, does not maintain reciprocity (not implemented)
        # viewable_area = total_area - surface.area
        # surface.view_factors = {s.boundary: s.area / viewable_area for s in self.surfaces}
        # surface.view_factors[surface.boundary] = 0

        # Option 2: surfaces see themselves, maintains reciprocity
        self.view_factors = [s.area / total_area for s in self.surfaces]

        # Check total of view factors - should sum to 1
        check = sum(self.view_factors)
        if abs(1 - check) > 0.01:
            raise ModelException('Error calculating view factors for {}.'.format(self.name))

    def update_infiltration(self, schedule, t_zone, h_limit, density):
        t_ext = schedule['ambient_dry_bulb']
        wind_speed = schedule['wind_speed']
        delta_t = t_ext - t_zone

        if self.infiltration_parameters['method'] == 'ASHRAE':
            # calculate Indoor infiltration flow (m^3/s)
            params = self.infiltration_parameters
            inf_c = Units.cfm2m3_s(params['inf_C_i']) / Units.inH2O2Pa(1.0) ** params['inf_n_i']
            inf_Cs = params['inf_stack_coef'] * Units.inH2O_R2Pa_K(1.0) ** params['inf_n_i']
            inf_Cw = params['inf_wind_coef'] * Units.inH2O_mph2Pas2_m2(1.0) ** params['inf_n_i']
            inf_flow_temp = inf_c * inf_Cs * abs(delta_t) ** params['inf_n_i']
            inf_flow_wind = inf_c * inf_Cw * (params['inf_sft'] * wind_speed) ** (2 * params['inf_n_i'])
            self.inf_flow = (inf_flow_temp ** 2 + inf_flow_wind ** 2) ** 0.5

            # TODO: for now, leave ventilation here. Will need to move once we add ventilation for other zones
            # calculate Indoor ventilation flow (m^3/s)
            vent_cfm = schedule['ventilation_rate'] * self.ventilation_max_flow_rate
            self.vent_flow = Units.cfm2m3_s(vent_cfm)

            # combine and calculate ACH, Indoor node only
            if self.ventilation_type == 'balanced':
                # Add balanced + unbalanced in quadrature, but both inf and balanced are same term
                total_flow = self.inf_flow + self.vent_flow
            else:
                total_flow = (self.inf_flow ** 2 + self.vent_flow ** 2) ** 0.5
            self.air_changes = total_flow / self.volume * 3600  # Air changes per hour

            # calculate indoor node sensible heat gain
            # TODO: For now we're only handling HRV, not ERV. Probably just need to sum sensible + latent, but Jeff to double check E+
            ventilation_erv_eff = self.ventilation_sens_eff + self.ventilation_lat_eff
            if ventilation_erv_eff > 0:
                # if effectiveness is 70%, 70% of heat is recovered so HX to space is (1-.7) = 30%
                self.inf_heat = self.inf_flow * delta_t * density * cp_air * 1000 + \
                                self.vent_flow * delta_t * density * cp_air * 1000 * (
                                        1 - ventilation_erv_eff)
            else:
                self.inf_heat = total_flow * delta_t * density * cp_air * 1000  # in W

        # calculate infiltration heat gain from foundation
        elif self.infiltration_parameters['method'] == 'ConstantACH':
            self.inf_flow = self.infiltration_parameters['ConstantACH'] * self.volume / 3600
            heat_flow = self.inf_flow * density * cp_air * 1000  # in W / K
            self.inf_heat = heat_flow * delta_t  # in W
            self.inf_heat = np.clip(self.inf_heat, -abs(h_limit), abs(h_limit))  # limit heat gain/loss based on ambient

        elif self.infiltration_parameters['method'] == 'ELA':
            # FUTURE: update attic wind coefficient based on height. Inputs are in properties file already
            # see https://bigladdersoftware.com/epx/docs/8-6/input-output-reference/group-airflow.html
            #  - zoneinfiltrationeffectiveleakagearea
            f = 1
            params = self.infiltration_parameters
            self.inf_flow = f * params['ELA'] / 1000 * (
                    params['inf_stack_coef'] * abs(delta_t) + params['inf_wind_coef'] * wind_speed ** 2) ** 0.5
            heat_flow = self.inf_flow * density * cp_air * 1000  # in W / K
            self.inf_heat = heat_flow * delta_t  # in W
            self.inf_heat = np.clip(self.inf_heat, -abs(h_limit), abs(h_limit))  # limit heat gain/loss based on ambient
            # TODO: add heavy ball convergence to fix high wind issues for attics
            # heat_flow = np.clip(heat_flow, 0, 400)
            # inf_heat = heat_flow * delta_t  # in W
            # # if abs(inf_heat) > max(abs(self.inf_heat), abs(h_limit) / 2):
            # #     # if heat gains are increasing, reduce rate of increase to prevent instability in zone temperature
            # #     inf_heat = self.inf_heat + 0.6 * (inf_heat - self.inf_heat)
            #
            # # limit heat gain/loss based on ambient
            # self.inf_heat = np.clip(inf_heat, -abs(h_limit), abs(h_limit))

        # FUTURE: add latent gains for other zones
        return self.inf_heat

    def calculate_interior_radiation(self, t_zone, t_boundaries):
        # calculate all internal surface temperatures and LWR radiation
        # check that other surface temperatures match the view factors
        assert len(t_boundaries) == len(self.view_factors)

        # Option 1: Use previous time step surface temperatures to calculate LWR, then recalculate surface temperatures
        #  - running multiple times per time step for longer time steps
        #  - adding heavy ball convergence to reduce instability in determining the temperature
        for _ in range(self.iterations):
            # total LWR gains to zone
            avg_t4 = sum([vf * Units.C2K(s.temperature) ** 4 for vf, s in zip(self.view_factors, self.surfaces)])
            for t_boundary, s in zip(t_boundaries, self.surfaces):
                # for each surface, update LWR gain, surface temperature
                h_lwr = s.e_factor * (avg_t4 - Units.C2K(s.temperature) ** 4)
                t_surf_init = s.radiation_frac * t_boundary + (1 - s.radiation_frac) * t_zone  # excludes irradiance
                t_surf_new = t_surf_init + h_lwr * s.radiation_res
                t_surf_new = np.clip(t_surf_new, s.temperature - 2, s.temperature + 2)
                t_surf = s.temperature + 0.2 * (t_surf_new - s.temperature) + 0.05 * (s.temperature - s.t_prev)
                s.t_prev = s.temperature
                s.temperature = t_surf

        avg_t4 = sum([vf * Units.C2K(s.temperature) ** 4 for vf, s in zip(self.view_factors, self.surfaces)])
        for s in self.surfaces:
            s.lwr_gain = s.e_factor * (avg_t4 - Units.C2K(s.temperature) ** 4)

        check = sum([s.lwr_gain for s in self.surfaces])
        if abs(check) > 100:
            print('Issue with internal radiation')
        # if abs(self.lwr_gain + self.solar_gain) > 10000:
        #     print('radiation issues...')


class Boundary:
    """
    Class for a specific boundary within a building envelope, e.g. exterior walls, garage walls, etc.
    """

    def __init__(self, label, all_res, all_cap, zones, **kwargs):
        self.label = label
        self.name, ext_zone, int_zone, int_zone_alt = BOUNDARIES[label]

        # Define boundary area
        if self.name == 'Exterior wall':
            self.area = kwargs['total wall area (m^2)']
        elif self.name == 'Windows':
            self.area = (kwargs['front window area (m^2)'] + kwargs['right window area (m^2)'] +
                         kwargs['back window area (m^2)'] + kwargs['left window area (m^2)'])
        elif self.name == 'Interior Wall':
            self.area = kwargs['interior wall area (m^2)']
        elif self.name == 'Living Furniture':
            self.area = kwargs['Living Furniture surface area (m2)']
        elif self.name == 'Roof':
            self.area = kwargs['front roof area (m^2)'] + kwargs['back roof area (m^2)']
        elif self.name == 'Gable wall':
            self.area = kwargs['left gable wall area (m^2)'] + kwargs['right gable wall area (m^2)']
        elif self.name in ['Floor', 'Ceiling roof', 'Foundation floor']:
            self.area = kwargs['finished floor area (m^2)'] / kwargs['num stories']
        elif self.name == 'Foundation wall':
            self.area = kwargs['basement wall area (m^2)'] if 'basement wall area (m^2)' in kwargs \
                else kwargs['crawlspace below grade wall area (m^2)']
        elif self.name == 'Crawlspace wall':
            self.area = kwargs['crawlspace above grade wall area (m^2)']
        elif self.name == 'Basement Furniture':
            self.area = kwargs['Basement Furniture surface area (m2)']
        elif self.name == 'Attached wall':
            self.area = kwargs['garage attached wall area (m^2)']
        elif self.name == 'Garage wall':
            self.area = (kwargs['garage front wall area (m^2)'] + kwargs['garage right wall area (m^2)'] +
                         kwargs['garage back wall area (m^2)'] + kwargs['garage left wall area (m^2)'])
        elif self.name == 'Garage Furniture':
            self.area = kwargs['Garage Furniture surface area (m2)']
        elif self.name in ['Garage roof', 'Garage floor']:
            self.area = kwargs['garage floor area (m^2)']
        else:
            raise ModelException('No area defined for {}'.format(self.name))

        # Define starting (exterior) and ending (interior) zones
        zone_labels = list(zones.keys()) + list(EXT_ZONES.keys())
        if ext_zone not in zone_labels:
            raise ModelException(
                'Cannot use {} boundary when {} zone is not included.'.format(self.name, ext_zone))

        if int_zone not in zone_labels:
            int_zone = int_zone_alt
        self.is_int = int_zone == 'LIV'
        if int_zone not in zone_labels:
            raise ModelException(
                'Cannot use {} boundary when {} zone is not included.'.format(self.name, int_zone))

        # parse RC parameters to be readable for RCModel, get number and names for nodes
        same_zones = ext_zone == int_zone
        cap_values, res_values = self.create_rc_data(all_res, all_cap, same_zones=same_zones)
        self.n_nodes = len(cap_values)
        self.all_nodes = [self.label + str(i + 1) for i in range(self.n_nodes)]

        # Create solar irradiance and LWR parameters for exterior and interior surfaces
        node_ext = self.all_nodes[0] if self.n_nodes else int_zone  # node name closest to external zone
        res_ext = all_res.get('R_' + label + '_ext', 0) / self.area
        if self.name != 'Windows':
            self.ext_surface = BoundarySurface(self, True, ext_zone, node_ext, res_film=res_ext,
                                               res_total=res_values[0], **kwargs)
        else:
            self.ext_surface = BoundarySurface(self, True, **kwargs)  # empty surface
        if not (same_zones or self.name == 'Windows'):
            res_int = all_res.get('R_' + label + '_int', 0) / self.area
            node_int = self.all_nodes[-1] if self.n_nodes else ext_zone  # node name closest to internal zone
            self.int_surface = BoundarySurface(self, False, int_zone, node_int, res_film=res_int,
                                               res_total=res_values[-1], **kwargs)
        else:
            self.int_surface = BoundarySurface(self, False, **kwargs)  # empty surface

        # Save all RC parameters, including internal and external resistance values
        self.capacitors = {'C_' + node: val for node, val in zip(self.all_nodes, cap_values)}

        # FUTURE: linearizing interior radiation for attic
        # for s, idx in zip([self.ext_surface, self.int_surface], [0, -1]):
        #     if s.zone == 'ATC':
        #         # update resistance
        #         t_op = 20 + 273  # assume 20C as operating point for all temperatures
        #         r_rad = 1 / (4 * s.emissivity * 5.6704e-8 * t_op ** 3 * (1 - s.radiation_frac) * self.area)  # in K/W
        #         res_values[idx] = 1 / (1 / res_values[idx] + 1 / r_rad)
        #
        #         # set emissivity to 0 to turn off non-linear radiation
        #         s.emissivity = 0

        # for resistances, convert boundary names to node names
        # e.g. R_CL1 -> R_ATC_CL1,  R_CL2 -> R_CL1_CL2,  R_CL3 -> R_CL2_LIV
        if same_zones:
            # do not include int_zone if int and ext zones are the same
            node_list = [ext_zone] + self.all_nodes
        else:
            node_list = [ext_zone] + self.all_nodes + [int_zone]
        self.resistors = {'_'.join(['R', node_list[i], node_list[i + 1]]): r for i, r in enumerate(res_values)}

    def create_rc_data(self, all_res, all_cap, same_zones=False):
        cap_list = [all_cap['C_' + self.label + str(i)] for i in range(10) if 'C_' + self.label + str(i) in all_cap]
        nodes = len(cap_list)
        assert nodes > 0

        res_list = [all_res['R_' + self.label + str(i)] for i in range(10) if 'R_' + self.label + str(i) in all_res]
        res_ext = all_res.get('R_' + self.label + '_ext', 0)
        res_int = all_res.get('R_' + self.label + '_int', 0)
        if len(res_list) != nodes:
            raise ModelException('Cannot parse RC data for {}. Number of resistors ({}) is not compatible with the '
                                 'number of capacitors ({})'.format(self.name, len(res_list), nodes))

        if same_zones:
            # if start and end zones are equal, cut boundary in half
            new_nodes = nodes // 2
            # self.area *= 2  # Not doubling the area - assuming it includes both sides of the boundary
            if nodes % 2 == 0:
                cap_list = cap_list[:new_nodes]
                res_list = res_list[:new_nodes]
            else:
                cap_list = cap_list[:new_nodes] + [cap_list[new_nodes] / 2]  # cut middle node in half
                res_list = res_list[:new_nodes] + [res_list[new_nodes] / 2]  # cut middle node in half
            nodes = len(cap_list)

        # R is split before and after given node - leads to 1 more R than C
        # exterior/interior film resistances included in first/last R
        res_list = [0] + res_list + [0]
        res_list = [(res_list[i] + res_list[i + 1]) / 2 for i in range(nodes + 1)]
        res_list[0] += res_ext
        res_list[-1] += res_int

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

        # update values with surface area
        cap_list = [c * self.area for c in cap_list]
        res_list = [r / self.area for r in res_list]

        return cap_list, res_list


class Envelope(RCModel):
    name = 'Envelope'

    def __init__(self, **kwargs):

        # Collect RC parameters from properties file
        all_res, all_cap = FileIO.get_rc_params(**kwargs)

        # Create lists of all zones and boundaries
        self.zones = {label: Zone(label, all_cap['C_' + label], **kwargs) for label in ZONES if 'C_' + label in all_cap}
        self.indoor_zone = self.zones['LIV']
        self.boundaries = [Boundary(label, all_res, all_cap, self.zones, **kwargs) for label in BOUNDARIES
                           if 'R_' + label + '1' in all_res]

        # Update zones to include boundary surface information
        for zone in self.zones.values():
            zone.create_surfaces(self.boundaries)

        # create dictionaries of exterior and interior boundaries
        self.ext_boundaries = [bd for bd in self.boundaries if bd.ext_surface.is_exterior]
        self.int_boundaries = [bd for bd in self.boundaries if bd.is_int]

        # define floor and interior wall boundaries (for solar through windows)
        self.floor = [bd for bd in self.boundaries if bd.name in ['Floor', 'Foundation floor']][0]
        self.int_walls = [bd for bd in self.boundaries if bd.name == 'Interior Wall'][0]

        # Generate RC Model
        super().__init__(ext_node_names=list(EXT_ZONES.keys()), **kwargs)
        self.t_ext_idx = self.input_names.index('T_EXT')
        self.t_gnd_idx = self.input_names.index('T_GND') if 'T_GND' in self.input_names else None

        # Set initial surface temperatures to zone temperature
        # and save state and input indices for faster updates
        for label, zone in self.zones.items():
            zone.t_idx = self.state_names.index('T_' + label)
            zone.h_idx = self.input_names.index('H_' + label)
            for surface in zone.surfaces:
                surface.temperature = self.states[zone.t_idx]
                surface.t_prev = self.states[zone.t_idx]
                if 'T_' + surface.node in self.state_names:
                    surface.t_idx = self.state_names.index('T_' + surface.node)
                    surface.h_idx = self.input_names.index('H_' + surface.node)
        for bd in self.ext_boundaries:
            bd.ext_surface.temperature = kwargs['initial_schedule']['ambient_dry_bulb']
            bd.ext_surface.t_prev = kwargs['initial_schedule']['ambient_dry_bulb']
            bd.ext_surface.t_idx = self.state_names.index('T_' + bd.ext_surface.node)
            bd.ext_surface.h_idx = self.input_names.index('H_' + bd.ext_surface.node)

        # Initialize humidity model
        t_init = self.states[self.indoor_zone.t_idx]
        self.liv_net_sensible_gains = 0
        self.humidity = HumidityModel(t_init, **kwargs)

        # Occupancy parameters
        occupancy_gain = kwargs['gain per occupant (W)'] * kwargs['number of occupants']
        self.occupancy_sensible_gain = (kwargs['occupants convective gainfrac'] +
                                        kwargs['occupants radiant gainfrac']) * occupancy_gain
        self.occupancy_latent_gain = kwargs['occupants latent gainfrac'] * occupancy_gain

        # Results parameters
        self.temp_deadband = (kwargs.get('heating deadband temperature (C)', 1),
                              kwargs.get('cooling deadband temperature (C)', 1))
        self.unmet_hvac_load = 0

    def load_rc_data(self, **kwargs):
        # combine rc data from all boundaries
        all_rc_data = {}
        for boundary in self.boundaries:
            rc_data = {**boundary.capacitors, **boundary.resistors}
            for key, val in rc_data.items():
                if key not in all_rc_data:
                    all_rc_data[key] = val
                else:
                    # combine in parallel
                    all_rc_data[key] = self.par(all_rc_data[key], val)

        # add capacitances from zones
        all_rc_data.update({'C_' + label: zone.capacitance for label, zone in self.zones.items()})
        return all_rc_data

    def load_initial_state(self, initial_temp_setpoint=None, **kwargs):
        # Sets all temperatures to the steady state value based on initial conditions
        # Adds random temperature if exact setpoint is not set
        # Note: initialization will update the initial state to more typical values
        outdoor_temp = kwargs['initial_schedule']['ambient_dry_bulb']
        ground_temp = kwargs['initial_schedule']['ground_temperature']

        # Indoor initial condition depends on ambient temperature - use heating/cooling setpoint when </> 15 deg C
        if initial_temp_setpoint is None:
            # select heating/cooling setpoint based on starting outdoor temperature
            if outdoor_temp > 12:
                deadband = kwargs.get('cooling deadband temperature (C)', 1)
                random_delta = np.random.uniform(low=-deadband / 2, high=deadband / 2)
                indoor_temp = kwargs['initial_schedule']['cooling_setpoint'] + random_delta
            else:
                deadband = kwargs.get('heating deadband temperature (C)', 1)
                random_delta = np.random.uniform(low=-deadband / 2, high=deadband / 2)
                indoor_temp = kwargs['initial_schedule']['heating_setpoint'] + random_delta
        elif isinstance(initial_temp_setpoint, str):
            assert initial_temp_setpoint in ['heating', 'cooling']
            deadband = kwargs.get(initial_temp_setpoint + ' deadband temperature (C)', 1)
            random_delta = np.random.uniform(low=-deadband / 2, high=deadband / 2)
            indoor_temp = kwargs['initial_schedule'][initial_temp_setpoint + '_setpoint'] + random_delta
        elif isinstance(initial_temp_setpoint, (int, float)):
            indoor_temp = initial_temp_setpoint
        else:
            raise ModelException('Unknown initial temperature setpoint: {}'.format(initial_temp_setpoint))

        # Update continuous time matrices to swap T_LIV from state to input
        x_idx = self.state_names.index('T_LIV')
        keep_states = np.ones(len(self.state_names), dtype=bool)
        keep_states[x_idx] = False
        keep_inputs = [self.input_names.index('T_EXT'), self.input_names.index('T_GND')]
        A = self.A_c[keep_states, :][:, keep_states]
        B = np.hstack([self.A_c[keep_states, :][:, [x_idx]], self.B_c[keep_states, :][:, keep_inputs]])
        u = np.array([indoor_temp, outdoor_temp, ground_temp])

        # Calculate steady state values (effectively interpolates from the input temperatures)
        x = - np.linalg.inv(A).dot(B).dot(u)
        x = np.insert(x, x_idx, indoor_temp)
        return super().load_initial_state(initial_states=x)

    def remove_unused_inputs(self, unused_inputs=None, **kwargs):
        # remove heat injection inputs that aren't into main nodes or nodes with a radiation injection
        unused_inputs = ['H_' + node for bd in self.boundaries for node in bd.all_nodes
                         if node not in [bd.int_surface.node, bd.ext_surface.node]]

        super().remove_unused_inputs(unused_inputs=unused_inputs, **kwargs)

    def update_radiation(self, schedule):
        # Calculate external radiation (solar and LWR) by boundary, excluding windows
        for boundary in self.ext_boundaries:
            surface = boundary.ext_surface
            t_boundary = self.states[surface.t_idx]
            h_radiation = surface.calculate_exterior_radiation(t_boundary, schedule['ambient_dry_bulb'],
                                                               schedule['sky_temperature'],
                                                               schedule['solar_' + boundary.label])
            self.inputs[surface.h_idx] += h_radiation * surface.radiation_frac

        # Update injections from windows, if they exist
        # FUTURE: add LWR to transmitted radiation through windows
        solar_gain = schedule.get('solar_WD', 0)  # Note: absorptivity already accounted for

        # add window solar gains to a mix of air node, interior walls, and floor
        # TODO: add fraction_to_furniture. maybe use ratio of areas
        fraction_to_air = 0.0
        fraction_to_walls = 0.66
        fraction_to_floor = 0.66

        self.inputs[self.indoor_zone.h_idx] += solar_gain * fraction_to_air
        if self.floor:
            self.inputs[self.floor.int_surface.h_idx] += solar_gain * fraction_to_floor
        if self.int_walls:
            self.inputs[self.int_walls.ext_surface.h_idx] += solar_gain * fraction_to_walls

        # Calculate internal radiation by zone
        for zone in self.zones.values():
            t_zone = self.states[zone.t_idx]

            # collect all boundary temperatures
            t_boundaries = [self.states[s.t_idx] if s.node != 'EXT' else 0 for s in zone.surfaces]

            # update radiation for all boundaries within zone
            zone.calculate_interior_radiation(t_zone, t_boundaries)
            for surface in zone.surfaces:
                self.inputs[surface.h_idx] += surface.lwr_gain * surface.radiation_frac
                self.inputs[zone.h_idx] += surface.lwr_gain * (1 - surface.radiation_frac)

    def update_infiltration(self, schedule):
        # calculate outdoor dry air density using moist air density and humidity ratio
        density = self.humidity.get_dry_air_density(schedule['ambient_dry_bulb'], schedule['ambient_humidity'],
                                                    schedule['ambient_pressure'])

        # calculate infiltration for all zones, add ventilation for Indoor zone only
        for zone in self.zones.values():
            t_zone = self.states[zone.t_idx]

            # solve for the maximum infiltration gain to achieve outdoor temperature
            ext_gains = self.inputs[zone.h_idx]
            h_limit = self.solve_for_input(zone.t_idx, zone.h_idx, schedule['ambient_dry_bulb'])
            # reduce magnitude based of inf heat gains from external heat injections (not for indoor zone)
            h_limit = np.clip(h_limit, -abs(h_limit + ext_gains), abs(h_limit + ext_gains))

            h_inf = zone.update_infiltration(schedule, t_zone, h_limit, density)
            self.inputs[zone.h_idx] += h_inf

    def reset_env_inputs(self, schedule):
        # reset to defaults
        self.inputs = self.default_inputs.copy()

        # Add occupancy heat and latent gains
        self.inputs[self.indoor_zone.h_idx] += schedule['Occupancy'] * self.occupancy_sensible_gain  # in W
        self.humidity.latent_gains += schedule['Occupancy'] * self.occupancy_latent_gain

        # Add external temperatures to inputs
        self.inputs[self.t_ext_idx] = schedule['ambient_dry_bulb']
        if self.t_gnd_idx is not None:
            self.inputs[self.t_gnd_idx] = schedule['ground_temperature']

        # Update solar radiation, external LWR, and internal LWR
        self.update_radiation(schedule)

        # Update infiltration and ventilation
        self.update_infiltration(schedule)

    def update(self, inputs=None, schedule=None, **kwargs):
        # Note: reset_env_inputs should have already been called
        # self.reset_env_inputs(schedule)

        self.liv_net_sensible_gains = self.inputs[self.indoor_zone.h_idx]

        # Run RC Model update
        super().update(reset_inputs=False)
        t_liv = self.states[self.indoor_zone.t_idx]

        # check that states are within reasonable range
        if (self.states < -30).any() or (self.states > 100).any() or not (10 < t_liv < 30):
            print('WARNING: Extreme envelope temperatures: {}'.format(self.get_states()))
        if (self.states < -40).any() or (self.states > 130).any() or not (0 < t_liv < 40):
            raise ModelException('Envelope temperatures are outside acceptable range: {}'.format(self.get_states()))

        # Run humidity update
        # NOTE: Only incorporating latent gains in LIV node
        self.humidity.update_humidity(t_liv, self.indoor_zone.air_changes, schedule['ambient_dry_bulb'],
                                      schedule['ambient_humidity'], schedule['ambient_pressure'])
        if t_liv < self.humidity.indoor_wet_bulb - 0.1:
            print('Warning: Wet bulb temp ({}), greater than dry bulb temp ({})'.format(
                self.humidity.indoor_wet_bulb, t_liv))
            self.humidity.indoor_wet_bulb = t_liv

        # Calculate unmet thermal loads - negative=below deadband, positive=above deadband
        # TODO: get setpoint and deadband from HVAC at every time step
        t_low = schedule['heating_setpoint'] - self.temp_deadband[0] / 2
        t_high = schedule['cooling_setpoint'] + self.temp_deadband[1] / 2
        self.unmet_hvac_load = t_liv - t_low if t_liv < t_low else max(t_liv - t_high, 0)

    def get_main_states(self):
        # get temperatures from main nodes, and living space humidity
        out = {zone.name: self.states[zone.t_idx] for zone in self.zones.values()}
        out['Indoor Wet Bulb'] = self.humidity.indoor_wet_bulb
        return out

    def generate_results(self, verbosity, to_ext=False):
        main_temps = self.get_main_states()

        if to_ext:
            # for now, use different format for external controller
            to_ext_control = {'T {}'.format(loc): val for loc, val in main_temps.items()}
            return to_ext_control
        else:
            results = {}
            if verbosity >= 1:
                results.update({'Temperature - {} (C)'.format(loc): val for loc, val in main_temps.items()})
                results.update({'Temperature - Outdoor (C)': self.inputs[self.t_ext_idx],
                                'Temperature - Ground (C)': self.inputs[self.t_gnd_idx]})
                results['Unmet HVAC Load (C)'] = self.unmet_hvac_load
            if verbosity >= 4:
                results.update({
                    'Relative Humidity - Indoor (-)': self.humidity.indoor_rh,
                    'Humidity Ratio - Indoor (-)': self.humidity.indoor_w,
                    'Indoor Net Sensible Heat Gain (W)': self.liv_net_sensible_gains,
                    'Indoor Net Latent Heat Gain (W)': self.humidity.latent_gains,

                    'Air Changes per Hour (1/hour)': self.indoor_zone.air_changes,
                    'Indoor Ventilation Flow Rate (m^3/s)': self.indoor_zone.vent_flow,
                })
                results.update({zone.name + ' Infiltration Flow Rate (m^3/s)': zone.inf_flow
                                for zone in self.zones.values()})
                results.update({zone.name + ' Infiltration Heat Gain (W)': zone.inf_heat
                                for zone in self.zones.values()})

            if verbosity >= 7:
                # add density
                results['Indoor Air Density (kg/m^3)'] = self.humidity.indoor_density

                # add heat injections into the living space (pos=heat injected)
                t_indoor = self.states[self.indoor_zone.t_idx]
                for bd in self.int_boundaries:
                    if bd.n_nodes == 0 and bd.ext_surface.zone == '':  # Windows - get ambient temperature
                        t_node = self.inputs[self.t_ext_idx]
                        r_liv = bd.resistors['R_EXT_LIV']
                    elif bd.int_surface.t_idx is None:  # Interior Walls - use exterior surface node
                        t_node = self.states[bd.ext_surface.t_idx]
                        r_liv = bd.ext_surface.resistance
                    else:
                        t_node = self.states[bd.int_surface.t_idx]
                        r_liv = bd.int_surface.resistance
                    results['Convection from {} (W)'.format(bd.name)] = (t_node - t_indoor) / r_liv

            if verbosity >= 8:
                results.update({**self.get_states(), **self.get_inputs()})

                # add surface temperature, solar and LWR gains for each exterior surface
                for bd in self.ext_boundaries:
                    results['{} Ext. Solar Gain (W)'.format(bd.name)] = bd.ext_surface.solar_gain
                    results['{} Ext. LWR Gain (W)'.format(bd.name)] = bd.ext_surface.lwr_gain
                    results['{} Ext. Surface Temperature (C)'.format(bd.name)] = bd.ext_surface.temperature

                # add surface temperature and LWR gains for each interior surface, by zone
                for zone in self.zones.values():
                    for surface in zone.surfaces:
                        results['{} {} LWR Gain (W)'.format(surface.boundary_name, zone.name)] = surface.lwr_gain
                        results['{} {} Surface Temperature (C)'
                                ''.format(surface.boundary_name, zone.name)] = surface.temperature

            return results
