import numpy as np
import datetime as dt

from ochre.utils import OCHREException
from ochre.utils.units import convert, degC_to_K, cfm_to_m3s
import ochre.utils.envelope as utils
from ochre.Models import RCModel, HumidityModel, ModelException
from ochre import Simulator

cp_air = 1.006  # kJ/kg-K
rho_air = 1.2041  # kg/m^3, used for determining capacitance only


class BoundarySurface:
    """
    Class for a surface of a boundary, e.g. the exterior surface of the roof.
    """

    def __init__(self, boundary, is_ext_surface, zone_label='', node_name='', res_material=1,
                 linearize_ext_radiation=False, **kwargs):
        self.boundary = boundary
        self.boundary_name = boundary.name
        self.zone_label = zone_label
        self.zone_name = utils.ZONES.get(zone_label, utils.EXT_ZONES.get(zone_label))
        self.is_exterior = self.zone_label == 'EXT'
        self.node = node_name
        self.area = boundary.area  # in m^2

        self.t_idx = None  # Index of envelope states for surface temperature
        self.h_idx = None  # Index of envelope inputs for surface heat gain
        self.window_view_factor = 0  # for solar radiation from windows, indoor zone only

        # TODO: add these to envelope input update, equipment sensible gain update
        self.radiation_to_zone = 0  # for reporting zone radiation per surface for component loads
        self.internal_gain = 0  # for equipment sensible heat to surface, e.g. HPWH

        # Radiation variables
        self.solar_gain = 0  # solar gain to surface, in W
        self.transmitted_gain = 0  # transmitted solar gain through surface to zone (for windows), in W
        self.lwr_gain = 0  # LWR gain to surface, in W
        self.temperature = None  # surface temp, in deg C
        self.t_prev = None  # previous surface temp, in deg C
        self.t_boundary = None  # temp of closest boundary node, in deg C
        self.iterations = kwargs['time_res'] // dt.timedelta(minutes=5) + 1  # number of iterations to calculate LWR

        if self.boundary_name == 'Window':
            # Note - window emissivity is for both interior and exterior LWR through windows, set to 0.84
            # see https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/window-calculation-module.html
            # step-5.-determine-layer-solar-reflectance
            # TODO: res_material -= res_ext??
            self.emissivity = 0.84
            window_shgc = kwargs['SHGC (-)'] * kwargs['Shading Fraction (-)']
            t, a = utils.calculate_window_parameters(window_shgc, kwargs['U Factor (W/m^2-K)'],
                                                     res_material)
            self.transmittance = t
            self.absorptivity = a
        else:
            e_default = 0.05 if self.zone_name == 'Attic' and kwargs.get('Radiant Barrier', False) else 0.90
            a_default = 0.05 if self.zone_name == 'Attic' and kwargs.get('Radiant Barrier', False) else 0.60
            if is_ext_surface:
                self.emissivity = kwargs.get('Exterior Emissivity (-)', e_default)  # unitless
                self.absorptivity = kwargs.get('Exterior Solar Absorptivity (-)', a_default)  # unitless
                self.transmittance = 0
            else:
                self.emissivity = kwargs.get('Interior Emissivity (-)', e_default)  # unitless
                self.absorptivity = kwargs.get('Interior Solar Absorptivity (-)', a_default)  # unitless
                self.transmittance = None

        if is_ext_surface:
            self.res_film = kwargs.get('Exterior Film Resistance (m^2-K/W)', 0)  # in m^2-K/W
        else:
            self.res_film = kwargs.get('Interior Film Resistance (m^2-K/W)', 0)  # in m^2-K/W

        if self.emissivity:
            self.e_factor = self.emissivity * 5.6704e-8 * self.area  # in W/K^4
        else:
            self.e_factor = None

        if linearize_ext_radiation and self.is_exterior:
            # add resistors to RC network to approximate exterior LWR radiation
            # linearizes radiation equation: H_ab = e_factor * (Ta^4 - Tb^4) -> dH/dT = e_factor * 4 * T^3 = 1/Rab
            # assumes operating point of T=15C for boundary and exterior temperature
            # note: radiation will be reported as combined with convection
            t_ref = 15 + degC_to_K
            res_radiation = 1 / (4 * self.e_factor * t_ref ** 3) * self.boundary.area
            self.res_film = RCModel.par(self.res_film, res_radiation)

        # Calculate radiation gain fraction and thermal resistance for surface temp calculation
        if self.res_film == 0 and not (self.zone_label in ['GND'] or self.node == ''):
            print(f'WARNING: No film resistance for {self.boundary_name}. May impact radiation.')
        self.radiation_frac = self.res_film / (self.res_film + res_material)  # unitless
        self.res_total = (self.res_film + res_material) / self.boundary.area  # in K/W
        # Note: radiation resistance assumes res_material >> res_film
        self.radiation_res = self.res_film / self.boundary.area  # in K/W
        # self.radiation_res = self.res_film * res_material / (self.res_film + res_material) / self.boundary.area # in K/W

        # Calculate exterior view factors
        # calculate fraction to sky, air, and ground. Note ground + air are combined since both use ambient temp
        # https://bigladdersoftware.com/epx/docs/8-0/engineering-reference/page-020.html#external-longwave-radiation
        if self.is_exterior:
            self.tilt = kwargs.get('Tilt (deg)', utils.get_boundary_tilt(self.boundary_name))
            default_azimuth = [0] if self.tilt == 0 else None
            self.azimuths = kwargs.get("Azimuth (deg)", default_azimuth)
            # sky view factor incorporates F and Beta from reference
            self.sky_view_factor = ((1 + np.cos(convert(self.tilt, 'deg', 'rad'))) / 2) ** 1.5
        else:
            self.tilt = None
            self.azimuths = None
            self.sky_view_factor = 0

    def calculate_exterior_radiation(self, t_ext, t_sky):
        # get inputs for LWR calculation: injected radiation, zone temperature, and surface temperature from conduction
        if t_sky is None or np.isnan(t_sky):
            h_lwr_inj = self.e_factor * (t_ext + degC_to_K) ** 4
        else:
            h_lwr_inj = self.e_factor * ((1 - self.sky_view_factor) * (t_ext + degC_to_K) ** 4 +
                                         self.sky_view_factor * (t_sky + degC_to_K) ** 4)
        t_surf_init = self.radiation_frac * self.t_boundary + (1 - self.radiation_frac) * t_ext  # excludes irradiance

        # calculate exterior surface temperature and radiation
        # Option 1: Use previous time step surface temperature to calculate LWR, then recalculate surface temperature
        #  - running multiple times per time step for longer time steps
        #  - adding heavy ball convergence to reduce instability in determining the temperature
        for _ in range(self.iterations):
            self.lwr_gain = h_lwr_inj - self.e_factor * (self.temperature + degC_to_K) ** 4
            t_surf_new = t_surf_init + (self.solar_gain + self.lwr_gain) * self.radiation_res
            t_surf_new = min(max(t_surf_new, self.temperature - 2), self.temperature + 2)
            t_surf = self.temperature + 0.5 * (t_surf_new - self.temperature) + 0.1 * (self.temperature - self.t_prev)
            self.t_prev = self.temperature
            self.temperature = t_surf
            t_error = abs(self.temperature - self.t_prev)
            if t_error < 0.01:
                break
            if t_error > 5:
                print(f'WARNING: Large fluctuations in {self.boundary_name} long-wave radiation')
        self.lwr_gain = h_lwr_inj - self.e_factor * (self.temperature + degC_to_K) ** 4

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


class ExteriorZone:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.t_idx = None  # Index of envelope inputs for zone temperature
        self.temperature = None


class Zone:
    """
    Class for interior air zone, e.g. main indoor zone, garage, attic, foundation.
    """

    def __init__(self, name, label, time_res, initial_schedule, **zone_args):
        self.name = name
        self.label = label
        self.zone_type = zone_args.get('Zone Type', 'Default')  # used for basement heat fraction
        self.t_idx = None  # Index of envelope outputs for zone temperature
        self.h_idx = None  # Index of envelope inputs for zone heat gain
        self.temperature = None

        # Internal gains parameters
        self.internal_sens_gain = 0  # for non-HVAC equipment sensible gains, in W
        self.internal_latent_gain = 0  # for non-HVAC equipment latent gains, in W
        self.hvac_sens_gain = 0  # for HVAC equipment sensible gains, in W
        self.hvac_latent_gain = 0  # for HVAC equipment latent gains, in W

        # Internal zone radiation parameters
        self.surfaces = []
        self.s_view_factors = None  # all surface view factors
        self.s_e_factors = None  # all surface emissivity factors
        self.s_rad_fractions = None  # all surface radiation fractions
        self.s_rad_resistances = None  # all surface radiation resistances
        # number of iterations to calculate LWR
        self.iterations = time_res // dt.timedelta(minutes=5) + 3
        self.radiation_heat = 0  # gains from windows/interior radiation, in W

        # get zone capacitance from RC parameters and multiplier
        self.volume = zone_args.get('Volume (m^3)')
        if self.volume is not None:
            capacitance_multiplier = zone_args.get('Thermal Mass Multiplier', 7)
            self.capacitance = self.volume * rho_air * cp_air * capacitance_multiplier  # in kJ/K
        else:
            self.capacitance = None

        # Humidity parameters - Indoor zone only for now
        humidity_default = True if self.name == 'Indoor' else False
        if zone_args.get('enable_humidity', humidity_default):
            assert self.volume is not None
            self.humidity = HumidityModel(time_res, initial_schedule, self.volume)
        else:
            self.humidity = None

        # Infiltration parameters
        self.inf_heat = 0  # in W
        self.inf_flow = 0  # in m^3/s
        self.infiltration_method = zone_args.get('Infiltration Method')
        assert self.volume is not None or self.infiltration_method is None
        inf_params = {
            'ASHRAE': ['inf_n_i', 'inf_sft', 'inf_c', 'inf_Cs', 'inf_Cw'],
            'ELA': ['ELA (cm^2)', 'ELA stack coefficient (L/s/cm^4/K)', 'ELA wind coefficient (L/s/cm^4/(m/s))'],
            'ACH': ['Air Changes (1/hour)'],
            None: [],
        }
        self.infiltration_parameters = {param: zone_args[param] for param in inf_params[self.infiltration_method]}

        # Forced ventilation parameters - Indoor zone only for now
        # self.max_flow_rate = self.volume / kwargs['time_res'].total_seconds()  # in m^3/s
        self.forced_vent_heat = 0  # in W
        self.forced_vent_flow = 0  # m^3/s
        self.ventilation_type = zone_args.get('Ventilation Type')
        self.balanced_ventilation = zone_args.get('Balanced Ventilation', True)
        self.sens_recovery_eff = zone_args.get('Sensible Recovery Efficiency (-)', 0)
        self.lat_recovery_eff = zone_args.get('Latent Recovery Efficiency (-)', 0)
        if self.sens_recovery_eff > 0:
            assert self.balanced_ventilation

        # Natural ventilation parameters - Indoor zone only
        # TODO: why use ASHRAE for infiltration and ELA for natural ventilation?
        self.nat_vent_heat = 0  # in W
        self.nat_vent_flow = 0  # m^3/s
        self.nat_vent_stack_coeff = zone_args.get('ELA stack coefficient (L/s/cm^4/K)')
        self.nat_vent_wind_coeff = zone_args.get('ELA wind coefficient (L/s/cm^4/(m/s))')
        self.open_window_area = None

    def create_surfaces(self, boundaries):
        if not boundaries:
            return

        # find surfaces connected to this zone
        self.surfaces.extend([b.int_surface for b in boundaries if b.int_surface.zone_label == self.label])
        self.surfaces.extend([b.ext_surface for b in boundaries if b.ext_surface.zone_label == self.label])

        # Check that all surfaces have emissivity defined
        for surface in self.surfaces:
            if surface.emissivity is None:
                raise ModelException(f'{surface.boundary_name} missing Interior Thermal Absorptivity.')

        # calculate view factors for LWR - account for area and emissivity/absorptivity of LWR
        # Option 1: surfaces see themselves, maintains reciprocity
        total_area_absorptivity = sum([s.area * s.emissivity for s in self.surfaces])
        self.s_view_factors = np.array([s.area * s.emissivity / total_area_absorptivity for s in self.surfaces])

        # Option 2: no self-viewing surfaces, does not maintain reciprocity (not implemented)
        # viewable_area = total_area - surface.area
        # s.zone_view_factor = s.area / viewable_area

        # save data from surfaces to zone
        self.s_e_factors = np.array([s.e_factor for s in self.surfaces])
        self.s_rad_fractions = np.array([s.radiation_frac for s in self.surfaces])
        self.s_rad_resistances = np.array([s.radiation_res for s in self.surfaces])

        # Check total of view factors - should sum to 1
        if abs(1 - self.s_view_factors.sum()) > 0.01:
            raise ModelException(f'Error calculating zone view factors for {self.name} Zone.')

        # calculate view factors for transmitted solar from windows - accounts for area and solar absorptivity
        # window_view_surfaces = [s for s in self.surfaces if s.boundary.label in ['FL', 'FF', 'IW', 'IM']]
        window_view_surfaces = self.surfaces
        total_area_absorptivity = sum([s.area * s.absorptivity for s in window_view_surfaces])
        for s in window_view_surfaces:
            s.window_view_factor = s.area * s.absorptivity / total_area_absorptivity

        # Check total of view factors - should sum to 1
        check = sum([s.window_view_factor for s in self.surfaces])
        if abs(1 - check) > 0.01:
            raise ModelException('Error calculating window view factors for {}.'.format(self.name))

        # get window area in zone, for natural ventilation
        # fractions based on operable windows, open window area fraction, and % of windows actually open
        window_area = sum([s.area for s in self.surfaces if s.boundary_name == 'Window'])
        if window_area > 0:
            self.open_window_area = window_area * 0.67 * 0.5 * 0.2  # in m^2

    def update_infiltration(self, t_ext, t_zone, wind_speed, density, w_amb=0, h_limit=None, vent_cfm=0, t_base=None):
        # Calculates flow rate and heat gain from infiltration and ventilation (forced and natural)
        # Calculate infiltration flow, depending on the infiltration method
        # See E+ EMS program in idf file: infil_program (inf_flow = Qinf)
        delta_t = t_ext - t_zone
        if self.infiltration_method == 'ASHRAE':
            params = self.infiltration_parameters
            inf_flow_temp = params['inf_c'] * params['inf_Cs'] * abs(delta_t) ** params['inf_n_i']
            inf_flow_wind = params['inf_c'] * params['inf_Cw'] * (params['inf_sft'] * wind_speed) ** (
                2 * params['inf_n_i'])
            self.inf_flow = (inf_flow_temp ** 2 + inf_flow_wind ** 2) ** 0.5

        elif self.infiltration_method == 'ACH':
            self.inf_flow = self.infiltration_parameters['Air Changes (1/hour)'] * self.volume / 3600

        elif self.infiltration_method == 'ELA':
            # FUTURE: update attic wind coefficient based on height. Inputs are in properties file already
            # see https://bigladdersoftware.com/epx/docs/8-6/input-output-reference/group-airflow.html
            #  - zoneinfiltrationeffectiveleakagearea
            f = 1
            params = self.infiltration_parameters
            self.inf_flow = f * params['ELA (cm^2)'] / 1000 * (
                params['ELA stack coefficient (L/s/cm^4/K)'] * abs(delta_t) +
                params['ELA wind coefficient (L/s/cm^4/(m/s))'] * wind_speed ** 2) ** 0.5

        elif self.infiltration_method is None:
            pass
        else:
            raise ModelException(f'Unknown infiltration method: {self.infiltration_method}')

        if self.name == 'Indoor':
            # calculate forced ventilation flow, indoor zone only
            self.forced_vent_flow = vent_cfm * cfm_to_m3s  # in m^3/s

            # calculate natural ventilation flow, indoor zone only
            # taken from ResStock, see apply_natural_ventilation_and_whole_house_fan
            # TODO: add occupancy logic (only on if occupancy > 0)
            max_oa_hr = 0.0115  # From BA HSP
            if t_base is None:
                t_base = convert(73, 'degF', 'degC')
            # max_oa_rh = 0.7 # Note: removing check for max RH
            run_nat_vent = (w_amb < max_oa_hr) and (t_zone > t_ext) and (t_zone > t_base)
            if run_nat_vent and self.open_window_area is not None:
                area = self.open_window_area * 0.6
                nat_vent_area = convert(area, 'ft^2', 'cm^2')
                max_nat_flow = convert(20 * self.volume, 'm^3/hr', 'm^3/s')  # max 20 ACH
                adj = (t_zone - t_base) / (t_zone - t_ext)
                adj = max(min(adj, 1), 0)
                nat_vent_flow = nat_vent_area * adj * ((((self.nat_vent_stack_coeff * abs(delta_t)) +
                                                         (self.nat_vent_wind_coeff * (wind_speed ** 2))) ** 0.5) / 1000)
                self.nat_vent_flow = min(nat_vent_flow, max_nat_flow)
            else:
                self.nat_vent_flow = 0

        # combine infiltration and ventilation flow
        total_nat_flow = self.inf_flow + self.nat_vent_flow
        if self.balanced_ventilation:
            # All flows are balanced, add in series
            # total_flow = self.inf_flow + self.forced_vent_flow
            sensible_flow = (total_nat_flow + self.forced_vent_flow * (1 - self.sens_recovery_eff))
            latent_flow = (total_nat_flow + self.forced_vent_flow * (1 - self.lat_recovery_eff))
        else:
            # Add balanced + unbalanced in quadrature, reduce natural flows so that the results sum to the total
            sensible_flow = (total_nat_flow ** 2 + self.forced_vent_flow ** 2) ** 0.5
            latent_flow = sensible_flow
            nat_flow_ratio = (sensible_flow - self.forced_vent_flow) / total_nat_flow
            self.inf_flow *= nat_flow_ratio  # for results only
            self.nat_vent_flow *= nat_flow_ratio  # for results only

        # calculate sensible heat gains
        sensible_gain = sensible_flow * density * cp_air * delta_t * 1000  # in W
        if h_limit is not None and abs(sensible_gain) > abs(h_limit):
            # clip sensible gain based on heat gain limit
            sensible_gain = h_limit

        # TODO: add heavy ball convergence to fix high wind issues for attics
        # heat_flow = np.clip(heat_flow, 0, 400)
        # inf_heat = heat_flow * delta_t  # in W
        # # if abs(inf_heat) > max(abs(self.inf_heat), abs(h_limit) / 2):
        # #     # if heat gains are increasing, reduce rate of increase to prevent instability in zone temperature
        # #     inf_heat = self.inf_heat + 0.6 * (inf_heat - self.inf_heat)
        #
        # # limit heat gain/loss based on ambient
        # self.inf_heat = np.clip(inf_heat, -abs(h_limit), abs(h_limit))

        # calculate component heat gains, proportional to flow rates, for results only
        self.inf_heat = sensible_gain * self.inf_flow / sensible_flow if sensible_flow != 0 else 0
        self.nat_vent_heat = sensible_gain * self.nat_vent_flow / sensible_flow if sensible_flow != 0 else 0
        self.forced_vent_heat = sensible_gain * self.forced_vent_flow / sensible_flow if sensible_flow != 0 else 0

        # calculate latent heat gains if humidity model exists
        if self.humidity is not None:
            latent_flow = min(latent_flow, self.humidity.max_latent_flow)
            latent_gains = latent_flow * self.humidity.h_vap * density * 1000 * (w_amb - self.humidity.w)  # in W
            self.humidity.latent_gains_init += latent_gains

        # FUTURE: add latent gains for other zones
        return sensible_gain

    def calculate_interior_radiation(self, t_zone):
        # calculate all internal surface temperatures and LWR radiation
        t_boundaries = np.array([s.t_boundary for s in self.surfaces])
        t_surf_no_rad = self.s_rad_fractions * t_boundaries + (1 - self.s_rad_fractions) * t_zone  # excludes radiation
        t_surf_min = min(t_surf_no_rad)
        t_surf_max = max(t_surf_no_rad)

        # TODO: try setting temperatures to weighted average of t_surf_no_rad
        # t_surfaces = np.ones(len(self.surfaces)) * t_surf_no_rad.dot(self.s_view_factors)
        # t_surfaces_prev = t_surfaces

        # Option 1: Use previous time step surface temperatures to calculate LWR, then recalculate surface temperatures
        #  - running multiple times per time step for longer time steps
        #  - adding heavy ball convergence to reduce instability in determining the temperature
        t_surfaces = np.array([s.temperature for s in self.surfaces])
        t_surfaces_prev = np.array([s.t_prev for s in self.surfaces])
        max_error = None
        for _ in range(self.iterations):
            # get total LWR heat from each surface and to each surface, based on current surface temperatures
            h_lwr_out = self.s_e_factors * (t_surfaces + degC_to_K) ** 4
            h_lwr_in = h_lwr_out.sum() * self.s_view_factors

            # update surface temperature based on difference in LWR gain, constrain to min/max values
            t_surfaces_new = t_surf_no_rad + (h_lwr_in - h_lwr_out) * self.s_rad_resistances
            t_surfaces_new = t_surfaces_new.clip(t_surf_min, t_surf_max)
            t_surfaces_new = t_surfaces + 0.3 * (t_surfaces_new - t_surfaces) + 0.2 * (t_surfaces - t_surfaces_prev)
            t_surfaces_prev = t_surfaces
            t_surfaces = t_surfaces_new
            max_error = np.abs(t_surfaces - t_surfaces_prev).max()
            if max_error < 0.01:
                break
        if max_error > 1:
            print(f'WARNING: Large fluctuations in {self.name} Zone internal radiation')
        # if t_surfaces.max() - t_surfaces.min() > 25:
        #     print(f'WARNING: Large differences in {self.name} Zone surface temperatures')

        # get final LWR gains
        h_lwr_out = self.s_e_factors * (t_surfaces + degC_to_K) ** 4
        h_lwr_in = h_lwr_out.sum() * self.s_view_factors
        h_lwr_net = h_lwr_in - h_lwr_out
        if abs(h_lwr_net.sum()) > 10:
            raise ModelException(f'{self.name} Zone internal radiation error')

        # update surface values
        for i, s in enumerate(self.surfaces):
            s.temperature = t_surfaces[i]
            s.t_prev = t_surfaces_prev[i]
            s.lwr_gain = h_lwr_net[i]


class Boundary:
    """
    Class for a specific boundary within a building envelope, e.g. exterior walls, garage walls, etc.
    """

    def __init__(self, name, location, **kwargs):
        self.name = name
        self.label = kwargs['Boundary Label']
        self.area = sum(kwargs['Area (m^2)'])
        self.ext_zone_label = kwargs['Exterior Zone Label']
        self.int_zone_label = kwargs['Interior Zone Label']

        # Get film resistances
        new_bd_properties = utils.calculate_film_resistances(self.name, kwargs, location)
        kwargs = {**new_bd_properties, **kwargs}

        # Define starting (exterior) and ending (interior) zones
        # zone_labels = [zone.label for zone in zones] + list(EXT_ZONES.keys())
        # if ext_zone not in zone_labels:
        #     raise ModelException(
        #         'Cannot use {} boundary when {} zone is not included.'.format(self.name, ext_zone))
        # if int_zone not in zone_labels:
        #     raise ModelException(
        #         'Cannot use {} boundary when {} zone is not included.'.format(self.name, int_zone))

        # parse RC parameters to be readable for RCModel, get number and names for nodes
        cap_values = [kwargs['C_' + self.label + str(i)] for i in range(10) if 'C_' + self.label + str(i) in kwargs]
        res_values = [kwargs['R_' + self.label + str(i)] for i in range(10) if 'R_' + self.label + str(i) in kwargs]
        if not len(cap_values) and self.name != 'Window':
            raise OCHREException(f'Missing RC coefficients for {self.name} with properties: {kwargs}')
        if len(res_values) != len(cap_values):
            raise ModelException(f'Cannot parse RC data for {self.name}. Number of resistors ({len(res_values)}) is not'
                                 f' compatible with the number of capacitors ({len(cap_values)})')
        same_zones = self.ext_zone_label == self.int_zone_label
        u_window = kwargs.get('U Factor (W/m^2-K)')

        # convert RC parameters into RC network, remove 0 capacitance values, etc.
        cap_values, res_values = utils.create_rc_data(cap_values, res_values, same_zones, u_window)
        self.n_nodes = len(cap_values)
        self.all_nodes = [self.label + str(i + 1) for i in range(self.n_nodes)]

        # Create exterior and interior boundary surfaces
        node_ext = self.all_nodes[0] if self.n_nodes else self.int_zone_label  # node name closest to external zone
        self.ext_surface = BoundarySurface(self, True, self.ext_zone_label, node_ext, res_material=res_values[0],
                                           **kwargs)
        if not same_zones:
            node_int = self.all_nodes[-1] if self.n_nodes else self.ext_zone_label  # node name closest to internal zone
            self.int_surface = BoundarySurface(self, False, self.int_zone_label, node_int, res_material=res_values[-1],
                                               **kwargs)
        else:
            self.int_surface = BoundarySurface(self, False, **kwargs)  # empty surface

        # Save capacitor parameters, converts kJ/m^2-K to kJ/K
        self.capacitors = {'C_' + node: val * self.area for node, val in zip(self.all_nodes, cap_values)}  # in kJ/K

        # get all resistance node names and values (including film resistances)
        node_list = [self.ext_zone_label, self.label + '-ext'] + self.all_nodes
        res_values = [self.ext_surface.res_film] + res_values
        if not same_zones:
            # do not include int_zone if int and ext zones are the same
            node_list += [self.label + '-int', self.int_zone_label]
            res_values += [self.int_surface.res_film]

        # for resistances, convert boundary names to node names,
        # e.g. R_CL1 -> R_ATC_CL1,  R_CL2 -> R_CL1_CL2,  R_CL3 -> R_CL2_LIV
        assert len(node_list) == len(res_values) + 1
        self.resistors = {f'R_{node_list[i]}_{node_list[i + 1]}': r / self.area
                          for i, r in enumerate(res_values)}  # in K/W


class Envelope(RCModel):
    name = 'Envelope'
    optional_inputs = [
        'Ambient Humidity Ratio (-)',
        'Ambient Pressure (kPa)',
        'Sky Temperature (C)',
        'Occupancy (Persons)',
        'Internal Gains (W)',
        'Ventilation Rate (cfm)',
    ]

    def __init__(self, zones, boundaries=None, occupancy=None, location=None, linearize_infiltration=False,
                 external_radiation_method='full', internal_radiation_method='full', **kwargs):
        # Options for radiation methods: full, linear, none
        self.run_external_rad = external_radiation_method == 'full'
        linearize_ext_radiation = external_radiation_method == 'linear'
        self.run_internal_rad = internal_radiation_method == 'full'
        self.linearize_int_radiation = internal_radiation_method == 'linear'
        self.linearize_infiltration = linearize_infiltration

        # Create interior zones
        self.zones = {
            name: Zone(name, label, kwargs['time_res'], kwargs['initial_schedule'], **zones[name])
            for label, name in utils.ZONES.items() if name in zones
        }
        self.indoor_zone = self.zones['Indoor']

        # Create boundaries using parameters from envelope files
        if boundaries is not None:
            # Get detailed boundary properties, e.g. RC coefficients
            boundaries = utils.get_boundary_rc_values(boundaries, **kwargs)
            self.boundaries = [Boundary(name, location, linearize_ext_radiation=linearize_ext_radiation, **properties)
                               for name, properties in boundaries.items()]
        else:
            self.boundaries = []

        # Save external zones (for getting zone temperature)
        ext_zone_labels = kwargs.get('ext_zone_labels', set([bd.ext_zone_label for bd in self.boundaries]))
        self.ext_zones = {name: ExteriorZone(name, label)
                          for label, name in utils.EXT_ZONES.items() if label in ext_zone_labels}

        # Create dictionaries of exterior and interior boundaries
        self.ext_boundaries = [bd for bd in self.boundaries if bd.ext_surface.is_exterior]

        # Update zones to include boundary surface information
        for zone in self.zones.values():
            zone.create_surfaces(self.boundaries)

        # Add required inputs for envelope schedule
        required_inputs = ['Ambient Dry Bulb (C)']
        if 'Ground' in self.ext_zones:
            required_inputs.append('Ground Temperature (C)')
        if any([zone.infiltration_method == 'ASHRAE' for zone in self.zones.values()]):
            required_inputs.append('Wind Speed (m/s)')
        required_inputs.extend([f'{bd.name} Irradiance (W)'
                                for bd in self.ext_boundaries if bd.name not in ['Raised Floor']])

        # remove heat injection inputs that aren't into main nodes or nodes with a radiation injection
        unused_inputs = [f'H_{node}' for bd in self.boundaries for node in bd.all_nodes
                         if node not in [bd.int_surface.node, bd.ext_surface.node]]
        if self.linearize_infiltration:
            # Note: this won't remove a 1-node internal boundary (e.g. Attic Floor)
            unused_inputs += [f'H_{s.node}' for zone in self.zones.values() for s in zone.surfaces
                              if s.boundary.n_nodes > 1]

        # Generate RC Model
        external_nodes = [zone.label for zone in self.ext_zones.values()]
        outputs = ['T_' + zone.label for zone in self.zones.values()]
        super().__init__(external_nodes=external_nodes, outputs=outputs, unused_inputs=unused_inputs,
                         required_inputs=required_inputs, **kwargs)

        # Check that envelope model is not reduced if non-linear radiation methods are used
        if self.reduced and (self.run_internal_rad or self.run_external_rad):
            raise ModelException('Cannot run non-linear radiation methods with a reduced Envelope model. '
                                 "Set external_radiation_method='linear' and internal_radiation_method='linear', "
                                 "or use a full envelope model.")

        # Print warnings for columns that aren't included in schedule
        if 'Ambient Pressure (kPa)' not in self.schedule:
            self.warn('Ambient pressure not in schedule. Using standard pressure of 1 atm (101.3 kPa).')
        if 'Occupancy (Persons)' not in self.schedule:
            self.warn('Occupancy not in schedule. Ignoring heat gains from occupants.')
        if self.run_external_rad and 'Sky Temperature (C)' not in self.schedule:
            self.warn('Sky temperature not in schedule. May impact external radiation method.')

        # Set initial surface temperatures based on closest zone
        #  and save state and input indices for faster updates
        for zone in self.zones.values():
            zone.t_idx = self.output_names.index('T_' + zone.label)
            zone.h_idx = self.input_names.index('H_' + zone.label)
            zone.temperature = self.outputs[zone.t_idx]
            for surface in zone.surfaces:
                surface.temperature = zone.temperature
                surface.t_prev = zone.temperature
                if 'T_' + surface.node in self.state_names:
                    surface.t_idx = self.state_names.index('T_' + surface.node)
                if 'H_' + surface.node in self.input_names:
                    surface.h_idx = self.input_names.index('H_' + surface.node)
        for bd in self.ext_boundaries:
            surface = bd.ext_surface
            surface.temperature = kwargs['initial_schedule']['Ambient Dry Bulb (C)']
            surface.t_prev = kwargs['initial_schedule']['Ambient Dry Bulb (C)']
            if 'T_' + surface.node in self.state_names:
                surface.t_idx = self.state_names.index('T_' + surface.node)
            if 'H_' + surface.node in self.input_names:
                surface.h_idx = self.input_names.index('H_' + surface.node)
        for zone in self.ext_zones.values():
            zone.t_idx = self.input_names.index('T_' + zone.label)
            zone.temperature = self.inputs[zone.t_idx]

        # Occupancy parameters, units are W/person
        if occupancy is None:
            occupancy = {}
        occupancy_gain = occupancy.get('Gain per Occupant (W)', convert(400, 'Btu/hour', 'W'))
        self.occupancy_sensible_gain = (occupancy.get('Convective Gain Fraction (-)', 0.563) +
                                        occupancy.get('Radiative Gain Fraction (-)', 0)) * occupancy_gain
        self.occupancy_latent_gain = occupancy.get('Latent Gain Fraction (-)', 0.437) * occupancy_gain

        # HVAC parameters
        self.heating_setpoint = None
        self.heating_deadband = None
        self.cooling_setpoint = None
        self.cooling_deadband = None
        self.unmet_hvac_load = 0  # units are C, equivalent to C * self.time_res

    def load_rc_data(self, **kwargs):
        # combine rc data from all boundaries
        def update_with_par(dict1, dict2):
            for key, val in dict2.items():
                if key not in dict1:
                    dict1[key] = val
                else:
                    # combine resistors in parallel
                    dict1[key] = self.par(dict1[key], val)
            return dict1

        all_resistances = {}
        all_capacitances = {}
        for boundary in self.boundaries:
            all_resistances = update_with_par(all_resistances, boundary.resistors)

            all_capacitances.update(**boundary.capacitors)

        # add capacitances from zones
        all_capacitances.update({'C_' + zone.label: zone.capacitance for zone in self.zones.values()})

        # For linear internal radiation, add resistors to RC network to approximate internal radiation
        # Note: linear external radiation updates are done in Boundary initialization
        if self.linearize_int_radiation:
            all_resistances = update_with_par(all_resistances, self.add_radiation_resistances())

        # For linear infiltration, add resistors to RC network to approximate infiltration
        if self.linearize_infiltration:
            all_resistances = update_with_par(all_resistances,
                                              self.add_infiltration_resistances(kwargs['initial_schedule']))

        # Convert capacitances from kJ to J
        all_capacitances = {key: val * 1000 for key, val in all_capacitances.items()}

        return {**all_resistances, **all_capacitances}

    @staticmethod
    def initialize_state(state_names, input_names, A_c, B_c, initial_temp_setpoint=None, **kwargs):
        # Sets all temperatures to the steady state value based on initial conditions
        # Adds random temperature if exact setpoint is not set
        # Note: initialization will update the initial state to more typical values
        outdoor_temp = kwargs['initial_schedule']['Ambient Dry Bulb (C)']
        ground_temp = kwargs['initial_schedule'].get('Ground Temperature (C)', 10)

        # get HVAC Heating/Cooling deadbands and setpoints
        options = ['HVAC Heating', 'HVAC Cooling']
        # TODO: Deadbands are broken. Need to decide whether to get from HPXML, dict, or schedule
        deadbands = [kwargs.get(option, {}).get('Deadband Temperature (C)', 1) for option in options]
        setpoints = [kwargs['initial_schedule'].get(f'{option} Setpoint (C)') for option in options]
        nones = sum([setpoint is None for setpoint in setpoints])
        if nones == 2 and not isinstance(initial_temp_setpoint, (int, float)):
            print('WARNING: No Heating or Cooling Setpoint in schedule, and initial_temp_setpoint is not defined.'
                  ' Setting initial setpoint to 21 degC.')
            setpoints = [21, 21]
        elif nones == 1:
            # use same setpoint for both
            s = [setpoint for setpoint in setpoints if setpoint is not None][0]
            setpoints = [s, s]

        # Get initial Indoor temperature
        if isinstance(initial_temp_setpoint, (int, float)):
            indoor_temp = initial_temp_setpoint
        elif initial_temp_setpoint in options:
            idx = options.index(initial_temp_setpoint)
            random_delta = np.random.uniform(low=-deadbands[idx] / 2, high=deadbands[idx] / 2)
            indoor_temp = setpoints[idx] + random_delta
        elif initial_temp_setpoint is None:
            # select heating/cooling setpoint based on starting outdoor temperature (above/below 12 C)
            idx = 1 if outdoor_temp > 12 else 0
            random_delta = np.random.uniform(low=-deadbands[idx] / 2, high=deadbands[idx] / 2)
            indoor_temp = setpoints[idx] + random_delta
        else:
            raise ModelException('Unknown initial temperature setpoint: {}'.format(initial_temp_setpoint))

        # Update continuous time matrices to swap T_LIV from state to input
        x_idx = state_names.index('T_LIV')
        keep_states = np.ones(len(state_names), dtype=bool)
        keep_states[x_idx] = False
        keep_inputs = [input_names.index('T_EXT')]
        input_values = [indoor_temp, outdoor_temp]
        if 'T_GND' in input_names:
            assert ground_temp is not None
            keep_inputs.append(input_names.index('T_GND'))
            input_values.append(ground_temp)
        A = A_c[keep_states, :][:, keep_states]
        B = np.hstack([A_c[keep_states, :][:, [x_idx]], B_c[keep_states, :][:, keep_inputs]])
        u = np.array(input_values)

        # Calculate steady state values (effectively interpolates from the input temperatures)
        x = - np.linalg.inv(A).dot(B).dot(u)
        x = np.insert(x, x_idx, indoor_temp)

        # Return states as a dictionary
        return {name: val for name, val in zip(state_names, x)}

    def get_input_weights(self):
        # TODO: Need to test if weights improve performance
        # Default weights of 1 for temperatures (in C) and 50 for heat gains (in W)
        weights = {name: 1 if 'T_' in name else 50 for name in self.input_names}

        # Update heat gain weights for zones to 1000, and for exterior nodes to 2000
        for zone in self.zones.values():
            weights[f'H_{zone.label}'] = 1000
        for bd in self.ext_boundaries:
            weights[f'H_{bd.ext_surface.node}'] = 2000

        return weights

    def add_radiation_resistances(self):
        # add resistors to RC network to approximate internal radiation
        # linearizes radiation equation: H_ab = e_factor * (Ta^4 - Tb^4)  (Ta="average" of surface temps, Tb=boundary)
        #   -> dH/dT = e_factor * 4 * T^3 = 1/Rab
        # assumes operating point of T=20C for all boundary/zone temperatures
        # Note: node <label>-rad is removed from envelope model using star-mesh transform
        t_ref = 20 + degC_to_K
        radiation_res = {}
        for zone in self.zones.values():
            for surface in zone.surfaces:
                radiation_res[f'R_{surface.node}_{zone.label}-rad'] = 1 / (4 * surface.e_factor * t_ref ** 3)

        return radiation_res

    def add_infiltration_resistances(self, initial_schedule):
        # add resistors to RC network to approximate infiltration
        # linearizes infiltration equation: H_ab = inf_heat = k * (T_ext - T_zone), k = 1/Rab
        # k depends on infiltration method
        # assumes operating point of vent_rate=<initial_value>, delta_t=10C, wind_speed=<initial_value>
        wind_speed_op = initial_schedule.get('Wind Speed (m/s)')
        vent_cfm_op = initial_schedule.get('Ventilation Rate (cfm)', 0)
        pressure_op = initial_schedule.get('Ambient Pressure (kPa)', 101.325)
        t_ext_op = 10  # in C
        t_zone_op = 20  # in C
        density = HumidityModel.get_dry_air_density(20, 0, pressure_op)

        radiation_res = {}
        for zone in self.zones.values():
            heat = zone.update_infiltration(t_ext_op, t_zone_op, wind_speed_op, density, vent_cfm=vent_cfm_op)
            if heat:
                res = (t_ext_op - t_zone_op) / heat
                radiation_res[f'R_{zone.label}_EXT'] = res

        return radiation_res

    def get_ebm_parameters(self, **kwargs):
        # TODO: calculates 1R-1C parameters for HVAC equivalent battery model
        # Create linear envelope, hourly resolution or use input param for time res
        # get capacitance as dt / B[t_liv, h_liv]
        # for R, maybe R = steady state increase in t_liv / small increase in h_liv, using A_c, B_c
        #  - see initialize state
        pass

    def update_radiation(self):
        t_ext = self.current_schedule['Ambient Dry Bulb (C)']
        t_sky = self.current_schedule.get('Sky Temperature (C)')

        # Calculate external radiation (solar and LWR) by boundary
        for boundary in self.ext_boundaries:
            surface = boundary.ext_surface
            solar_gain = self.current_schedule.get(f'{boundary.name} Irradiance (W)', 0)

            # get surface absorbed solar gain
            surface.solar_gain = solar_gain * surface.absorptivity
            h_radiation = surface.solar_gain

            # get long wave exterior radiation gain
            if not self.reduced:
                assert surface.t_idx is not None
                surface.t_boundary = self.states[surface.t_idx]
            if self.run_external_rad:
                surface.calculate_exterior_radiation(t_ext, t_sky)
                h_radiation += surface.lwr_gain

            # add solar and exterior radiation gains to inputs
            if boundary.n_nodes == 0:
                # assert that surface.h_idx is the internal zone, add to zone.radiation_heat (e.g. for windows)
                zone = self.zones[boundary.int_surface.zone_name]
                assert zone.h_idx == surface.h_idx
                surface.radiation_to_zone += h_radiation * surface.radiation_frac
                zone.radiation_heat += h_radiation * surface.radiation_frac
            else:
                self.inputs_init[surface.h_idx] += h_radiation * surface.radiation_frac

            # Get solar radiation transmitted through windows, injected into zone and other boundaries
            surface.transmitted_gain = solar_gain * surface.transmittance
            if surface.transmitted_gain > 0:
                zone = self.zones[boundary.int_surface.zone_name]

                # add fraction of window solar gains to each surface, based on area and absorptivity ratios
                # Note: some heat is reflected back out of the windows and lost
                for s in zone.surfaces:
                    if s.h_idx is not None:
                        self.inputs_init[s.h_idx] += surface.transmitted_gain * s.window_view_factor * s.radiation_frac
                    surface.radiation_to_zone += surface.transmitted_gain * s.window_view_factor * (1 - s.radiation_frac)
                    zone.radiation_heat += surface.transmitted_gain * s.window_view_factor * (1 - s.radiation_frac)

        # Calculate internal radiation by zone
        for zone in self.zones.values():
            if not self.reduced:
                for surface in zone.surfaces:
                    # collect all surface boundary temperatures
                    if surface.t_idx is not None:
                        surface.t_boundary = self.states[surface.t_idx]
                    else:
                        # boundary has no nodes and has opposite surface at external node (e.g. windows)
                        # set boundary temp to opposite surface temperature (should be updated from external radiation)
                        assert surface == surface.boundary.int_surface
                        opposite_surface = surface.boundary.ext_surface
                        surface.t_boundary = opposite_surface.temperature

            if self.run_internal_rad:
                # update radiation for all boundaries within zone
                zone.calculate_interior_radiation(zone.temperature)
                for surface in zone.surfaces:
                    if surface.t_idx is not None:
                        # for windows, some of the heat is lost through the boundary
                        self.inputs_init[surface.h_idx] += surface.lwr_gain * surface.radiation_frac
                    surface.radiation_to_zone += surface.lwr_gain * (1 - surface.radiation_frac)
                    zone.radiation_heat += surface.lwr_gain * (1 - surface.radiation_frac)

            # Update zone inputs with total radiation heat gains (windows + internal radiation)
            self.inputs_init[zone.h_idx] += zone.radiation_heat

    def update_infiltration(self):
        if self.linearize_infiltration:
            # Don't run infiltration update - infiltration accounted for in RC network
            return

        # Get schedule values
        t_ext = self.current_schedule['Ambient Dry Bulb (C)']
        wind_speed = self.current_schedule.get('Wind Speed (m/s)')
        w_amb = self.current_schedule.get('Ambient Humidity Ratio (-)')
        pressure = self.current_schedule.get('Ambient Pressure (kPa)', 101.325)

        # Calculate outdoor dry air density, used for calculating infiltration/ventilation heat gains
        if w_amb is not None:
            density = HumidityModel.get_dry_air_density(t_ext, w_amb, pressure)
        else:
            # use typical value
            density = 1.225

        # calculate infiltration for all zones, ventilation for Indoor zone only
        for zone in self.zones.values():
            # solve for the maximum infiltration gain to achieve outdoor temperature
            h_limit = self.solve_for_input(zone.t_idx, zone.h_idx, t_ext, solve_as_output=True)

            # adjust infiltration heat gain limit if external gains cancels out the effects of infiltration
            # Note: doesn't depend on the sign of h_limit
            ext_gains = self.inputs_init[zone.h_idx]
            if (zone.temperature < t_ext) ^ (ext_gains > 0):
                h_limit -= ext_gains

            if zone.name == 'Indoor':
                vent_cfm = self.current_schedule.get('Ventilation Rate (cfm)', 0)
                if self.heating_setpoint is not None and self.cooling_setpoint is not None:
                    t_base = (self.heating_setpoint + self.cooling_setpoint) / 2
                else:
                    t_base = None
                h_inf = zone.update_infiltration(t_ext, zone.temperature, wind_speed, density, w_amb, h_limit, vent_cfm,
                                                 t_base)
            else:
                h_inf = zone.update_infiltration(t_ext, zone.temperature, wind_speed, density, w_amb, h_limit)
            self.inputs_init[zone.h_idx] += h_inf

    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)

        # reset all inputs to defaults, including latent gains
        self.inputs_init = np.zeros(len(self.input_names))
        for zone in self.zones.values():
            if zone.humidity is not None:
                zone.humidity.latent_gains_init = 0
                zone.humidity.pressure = self.current_schedule.get('Ambient Pressure (kPa)', 101.325) * 1000  # in Pa
            zone.radiation_heat = 0
            zone.internal_sens_gain = 0
            zone.internal_latent_gain = 0
            zone.hvac_sens_gain = 0
            zone.hvac_latent_gain = 0
            for surface in zone.surfaces:
                surface.internal_gain = 0
                surface.radiation_to_zone = 0

        # Add external temperatures to inputs
        self.ext_zones['Outdoor'].temperature = self.current_schedule['Ambient Dry Bulb (C)']
        if 'Ground' in self.ext_zones:
            self.ext_zones['Ground'].temperature = self.current_schedule['Ground Temperature (C)']
        for zone in self.ext_zones.values():
            self.inputs_init[zone.t_idx] = zone.temperature

        # Add occupancy sensible and latent heat gains to indoor zone
        occupancy = self.current_schedule.get('Occupancy (Persons)', 0)
        self.inputs_init[self.indoor_zone.h_idx] += occupancy * self.occupancy_sensible_gain  # in W
        if self.indoor_zone.humidity is not None:
            self.indoor_zone.humidity.latent_gains_init += occupancy * self.occupancy_latent_gain

        # Add other internal gains to indoor zone
        # Note: adding to indoor_zone.internal_sens_gain, will get set to 0 if Dwelling power is shut off
        other_gains = self.current_schedule.get('Internal Gains (W)', 0)
        self.indoor_zone.internal_sens_gain += other_gains  # in W
        # self.inputs_init[self.indoor_zone.h_idx] += other_gains  # in W

        # Update solar radiation, external LWR, and internal LWR
        self.update_radiation()

        # Update infiltration and ventilation (best if done last)
        self.update_infiltration()

    def update_model(self, control_signal=None):
        assert not control_signal  # no Envelope controls allowed, will get overwritten

        # Add zone and surface sensible gains to envelope inputs
        equipment_sens_gains = np.zeros(self.nu, dtype=float)
        for zone in self.zones.values():
            equipment_sens_gains[zone.h_idx] += zone.internal_sens_gain + zone.hvac_sens_gain
            for surface in zone.surfaces:
                equipment_sens_gains[surface.h_idx] += surface.internal_gain

            # Add zone latent gains to humidity models
            if zone.humidity is not None:
                zone.humidity.latent_gains = zone.humidity.latent_gains_init 
                zone.humidity.latent_gains += zone.internal_latent_gain + zone.hvac_latent_gain

        control_signal = self.inputs_init + equipment_sens_gains
        return super().update_model(control_signal)

    def update_results(self):
        # check that states are within reasonable range
        t_liv = self.next_outputs[self.indoor_zone.t_idx]
        t_state_min = self.next_states.min()
        t_state_max = self.next_states.max()
        if ((self.cooling_setpoint is not None and t_liv > 30) or
                (self.heating_setpoint is not None and t_liv < 10) or
                (not self.reduced and ((t_state_min < -40) or (t_state_max > 110)))):
            bad_temps = {name: t for name, t in zip(self.state_names, self.next_states) if (t < -40) or (t > 110)}
            self.warn(f'Extreme envelope temperatures. Indoor temp={t_liv}. Extreme temps: {bad_temps}')
        if not (-20 < t_liv < 50) or (not self.reduced and ((t_state_min < -55) or (t_state_max > 130))):
            bad_temps = {name: t for name, t in zip(self.state_names, self.next_states) if (t < -55) or (t > 130)}
            raise ModelException(f'Envelope temperatures are outside acceptable range. '
                                 f'Indoor temp={t_liv}. Extreme temps: {bad_temps}')

        current_results = super().update_results()

        for zone in self.zones.values():
            # Update zone temperatures from outputs
            zone.temperature = self.outputs[zone.t_idx]

            # Run humidity update - indoor zone only
            if zone.humidity is not None:
                zone.humidity.update_humidity(self.outputs[zone.t_idx])

        # Calculate unmet thermal loads - negative=cold/unmet heating load, positive=hot/unmet cooling load
        self.unmet_hvac_load = 0
        if self.cooling_setpoint is not None:
            hot_comfort_temp = self.cooling_setpoint + self.cooling_deadband / 2
            self.unmet_hvac_load += max(t_liv - hot_comfort_temp, 0)
        if self.heating_setpoint is not None:
            cold_comfort_temp = self.heating_setpoint - self.heating_deadband / 2
            self.unmet_hvac_load -= max(cold_comfort_temp - t_liv, 0)

        return current_results

    def get_zone_temperature(self, zone_name):
        # get temperatures from interior or exterior zone
        if zone_name is None:
            return None
        if zone_name in self.zones:
            return self.zones[zone_name].temperature
        elif zone_name in self.ext_zones:
            return self.ext_zones[zone_name].temperature
        else:
            raise OCHREException(f'Unknown zone name {zone_name}.')

    def add_component_loads(self):
        # TODO
        # add conduction components, compare to a model with constant temperature across all states
        constant_state = np.ones(self.nx) * self.indoor_zone.temperature
        # m_i_inv.dot(y_values - c_i.dot(self.A.dot(self.states - constant_state)
        # same for outdoor temp, ground temp
        # assign a component (one for each boundary next to a conditioned space) to each state/temperature
        # will need to assign ratios for multiple components for outdoor/ground temps
        # maybe use total resistance of a boundary for the ratios, ignore capacitance

        # add internal radiation components
        # maybe convert zone radiation to dictionary, keys are surfaces

        # add external radiation and solar components (to window only)
        # dictionary for zone radiation

        # add infiltration and ventilation components
        # take directly from outputs

        # add internal gains
        # take directly from outputs
        return {}

    def generate_results(self):
        # Note: most results are included in Dwelling/HVAC. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 3:
            # Indoor temperature and unmet loads
            results["Temperature - Indoor (C)"] = self.indoor_zone.temperature
            results['Unmet HVAC Load (C)'] = self.unmet_hvac_load

        if self.verbosity >= 5:
            # All zone temperatures
            results.update({f'Temperature - {name} (C)': zone.temperature for name, zone in self.zones.items()})
            results.update({f'Temperature - {name} (C)': zone.temperature for name, zone in self.ext_zones.items()})

            # All component loads (for indoor zone) and net load
            # Net sensible gains =  occupancy + HVAC + equipment
            #                     + infiltration + forced ventilation + natural ventilation
            #                     + absorbed ext. radiation (windows) + transmitted window gains + interior radiation
            results.update({f'Net Sensible Heat Gain - {name} (W)': self.inputs[zone.h_idx]
                            for name, zone in self.zones.items()})
            if not self.linearize_infiltration:
                results["Infiltration Heat Gain - Indoor (W)"] = self.indoor_zone.inf_heat
            results["Forced Ventilation Heat Gain - Indoor (W)"] = self.indoor_zone.forced_vent_heat
            results["Natural Ventilation Heat Gain - Indoor (W)"] = self.indoor_zone.nat_vent_heat
            occupant_gain = self.current_schedule.get('Occupancy (Persons)', 0) * self.occupancy_sensible_gain
            # internal gains = occupancy + non-HVAC equipment only
            results["Internal Heat Gain - Indoor (W)"] = (
                occupant_gain + self.indoor_zone.internal_sens_gain
            )

            # Add window transmittance (note, gains go to indoor zone and to interior boundaries)
            windows = [bd for bd in self.ext_boundaries if bd.name == 'Window']
            if windows:
                results['Window Transmitted Solar Gain (W)'] = windows[0].ext_surface.transmitted_gain

            # TODO: add other component loads
            if not self.reduced:
                results.update(self.add_component_loads())
                # add heat injections from boundaries into zones (pos=heat injected to zone)
                # outdoor_zone = self.ext_zones['Outdoor']
                # zone_surfaces = [(outdoor_zone, bd.ext_surface) for bd in self.ext_boundaries]
                # zone_surfaces += [(zone, s) for zone in self.zones.values() for s in zone.surfaces]
                # for zone, surface in zone_surfaces:
                #     if not surface.node or surface.t_boundary is None:
                #         continue
                #     convection = (surface.t_boundary - zone.temperature) / surface.res_total
                #     results[f'Convection from {surface.boundary_name} to {zone.name} (W)'] = convection

        if self.verbosity >= 8:
            results['Occupancy (Persons)'] = self.current_schedule.get('Occupancy (Persons)', 0)
            # Add detailed heat gain results for each zone
            for name, zone in self.zones.items():
                if not self.linearize_infiltration:
                    results[f'Infiltration Flow Rate - {name} (m^3/s)'] = zone.inf_flow
                    results[f'Infiltration Heat Gain - {name} (W)'] = zone.inf_heat

                if name == 'Indoor':
                    results[f'Forced Ventilation Flow Rate - {name} (m^3/s)'] = zone.forced_vent_flow
                    results[f'Natural Ventilation Flow Rate - {name} (m^3/s)'] = zone.nat_vent_flow
                    air_changes = (zone.inf_flow + zone.forced_vent_flow + zone.nat_vent_flow) / zone.volume * 3600
                    results[f'Air Changes per Hour - {name} (1/hour)'] = air_changes

                    occupant_gain = self.current_schedule.get('Occupancy (Persons)', 0) * self.occupancy_sensible_gain
                    results[f'Occupancy Heat Gain - {name} (W)'] = occupant_gain
                else:
                    occupant_gain = 0

                if occupant_gain + zone.internal_sens_gain > 0:
                    # occupancy + non-HVAC equipment only
                    results[f'Internal Heat Gain - {name} (W)'] = occupant_gain + zone.internal_sens_gain
                
                # add radiation gain from windows and internal radiation, in W
                if self.run_internal_rad:
                    results[f'Radiation Heat Gain - {name} (W)'] = zone.radiation_heat

                if zone.humidity is not None:
                    # Indoor is the only zone with humidity or ventilation (for now)
                    results[f'Relative Humidity - {name} (-)'] = zone.humidity.rh
                    results[f'Wet Bulb - {name} (C)'] = zone.humidity.wet_bulb
                    results[f'Humidity Ratio - {name} (-)'] = zone.humidity.w
                    results[f'Net Latent Heat Gain - {name} (W)'] = zone.humidity.latent_gains
                    results[f'Air Density - {name} (kg/m^3)'] = zone.humidity.density

        if self.verbosity >= 9:
            if self.run_external_rad:
                # add surface temperature, solar and LWR gains for each exterior surface
                for bd in self.ext_boundaries:
                    surface = bd.ext_surface
                    results[f'{bd.name} Ext. Solar Gain (W)'] = surface.solar_gain
                    results[f'{bd.name} Ext. LWR Gain (W)'] = surface.lwr_gain
                    results[f'{bd.name} Ext. Surface Temperature (C)'] = surface.temperature
                    results[f'{bd.name} Ext. Film Coefficient (m^2-K/W)'] = surface.res_film

            if self.run_internal_rad:
                # add surface temperature and LWR gains for each interior surface, by zone
                for name, zone in self.zones.items():
                    for surface in zone.surfaces:
                        bd_name = surface.boundary_name
                        results[f'{bd_name} {name} LWR Gain (W)'] = surface.lwr_gain
                        results[f'{bd_name} {name} Surface Temperature (C)'] = surface.temperature
                        results[f'{bd_name} {name} Film Coefficient (m^2-K/W)'] = surface.res_film

        return results
