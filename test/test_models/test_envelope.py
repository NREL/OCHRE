import unittest
import datetime as dt
import numpy as np

from ochre.Models import Envelope

zone_init_args = {
    'time_res': dt.timedelta(minutes=5),
    'label': 'ATC',
    'capacitance': 1,
    'attic volume (m^3)': 10,
    'attic infiltration method': 'ELA',
    'attic ELA (cm^2)': 1,
    'attic stack coefficient {(L/s)/(cm^4-K)}': 0.1,
    'attic wind coefficient {(L/s)/(cm^4-(m/s))}': 0.1,
}

properties = {
    'Exterior Emissivity': 0.9,
    'Exterior Solar Absorptivity': 0.9,
    'Interior Thermal Absorptivity': 0.9,
    'Interior Solar Absorptivity': 0.9,
}

init_args = {
    'time_res': dt.timedelta(minutes=5),
    'R_EW_ext': 1.0,
    'R_EW1': 1.0,
    'C_EW1': 10,
    'R_EW2': 1.0,
    'C_EW2': 10,
    'R_EW_int': 1.0,
    'R_WD1': 1.0,
    'C_WD1': 0,

    'R_IW_ext': 1.0,
    'R_IW1': 1.0,
    'C_IW1': 10,
    'R_IW2': 1.0,
    'C_IW2': 10,
    'R_IW3': 1.0,
    'C_IW3': 10,
    'R_IW_int': 1.0,

    # attic
    'R_CL_ext': 1.0,
    'R_CL1': 1.0,
    'C_CL1': 10,
    'R_CL_int': 1.0,
    'R_RG_ext': 1.0,
    'R_RG1': 1.0,
    'C_RG1': 10,
    'R_RG_int': 1.0,
    'R_RF_ext': 1.0,
    'R_RF1': 1.0,
    'C_RF1': 10,
    'R_RF_int': 1.0,

    # foundation
    'R_FL_ext': 1.0,
    'R_FL1': 1.0,
    'C_FL1': 10,
    'R_FL_int': 1.0,
    'R_FW_ext': 0,
    'R_FW1': 1.0,
    'C_FW1': 10,
    'R_FW_int': 1.0,
    'R_CW_ext': 1.0,
    'R_CW1': 1.0,
    'C_CW1': 10,
    'R_CW_int': 1.0,
    'R_FF_ext': 0,
    'R_FF1': 1.0,
    'C_FF1': 10,
    'R_FF2': 1.0,
    'C_FF2': 10,
    'R_FF_int': 1.0,

    # garage
    'R_AW_ext': 1.0,
    'R_AW1': 1.0,
    'C_AW1': 10,
    'R_AW_int': 1.0,
    'R_GW_ext': 1.0,
    'R_GW1': 1.0,
    'C_GW1': 10,
    'R_GW_int': 1.0,
    'R_GR_ext': 1.0,
    'R_GR1': 1.0,
    'C_GR1': 10,
    'R_GR_int': 1.0,
    'R_GF_ext': 1.0,
    'R_GF1': 1.0,
    'C_GF1': 10,
    'R_GF_int': 1.0,

    # zones
    'C_LIV': 1000,
    'C_ATC': 100,
    'C_GAR': 100,
    'C_FND': 100,

    # 'building length (m)': 10,
    # 'building width (m)': 10,
    'finished floor area (m^2)': 100,
    'num stories': 1,
    'building volume (m^3)': 100,
    'total wall area (m^2)': 10,
    'front window area (m^2)': 1,
    'back window area (m^2)': 1,
    'right window area (m^2)': 1,
    'left window area (m^2)': 1,
    'interior wall area (m^2)': 10,

    'attic floor area (m^2)': 100,
    'attic volume (m^3)': 10,
    'front roof area (m^2)': 10,
    'back roof area (m^2)': 10,
    'left gable wall area (m^2)': 10,
    'right gable wall area (m^2)': 10,
    'roof pitch': 0,

    'basement floor area (m^2)': 100,
    'basement wall area (m^2)': 10,
    'crawlspace above grade wall area (m^2)': 10,
    'crawlspace below grade wall area (m^2)': 10,
    'crawlspace volume (m^3)': 10,

    'garage attached wall area (m^2)': 10,
    'garage front wall area (m^2)': 10,
    'garage back wall area (m^2)': 0,
    'garage right wall area (m^2)': 0,
    'garage left wall area (m^2)': 0,
    'garage floor area (m^2)': 10,
    'garage volume (m^3)': 10,
    'Garage Furniture surface area (m2)': 10,
    'Indoor Furniture surface area (m2)': 100,

    'number of occupants': 2,
    'gain per occupant (W)': 1,
    'occupants convective gainfrac': 0.3,
    'occupants radiant gainfrac': 0.3,
    'occupants latent gainfrac': 0.3,

    'ventilation cfm': 1,
    'ventilation type': 'balanced',
    'erv sensible effectiveness': 0.713019174966,
    'erv latent effectiveness': 0,

    'infiltration method': 'ASHRAE',
    'inf_f_t': 1,
    'inf_C_i': 1000,
    'inf_n_i': 1,
    'inf_stack_coef': 0.001,
    'inf_wind_coef': 0.001,
    'ws_S_wo': 1,
    'inf_Y_i': 0.1,
    'inf_S_wflue': 1,

    'attic infiltration method': 'ELA',
    'attic ELA (cm^2)': 0.1,
    'attic stack coefficient {(L/s)/(cm^4-K)}': 0.1,
    'attic wind coefficient {(L/s)/(cm^4-(m/s))}': 0.1,

    'foundation infiltration method': 'ACH',
    'Air Changes (ACH):': 1,

    'garage infiltration method': 'ELA',
    'garage ELA (cm^2)': 100.0,
    'garage stack coefficient {(L/s)/(cm^4-K)}': 0.000177467803711,
    'garage wind coefficient {(L/s)/(cm^4-(m/s))}': 0.000166437954278,

    'initial_temp_setpoint': 21,
    'Exterior wall properties': properties,
    'Window properties': {**properties, **{'SHGC (-)': 0.5, 'U-value (W/m^2-K)': 2}},
    'Interior Wall properties': properties,
    'Roof properties': properties,
    'Ceiling roof properties': properties,
    'Gable wall properties': properties,
    'Floor properties': properties,
    'Foundation floor properties': properties,
    'Foundation wall properties': properties,
    'Crawlspace wall properties': properties,
    'Attached wall properties': properties,
    'Garage wall properties': properties,
    'Garage roof properties': properties,
    'Garage floor properties': properties,
    'Garage Furniture properties': {'R-value (m^2-K/W)': 1, 'Capacitance (kJ/m^2-K)': 10, **properties},
    'Indoor Furniture properties': {'R-value (m^2-K/W)': 1, 'Capacitance (kJ/m^2-K)': 9, **properties},
}

update_args1 = {
    'solar_EW': 0,
    'solar_WD': 0,
    'solar_RF': 0,
    'solar_RG': 0,
    'solar_CW': 0,
    'solar_GW': 0,
    'solar_GR': 0,
    'occupants': 0,
    'wind_speed': 1,
    'ventilation_rate': 0,
    'ambient_dry_bulb': 15,
    'ambient_humidity': 0.007,
    'ambient_pressure': 101,
    'ground_temperature': 10,
    'sky_temperature': 0,
    'heating_setpoint': 20,
    'cooling_setpoint': 21,
}
init_args['initial_schedule'] = update_args1.copy()

update_args2 = {
    'solar_EW': 1000,
    'solar_WD': 100,
    'solar_RF': 1000,
    'solar_RG': 100,
    'solar_CW': 100,
    'solar_GW': 100,
    'solar_GR': 100,
    'occupants': 4,
    'wind_speed': 5,
    'ventilation_rate': 1,
    'ambient_dry_bulb': 30,
    'ambient_humidity': 0.007,
    'ambient_pressure': 101,
    'ground_temperature': 15,
    'sky_temperature': 0,
    'heating_setpoint': 20,
    'cooling_setpoint': 21,
}


class SurfaceTestCase(unittest.TestCase):
    def setUp(self):
        self.env = Envelope(**init_args)

        self.rf_ext = [b for b in self.env.boundaries if b.name == 'Roof'][0].ext_surface
        self.iw_ext = [b for b in self.env.boundaries if b.name == 'Interior Wall'][0].ext_surface
        self.aw_ext = [b for b in self.env.boundaries if b.name == 'Attached wall'][0].ext_surface
        self.aw_int = [b for b in self.env.boundaries if b.name == 'Attached wall'][0].int_surface

    def test_initialize(self):
        self.assertEqual(self.rf_ext.boundary.label, 'RF')
        self.assertEqual(self.rf_ext.zone_label, 'EXT')
        self.assertEqual(self.rf_ext.area, 20)
        self.assertEqual(self.rf_ext.node, 'RF1')
        self.assertEqual(self.env.state_names[self.rf_ext.t_idx], 'T_RF1')
        self.assertEqual(self.env.input_names[self.rf_ext.h_idx], 'H_RF1')
        self.assertAlmostEqual(self.rf_ext.radiation_frac, 0.67, places=2)
        self.assertAlmostEqual(self.rf_ext.radiation_res, 0.025 * 2 / 3)
        self.assertAlmostEqual(self.rf_ext.temperature, 15, places=1)
        self.assertEqual(self.rf_ext.emissivity, 0.9)
        self.assertEqual(self.rf_ext.absorptivity, 0.9)

        self.assertEqual(self.iw_ext.zone_label, 'LIV')
        self.assertEqual(self.iw_ext.area, 10)
        self.assertEqual(self.iw_ext.is_exterior, False)
        self.assertEqual(self.iw_ext.node, 'IW1')
        self.assertEqual(self.env.state_names[self.iw_ext.t_idx], 'T_IW1')
        self.assertEqual(self.env.input_names[self.iw_ext.h_idx], 'H_IW1')
        self.assertAlmostEqual(self.iw_ext.radiation_frac, 0.67, places=2)
        self.assertAlmostEqual(self.iw_ext.radiation_res, 0.05 * 2 / 3)
        self.assertAlmostEqual(self.iw_ext.temperature, 21, places=1)

        self.assertEqual(self.aw_ext.zone_label, 'GAR')
        self.assertAlmostEqual(self.aw_ext.temperature, 15, places=0)

        self.assertEqual(self.aw_int.zone_label, 'LIV')
        self.assertAlmostEqual(self.aw_int.temperature, 21, places=1)

    def test_calculate_external_radiation(self):
        # equal temperatures
        self.rf_ext.solar_gain = 0
        self.rf_ext.t_boundary = 15
        self.rf_ext.calculate_exterior_radiation(15, 0)
        self.assertAlmostEqual(self.rf_ext.lwr_gain, 0, places=-1)
        self.assertAlmostEqual(self.rf_ext.temperature, 15, places=-1)

        # unequal temperatures
        self.rf_ext.t_boundary = 18
        self.rf_ext.calculate_exterior_radiation(15, 0)
        self.rf_ext.calculate_exterior_radiation(15, 0)  # run twice to update temperature
        self.assertAlmostEqual(self.rf_ext.lwr_gain, -70, places=-1)
        self.assertAlmostEqual(self.rf_ext.temperature, 16, places=0)

        # with solar
        self.rf_ext.solar_gain = 1000
        self.rf_ext.temperature = 35
        self.rf_ext.t_boundary = 35
        self.rf_ext.calculate_exterior_radiation(30, 0)  # run twice to update temperature
        self.rf_ext.calculate_exterior_radiation(30, 0)
        self.assertAlmostEqual(self.rf_ext.lwr_gain, -804, places=-1)
        self.assertAlmostEqual(self.rf_ext.temperature, 36.8, places=1)

        # test multiple iterations
        self.rf_ext.iterations = 3
        self.rf_ext.temperature = 35
        self.rf_ext.t_boundary = 35
        self.rf_ext.calculate_exterior_radiation(30, 0)
        self.assertAlmostEqual(self.rf_ext.lwr_gain, -796, places=-1)
        self.assertAlmostEqual(self.rf_ext.temperature, 36.8, places=1)


class ZoneTestCase(unittest.TestCase):
    def setUp(self):
        self.env = Envelope(**init_args)

        self.liv = self.env.indoor_zone
        self.fnd = self.env.zones['FND']
        self.gar = self.env.zones['GAR']

    def test_initialize(self):
        self.assertEqual(len(self.liv.surfaces), 7)
        self.assertEqual(self.env.state_names[self.liv.t_idx], 'T_LIV')
        self.assertEqual(self.env.input_names[self.liv.h_idx], 'H_LIV')
        self.assertAlmostEqual(self.liv.capacitance, 1e6)
        self.assertAlmostEqual(self.liv.volume, 100)
        self.assertEqual(len(self.liv.infiltration_parameters), 6)
        self.assertAlmostEqual(self.liv.max_flow_rate, 1 / 3)
        self.assertEqual(self.liv.balanced_ventilation, 'balanced')

        self.assertEqual(len(self.fnd.surfaces), 4)
        self.assertEqual(self.env.state_names[self.fnd.t_idx], 'T_FND')
        self.assertEqual(self.env.input_names[self.fnd.h_idx], 'H_FND')
        self.assertAlmostEqual(self.fnd.capacitance, 1e5)
        self.assertAlmostEqual(self.fnd.volume, 10)
        self.assertEqual(len(self.fnd.infiltration_parameters), 2)
        self.assertAlmostEqual(self.fnd.max_flow_rate, 1 / 30)
        self.assertEqual(self.fnd.balanced_ventilation, None)

        self.assertEqual(len(self.gar.surfaces), 5)
        self.assertEqual(self.env.state_names[self.gar.t_idx], 'T_GAR')
        self.assertEqual(self.env.input_names[self.gar.h_idx], 'H_GAR')
        self.assertAlmostEqual(self.gar.capacitance, 1e5)
        self.assertAlmostEqual(self.gar.volume, 10)
        self.assertEqual(len(self.gar.infiltration_parameters), 4)
        self.assertAlmostEqual(self.gar.max_flow_rate, 1 / 30)
        self.assertEqual(self.gar.balanced_ventilation, None)

    def test_update_infiltration(self):
        # living space
        self.liv.update_infiltration(update_args2, 20, 1000)
        self.assertAlmostEqual(self.liv.inf_flow, 0.07, places=2)
        self.assertAlmostEqual(self.liv.inf_heat, 760, places=-1)
        self.assertAlmostEqual(self.liv.forced_vent_flow, 0.0005, places=4)
        self.assertAlmostEqual(self.liv.air_changes, 2.4, places=-1)

        # living space, exhaust ventilation
        self.liv.ventilation_type = 'exhaust'
        self.liv.sens_recovery_eff = 0
        self.liv.balanced_ventilation = False
        self.liv.update_infiltration(update_args2, 20, 1000)
        self.assertAlmostEqual(self.liv.inf_flow, 0.07, places=2)
        self.assertAlmostEqual(self.liv.inf_heat, 760, places=-1)
        self.assertAlmostEqual(self.liv.forced_vent_flow, 0.0005, places=4)
        self.assertAlmostEqual(self.liv.air_changes, 2.4, places=-1)

        # foundation
        self.fnd.update_infiltration(update_args2, 20, 1000)
        self.assertAlmostEqual(self.fnd.inf_flow, 0.0028, places=4)
        self.assertAlmostEqual(self.fnd.inf_heat, 32, places=0)

        # garage, cold
        self.gar.update_infiltration(update_args1, 17, 1000)
        self.assertAlmostEqual(self.gar.inf_flow, 0.0023, places=4)
        self.assertAlmostEqual(self.gar.inf_heat, -6, places=0)

        # garage, warm
        self.gar.update_infiltration(update_args2, 20, 1000)
        self.assertAlmostEqual(self.gar.inf_flow, 0.0077, places=4)
        self.assertAlmostEqual(self.gar.inf_heat, 89, places=0)

        # garage with h_limit
        self.gar.update_infiltration(update_args2, 20, 50)
        self.assertAlmostEqual(self.gar.inf_flow, 0.0077, places=4)
        self.assertAlmostEqual(self.gar.inf_heat, 50, places=0)

    def test_calculate_interior_radiation(self):
        # indoor radiation, cold EW
        ew = [s for s in self.liv.surfaces if s.boundary.label == 'EW'][0]
        ew.temperature = 19
        for s in self.liv.surfaces:
            s.t_boundary = 21
        self.liv.calculate_interior_radiation(21)
        # self.liv.calculate_interior_radiation(21, t_boundaries)  # run twice to update temperature
        self.assertEqual(ew.solar_gain, 0)
        self.assertAlmostEqual(ew.lwr_gain, 50, places=-1)
        self.assertAlmostEqual(ew.temperature, 20, places=0)

        # indoor radiation, colder indoor temp
        ew.temperature = 23
        self.liv.calculate_interior_radiation(19)
        # self.liv.calculate_interior_radiation(21, t_boundaries)  # run twice to update temperature
        self.assertAlmostEqual(ew.lwr_gain, -80, places=-1)
        self.assertAlmostEqual(ew.temperature, 22, places=0)

        # garage radiation, hot AW
        aw = [s for s in self.gar.surfaces if s.boundary.label == 'AW'][0]
        aw.temperature = 17
        for s in self.gar.surfaces:
            s.t_boundary = 14
        self.gar.calculate_interior_radiation(14)
        # self.gar.calculate_interior_radiation(14, t_boundaries)  # run twice to update temperature
        self.assertAlmostEqual(aw.lwr_gain, -40, places=-1)
        self.assertAlmostEqual(aw.temperature, 16, places=0)

        # garage radiation, hot
        aw = [s for s in self.gar.surfaces if s.boundary.label == 'AW'][0]
        aw.temperature = 21
        for s in self.gar.surfaces:
            s.t_boundary = 27
        self.gar.calculate_interior_radiation(25)
        # self.gar.calculate_interior_radiation(25, t_boundaries)  # run twice to update temperature
        self.assertAlmostEqual(aw.lwr_gain, -170, places=-1)
        self.assertAlmostEqual(aw.temperature, 20, places=0)


class BoundaryTestCase(unittest.TestCase):
    def setUp(self):
        self.env = Envelope(**init_args)

        self.ew = [b for b in self.env.boundaries if b.name == 'Exterior wall'][0]
        self.ff = [b for b in self.env.boundaries if b.name == 'Foundation floor'][0]
        self.aw = [b for b in self.env.boundaries if b.name == 'Attached wall'][0]
        self.iw = [b for b in self.env.boundaries if b.name == 'Interior Wall'][0]

    def test_initialize(self):
        self.assertAlmostEqual(self.ew.area, 10)
        self.assertEqual(self.ew.is_int, True)
        self.assertEqual(self.ew.n_nodes, 2)
        self.assertEqual(len(self.ew.capacitors), 2)
        self.assertAlmostEqual(self.ew.capacitors['C_EW1'], 1e5)
        self.assertEqual(len(self.ew.resistors), 3)
        self.assertAlmostEqual(self.ew.resistors['R_EXT_EW1'], 0.15)
        self.assertEqual(self.ew.ext_surface.zone_label, 'EXT')
        self.assertEqual(self.ew.int_surface.zone_label, 'LIV')

        self.assertEqual(len(self.iw.capacitors), 2)
        self.assertAlmostEqual(self.iw.capacitors['C_IW1'], 1e5)
        self.assertEqual(len(self.iw.resistors), 2)
        self.assertAlmostEqual(self.iw.resistors['R_LIV_IW1'], 0.15)
        self.assertAlmostEqual(self.iw.resistors['R_IW1_IW2'], 0.1)
        self.assertEqual(self.iw.ext_surface.zone_label, 'LIV')
        self.assertEqual(self.iw.int_surface.zone_label, '')

        self.assertEqual(len(self.aw.capacitors), 1)
        self.assertAlmostEqual(self.aw.capacitors['C_AW1'], 1e5)
        self.assertEqual(len(self.aw.resistors), 2)
        self.assertAlmostEqual(self.aw.resistors['R_GAR_AW1'], 0.15)
        self.assertAlmostEqual(self.aw.resistors['R_AW1_LIV'], 0.15)

        self.assertEqual(self.ff.is_int, False)
        self.assertEqual(self.aw.is_int, True)
        self.assertEqual(self.iw.is_int, True)

    # TODO: test linearization
    # def test_initialize(self):
    #     self.assertAlmostEqual(self.ew.area, 10)
    #     self.assertEqual(self.ew.is_int, True)
    #     self.assertEqual(self.ew.n_nodes, 2)
    #     self.assertEqual(len(self.ew.capacitors), 2)
    #     self.assertAlmostEqual(self.ew.capacitors['C_EW1'], 1e5)
    #     self.assertEqual(len(self.ew.resistors), 5)
    #     self.assertAlmostEqual(self.ew.resistors['R_EXT_EW-ext'], 0.1)
    #     self.assertAlmostEqual(self.ew.resistors['R_EW-ext_EW1'], 0.05)
    #     self.assertEqual(self.ew.ext_surface.zone, 'EXT')
    #     self.assertEqual(self.ew.int_surface.zone, 'LIV')
    #
    #     self.assertEqual(len(self.iw.capacitors), 2)
    #     self.assertAlmostEqual(self.iw.capacitors['C_IW1'], 1e5)
    #     self.assertEqual(len(self.iw.resistors), 3)
    #     self.assertAlmostEqual(self.iw.resistors['R_LIV_IW-ext'], 0.1)
    #     self.assertAlmostEqual(self.iw.resistors['R_IW1_IW2'], 0.1)
    #     self.assertEqual(self.iw.ext_surface.zone, 'LIV')
    #     self.assertEqual(self.iw.int_surface.zone, '')
    #
    #     self.assertEqual(len(self.aw.capacitors), 1)
    #     self.assertAlmostEqual(self.aw.capacitors['C_AW1'], 1e5)
    #     self.assertEqual(len(self.aw.resistors), 4)
    #     self.assertAlmostEqual(self.aw.resistors['R_GAR_AW-ext'], 0.1)
    #     self.assertAlmostEqual(self.aw.resistors['R_AW1_AW-int'], 0.05)
    #
    #     self.assertEqual(self.ff.is_int, False)
    #     self.assertEqual(self.aw.is_int, True)
    #     self.assertEqual(self.iw.is_int, True)
    #

class EnvelopeTestCase(unittest.TestCase):
    """
    Test Case to test the Envelope Model class.
    """

    def setUp(self):
        self.env = Envelope(**init_args)

    def test_initialize(self):
        # Zones and boundaries
        self.assertEqual(len(self.env.zones), 4)
        self.assertEqual(len(self.env.boundaries), 16)
        self.assertEqual(len(self.env.ext_boundaries), 7)
        self.assertEqual(len(self.env.int_boundaries), 7)
        self.assertIsNotNone(self.env.indoor_zone)

        # Furniture capacitance
        self.assertAlmostEqual(self.env.indoor_zone.capacitance, 1e6)

        # States and Inputs
        self.assertIn('T_LIV', self.env.state_names)
        self.assertIn('T_GW1', self.env.state_names)
        self.assertIn('T_AW1', self.env.state_names)
        self.assertIn('T_IW2', self.env.state_names)
        self.assertIn('T_GM1', self.env.state_names)
        self.assertIn('H_LIV', self.env.input_names)
        self.assertIn('H_GW1', self.env.input_names)
        self.assertNotIn('H_IW2', self.env.input_names)
        self.assertIn('T_EXT', self.env.input_names)
        self.assertEqual(len(self.env.states), 22)
        self.assertEqual(len(self.env.inputs), 23)

        liv_idx = self.env.indoor_zone.t_idx
        self.assertAlmostEqual(self.env.states[liv_idx], 21)
        self.assertLess(self.env.states.max(), 30)
        self.assertGreater(self.env.states.min(), 10)

        # Matrices
        self.assertTrue(all(self.env.A.diagonal() < 1))
        self.assertGreater(self.env.A[liv_idx, liv_idx], 0.5)

        # Occupancy
        self.assertEqual(self.env.occupancy_sensible_gain, 1.2)

    def test_update_radiation(self):
        liv_idx = self.env.indoor_zone.h_idx

        # test with no solar radiation
        self.env.update_radiation(update_args1)
        self.assertAlmostEqual(self.env.inputs[liv_idx], -40, places=-1)

        # test with solar + LWR
        self.env.update_radiation(update_args2)
        self.assertAlmostEqual(self.env.inputs[liv_idx], -70, places=-1)

    def test_update_infiltration(self):
        # cold update
        self.env.update_infiltration(update_args1)
        self.assertAlmostEqual(self.env.indoor_zone.inf_flow, 0.01, places=2)
        self.assertAlmostEqual(self.env.indoor_zone.forced_vent_flow, 0, places=5)
        self.assertAlmostEqual(self.env.indoor_zone.air_changes, 0.2, places=1)
        self.assertAlmostEqual(self.env.indoor_zone.inf_heat, -40, places=-1)

        # warm update
        self.env.update_infiltration(update_args2)
        self.assertAlmostEqual(self.env.indoor_zone.inf_flow, 0.07, places=2)
        self.assertAlmostEqual(self.env.indoor_zone.inf_heat, 680, places=-1)

    def test_reset_env_inputs(self):
        self.env.inputs += 10
        liv_idx = self.env.indoor_zone.h_idx

        self.env.update_inputs(update_args1)
        self.assertEqual(self.env.inputs[self.env.ext_zones['EXT'].t_idx], 15)
        self.assertAlmostEqual(self.env.inputs[liv_idx], -80, places=-1)

        self.env.update_inputs(update_args2)
        self.assertEqual(self.env.inputs[self.env.ext_zones['EXT'].t_idx], 30)
        self.assertAlmostEqual(self.env.inputs[liv_idx], 660, places=-1)

    def test_update(self):
        liv_idx = self.env.indoor_zone.t_idx

        # Cold update
        self.env.update_inputs(update_args1)
        temp = self.env.states[liv_idx]
        self.env.update(schedule=update_args1)
        self.assertLess(self.env.states[liv_idx], temp)
        self.assertEqual(self.env.states[liv_idx], self.env.indoor_zone.temperature)
        self.assertEqual(self.env.states[liv_idx], self.env.zones['LIV'].temperature)

        # Hot update
        self.env.update_inputs(update_args2)
        temp = self.env.states[liv_idx]
        self.env.update(schedule=update_args2)
        self.assertGreater(self.env.states[liv_idx], temp)

        # Unmet loads
        self.env.states[liv_idx] = 25
        self.env.indoor_zone.temperature = 25
        self.env.update_inputs(update_args2)
        self.env.update(schedule=update_args2)
        self.assertAlmostEqual(self.env.unmet_hvac_load, 3.3, places=1)

    def test_linear_radiation(self):
        self.env_linear = Envelope(use_linear_radiation=True, **init_args)

        # check for same state/input names, but not same state values
        self.assertListEqual(self.env.state_names, self.env_linear.state_names)
        self.assertListEqual(self.env.input_names, self.env_linear.input_names)
        self.assertFalse(np.equal(self.env.states, self.env_linear.states).all())

        # check that indoor zone dynamics are similar but not the same
        env_indoor_a = self.env.A[self.env.indoor_zone.t_idx, self.env.indoor_zone.t_idx]
        env_linear_indoor_a = self.env_linear.A[self.env_linear.indoor_zone.t_idx, self.env_linear.indoor_zone.t_idx]
        self.assertGreater(env_indoor_a, env_linear_indoor_a)
        self.assertAlmostEqual(env_indoor_a - env_linear_indoor_a, 0.02, places=2)

    def test_get_main_states(self):
        result = self.env.get_zone_temperature()
        self.assertEqual(len(result), 6)

    def test_generate_results(self):
        results = self.env.generate_results(1)
        self.assertEqual(len(results), 9)
        self.assertIn('Temperature - Indoor (C)', results)
        self.assertEqual(results['Temperature - Indoor (C)'], self.env.indoor_zone.temperature)

        results = self.env.generate_results(4)
        self.assertEqual(len(results), 23)
        self.assertIn('Relative Humidity - Indoor (-)', results)
        self.assertEqual(results['Relative Humidity - Indoor (-)'], self.env.indoor_zone.humidity.rh)

        results = self.env.generate_results(7)
        self.assertEqual(len(results), 25)
        self.assertIn('Indoor Air Density (kg/m^3)', results)

        results = self.env.generate_results(8)
        self.assertEqual(len(results), 129)
        self.assertIn('T_LIV', results)
        self.assertEqual(results['T_LIV'], self.env.indoor_zone.temperature)
        self.assertIn('H_LIV', results)
        self.assertEqual(results['H_LIV'], self.env.inputs[self.env.indoor_zone.h_idx])


if __name__ == '__main__':
    unittest.main()
