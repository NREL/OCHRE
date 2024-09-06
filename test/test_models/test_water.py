import unittest
import datetime as dt
import numpy as np

from ochre.Models import StratifiedWaterModel, OneNodeWaterModel, TwoNodeWaterModel, IdealWaterModel, \
    ModelException

water_init_args = {
    'time_res': dt.timedelta(minutes=1),
    'Heat Transfer Coefficient (W/m^2/K)': 1,
    'tank radius (m)': 0.2,
    'tank height (m)': 1,
}

update_args_no_draw = {
    'mains_temperature': 10,
    'ambient_dry_bulb': 20,
    'ambient_wet_bulb': 18,
    'ambient_humidity': 0.005,
    'wh_setpoint': 51.666667,
    'sinks': 0,
    'showers': 0,
    'baths': 0,
    'clothes_washer': 0,
    'dishwasher': 0,
}
update_args_small_draw = update_args_no_draw.copy()
update_args_small_draw['dishwasher'] = 1
update_args_tempered_draw = update_args_no_draw.copy()
update_args_tempered_draw['showers'] = 1
update_args_large_draw = update_args_no_draw.copy()
update_args_large_draw['dishwasher'] = 100


class StratifiedWaterModelTestCase(unittest.TestCase):
    """
    Test Case to test the Water Model class. Uses the 12-node water tank model by default
    """

    def setUp(self):
        self.model = StratifiedWaterModel(ext_node_names='AMB', **water_init_args)

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 12)
        self.assertAlmostEqual(self.model.volume, np.pi * 1000 * 0.04)

        # States and Inputs
        self.assertIn('T_WH1', self.model.state_names)
        self.assertIn('T_WH12', self.model.state_names)
        self.assertIn('H_WH12', self.model.input_names)

        self.assertAlmostEqual(self.model.states[0], 51.1, places=1)

        # Matrices
        self.assertTrue(all(self.model.A.diagonal() < 1))
        self.assertTrue(all(self.model.A.diagonal() > 0.9))
        self.assertTrue(all(self.model.B[:, 0] > 0))

    def test_load_rc_data(self):
        rc_params = self.model.load_rc_data(**water_init_args)

        self.assertIn('C_WH1', rc_params)
        self.assertIn('C_WH12', rc_params)
        self.assertIn('R_WH1_WH2', rc_params)
        self.assertIn('R_WH11_WH12', rc_params)
        self.assertIn('R_WH11_AMB', rc_params)

        self.assertAlmostEqual(rc_params['C_WH1'], rc_params['C_WH12'])
        self.assertAlmostEqual(rc_params['C_WH1'], 43804, places=0)
        self.assertAlmostEqual(rc_params['R_WH2_WH3'], rc_params['R_WH10_WH11'])
        self.assertAlmostEqual(rc_params['R_WH2_WH3'], 1.04, places=2)
        self.assertAlmostEqual(rc_params['R_WH1_AMB'], rc_params['R_WH12_AMB'])
        self.assertAlmostEqual(rc_params['R_WH1_AMB'], 4.34, places=2)
        self.assertAlmostEqual(rc_params['R_WH2_AMB'], 9.55, places=2)

    def test_update_water_draw(self):
        top_temperature = self.model.states[0]

        # No water draw
        result = self.model.update_water_draw(update_args_no_draw)
        self.assertEqual(self.model.draw_total, 0)
        self.assertEqual(self.model.h_delivered, 0)
        self.assertListEqual(result.tolist(), [0] * self.model.n_nodes)

        # Small water draw
        result = self.model.update_water_draw(update_args_small_draw)
        self.assertEqual(self.model.draw_total, 1)
        self.assertAlmostEqual(self.model.h_delivered, 2866, places=0)
        self.assertAlmostEqual(result[-1], -2866, places=0)
        self.assertAlmostEqual(self.model.outlet_temp, top_temperature)

        # Large water draw
        self.model.states[6:] = 45
        result = self.model.update_water_draw(update_args_large_draw)
        self.assertLess(self.model.outlet_temp, top_temperature)
        self.assertAlmostEqual(self.model.h_delivered, 270778, places=0)
        self.assertLess(result[0], 0)

        # Tempered water draw - low setpoint
        self.model.tempered_draw_temp = 40
        result = self.model.update_water_draw(update_args_tempered_draw)
        self.assertLess(self.model.draw_total, 1)
        self.assertAlmostEqual(self.model.h_delivered, 2091.5, places=0)
        self.assertEqual(self.model.h_unmet_load, 0)
        self.assertLess(result[-1], 0)

        # Tempered water draw - high setpoint
        self.model.states[0] = 55  # reset state
        self.model.tempered_draw_temp = 60
        self.model.update_water_draw(update_args_tempered_draw)
        self.assertEqual(self.model.draw_total, 1)
        self.assertAlmostEqual(self.model.h_unmet_load, 348.6, places=1)

    def test_inversion_mixing(self):
        # test with no mixing
        self.model.next_states = np.arange(40, 28, -1, dtype=float)
        self.model.run_inversion_mixing_rule()
        self.assertListEqual(list(self.model.next_states), list(range(40, 28, -1)))

        # test with full mixing
        self.model.next_states = np.arange(28, 40, dtype=float)
        self.model.run_inversion_mixing_rule()
        self.assertAlmostEqual(self.model.next_states[0], 33.5)
        self.assertAlmostEqual(self.model.next_states[-1], 33.5)

        # test with partial mixing
        self.model.next_states = np.array([34, 35, 34, 33, 32, 31, 32, 33, 32, 31, 30, 30], dtype=float)
        self.model.run_inversion_mixing_rule()
        self.assertAlmostEqual(self.model.next_states[0], 34.5)
        self.assertAlmostEqual(self.model.next_states[2], 34)
        self.assertAlmostEqual(self.model.next_states[5], 32)
        self.assertAlmostEqual(self.model.next_states[7], 32)
        self.assertAlmostEqual(self.model.next_states[-1], 30)

    def test_update(self):
        # No water draw update
        temp = self.model.states[0]
        result = self.model.update(schedule=update_args_no_draw)
        self.assertEqual(self.model.h_injections, 0)
        self.assertLess(self.model.next_states[0], temp)
        self.assertGreater(result, 0)

        # Small water draw update
        result = self.model.update(schedule=update_args_small_draw)
        self.assertGreater(result, 0)

        # Large water draw update
        result = self.model.update(schedule=update_args_large_draw)
        self.assertAlmostEqual(self.model.next_states[0], 51, places=0)
        self.assertAlmostEqual(self.model.next_states[2], 29, places=0)
        self.assertAlmostEqual(self.model.next_states[9], 10, places=0)
        self.assertAlmostEqual(self.model.next_states[-1], 10, places=0)
        self.assertGreater(result, 0)

        # Water heater injection
        self.model.states = self.model.next_states
        heats = np.zeros(self.model.n_nodes)
        heats[9] = 10000
        self.model.update(update_args_no_draw, heats)
        self.assertAlmostEqual(self.model.next_states[0], 51, places=0)
        self.assertAlmostEqual(self.model.next_states[2], 29, places=0)
        self.assertAlmostEqual(self.model.next_states[9], 12, places=0)
        self.assertAlmostEqual(self.model.next_states[-1], 10, places=0)

        # Check high temperature error
        self.model.states[0] = 110
        with self.assertRaises(ModelException):
            self.model.update(schedule=update_args_no_draw)

    def test_generate_results(self):
        results = self.model.generate_results(3)
        self.assertEqual(len(results), 4)

        results = self.model.generate_results(6)
        self.assertEqual(len(results), 10)

        results = self.model.generate_results(9)
        self.assertEqual(len(results), 35)


class OneNodeWaterModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model = OneNodeWaterModel(**water_init_args)

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 1)
        self.assertListEqual(self.model.vol_fractions.tolist(), [1])

    def test_load_rc_data(self):
        rc_params = self.model.load_rc_data(**water_init_args)

        self.assertEqual(len(rc_params), 2)
        self.assertAlmostEqual(rc_params['C_WH1'], 525651, places=0)
        self.assertAlmostEqual(rc_params['R_WH1_AMB'], 0.66, places=2)

    def test_update(self):
        # Small water draw update
        self.model.update(schedule=update_args_small_draw)
        self.assertAlmostEqual(self.model.next_states[0], 50.8, places=1)

        # Large water draw update
        self.model.update(schedule=update_args_large_draw)
        self.assertAlmostEqual(self.model.next_states[0], 18.4, places=1)


class TwoNodeWaterModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model = TwoNodeWaterModel(**water_init_args)

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 2)
        self.assertListEqual(self.model.vol_fractions.tolist(), [1 / 3, 2 / 3])

    def test_load_rc_data(self):
        rc_params = self.model.load_rc_data(**water_init_args)

        self.assertEqual(len(rc_params), 5)
        self.assertIn('C_WH1', rc_params)
        self.assertIn('R_WH1_WH2', rc_params)
        self.assertIn('R_WH1_AMB', rc_params)

        self.assertAlmostEqual(rc_params['C_WH1'] * 2, rc_params['C_WH2'])
        self.assertAlmostEqual(rc_params['C_WH1'], 175217, places=0)
        self.assertAlmostEqual(rc_params['R_WH1_WH2'], 6.21, places=2)
        self.assertAlmostEqual(rc_params['R_WH1_AMB'], 1.84, places=2)

    def test_update(self):
        # test 2 node water draw using empirical fraction

        # Small water draw update
        self.model.update(schedule=update_args_small_draw)
        self.assertAlmostEqual(self.model.next_states[0], 51.1, places=1)
        self.assertAlmostEqual(self.model.next_states[1], 50.6, places=1)

        # Large water draw update
        self.model.update(schedule=update_args_large_draw)
        self.assertAlmostEqual(self.model.next_states[0], 35.2, places=1)
        self.assertAlmostEqual(self.model.next_states[1], 10, places=1)


class IdealWaterTestCase(unittest.TestCase):
    def setUp(self):
        self.model = IdealWaterModel(**water_init_args)

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 1)
        self.assertListEqual(self.model.vol_fractions.tolist(), [1])

        self.assertTupleEqual(self.model.A.shape, (1, 1))
        self.assertListEqual(self.model.input_names, ['T_AMB', 'H_WH1'])
        self.assertAlmostEqual(self.model.A[0, 0], 1)

    def test_update(self):
        temp = self.model.states[0]
        result = self.model.update(schedule=update_args_no_draw)
        self.assertAlmostEqual(self.model.next_states[0], temp)
        self.assertAlmostEqual(result, 0, places=2)

        result = self.model.update(schedule=update_args_small_draw)
        self.assertLess(self.model.next_states[0], temp)
        self.assertAlmostEqual(result, 0, places=2)


if __name__ == '__main__':
    unittest.main()
