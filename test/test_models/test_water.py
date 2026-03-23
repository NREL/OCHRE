import unittest
import datetime as dt
import numpy as np
import pandas as pd

from ochre.Models import StratifiedWaterModel, OneNodeWaterModel, TwoNodeWaterModel, IdealWaterModel, ModelException

# Common simulation parameters required by Simulator base class
sim_params = {
    "start_time": dt.datetime(2020, 1, 1),
    "duration": dt.timedelta(hours=1),
    "verbosity": 0,  # suppress output
}

# Calculate volume from radius and height: V = pi * r^2 * h in m^3, then convert to liters
# radius=0.2m, height=1m => V = pi * 0.04 * 1 = 0.1257 m^3 = 125.7 L
tank_volume = np.pi * 0.04 * 1000  # in L

# Schedule inputs use new key names
schedule_no_draw = {
    "Mains Temperature (C)": 10,
    "Zone Temperature (C)": 20,
    "Water Heating (L/min)": 0,  # sinks, showers, baths combined
    "Clothes Washer (L/min)": 0,
    "Dishwasher (L/min)": 0,
}
schedule_small_draw = schedule_no_draw.copy()
schedule_small_draw["Dishwasher (L/min)"] = 1
schedule_tempered_draw = schedule_no_draw.copy()
schedule_tempered_draw["Water Heating (L/min)"] = 1
schedule_large_draw = schedule_no_draw.copy()
schedule_large_draw["Dishwasher (L/min)"] = 100


def make_schedule_df(schedule_dict, num_rows=60):
    """Create a schedule DataFrame from a dict of values."""
    return pd.DataFrame(
        {k: [v] * num_rows for k, v in schedule_dict.items()},
        index=pd.date_range("2020-01-01", periods=num_rows, freq="1min"),
    )


def get_water_init_args(schedule_dict=None):
    """Get water init args, optionally with a schedule DataFrame."""
    args = {
        "time_res": dt.timedelta(minutes=1),
        "Heat Transfer Coefficient (W/m^2/K)": 1,
        "Tank Height (m)": 1,
        "Tank Volume (L)": tank_volume,
        **sim_params,
    }
    if schedule_dict is not None:
        args["schedule"] = make_schedule_df(schedule_dict)
    return args


class StratifiedWaterModelTestCase(unittest.TestCase):
    """
    Test Case to test the Water Model class. Uses the 12-node water tank model by default
    """

    def setUp(self):
        # Use schedule_no_draw as default schedule
        self.model = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_no_draw))

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 12)
        self.assertAlmostEqual(self.model.volume, tank_volume)

        # States and Inputs
        self.assertIn("T_WH1", self.model.state_names)
        self.assertIn("T_WH12", self.model.state_names)
        self.assertIn("H_WH12", self.model.input_names)

        self.assertAlmostEqual(self.model.states[0], 51.1, places=1)

        # Matrices
        self.assertTrue(all(self.model.A.diagonal() < 1))
        self.assertTrue(all(self.model.A.diagonal() > 0.9))
        self.assertTrue(all(self.model.B[:, 0] > 0))

    def test_load_rc_data(self):
        rc_params_tuple = self.model.load_rc_data(**get_water_init_args())
        # load_rc_data now returns (capacitances, resistances) tuple
        self.assertIsInstance(rc_params_tuple, tuple)
        capacitances, resistances = rc_params_tuple

        self.assertIn("WH1", capacitances)
        self.assertIn("WH12", capacitances)
        self.assertIn(("WH1", "WH2"), resistances)
        self.assertIn(("WH11", "WH12"), resistances)
        self.assertIn(("WH11", "AMB"), resistances)

        self.assertAlmostEqual(capacitances["WH1"], capacitances["WH12"])
        self.assertAlmostEqual(capacitances["WH1"], 43804, places=0)
        self.assertAlmostEqual(resistances[("WH2", "WH3")], resistances[("WH10", "WH11")])
        self.assertAlmostEqual(resistances[("WH2", "WH3")], 1.04, places=2)
        self.assertAlmostEqual(resistances[("WH1", "AMB")], resistances[("WH12", "AMB")])
        self.assertAlmostEqual(resistances[("WH1", "AMB")], 4.34, places=2)
        self.assertAlmostEqual(resistances[("WH2", "AMB")], 9.55, places=2)

    def test_update_water_draw(self):
        top_temperature = self.model.states[0]

        # No water draw - update_inputs reads from schedule
        self.model.update_inputs()
        result = self.model.update_water_draw()
        self.assertEqual(self.model.draw_total, 0)
        self.assertEqual(self.model.h_delivered, 0)
        self.assertListEqual(result.tolist(), [0] * self.model.n_nodes)

        # Small water draw - create new model with small_draw schedule
        model_small = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_small_draw))
        model_small.update_inputs()
        result = model_small.update_water_draw()
        self.assertEqual(model_small.draw_total, 1)
        self.assertAlmostEqual(model_small.h_delivered, 2866, places=0)
        self.assertAlmostEqual(result[-1], -2866, places=0)
        self.assertAlmostEqual(model_small.outlet_temp, top_temperature)

        # Large water draw
        model_large = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_large_draw))
        model_large.states[6:] = 45
        model_large.update_inputs()
        result = model_large.update_water_draw()
        self.assertLess(model_large.outlet_temp, top_temperature)
        self.assertAlmostEqual(model_large.h_delivered, 270778, places=0)
        self.assertLess(result[0], 0)

        # Tempered water draw - low setpoint
        model_tempered = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_tempered_draw))
        model_tempered.tempered_draw_temp = 40
        model_tempered.update_inputs()
        result = model_tempered.update_water_draw()
        self.assertLess(model_tempered.draw_total, 1)
        self.assertAlmostEqual(model_tempered.h_delivered, 2091.5, places=0)
        self.assertEqual(model_tempered.h_unmet_load, 0)
        self.assertLess(result[-1], 0)

        # Tempered water draw - high setpoint
        model_tempered2 = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_tempered_draw))
        model_tempered2.states[0] = 55  # reset state
        model_tempered2.tempered_draw_temp = 60
        model_tempered2.update_inputs()
        model_tempered2.update_water_draw()
        self.assertEqual(model_tempered2.draw_total, 1)
        self.assertAlmostEqual(model_tempered2.h_unmet_load, 348.6, places=1)

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
        # No water draw update - use schedule from init
        temp = self.model.states[0]
        result = self.model.update()
        self.assertEqual(self.model.h_injections, 0)
        self.assertLess(self.model.next_states[0], temp)
        # result is now a dict from generate_results, check it's not empty or has expected key
        self.assertIsInstance(result, dict)

        # Small water draw update - create new model
        model_small = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_small_draw))
        result = model_small.update()
        self.assertIsInstance(result, dict)

        # Large water draw update
        model_large = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_large_draw))
        result = model_large.update()
        self.assertAlmostEqual(model_large.next_states[0], 51, places=0)
        self.assertAlmostEqual(model_large.next_states[2], 29, places=0)
        self.assertAlmostEqual(model_large.next_states[9], 10, places=0)
        self.assertAlmostEqual(model_large.next_states[-1], 10, places=0)
        self.assertIsInstance(result, dict)

        # Water heater injection - create new model with no-draw schedule for injection
        # (simulating the original test which passed no_draw for the injection step)
        model_injection = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_no_draw))
        # Set initial state to post-large-draw state
        model_injection.states = model_large.next_states.copy()
        heats = np.zeros(model_injection.n_nodes)
        heats[9] = 10000
        model_injection.update(control_signal=heats)
        self.assertAlmostEqual(model_injection.next_states[0], 51, places=0)
        self.assertAlmostEqual(model_injection.next_states[2], 29, places=0)
        self.assertAlmostEqual(model_injection.next_states[9], 12, places=0)
        self.assertAlmostEqual(model_injection.next_states[-1], 10, places=0)

        # Check high temperature error
        model_error = StratifiedWaterModel(ext_node_names="AMB", **get_water_init_args(schedule_no_draw))
        model_error.states[0] = 110
        with self.assertRaises(ModelException):
            model_error.update()

    def test_generate_results(self):
        # generate_results no longer takes verbosity argument, it uses self.verbosity
        # Test that it returns a dictionary
        results = self.model.generate_results()
        self.assertIsInstance(results, dict)


class OneNodeWaterModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model = OneNodeWaterModel(**get_water_init_args(schedule_small_draw))

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 1)
        self.assertListEqual(self.model.vol_fractions.tolist(), [1])

    def test_load_rc_data(self):
        rc_params_tuple = self.model.load_rc_data(**get_water_init_args())
        # load_rc_data now returns (capacitances, resistances) tuple
        capacitances, resistances = rc_params_tuple

        self.assertEqual(len(capacitances) + len(resistances), 2)
        self.assertAlmostEqual(capacitances["WH1"], 525651, places=0)
        self.assertAlmostEqual(resistances[("WH1", "AMB")], 0.66, places=2)

    def test_update(self):
        # Small water draw update
        self.model.update()
        self.assertAlmostEqual(self.model.next_states[0], 50.8, places=1)

        # Large water draw update - create new model
        model_large = OneNodeWaterModel(**get_water_init_args(schedule_large_draw))
        model_large.update()
        self.assertAlmostEqual(model_large.next_states[0], 18.4, places=1)


class TwoNodeWaterModelTestCase(unittest.TestCase):
    def setUp(self):
        self.model = TwoNodeWaterModel(**get_water_init_args(schedule_small_draw))

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 2)
        self.assertListEqual(self.model.vol_fractions.tolist(), [1 / 3, 2 / 3])

    def test_load_rc_data(self):
        rc_params_tuple = self.model.load_rc_data(**get_water_init_args())
        # load_rc_data now returns (capacitances, resistances) tuple
        capacitances, resistances = rc_params_tuple

        self.assertEqual(len(capacitances) + len(resistances), 5)
        self.assertIn("WH1", capacitances)
        self.assertIn(("WH1", "WH2"), resistances)
        self.assertIn(("WH1", "AMB"), resistances)

        self.assertAlmostEqual(capacitances["WH1"] * 2, capacitances["WH2"])
        self.assertAlmostEqual(capacitances["WH1"], 175217, places=0)
        self.assertAlmostEqual(resistances[("WH1", "WH2")], 6.21, places=2)
        self.assertAlmostEqual(resistances[("WH1", "AMB")], 1.84, places=2)

    def test_update(self):
        # Small water draw update
        self.model.update()
        self.assertAlmostEqual(self.model.next_states[0], 51.1, places=1)
        self.assertAlmostEqual(self.model.next_states[1], 50.6, places=1)

        # Large water draw update - create new model
        model_large = TwoNodeWaterModel(**get_water_init_args(schedule_large_draw))
        model_large.update()
        self.assertAlmostEqual(model_large.next_states[0], 35.2, places=1)
        self.assertAlmostEqual(model_large.next_states[1], 10, places=1)


class IdealWaterTestCase(unittest.TestCase):
    def setUp(self):
        self.model = IdealWaterModel(**get_water_init_args(schedule_no_draw))

    def test_initialize(self):
        self.assertEqual(self.model.n_nodes, 1)
        self.assertListEqual(self.model.vol_fractions.tolist(), [1])

        self.assertTupleEqual(self.model.A.shape, (1, 1))
        self.assertListEqual(self.model.input_names, ["T_AMB", "H_WH1"])
        self.assertAlmostEqual(self.model.A[0, 0], 1)

    def test_update(self):
        temp = self.model.states[0]
        result = self.model.update()
        self.assertAlmostEqual(self.model.next_states[0], temp)
        # result is now a dict
        self.assertIsInstance(result, dict)

        # Small draw - create new model
        model_small = IdealWaterModel(**get_water_init_args(schedule_small_draw))
        model_small.update()
        self.assertLess(model_small.next_states[0], temp)


if __name__ == "__main__":
    unittest.main()
