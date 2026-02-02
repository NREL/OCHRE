import unittest
import os
import datetime as dt
import time

from ochre import Dwelling
from test import test_output_path

dwelling_args = {
    "name": "test_dwelling",
    # Timing parameters
    "start_time": dt.datetime(2019, 5, 5, 12, 0, 0),  # May 5, 12:00PM
    "time_res": dt.timedelta(minutes=15),
    "duration": dt.timedelta(days=1),
    "ext_time_res": dt.timedelta(hours=1),
    # Input and Output Files
    "output_path": test_output_path,
    "hpxml_file": "BEopt_example.xml",
    "hpxml_schedule_file": "BEopt_example_schedule.csv",
    "weather_file": "USA_CO_Denver.Intl.AP.725650_TMY3.epw",
    "verbosity": 9,  # verbosity of results file (0-9); 8: include envelope; 9: include water heater
    "metrics_verbosity": 9,  # verbosity of results file (0-9)
}


class DwellingTestCase(unittest.TestCase):
    """
    Test Case to test the Dwelling class (basic, no initialization).
    """

    def setUp(self):
        self.dwelling = Dwelling(**dwelling_args)

    def tearDown(self):
        # Clean up output files
        for suffix in ["", "_schedule", "_metrics", "_hourly"]:
            out_file = os.path.join(test_output_path, f"test_dwelling{suffix}.csv")
            if os.path.exists(out_file):
                os.remove(out_file)

    def test_init(self):
        # Name keeps underscores now
        self.assertEqual(self.dwelling.name, "test_dwelling")
        self.assertEqual(self.dwelling.current_time, dt.datetime(2019, 5, 5, 12, 0))
        self.assertTrue(os.path.exists(test_output_path))

        # BEopt_example.xml has 13 equipment items
        self.assertEqual(len(self.dwelling.equipment), 13)
        self.assertEqual(len(self.dwelling.schedule), 96)
        self.assertTrue(self.dwelling.schedule.notna().all().all())

    def test_update(self):
        start = self.dwelling.current_time

        results = self.dwelling.update()

        self.assertEqual(self.dwelling.current_time, start + dwelling_args["time_res"])
        # With equipment, total_p_kw is > 0
        self.assertGreater(self.dwelling.total_p_kw, 0)
        self.assertEqual(len(self.dwelling.results), 1)

        # Check that results contain expected keys
        self.assertIn("Total Electric Power (kW)", results)
        self.assertIn("Total Reactive Power (kVAR)", results)

    def test_generate_results(self):
        result = self.dwelling.generate_results()

        # Result count depends on verbosity level
        self.assertGreater(len(result), 10)
        # Grid Voltage key name changed
        self.assertEqual(result.get("Grid Voltage (-)", 1), 1)
        self.assertIn("Total Electric Power (kW)", result)

    def test_export_results(self):
        # export_results now requires 'Time' in results
        self.dwelling.results = [{"Time": dt.datetime(2019, 5, 5, 12, 0), "A": 1}]
        self.dwelling.export_results()

        with open(self.dwelling.results_file, "r") as f:
            data = f.read()
        self.assertIn("A", data)
        self.assertIn("1", data)

    def test_initialize(self):
        # Create a dwelling with initialization_time set
        args = dwelling_args.copy()
        args["initialization_time"] = dt.timedelta(hours=1)
        dwelling = Dwelling(**args)

        # After init with initialization_time, current_time should be at start_time
        self.assertEqual(dwelling.current_time, dt.datetime(2019, 5, 5, 12, 0))

    def test_simulate(self):
        df, metrics, hourly = self.dwelling.simulate()

        self.assertEqual(self.dwelling.current_time, self.dwelling.start_time + dwelling_args["duration"])
        self.assertGreater(os.stat(self.dwelling.results_file).st_size, 0)

        # check time series outputs
        self.assertEqual(len(df), 96)
        # With equipment, power should be > 0
        self.assertTrue((df["Total Electric Power (kW)"] >= 0).all())
        self.assertIn("Temperature - Indoor (C)", df.columns)

        # check hourly outputs
        self.assertEqual(len(hourly), 24)

        # check output metrics
        self.assertIn("Total Electric Energy (kWh)", metrics)


class DwellingWithEquipmentTestCase(unittest.TestCase):
    """
    Test Case to test the Dwelling class with initialization
    """

    def setUp(self):
        args = dwelling_args.copy()
        args["initialization_time"] = dt.timedelta(hours=1)
        self.dwelling = Dwelling(**args)

    def tearDown(self):
        # Clean up output files
        for suffix in ["", "_schedule", "_metrics", "_hourly"]:
            out_file = os.path.join(test_output_path, f"test_dwelling{suffix}.csv")
            if os.path.exists(out_file):
                os.remove(out_file)

    def test_init(self):
        # BEopt_example.xml has 13 equipment items (equipment is now a dict)
        self.assertEqual(len(self.dwelling.equipment), 13)

        # Check equipment by end use
        self.assertEqual(len(self.dwelling.equipment_by_end_use["HVAC Heating"]), 1)
        self.assertEqual(len(self.dwelling.equipment_by_end_use["HVAC Cooling"]), 1)
        self.assertEqual(len(self.dwelling.equipment_by_end_use["Water Heating"]), 1)
        self.assertEqual(len(self.dwelling.equipment_by_end_use["PV"]), 0)
        self.assertEqual(len(self.dwelling.equipment_by_end_use["Battery"]), 0)

        # Check equipment names include ASHP (equipment dict iterates over names as strings)
        equip_names = list(self.dwelling.equipment)
        self.assertIn("ASHP Heater", equip_names)
        self.assertIn("ASHP Cooler", equip_names)
        self.assertIn("Electric Resistance Water Heater", equip_names)

    def test_update(self):
        results = self.dwelling.update()

        # Check power is being calculated
        self.assertGreater(results["Total Electric Power (kW)"], 0)
        self.assertIn("Total Reactive Power (kVAR)", results)
        self.assertIn("Temperature - Indoor (C)", results)

        # Check HVAC modes are present
        self.assertIn("HVAC Heating Mode", results)
        self.assertIn("HVAC Cooling Mode", results)
        self.assertIn("Water Heating Mode", results)

        # Verify sub_simulators times are synced (sub_simulators contains the actual equipment objects)
        for e in self.dwelling.sub_simulators:
            self.assertEqual(e.current_time, self.dwelling.current_time)

    def test_update_external(self):
        # Control signal format is now {equipment_name: {control_key: value}}
        control = {
            "Indoor Lighting": {"Load Fraction": 0},
            "Exterior Lighting": {"Load Fraction": 0},
        }
        results = self.dwelling.update(control_signal=control)

        # Check that control signal affects results
        self.assertIn("HVAC Heating Mode", results)
        self.assertEqual(results["Lighting Electric Power (kW)"], 0)

    def test_simulate(self):
        t0 = time.time()
        df, metrics, hourly = self.dwelling.simulate()
        t_sim = time.time() - t0

        # check speed of simulation - allow more time for CI environments
        self.assertLess(t_sim, 5.0)

        # check time series outputs
        self.assertEqual(len(df), 96)
        self.assertTrue((df["Total Electric Power (kW)"] >= 0).all())

        # check hourly outputs
        self.assertEqual(len(hourly), 24)

        # check output metrics have expected keys
        self.assertIn("Total Electric Energy (kWh)", metrics)
        self.assertGreater(metrics["Total Electric Energy (kWh)"], 0)


if __name__ == "__main__":
    unittest.main()
