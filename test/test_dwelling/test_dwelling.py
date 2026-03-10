import unittest
import os
import shutil
import datetime as dt
import time

import pandas as pd

from ochre import Dwelling
from ochre.utils.resstock import _parse_unit, convert_units, load_crosswalk
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

        # check speed of simulation
        self.assertLess(t_sim, 5.0)

        # check time series outputs
        self.assertEqual(len(df), 96)
        self.assertTrue((df["Total Electric Power (kW)"] >= 0).all())

        # check hourly outputs
        self.assertEqual(len(hourly), 24)

        # check output metrics have expected keys
        self.assertIn("Total Electric Energy (kWh)", metrics)
        self.assertGreater(metrics["Total Electric Energy (kWh)"], 0)


class ResStockOutputTestCase(unittest.TestCase):
    """Test that ResStock output format produces equivalent results to OCHRE format.

    Runs two simulations with the same dwelling config and explicit seed:
    - One with output_format="ochre" (default)
    - One with output_format="resstock"
    Then verifies that the ResStock output matches the OCHRE output after unit
    conversion, and that the ResStock output files have correct structure.
    """

    ochre_output = os.path.join(test_output_path, "ochre_run")
    resstock_output = os.path.join(test_output_path, "resstock_run")

    @classmethod
    def setUpClass(cls):
        base_args = dwelling_args.copy()
        base_args["initialization_time"] = dt.timedelta(hours=1)
        base_args["seed"] = 42

        # Run OCHRE simulation
        ochre_args = base_args.copy()
        ochre_args["output_path"] = cls.ochre_output
        cls.ochre_dwelling = Dwelling(**ochre_args)
        cls.ochre_df, cls.ochre_metrics, cls.ochre_hourly = cls.ochre_dwelling.simulate()

        # Run ResStock simulation
        resstock_args = base_args.copy()
        resstock_args["output_format"] = "resstock"
        resstock_args["output_path"] = cls.resstock_output
        cls.resstock_dwelling = Dwelling(**resstock_args)
        cls.resstock_ts, cls.resstock_annual, cls.resstock_hourly = cls.resstock_dwelling.simulate()

        cls.crosswalk = load_crosswalk()
        cls.hours_per_step = base_args["time_res"].total_seconds() / 3600

    @classmethod
    def tearDownClass(cls):
        for path in [cls.ochre_output, cls.resstock_output]:
            if os.path.isdir(path):
                shutil.rmtree(path)

    def test_resstock_files_exist(self):
        self.assertTrue(os.path.isfile(os.path.join(self.resstock_output, "results_timeseries.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.resstock_output, "results_annual.csv")))

    def test_timeseries_has_units_row(self):
        with open(os.path.join(self.resstock_output, "results_timeseries.csv")) as f:
            lines = f.readlines()
        # Row 0 = header, Row 1 = units, Row 2+ = data
        self.assertGreater(len(lines), 2)
        # Units row should have same number of fields as header
        self.assertEqual(len(lines[0].split(",")), len(lines[1].split(",")))

    def test_timeseries_row_count(self):
        self.assertEqual(len(self.resstock_ts), len(self.ochre_df))

    def test_total_electric_energy_matches(self):
        ochre_kwh = (self.ochre_df["Total Electric Power (kW)"] * self.hours_per_step).sum()
        resstock_kwh = self.resstock_ts["Fuel Use: Electricity: Total"].sum()
        self.assertAlmostEqual(ochre_kwh, resstock_kwh, places=4)

    def test_end_use_energy_matches(self):
        """Per-end-use comparison for columns present in both outputs."""
        valid = self.crosswalk[
            self.crosswalk["OCHRE"].notna()
            & (self.crosswalk["OCHRE"] != "")
            & self.crosswalk["ResStock Timeseries"].notna()
            & (self.crosswalk["ResStock Timeseries"] != "")
        ]

        checked = 0
        for _, row in valid.iterrows():
            ochre_col = row["OCHRE"]
            rs_col = row["ResStock Timeseries"]
            target_unit = row.get("ResStock Timeseries Unit", "")
            if pd.isna(target_unit):
                target_unit = ""

            if ochre_col not in self.ochre_df.columns:
                continue
            if rs_col not in self.resstock_ts.columns:
                continue

            from_unit = _parse_unit(ochre_col)
            expected = convert_units(self.ochre_df[ochre_col], from_unit, target_unit, self.hours_per_step)
            actual = self.resstock_ts[rs_col]

            pd.testing.assert_series_equal(
                actual.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
                atol=1e-6,
                rtol=0,
            )
            checked += 1

        # Ensure we actually checked some columns
        self.assertGreater(checked, 5)

    def test_temperature_conversion(self):
        expected_f = self.ochre_df["Temperature - Indoor (C)"] * 9.0 / 5.0 + 32.0
        actual_f = self.resstock_ts["Temperature: Conditioned Space"]
        pd.testing.assert_series_equal(
            actual_f.reset_index(drop=True),
            expected_f.reset_index(drop=True),
            check_names=False,
            atol=1e-6,
            rtol=0,
        )

    def test_annual_totals_consistent_with_timeseries(self):
        """Annual sums should match timeseries column sums after unit conversion."""
        ts_to_annual = {}
        for _, row in self.crosswalk.iterrows():
            ts = row.get("ResStock Timeseries", "")
            annual = row.get("ResStock Annual", "")
            if pd.notna(ts) and ts and pd.notna(annual) and annual:
                ts_to_annual[ts] = annual

        annual_dict = dict(zip(self.resstock_annual["Metric"], self.resstock_annual["Value"]))

        checked = 0
        for ts_col, annual_col in ts_to_annual.items():
            if ts_col not in self.resstock_ts.columns:
                continue
            if annual_col not in annual_dict:
                continue

            match = self.crosswalk[self.crosswalk["ResStock Timeseries"] == ts_col]
            ts_unit = match["ResStock Timeseries Unit"].iloc[0]
            if pd.isna(ts_unit):
                ts_unit = ""
            annual_unit = _parse_unit(annual_col)

            ts_sum = self.resstock_ts[ts_col].sum()
            expected_annual = convert_units(ts_sum, ts_unit, annual_unit)

            self.assertAlmostEqual(
                annual_dict[annual_col],
                expected_annual,
                places=3,
                msg=f"{annual_col}: annual={annual_dict[annual_col]}, expected={expected_annual}",
            )
            checked += 1

        self.assertGreater(checked, 3)

    def test_hourly_row_count(self):
        self.assertEqual(len(self.resstock_hourly), len(self.ochre_hourly))


if __name__ == "__main__":
    unittest.main()
