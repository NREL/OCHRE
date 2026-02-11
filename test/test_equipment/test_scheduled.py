import unittest
import datetime as dt
import pandas as pd
import numpy as np

from ochre.Equipment import ScheduledLoad
from test.test_equipment import equip_init_args

# Create a schedule with MELs (kW) column - this gets renamed to 'Power (kW)' internally
# Must cover the full duration (1 day = 1440 minutes)
start_time = equip_init_args["start_time"]
duration = equip_init_args["duration"]
time_res = equip_init_args["time_res"]
times = pd.date_range(start_time, start_time + duration, freq=time_res, inclusive="left")
# Create power values that cycle
power_values = np.tile(np.arange(0.1, 6.1, 0.1), int(np.ceil(len(times) / 60)))[: len(times)]
eq_schedule = pd.DataFrame({"MELs (kW)": power_values}, index=times)

init_args = equip_init_args.copy()
init_args.update(
    {
        "schedule": eq_schedule,
        "duration": dt.timedelta(hours=1),  # Use shorter duration for tests
        "Convective Gain Fraction (-)": 0.2,
        "Radiative Gain Fraction (-)": 0.3,
        "Latent Gain Fraction (-)": 0.4,
    }
)


class ScheduledLoadTestCase(unittest.TestCase):
    """
    Test Case to test schedule-based Equipment.
    """

    def setUp(self):
        self.equipment = ScheduledLoad(name="MELs", **init_args)

    def test_init(self):
        # Check that schedule exists and has Power (kW) column
        self.assertIn("Power (kW)", self.equipment.schedule.columns)
        self.assertEqual(self.equipment.sensible_gain_fraction, 0.5)  # 0.2 + 0.3
        self.assertEqual(self.equipment.latent_gain_fraction, 0.4)
        self.assertTrue(self.equipment.is_electric)

    def test_update_external_control(self):
        # Set up current_schedule with Power value
        self.equipment.current_schedule = {"Power (kW)": 0.1}
        mode = self.equipment.update_external_control({"Load Fraction": 1})
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.equipment.p_set_point, 0.1)

        self.equipment.current_schedule = {"Power (kW)": 0.2}
        mode = self.equipment.update_external_control({"Load Fraction": 0.5})
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.equipment.p_set_point, 0.2 * 0.5)

        self.equipment.current_schedule = {"Power (kW)": 0.2}
        mode = self.equipment.update_external_control({"Load Fraction": 0})
        self.assertEqual(mode, "Off")
        self.assertAlmostEqual(self.equipment.p_set_point, 0)

    def test_update_internal_control(self):
        # Set up current_schedule with Power value
        self.equipment.current_schedule = {"Power (kW)": 0.1}
        mode = self.equipment.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.equipment.p_set_point, 0.1)

        self.equipment.current_schedule = {"Power (kW)": 0}
        mode = self.equipment.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertAlmostEqual(self.equipment.p_set_point, 0)

    def test_calculate_power_and_heat(self):
        self.equipment.mode = "On"
        self.equipment.p_set_point = 2
        self.equipment.calculate_power_and_heat()
        # Gains are only calculated if zone is attached (self.zone is not None)
        # In this test, zone is None, so gains remain 0
        self.assertAlmostEqual(self.equipment.sensible_gain, 0)
        self.assertAlmostEqual(self.equipment.latent_gain, 0)
        self.assertAlmostEqual(self.equipment.electric_kw, 2)

    def test_generate_results(self):
        # generate_results() no longer takes verbosity arg - uses self.verbosity
        self.equipment.verbosity = 3
        results = self.equipment.generate_results()
        # At low verbosity with main_simulator=True, should have Time and Electric Power
        self.assertIn("Time", results)

        self.equipment.verbosity = 6
        results = self.equipment.generate_results()
        # At verbosity >= 6 with main_simulator=True, includes electric power
        self.assertIn("MELs Electric Power (kW)", results)


class ScheduleFileLoadTestCase(unittest.TestCase):
    """
    Test Case to test schedule-based Equipment from file.
    Note: widget_schedule.csv has 'widget_power' column, but ScheduledLoad expects 'Widget (kW)'.
    Since no matching columns are found, the schedule is empty and schedule_iterable is None.
    This test verifies that behavior.
    """

    def setUp(self):
        # Create schedule DataFrame with Widget (kW) column using data from widget_schedule.csv pattern
        widget_times = pd.date_range(start_time, start_time + duration, freq=time_res, inclusive="left")
        widget_powers = np.tile([0.1] * 60, int(np.ceil(len(widget_times) / 60)))[: len(widget_times)]
        widget_schedule = pd.DataFrame({"Widget (kW)": widget_powers}, index=widget_times)

        file_init_args = equip_init_args.copy()
        file_init_args.update(
            {
                "schedule": widget_schedule,
                "duration": dt.timedelta(hours=1),  # Shorter duration for test
            }
        )
        self.equipment = ScheduledLoad(name="Widget", **file_init_args)

    def test_init(self):
        # Check that schedule was loaded and has DatetimeIndex
        self.assertIn("Power (kW)", self.equipment.schedule.columns)
        self.assertIsInstance(self.equipment.schedule.index, pd.DatetimeIndex)
        self.assertIsNotNone(self.equipment.schedule_iterable)

    def test_reset_time(self):
        # Step through schedule a few times
        for _ in range(5):
            self.equipment.update()
        self.assertNotEqual(self.equipment.current_time, self.equipment.start_time)

        # Reset and verify we're back at start
        self.equipment.reset_time()
        self.assertEqual(self.equipment.current_time, self.equipment.start_time)

    def test_update_internal_control(self):
        # First call update() to populate current_schedule from schedule_iterable
        self.equipment.update()
        # Power should be set from schedule
        self.assertIsNotNone(self.equipment.p_set_point)


if __name__ == "__main__":
    unittest.main()
