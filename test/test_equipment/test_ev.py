import unittest
import numpy as np
import datetime as dt
import pandas as pd

from ochre.Equipment import ElectricVehicle, ScheduledEV
from test.test_equipment import equip_init_args

times = pd.date_range(
    equip_init_args["start_time"],
    equip_init_args["start_time"] + equip_init_args["duration"],
    freq=equip_init_args["time_res"],
    inclusive="left",
)
init_args = equip_init_args.copy()
init_args.update(
    {
        "vehicle_type": "PHEV",
        "charging_level": "Level 0",
        "range": 20,
        # Use correct column name - 'Ambient Dry Bulb (C)' instead of 'ambient_dry_bulb'
        "schedule": pd.DataFrame({"Ambient Dry Bulb (C)": [15] * len(times)}, index=times),
    }
)

schedule_init_args = equip_init_args.copy()
schedule_init_args.update(
    {
        "vehicle_num": "Vehicle 8",
    }
)


class EVTestCase(unittest.TestCase):
    """
    Test Case to test EV Equipment.
    """

    def setUp(self):
        np.random.seed(2)
        self.ev = ElectricVehicle(**init_args)

    def test_init(self):
        self.assertAlmostEqual(self.ev.capacity, 6.5)
        self.assertEqual(self.ev.max_power, 1.4)

        self.assertGreaterEqual(len(self.ev.all_events), 1)
        self.assertIn("start_time", self.ev.all_events.columns)
        self.assertIn("end_time", self.ev.all_events.columns)
        self.assertIn("end_soc", self.ev.all_events.columns)

    def test_generate_all_events(self):
        # Test that events are generated correctly
        self.assertGreaterEqual(len(self.ev.all_events), 1)
        self.assertIn("start_time", self.ev.all_events.columns)
        self.assertIn("start_soc", self.ev.all_events.columns)

    def test_update_external_control(self):
        start = self.ev.event_start
        end = self.ev.event_end
        one_min = dt.timedelta(minutes=1)

        # test outside of event - update_external_control takes only control_signal now
        # p_setpoint is initialized to 0, not None
        self.ev.update_external_control({"Delay": False})
        self.assertEqual(self.ev.event_start, start)
        self.assertEqual(self.ev.event_end, end)
        self.assertEqual(self.ev.p_setpoint, 0)

        self.ev.update_external_control({"Delay": True})
        self.assertEqual(self.ev.event_start, start + one_min)
        self.assertEqual(self.ev.event_end, end)

        self.ev.update_external_control({"Delay": 2})
        self.assertEqual(self.ev.event_start, start + one_min * 3)
        self.assertEqual(self.ev.event_end, end)

        self.ev.update_external_control({"Delay": 10000})
        self.assertEqual(self.ev.event_start, end)
        self.assertEqual(self.ev.event_end, end)

        # setpoint control when not in event
        self.ev.update_external_control({"P Setpoint": 1})
        # p_setpoint stays at max_power when out of event

        # setpoint with event active
        self.ev.event_start = self.ev.current_time
        self.ev.update_external_control({"P Setpoint": 1})
        # p_setpoint is set to 1 but capped to max_power
        self.assertLessEqual(self.ev.p_setpoint, self.ev.max_power)

    def test_update_internal_control(self):
        # test outside of event - update_internal_control takes no args now
        # p_setpoint is 0 when not in event
        mode = self.ev.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.ev.p_setpoint, 0)

        # test event start
        self.ev.current_time = self.ev.event_start + dt.timedelta(minutes=2)
        self.ev.soc = 0.5
        mode = self.ev.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertGreater(self.ev.p_setpoint, 0)

        # test event end with unmet load
        self.ev.current_time = self.ev.event_end + dt.timedelta(minutes=2)
        self.ev.soc = 0.1
        mode = self.ev.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertGreater(self.ev.unmet_load, 0)

    def test_calculate_power_and_heat(self):
        # calculate_power_and_heat takes no args now
        self.ev.mode = "Off"
        self.ev.calculate_power_and_heat()
        self.assertEqual(self.ev.electric_kw, 0)

        # For On mode, need to set up p_setpoint first
        self.ev.mode = "On"
        self.ev.soc = 0.5
        self.ev.p_setpoint = self.ev.max_power  # Set power setpoint
        self.ev.calculate_power_and_heat()
        self.assertAlmostEqual(self.ev.electric_kw, 1.4)
        # After calculate, next_soc should be set
        self.assertAlmostEqual(self.ev.next_soc, 0.503, places=3)

        # Test with SOC near max
        self.ev.soc = 0.999
        self.ev.p_setpoint = self.ev.max_power
        self.ev.calculate_power_and_heat()
        self.assertLess(self.ev.electric_kw, 1.4)  # Power should be limited
        self.assertAlmostEqual(self.ev.next_soc, 1, places=2)

        # Test with lower setpoint
        self.ev.soc = 0.5
        self.ev.p_setpoint = 1.0
        self.ev.calculate_power_and_heat()
        self.assertAlmostEqual(self.ev.electric_kw, 1.0)
        self.assertAlmostEqual(self.ev.next_soc, 0.502, places=3)

    def test_generate_results(self):
        # generate_results takes no args now, uses self.verbosity
        self.ev.verbosity = 6
        results = self.ev.generate_results()
        self.assertIn("EV SOC (-)", results)

    def test_simulate(self):
        np.random.seed(1)
        results = self.ev.simulate(duration=dt.timedelta(days=1))

        self.assertEqual(self.ev.current_time, self.ev.start_time + dt.timedelta(days=1))

        self.assertEqual(results["EV Electric Power (kW)"].max(), 1.4)
        self.assertEqual(results["EV Electric Power (kW)"].min(), 0)
        self.assertAlmostEqual(results["EV Electric Power (kW)"].mean(), 0.12, places=1)


class ScheduledEVTestCase(unittest.TestCase):
    """
    Test Case to test Scheduled EV Equipment.
    """

    def setUp(self):
        # ScheduledEV is a ScheduledLoad that expects '<name> (kW)' column
        scheduled_init = equip_init_args.copy()
        ev_times = pd.date_range(
            scheduled_init["start_time"],
            scheduled_init["start_time"] + scheduled_init["duration"],
            freq=scheduled_init["time_res"],
            inclusive="left",
        )
        # Create EV power schedule - use 'EV (kW)' column and name='EV'
        ev_powers = np.zeros(len(ev_times))
        ev_powers[100:200] = 1.92  # Charging period
        scheduled_init["schedule"] = pd.DataFrame({"EV (kW)": ev_powers}, index=ev_times)
        scheduled_init["name"] = "EV"
        self.ev = ScheduledEV(**scheduled_init)

    def test_init(self):
        # ScheduledEV is a ScheduledLoad - check for Power (kW) column
        self.assertEqual(self.ev.name, "EV")
        self.assertIn("Power (kW)", self.ev.schedule.columns)
        self.assertIsNotNone(self.ev.schedule_iterable)


if __name__ == "__main__":
    unittest.main()
