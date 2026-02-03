import datetime as dt
import unittest

from ochre.Equipment import Equipment
from test.test_equipment import equip_init_args


class TestEquipment(Equipment):
    """
    Simple equipment class with internal controller that sets power based on minute of the hour
    """

    name = "Test Equipment"

    def __init__(self, max_p, **kwargs):
        super().__init__(**kwargs)

        self.max_p = max_p

    def update_internal_control(self):
        # Turns on for 5 minutes, then off for 5 minutes
        if self.current_time.minute % 10 >= 5:
            return "Off"
        else:
            return "On"

    def calculate_power_and_heat(self):
        if self.mode == "On":
            self.electric_kw = min(self.current_time.minute, self.max_p)
        else:
            self.electric_kw = 0


class EquipmentTestCase(unittest.TestCase):
    """
    Test Case to test the Equipment class.
    """

    def setUp(self):
        self.equipment = TestEquipment(15, **equip_init_args)

    def test_initialize(self):
        self.assertEqual(self.equipment.mode, "Off")
        self.assertListEqual(self.equipment.modes, ["On", "Off"])
        self.assertEqual(self.equipment.current_time, equip_init_args["start_time"])
        self.assertEqual(self.equipment.zone_name, "Indoor")
        self.assertDictEqual(self.equipment.parameters, {})

    def test_reset_time(self, start_time=None):
        self.equipment.update()
        self.assertNotEqual(self.equipment.current_time, self.equipment.start_time)

        self.equipment.reset_time()
        self.assertEqual(self.equipment.current_time, self.equipment.start_time)

    def test_update(self):
        # run for 3 time steps
        for _ in range(3):
            self.equipment.update()
        self.assertEqual(self.equipment.current_time, equip_init_args["start_time"] + equip_init_args["time_res"] * 3)
        self.assertEqual(self.equipment.mode, "On")
        self.assertEqual(self.equipment.time_in_mode, equip_init_args["time_res"] * 3)
        self.assertDictEqual(self.equipment.mode_cycles, {"On": 1, "Off": 0})
        self.assertEqual(self.equipment.electric_kw, 2)

        # run for 5 time steps
        for _ in range(5):
            self.equipment.update()
        self.assertEqual(self.equipment.current_time, equip_init_args["start_time"] + equip_init_args["time_res"] * 8)
        self.assertEqual(self.equipment.mode, "Off")
        self.assertEqual(self.equipment.time_in_mode, equip_init_args["time_res"] * 3)
        self.assertDictEqual(self.equipment.mode_cycles, {"On": 1, "Off": 1})
        self.assertEqual(self.equipment.electric_kw, 0)

        # Test with minimum on/off times
        self.equipment.mode = "On"
        self.equipment.time_in_mode = dt.timedelta(minutes=0)
        self.equipment.min_time_in_mode = {"On": dt.timedelta(minutes=2), "Off": dt.timedelta(minutes=2)}

        self.equipment.update()
        self.assertEqual(self.equipment.mode, "On")
        self.assertEqual(self.equipment.time_in_mode, equip_init_args["time_res"])

        self.equipment.time_in_mode = dt.timedelta(minutes=2)
        self.equipment.update()
        self.assertEqual(self.equipment.mode, "Off")
        self.assertEqual(self.equipment.time_in_mode, equip_init_args["time_res"])

    def test_simulate(self):
        results = self.equipment.simulate(duration=dt.timedelta(hours=1))
        self.assertEqual(len(results), 60)
        self.assertIn("Test Equipment Electric Power (kW)", results.columns)
        self.assertIn("Test Equipment Mode", results.columns)
        self.assertNotIn("Test Equipment Gas Power (therms/hour)", results.columns)

        modes = (["On"] * 5 + ["Off"] * 5) * 6
        self.assertListEqual(results["Test Equipment Mode"].values.tolist(), modes)

        powers = [min(i, 15) if m == "On" else 0 for i, m in enumerate(modes)]
        self.assertListEqual(results["Test Equipment Electric Power (kW)"].values.tolist(), powers)

    def test_generate_results(self):
        self.equipment.update()

        # low verbosity - generate_results() uses self.verbosity
        # Note: main_simulator=True by default, so Time and Electric Power are included
        self.equipment.verbosity = 1
        results = self.equipment.generate_results()
        self.assertIn("Time", results)
        self.assertIn("Test Equipment Electric Power (kW)", results)
        self.assertNotIn("Test Equipment Mode", results)

        # high verbosity (>=7) includes Mode
        self.equipment.verbosity = 9
        results = self.equipment.generate_results()
        self.assertIn("Test Equipment Mode", results)
        self.assertEqual(results["Test Equipment Mode"], "On")

    def test_calculate_mode_priority(self):
        self.assertDictEqual(self.equipment.ext_mode_counters, {mode: dt.timedelta(0) for mode in self.equipment.modes})
        self.equipment.current_time += dt.timedelta(minutes=1)

        duty_cycle = 1 / 2
        self.equipment.mode = "Off"
        mode_priority = self.equipment.calculate_mode_priority(duty_cycle)
        self.assertListEqual(mode_priority, ["Off", "On"])

        self.equipment.mode = "On"
        self.equipment.ext_mode_counters["On"] = dt.timedelta(minutes=7)
        mode_priority = self.equipment.calculate_mode_priority(duty_cycle)
        self.assertListEqual(mode_priority, ["On", "Off"])

        self.equipment.ext_mode_counters["On"] = dt.timedelta(minutes=8)
        mode_priority = self.equipment.calculate_mode_priority(duty_cycle)
        self.assertListEqual(mode_priority, ["Off"])

        duty_cycle = 1 / 5
        self.equipment.mode = "On"
        self.equipment.ext_mode_counters["On"] = dt.timedelta(minutes=2)
        mode_priority = self.equipment.calculate_mode_priority(duty_cycle)
        self.assertListEqual(mode_priority, ["On", "Off"])

        self.equipment.ext_mode_counters["On"] = dt.timedelta(minutes=3)
        mode_priority = self.equipment.calculate_mode_priority(duty_cycle)
        self.assertListEqual(mode_priority, ["Off"])

        duty_cycle = 1
        self.equipment.mode = "Off"
        mode_priority = self.equipment.calculate_mode_priority(duty_cycle)
        self.assertListEqual(mode_priority, ["On"])

    def test_run_zip(self):
        import numpy as np

        pf_multiplier = np.tan(np.arccos(0.9))  # ~0.48432210483785254

        self.equipment.electric_kw = 2
        self.equipment.run_zip(1)
        self.assertEqual(self.equipment.electric_kw, 2)
        self.assertAlmostEqual(self.equipment.reactive_kvar, pf_multiplier * 2)

        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.electric_kw, 2)
        self.assertAlmostEqual(self.equipment.reactive_kvar, pf_multiplier * 2)

        # Negative power factor (inductive)
        self.equipment.zip_data = (
            np.array([0, 0, 1]),  # Zp, Ip, Pp
            np.array([0, 0, 1]),  # Zq, Iq, Pq
            -pf_multiplier,  # negative pf_mult for inductive
        )
        self.equipment.electric_kw = 2
        self.equipment.run_zip(1)
        self.assertEqual(self.equipment.electric_kw, 2)
        self.assertAlmostEqual(self.equipment.reactive_kvar, -pf_multiplier * 2)

        # Test with Ip=1 (current-dependent)
        self.equipment.zip_data = (
            np.array([0, 1, 0]),  # Zp=0, Ip=1, Pp=0
            np.array([0, 1, 0]),  # Zq=0, Iq=1, Pq=0
            0,  # pf_mult=0 (unity power factor)
        )
        self.equipment.electric_kw = 2
        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.electric_kw, 2.2)
        self.assertEqual(self.equipment.reactive_kvar, 0)

        # Test with Zp=1 (voltage-squared dependent)
        self.equipment.zip_data = (
            np.array([1, 0, 0]),  # Zp=1, Ip=0, Pp=0
            np.array([1, 0, 0]),  # Zq=1, Iq=0, Pq=0
            0,  # pf_mult=0
        )
        self.equipment.electric_kw = 2
        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.electric_kw, 2.42)
        self.assertEqual(self.equipment.reactive_kvar, 0)

        # Test reactive with Zq=1
        self.equipment.zip_data = (
            np.array([1, 0, 0]),  # Zp=1, Ip=0, Pp=0
            np.array([1, 0, 0]),  # Zq=1, Iq=0, Pq=0
            pf_multiplier,
        )
        self.equipment.electric_kw = 2
        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.reactive_kvar, pf_multiplier * 2.42)


if __name__ == "__main__":
    unittest.main()
