import unittest

from ochre.Equipment import GasGenerator, GasFuelCell
from test.test_equipment import equip_init_args

init_args = equip_init_args.copy()


class GasGeneratorTestCase(unittest.TestCase):
    """
    Test Case to test GasGenerator Equipment.
    """

    def setUp(self):
        self.generator = GasGenerator(**init_args)

    def test_init(self):
        self.assertAlmostEqual(self.generator.capacity, 6)
        self.assertAlmostEqual(self.generator.efficiency_rated, 0.95)
        # Generator uses self_consumption_mode (bool), not control_type
        self.assertEqual(self.generator.self_consumption_mode, False)
        self.assertEqual(self.generator.mode, "Off")

    def test_update_external_control(self):
        # test setpoint control - update_external_control takes only control_signal
        mode = self.generator.update_external_control({"P Setpoint": -2})
        self.assertEqual(mode, "On")
        self.assertEqual(self.generator.power_setpoint, -2)

        mode = self.generator.update_external_control({"P Setpoint": 0})
        self.assertEqual(mode, "Off")

        # test self-consumption mode control
        control_signal = {"Self Consumption Mode": True}
        mode = self.generator.update_external_control(control_signal)
        self.assertEqual(self.generator.self_consumption_mode, True)

        control_signal = {"Self Consumption Mode": False}
        mode = self.generator.update_external_control(control_signal)
        self.assertEqual(self.generator.self_consumption_mode, False)

        # test import/export limits
        mode = self.generator.update_external_control({"Max Import Limit": 5})
        self.assertEqual(self.generator.import_limit, 5)

        mode = self.generator.update_external_control({"Max Export Limit": 3})
        self.assertEqual(self.generator.export_limit, 3)

    def test_update_internal_control(self):
        # test schedule-based control (default, not self-consumption mode)
        # When not in self-consumption mode, uses schedule for power setpoint
        self.generator.self_consumption_mode = False
        mode = self.generator.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.generator.power_setpoint, 0)

        # Set power schedule (uses end_use name in key)
        self.generator.current_schedule = {"Gas Generator Electric Power (kW)": -3}
        mode = self.generator.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertEqual(self.generator.power_setpoint, -3)

        # test self-consumption control
        self.generator.self_consumption_mode = True
        self.generator.current_schedule = {"net_power": 2}  # house consuming 2kW
        mode = self.generator.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertEqual(self.generator.power_setpoint, -2)  # generator produces to offset

        self.generator.current_schedule = {"net_power": -1}  # house exporting 1kW
        mode = self.generator.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertEqual(self.generator.power_setpoint, 1)  # generator reduces to limit export

        # test with no net_power (should warn and set to 0)
        self.generator.current_schedule = {}
        mode = self.generator.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.generator.power_setpoint, 0)

    def test_get_power_limits(self):
        # test without ramp rate
        self.generator.ramp_rate = None
        p_min, p_max = self.generator.get_power_limits()
        self.assertEqual(p_min, -self.generator.capacity)
        self.assertEqual(p_max, 0)

        # test with ramp rate
        self.generator.ramp_rate = 0.5
        self.generator.electric_kw = 0
        p_min, p_max = self.generator.get_power_limits()
        self.assertEqual(p_min, -0.5)

        self.generator.electric_kw = -self.generator.capacity + 0.1
        p_min, p_max = self.generator.get_power_limits()
        self.assertEqual(p_min, -self.generator.capacity)

        # test with minimum capacity
        self.generator.ramp_rate = None
        self.generator.self_consumption_mode = True
        self.generator.capacity_min = 1
        p_min, p_max = self.generator.get_power_limits()
        self.assertEqual(p_min, -self.generator.capacity)
        self.assertEqual(p_max, -1)

    def test_calculate_power_and_heat(self):
        # calculate_power_and_heat takes no args now
        self.generator.mode = "Off"
        self.generator.calculate_power_and_heat()
        self.assertEqual(self.generator.electric_kw, 0)
        self.assertEqual(self.generator.sensible_gain, 0)

        # test generation - with ramp rate
        self.generator.mode = "On"
        self.generator.power_setpoint = -2
        self.generator.electric_kw = -1
        self.generator.calculate_power_and_heat()
        self.assertAlmostEqual(self.generator.electric_kw, -1.1, places=1)
        self.assertGreater(self.generator.sensible_gain, 0)

        # test consumption - not allowed for generators
        self.generator.mode = "On"
        self.generator.power_setpoint = 2
        self.generator.calculate_power_and_heat()
        self.assertAlmostEqual(self.generator.electric_kw, 0)
        self.assertAlmostEqual(self.generator.sensible_gain, 0, places=1)

    def test_generate_results(self):
        # generate_results takes no args now, uses self.verbosity
        self.generator.verbosity = 6
        results = self.generator.generate_results()
        self.assertIn("Time", results)


class GasFuelCellTestCase(unittest.TestCase):
    """
    Test Case to test GasFuelCell Equipment.
    """

    def setUp(self):
        self.fc = GasFuelCell(**init_args)

    def test_get_efficiency(self):
        eff = self.fc.calculate_efficiency(6)
        self.assertEqual(eff, self.fc.efficiency_rated)

        eff = self.fc.calculate_efficiency(3)
        self.assertEqual(eff, self.fc.efficiency_rated)

        eff = self.fc.calculate_efficiency(2)
        self.assertEqual(eff, self.fc.efficiency_rated * 2 / 3)

        eff = self.fc.calculate_efficiency(0)
        self.assertAlmostEqual(eff, 0, places=2)


if __name__ == "__main__":
    unittest.main()
