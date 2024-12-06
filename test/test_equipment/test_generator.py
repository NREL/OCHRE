import unittest
import datetime as dt

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
        self.assertEqual(self.generator.control_type, 'Off')

    def test_parse_control_signal(self):
        # test setpoint control
        mode = self.generator.parse_control_signal({}, {'P Setpoint': -2})
        self.assertEqual(mode, 1
        self.assertEqual(self.generator.power_setpoint, -2)

        mode = self.generator.parse_control_signal({}, {'P Setpoint': 0})
        self.assertEqual(mode, 0)

        # test control type
        control_signal = {'Control Type': 'Schedule'}
        mode = self.generator.parse_control_signal({}, control_signal)
        self.assertEqual(mode, 0)
        self.assertEqual(self.generator.control_type, 'Schedule')

        control_signal = {'Control Type': 'Other'}
        mode = self.generator.parse_control_signal({}, control_signal)
        self.assertEqual(mode, 0)
        self.assertEqual(self.generator.control_type, 'Schedule')

        # test parameter update
        control_signal = {'Parameters': {'charge_start_hour': 0}}
        mode = self.generator.parse_control_signal({}, control_signal)
        self.assertEqual(mode, 1)
        self.assertEqual(self.generator.control_type, 'Schedule')
        self.assertEqual(self.generator.power_setpoint, 1)

    def test_run_internal_control(self):
        # test schedule-based control
        self.generator.control_type = 'Schedule'
        mode = self.generator.run_internal_control({})
        self.assertEqual(mode, 0)
        self.assertEqual(self.generator.power_setpoint, 0)

        self.generator.current_time = init_args['start_time'] + dt.timedelta(
            hours=self.generator.parameters['discharge_start_hour'])
        mode = self.generator.run_internal_control({})
        self.assertEqual(mode, 1)
        self.assertEqual(self.generator.power_setpoint, - self.generator.parameters['discharge_power'])

        # test self-consumption control
        self.generator.control_type = 'Self-Consumption'
        mode = self.generator.run_internal_control({})
        self.assertEqual(mode, 0)

        mode = self.generator.run_internal_control({'net_power': 2})
        self.assertEqual(mode, 1)
        self.assertEqual(self.generator.power_setpoint, -2)

        mode = self.generator.run_internal_control({'net_power': -1})
        self.assertEqual(mode, 1)
        self.assertEqual(self.generator.power_setpoint, 1)

        # test self-consumption with export limit
        self.generator.parameters['export_limit'] = 1
        mode = self.generator.run_internal_control({'net_power': 3})
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(self.generator.power_setpoint, -2)

        mode = self.generator.run_internal_control({'net_power': 0.9})
        self.assertEqual(mode, 0)
        self.assertAlmostEqual(self.generator.power_setpoint, 0)

        # test off
        self.generator.control_type = 'Off'
        mode = self.generator.run_internal_control({})
        self.assertEqual(mode, 0)

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
        self.generator.control_type = 'Self-Consumption'
        self.generator.capacity_min = 1
        p_min, p_max = self.generator.get_power_limits()
        self.assertEqual(p_min, -self.generator.capacity)
        self.assertEqual(p_max, -1)

    def test_calculate_power_and_heat(self):
        self.generator.mode = 0
        self.generator.calculate_power_and_heat({})
        self.assertEqual(self.generator.electric_kw, 0)
        self.assertEqual(self.generator.sensible_gain, 0)

        # test generation - with ramp rate
        self.generator.on = 'O1
        self.generator.power_setpoint = -2
        self.generator.electric_kw = -1
        self.generator.calculate_power_and_heat({})
        self.assertAlmostEquals(self.generator.electric_kw, -1.1)
        self.assertAlmostEquals(self.generator.sensible_gain, 58, places=0)

        # test consumption - not allowed for generators
        self.generator.on = 'O1
        self.generator.power_setpoint = 2
        self.generator.calculate_power_and_heat({})
        self.assertAlmostEquals(self.generator.electric_kw, 0)
        self.assertAlmostEquals(self.generator.sensible_gain, 0, places=1)

    def test_generate_results(self):
        results = self.generator.generate_results(6)
        self.assertEqual(len(results), 3)


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


if __name__ == '__main__':
    unittest.main()
