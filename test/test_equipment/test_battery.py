import unittest
import numpy as np

from ochre.Equipment import Battery
from ochre.Models import Envelope
from test.test_equipment import equip_init_args
from test.test_models.test_envelope import init_args as env_init_args


envelope = Envelope(**{**env_init_args, 'initial_temp_setpoint': 20})

init_args = equip_init_args.copy()
init_args.update({
    'soc_init': 0.5,
    'r_cell': 0.002,
    'envelope_model': envelope,
})

update_args = {
    # 'Zone Temperature': 20,
}


class BatteryTestCase(unittest.TestCase):
    """
    Test Case to test Battery Equipment.
    """

    def setUp(self):
        self.battery = Battery(**init_args)

    def test_init(self):
        self.assertAlmostEqual(self.battery.capacity, 5)
        self.assertAlmostEqual(self.battery.soc, 0.5)
        self.assertEqual(self.battery.control_type, 'Off')
        self.assertAlmostEqual(self.battery.n_series, 14)
        self.assertAlmostEqual(self.battery.r_internal, 0.0099, places=4)

        # defaults - no thermal model, yes degradation model
        self.assertIsNone(self.battery.thermal_model)
        self.assertIsNotNone(self.battery.degradation_states)
        self.assertAlmostEqual(self.battery.voc_curve(0.5), 3.7, places=1)
        self.assertAlmostEqual(self.battery.uneg_curve(0.5), 0.12, places=2)

    def test_parse_control_signal(self):
        # test SOC Rate control
        self.battery.parse_control_signal({}, {'SOC Rate': 0.2})
        self.assertAlmostEqual(self.battery.power_setpoint, 2.077, places=3)

        self.battery.parse_control_signal({}, {'SOC Rate': -0.2})
        self.assertAlmostEqual(self.battery.power_setpoint, -1.926, places=3)

    def test_update_internal_control(self):
        # test self-consumption with charge_from_solar
        self.battery.control_type = 'Self-Consumption'
        self.battery.parameters['charge_from_solar'] = 1
        mode = self.battery.update_internal_control({'net_power': -1})
        self.assertEqual(mode, 'Off')
        self.assertAlmostEqual(self.battery.power_setpoint, 0)

        mode = self.battery.update_internal_control({'net_power': -1, 'pv_power': -2})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.battery.power_setpoint, 1)

        mode = self.battery.update_internal_control({'net_power': -2, 'pv_power': -1})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.battery.power_setpoint, 1)

        self.battery.parameters['charge_from_solar'] = 0
        mode = self.battery.update_internal_control({'net_power': -2, 'pv_power': -1})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.battery.power_setpoint, 2)

        # test SOC limits
        self.battery.soc = self.battery.soc_max
        mode = self.battery.update_internal_control({'net_power': -1})
        self.assertEqual(mode, 'Off')
        self.assertEqual(self.battery.power_setpoint, 0)

    def test_get_power_limits(self):
        self.battery.soc = self.battery.soc_max
        p_min, p_max = self.battery.get_power_limits()
        self.assertEqual(p_max, 0)
        self.assertAlmostEqual(p_min, -5)

        self.battery.soc = self.battery.soc_min
        p_min, p_max = self.battery.get_power_limits()
        self.assertEqual(p_min, 0)
        self.assertAlmostEqual(p_max, 5)

        self.battery.soc = self.battery.soc_max - 0.001
        p_min, p_max = self.battery.get_power_limits()
        self.assertAlmostEqual(p_max, 0.62, places=2)
        self.assertAlmostEqual(p_min, -5)

    def test_calculate_power_and_heat(self):
        self.battery.soc = 0.5
        self.battery.power_setpoint = -10
        self.battery.mode = 'On'
        self.battery.calculate_power_and_heat({})
        self.assertAlmostEqual(self.battery.capacity_kwh_nominal, self.battery.parameters['capacity_kwh'])
        self.assertAlmostEqual(self.battery.capacity_kwh, self.battery.parameters['capacity_kwh'])
        self.assertAlmostEquals(self.battery.electric_kw, -5)  # max capacity
        self.assertAlmostEquals(self.battery.sensible_gain, 251, places=-1)
        self.assertEqual(len(self.battery.degradation_data), 1)
        self.assertTupleEqual(self.battery.degradation_data[0], (0.5, 25))

    def test_calculate_degradation(self):
        # test with no cycles
        self.battery.degradation_data = [(0.5, 26)] * 1000
        self.battery.calculate_degradation()
        self.assertAlmostEqual(len(self.battery.degradation_data), 0)
        self.assertAlmostEqual(self.battery.degradation_states[0], 0.00024, places=5)
        self.assertEqual(self.battery.degradation_states[1], 0)
        self.assertAlmostEqual(self.battery.degradation_states[2], -0.00410, places=5)
        self.assertAlmostEqual(self.battery.capacity_kwh_nominal, 10.04, places=2)

        # test with cycles
        self.battery.reset_time()
        ramp = [(soc, 25) for soc in np.arange(0.1, 0.8, 0.005)]
        cycle = ramp + ramp[::-1]  # charge and discharge
        self.battery.degradation_data = cycle * 100
        self.battery.calculate_degradation()
        self.assertAlmostEqual(self.battery.degradation_states[0], 0.019, places=3)
        self.assertAlmostEqual(self.battery.degradation_states[1], 0.00011, places=5)
        self.assertAlmostEqual(self.battery.degradation_states[2], -0.03053, places=5)
        self.assertAlmostEqual(self.battery.capacity_kwh_nominal, 10.11, places=2)

    def test_generate_results(self):
        results = self.battery.generate_results(3)
        self.assertEqual(len(results), 1)

        results = self.battery.generate_results(6)
        self.assertEqual(len(results), 4)

        results = self.battery.generate_results(9)
        self.assertEqual(len(results), 9)

    def test_update_model(self):
        self.battery.soc = 0.5
        self.battery.power_input = -10
        self.battery.update_model({})
        self.assertAlmostEquals(self.battery.soc, 0.483, places=3)

        # test SOC limit
        self.battery.soc = self.battery.soc_max - 0.005
        self.battery.power_setpoint = 5
        self.battery.mode = 'On'
        self.battery.calculate_power_and_heat({})
        self.assertAlmostEquals(self.battery.electric_kw, 3.1, places=1)
        self.battery.update_model({})
        self.assertAlmostEquals(self.battery.soc, 0.95, places=3)

    def test_get_kwh_remaining(self):
        self.battery.soc = self.battery.soc_min
        kwh_rem = self.battery.get_kwh_remaining()
        self.assertEqual(kwh_rem, 0)

        self.battery.soc = self.battery.soc_min + 1 / self.battery.capacity_kwh
        kwh_rem = self.battery.get_kwh_remaining(include_efficiency=False)
        self.assertEqual(kwh_rem, 1.0)

        kwh_rem = self.battery.get_kwh_remaining(include_efficiency=True)
        self.assertAlmostEqual(kwh_rem, 0.951, places=3)

        kwh_rem = self.battery.get_kwh_remaining(include_efficiency=True, max_power=0.1)
        self.assertAlmostEqual(kwh_rem, 0.970, places=3)

        kwh_rem = self.battery.get_kwh_remaining(discharge=False, include_efficiency=False)
        self.assertAlmostEqual(kwh_rem, 7.0, places=1)


class BatteryThermalModelTestCase(unittest.TestCase):
    """
    Test Case to test Battery Equipment with a thermal model.
    """

    def setUp(self):
        self.battery = Battery(zone='LIV', **init_args)

    def test_init(self):
        self.assertIsNotNone(self.battery.thermal_model)
        self.assertAlmostEqual(self.battery.thermal_model.states[self.battery.t_idx], 20)

    def test_calculate_power_and_heat(self):
        self.battery.soc = 0.5
        self.battery.power_setpoint = -15
        self.battery.mode = 'On'
        self.battery.thermal_model.states[0] = 20
        self.battery.calculate_power_and_heat(update_args)
        self.assertAlmostEqual(self.battery.capacity_kwh_nominal, self.battery.parameters['capacity_kwh'])
        self.assertAlmostEqual(self.battery.capacity_kwh, 9.7, places=1)
        self.assertGreater(self.battery.thermal_model.next_states[0], 20)
        self.assertAlmostEquals(self.battery.sensible_gain, 251, places=-1)
        self.assertEqual(len(self.battery.degradation_data), 1)
        self.assertTupleEqual(self.battery.degradation_data[0], (0.5, 20))

        # test min soc, low temp
        self.battery.soc = self.battery.soc_min + 0.001
        self.battery.power_setpoint = -15
        self.battery.thermal_model.states[0] = -5  # same as ambient
        self.battery.calculate_power_and_heat(update_args)
        self.battery.update_model(update_args)
        self.assertAlmostEqual(self.battery.soc, self.battery.soc_min, places=5)

        # test max soc, high temp
        self.battery.soc = self.battery.soc_max - 0.001
        self.battery.power_setpoint = 15
        self.battery.thermal_model.states[0] = 35  # same as ambient
        self.battery.calculate_power_and_heat(update_args)
        self.battery.update_model(update_args)
        self.assertAlmostEqual(self.battery.soc, self.battery.soc_max, places=5)

    def test_generate_results(self):
        results = self.battery.generate_results(6)
        self.assertEqual(len(results), 5)

    def test_update_model(self):
        # test thermal update
        self.battery.thermal_model.next_states[0] = 30
        self.battery.update_model({})
        self.assertAlmostEquals(self.battery.thermal_model.states[0], 30)


if __name__ == '__main__':
    unittest.main()
