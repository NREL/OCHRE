import unittest
import datetime as dt

from ochre.Equipment import HeatPumpWaterHeater, ElectricResistanceWaterHeater, \
    GasWaterHeater, TanklessWaterHeater, GasTanklessWaterHeater, WaterHeater
from ochre.Models import TwoNodeWaterModel, IdealWaterModel
from test.test_models.test_water import water_init_args, update_args_no_draw, update_args_small_draw, \
    update_args_large_draw
from test.test_equipment import equip_init_args

init_args = equip_init_args.copy()
init_args.update(water_init_args)
init_args.update({
    'rated input power (W)': 5000,  # capacity = 4000
    'eta_c': 0.8,
    'Energy Factor (-)': 0.9,  # for gas WH
    'initial_schedule': update_args_no_draw.copy(),
    'number of bedrooms': 2,
})

hpwh_init_args = init_args.copy()
hpwh_init_args.update({
    'HPWH COP': 2,
    'HPWH Power': 979,
    'HPWH SHR': 0.98,
    'HPWH Parasitics (W)': 3.0,
    'HPWH Fan Power (W)': 0.0462 * 181,
})


class WaterHeaterTestCase(unittest.TestCase):
    """
    Test Case to test Water Heater Equipment with static (non-ideal) capacity.
    """

    def setUp(self):
        self.wh = WaterHeater(**init_args)

    def test_init(self):
        self.assertFalse(self.wh.use_ideal_mode
        self.assertTrue(isinstance(self.wh.model, TwoNodeWaterModel))
        self.assertEqual(self.wh.h_upper_idx, 0)
        self.assertEqual(self.wh.h_lower_idx, 1)
        self.assertIsNone(self.wh.zone_name)

        self.assertNotEqual(self.wh.setpoint_temp, self.wh.model.states[0])

    def test_parse_control_signal(self):
        # test load fraction
        self.wh.mode = 'Off'
        control_signal = {'Load Fraction': 1}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, None)

        self.wh.mode = 'On'
        control_signal = {'Load Fraction': 0}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Off')

        control_signal = {'Load Fraction': 0.5}
        with self.assertRaises(Exception):
            self.wh.parse_control_signal(update_args_no_draw, control_signal)

        # test with setpoint and deadband
        self.wh.mode = 'Off'
        mode = self.wh.parse_control_signal(update_args_no_draw, {'Setpoint': 55, 'Deadband': 5})
        self.assertEqual(mode, None)
        self.assertEqual(self.wh.setpoint_temp, 55)
        self.assertEqual(self.wh.deadband_temp, 5)

        mode = self.wh.parse_control_signal(update_args_no_draw, {'Deadband': 4})
        self.assertEqual(mode, None)
        self.assertEqual(self.wh.setpoint_temp, 51.666667)
        self.assertEqual(self.wh.deadband_temp, 4)

        mode = self.wh.parse_control_signal(update_args_no_draw, {'Setpoint': 60})
        self.assertEqual(mode, 'On')
        self.assertEqual(self.wh.setpoint_temp, 60)
        self.assertEqual(self.wh.deadband_temp, 4)

    def test_run_duty_cycle_control(self):
        self.wh.mode = 'Off'
        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': 0.5})
        self.assertEqual(mode, 'Off')

        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': [1, 0]})
        self.assertEqual(mode, 'On')

        self.wh.mode = 'On'
        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': 0.5})
        self.assertEqual(mode, 'On')

        self.wh.mode = 'On'
        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp + 1
        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': 0.5})
        self.assertEqual(mode, 'Off')

    def test_run_thermostat_control(self):
        self.wh.mode = 'Off'
        mode = self.wh.run_thermostat_control(update_args_no_draw)
        self.assertEqual(mode, None)

        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp - self.wh.deadband_temp - 1
        mode = self.wh.run_thermostat_control(update_args_no_draw)
        self.assertEqual(mode, 'On')

        self.wh.mode = 'On'
        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp + 1
        mode = self.wh.run_thermostat_control(update_args_no_draw)
        self.assertEqual(mode, 'Off')

    def test_add_heat_from_mode(self):
        result = self.wh.add_heat_from_mode('On')
        self.assertListEqual(list(result), [0, 4000])

        result = self.wh.add_heat_from_mode('Off')
        self.assertListEqual(list(result), [0, 0])

        result = self.wh.add_heat_from_mode('On', duty_cycle=0.5)
        self.assertListEqual(list(result), [0, 2000])

    def test_calculate_power_and_heat(self):
        self.wh.mode = 'Off'
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 47, places=0)
        self.assertEqual(self.wh.delivered_heat, 0)
        self.assertEqual(self.wh.electric_kw, 0)
        self.assertEqual(self.wh.gas_therms_per_hour, 0)

        self.wh.mode = 'On'
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 1050, places=-1)
        self.assertAlmostEqual(self.wh.delivered_heat, self.wh.capacity_rated)
        self.assertAlmostEqual(self.wh.electric_kw, 5.0)
        self.assertEqual(self.wh.gas_therms_per_hour, 0)

    def test_generate_results(self):
        results = self.wh.generate_results(1)
        self.assertDictEqual(results, {})

        results = self.wh.generate_results(3)
        self.assertEqual(len(results), 5)
        self.assertIn('Water Heating Delivered (W)', results)

        results = self.wh.generate_results(6)
        self.assertEqual(len(results), 16)
        self.assertIn('Water Heating Mode', results)
        self.assertIn('Hot Water Delivered (L/min)', results)
        self.assertIn('Water Heating COP (-)', results)


class IdealWaterHeaterTestCase(unittest.TestCase):
    """
    Test Case to test Water Heater Equipment with ideal capacity.
    """

    def setUp(self):
        self.wh = WaterHeater(use_ideal_mode=True, **init_args)

        # update initial state to top of deadband (for 1-node model)
        self.wh.model.states[self.wh.t_upper_idx] = self.wh.setpoint_temp

    def test_init(self):
        self.assertTrue(self.wh.use_ideal_mode
        self.assertTrue(isinstance(self.wh.model, TwoNodeWaterModel))
        self.assertEqual(self.wh.h_lower_idx, 1)
        self.assertEqual(self.wh.h_upper_idx, 0)

    def test_run_duty_cycle_control(self):
        self.wh.mode = 'Off'
        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': 0.5})
        self.assertEqual(self.wh.duty_cycle_by_mode['On'], 0.5)
        self.assertEqual(mode, 'On')

        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': [1, 0]})
        self.assertEqual(self.wh.duty_cycle_by_mode['On'], 1)
        self.assertEqual(mode, 'On')

        mode = self.wh.parse_control_signal(update_args_no_draw, {'Duty Cycle': 0})
        self.assertEqual(self.wh.duty_cycle_by_mode['On'], 0)
        self.assertEqual(mode, 'Off')

    def test_update_internal_control(self):
        self.wh.mode = 'Off'
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.wh.duty_cycle_by_mode['On'], 0.823, places=2)

        # test with draw
        mode = self.wh.update_internal_control(update_args_small_draw)
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.wh.duty_cycle_by_mode['On'], 1, places=2)

        # test with temperature change
        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp - 0.1
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.wh.duty_cycle_by_mode['On'], 0.16, places=2)

        # test off
        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp + 1
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Off')
        self.assertEqual(self.wh.duty_cycle_by_mode['On'], 0)

    def test_calculate_power_and_heat(self):
        # test with no draw
        self.wh.mode = self.wh.update_internal_control(update_args_no_draw)
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.electric_kw, 4.115, places=2)
        self.assertAlmostEqual(self.wh.sensible_gain, 870, places=0)
        self.assertAlmostEqual(self.wh.model.next_states[0], self.wh.setpoint_temp, places=1)

        # test with draw
        self.wh.update_internal_control(update_args_small_draw)
        self.wh.calculate_power_and_heat(update_args_small_draw)
        self.assertAlmostEqual(self.wh.electric_kw, 5, places=2)
        self.assertAlmostEqual(self.wh.sensible_gain, 1050, places=-1)
        self.assertAlmostEqual(self.wh.model.next_states[0], self.wh.setpoint_temp, places=0)

        # test with large draw
        self.wh.update_internal_control(update_args_large_draw)
        self.wh.calculate_power_and_heat(update_args_large_draw)
        self.assertAlmostEqual(self.wh.electric_kw, 5, places=2)
        self.assertAlmostEqual(self.wh.model.next_states[0], 35, places=0)


class ERWaterHeaterTestCase(unittest.TestCase):

    def setUp(self):
        self.wh = ElectricResistanceWaterHeater(**init_args)

    def test_parse_control_signal(self):
        self.wh.mode = 'Off'
        control_signal = {'Duty Cycle': 1}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Upper On')
        self.assertEqual(self.wh.ext_mode_counters['Upper On'], dt.timedelta(minutes=0))  # gets updated later
        self.assertEqual(self.wh.ext_mode_counters['Lower On'], dt.timedelta(minutes=1))

        # test swap from upper to lower
        self.wh.mode = 'Upper On'
        self.wh.model.states[self.wh.t_upper_idx] = self.wh.setpoint_temp + 1
        control_signal = {'Duty Cycle': 1}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Lower On')

    def test_update_internal_control(self):
        self.wh.mode = 'Lower On'
        self.wh.model.states[self.wh.t_upper_idx] = self.wh.setpoint_temp - self.wh.deadband_temp - 1
        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp - self.wh.deadband_temp - 1
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Upper On')  # Upper element gets priority

        self.wh.mode = 'Upper On'
        self.wh.model.states[self.wh.t_upper_idx] = self.wh.setpoint_temp + 1
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Lower On')  # Lower turns on after 1 turn

        self.wh.mode = 'Lower On'
        self.wh.model.states[self.wh.t_lower_idx] = self.wh.setpoint_temp + 1
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Off')


class HPWaterHeaterTestCase(unittest.TestCase):

    def setUp(self):
        self.wh = HeatPumpWaterHeater(**hpwh_init_args)

    def test_parse_control_signal(self):
        self.wh.mode = 'Off'
        control_signal = {'HP Duty Cycle': 0, 'ER Duty Cycle': 0.9}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Off')

        self.wh.mode = 'Off'
        control_signal = {'HP Duty Cycle': 0.6, 'ER Duty Cycle': 0.4}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Heat Pump On')

        self.wh.mode = 'Upper On'
        control_signal = {'HP Duty Cycle': 0.5, 'ER Duty Cycle': 0}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Heat Pump On')

        self.wh.mode = 'Heat Pump On'
        self.wh.model.states[self.wh.t_upper_idx] = 60
        control_signal = {'HP Duty Cycle': 0.5, 'ER Duty Cycle': 0}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Off')

        # test HP only mode
        self.wh.hp_only_mode = True
        self.wh.mode = 'Off'
        control_signal = {'HP Duty Cycle': 1, 'ER Duty Cycle': 1}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Heat Pump On')

        self.wh.mode = 'Off'
        control_signal = {'HP Duty Cycle': 0, 'ER Duty Cycle': 1}
        mode = self.wh.parse_control_signal(update_args_no_draw, control_signal)
        self.assertEqual(mode, 'Off')

    def test_update_internal_control(self):
        # TODO: Jeff - may need more tests here to make sure HP thermostat control works
        self.assertEqual(self.wh.model.n_nodes, 12)

        self.wh.mode = 'Off'
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, None)

        self.wh.mode = 'Heat Pump On'
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, None)

        self.wh.mode = 'Upper On'
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Upper On')

        self.wh.mode = 'Off'
        self.wh.model.states[self.wh.t_lower_idx] = 20
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Heat Pump On')

        self.wh.mode = 'Off'
        self.wh.model.states[self.wh.t_upper_idx] = 30
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Upper On')

        # test HP only mode
        self.wh.hp_only_mode = True
        self.wh.mode = 'Off'
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Heat Pump On')

        self.wh.mode = 'Heat Pump On'
        self.wh.model.states[self.wh.t_upper_idx] = 60
        self.wh.model.states[self.wh.t_lower_idx] = 60
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertEqual(mode, 'Off')

    def test_add_heat_from_mode(self):
        result = self.wh.add_heat_from_mode('Heat Pump On')
        self.assertEqual(len(result), 12)
        self.assertEqual(sum(result), 979 * 2)
        self.assertAlmostEqual(result[5], 979 * 2 * 5 / 110)
        self.assertAlmostEqual(result[8], 979 * 2 * 20 / 110)

    def test_calculate_power_and_heat(self):
        self.wh.mode = 'Upper On'
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertEqual(self.wh.delivered_heat, 4000)
        self.assertAlmostEqual(self.wh.sensible_gain, 1046, places=-1)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertAlmostEqual(self.wh.electric_kw, 5.003)

        self.wh.mode = 'Heat Pump On'
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 2200, places=-1)
        self.assertAlmostEqual(self.wh.sensible_gain, -1075, places=-1)
        self.assertAlmostEqual(self.wh.latent_gain, -23, places=0)
        self.assertAlmostEqual(self.wh.hp_cop, 2.1, places=1)
        self.assertAlmostEqual(self.wh.hp_capacity, 2200, places=-1)
        self.assertAlmostEqual(self.wh.electric_kw, 1.05, places=2)

        # test with ideal capacity
        self.wh.use_ideal_moderue
        self.wh.duty_cycle_by_mode['Heat Pump On'] = 0
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 0, places=-1)
        self.assertAlmostEqual(self.wh.sensible_gain, 50, places=-1)
        self.assertAlmostEqual(self.wh.latent_gain, 0, places=0)
        self.assertAlmostEqual(self.wh.electric_kw, 0.003, places=3)

        self.wh.duty_cycle_by_mode['Heat Pump On'] = 0.8
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 1760, places=-1)
        self.assertAlmostEqual(self.wh.sensible_gain, -850, places=0)
        self.assertAlmostEqual(self.wh.electric_kw, 0.844, places=3)

        self.wh.duty_cycle_by_mode['Heat Pump On'] = 0.8
        self.wh.duty_cycle_by_mode['Upper On'] = 0.2
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 2560, places=-1)
        self.assertAlmostEqual(self.wh.sensible_gain, -650, places=0)
        self.assertAlmostEqual(self.wh.electric_kw, 1.84, places=2)

        self.wh.duty_cycle_by_mode['Heat Pump On'] = 0
        self.wh.duty_cycle_by_mode['Upper On'] = 1
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 4000, places=-1)
        self.assertAlmostEqual(self.wh.sensible_gain, 1050, places=-1)
        self.assertAlmostEqual(self.wh.electric_kw, 5.003, places=3)

    def test_generate_results(self):
        results = self.wh.generate_results(6)
        self.assertEqual(len(results), 19)


class GasWaterHeaterTestCase(unittest.TestCase):

    def setUp(self):
        self.wh = GasWaterHeater(**init_args)

    def test_calculate_power_and_heat(self):
        self.wh.mode = 'On'
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 4000, places=0)
        self.assertAlmostEqual(self.wh.gas_therms_per_hour, 0.171, places=2)
        self.assertAlmostEqual(self.wh.sensible_gain, 45, places=0)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertEqual(self.wh.electric_kw, 0)

        # Test with ideal capacity
        self.wh.use_ideal_moderue
        self.wh.duty_cycle_by_mode['On'] = 0.125
        self.wh.mode = 'On'
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.delivered_heat, 500, places=0)
        self.assertAlmostEqual(self.wh.gas_therms_per_hour, 0.021, places=2)
        self.assertAlmostEqual(self.wh.sensible_gain, 45, places=0)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertEqual(self.wh.electric_kw, 0)


class TanklessWaterHeaterTestCase(unittest.TestCase):

    def setUp(self):
        self.wh = TanklessWaterHeater(**init_args)

    def test_init(self):
        self.assertEqual(self.wh.use_ideal_modeue)
        self.assertTrue(isinstance(self.wh.model, IdealWaterModel))

    def test_update_internal_control(self):
        mode = self.wh.update_internal_control(update_args_no_draw)
        self.assertAlmostEqual(self.wh.duty_cycle_by_mode['On'], 0, places=0)
        self.assertEqual(mode, 'Off')

        mode = self.wh.update_internal_control(update_args_small_draw)
        self.assertAlmostEqual(self.wh.duty_cycle_by_mode['On'], 1, places=-1)
        self.assertEqual(mode, 'On')

    def test_calculate_power_and_heat(self):
        self.wh.delivered_heat = 0
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 0, places=4)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertEqual(self.wh.delivered_heat, 0)
        self.assertEqual(self.wh.electric_kw, 0)

        # Test with large water draw
        self.wh.mode = 'On'
        self.wh.heat_from_draw = 6000
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 0, places=4)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertAlmostEqual(self.wh.delivered_heat, 3680, places=0)
        self.assertAlmostEqual(self.wh.electric_kw, 5, places=2)

        # Test with small water draw
        self.wh.heat_from_draw = 500
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 0, places=4)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertAlmostEqual(self.wh.delivered_heat, 500, places=0)
        self.assertAlmostEqual(self.wh.electric_kw, 0.68, places=2)


class GasTanklessWaterHeaterTestCase(unittest.TestCase):

    def setUp(self):
        self.wh = GasTanklessWaterHeater(**init_args)

    def test_calculate_power_and_heat(self):
        # test off
        self.wh.heat_from_draw = 0
        self.wh.calculate_power_and_heat(update_args_no_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 0, places=4)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertEqual(self.wh.delivered_heat, 0)
        self.assertEqual(self.wh.gas_therms_per_hour, 0)
        self.assertAlmostEqual(self.wh.electric_kw, 0.007, places=3)

        # test on
        self.wh.mode = 'On'
        self.wh.heat_from_draw = 500
        self.wh.calculate_power_and_heat(update_args_small_draw)
        self.assertAlmostEqual(self.wh.sensible_gain, 0, places=4)
        self.assertEqual(self.wh.latent_gain, 0)
        self.assertAlmostEqual(self.wh.delivered_heat, 500, places=-1)
        self.assertAlmostEqual(self.wh.gas_therms_per_hour, 0.023, places=3)
        self.assertAlmostEqual(self.wh.electric_kw, 0.007, places=3)


if __name__ == '__main__':
    unittest.main()
