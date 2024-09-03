import unittest

from ochre.Equipment import Equipment
from test.test_equipment import *


class TestEquipment(Equipment):
    """
    Simple equipment class with internal controller that sets power based on minute of the hour
    """
    name = 'Test Equipment'

    def __init__(self, max_p, **kwargs):
        super().__init__(**kwargs)

        self.max_p = max_p

    def run_internal_control(self):
        # Turns on for 5 minutes, then off for 5 minutes
        if self.current_time.minute % 10 >= 5:
            return 'Off'
        else:
            return 'On'

    def calculate_power_and_heat(self):
        if self.mode == 'On':
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
        self.assertEqual(self.equipment.mode, 'Off')
        self.assertListEqual(self.equipment.modes, ['On', 'Off'])
        self.assertEqual(self.equipment.current_time, equip_init_args['start_time'])
        self.assertEqual(self.equipment.zone_name, 'Indoor')
        self.assertDictEqual(self.equipment.parameters, {})

    def test_reset_time(self, start_time=None):
        self.equipment.update(1, {}, {})
        self.equipment.update_model({})
        self.assertNotEqual(self.equipment.current_time, self.equipment.start_time)

        self.equipment.reset_time()
        self.assertEqual(self.equipment.current_time, self.equipment.start_time)

    def test_update(self):
        # run for 3 time steps
        for _ in range(3):
            self.equipment.update(1, {}, {})
            self.equipment.update_model({})
        self.assertEqual(self.equipment.current_time, equip_init_args['start_time'] + equip_init_args['time_res'] * 3)
        self.assertEqual(self.equipment.mode, 'On')
        self.assertEqual(self.equipment.time_in_mode, equip_init_args['time_res'] * 3)
        self.assertDictEqual(self.equipment.mode_cycles, {'On': 1, 'Off': 0})
        self.assertEqual(self.equipment.electric_kw, 2)

        # run for 5 time steps
        for _ in range(5):
            self.equipment.update(1, {}, {})
            self.equipment.update_model({})
        self.assertEqual(self.equipment.current_time, equip_init_args['start_time'] + equip_init_args['time_res'] * 8)
        self.assertEqual(self.equipment.mode, 'Off')
        self.assertEqual(self.equipment.time_in_mode, equip_init_args['time_res'] * 3)
        self.assertDictEqual(self.equipment.mode_cycles, {'On': 1, 'Off': 1})
        self.assertEqual(self.equipment.electric_kw, 0)

        # Test with minimum on/off times
        self.equipment.mode = 'On'
        self.equipment.time_in_mode = dt.timedelta(minutes=0)
        self.equipment.min_time_in_mode = {'On': dt.timedelta(minutes=2),
                                           'Off': dt.timedelta(minutes=2)}

        self.equipment.update(1, {}, {})
        self.assertEqual(self.equipment.mode, 'On')
        self.assertEqual(self.equipment.time_in_mode, equip_init_args['time_res'])

        self.equipment.time_in_mode = dt.timedelta(minutes=2)
        self.equipment.update(1, {}, {})
        self.assertEqual(self.equipment.mode, 'Off')
        self.assertEqual(self.equipment.time_in_mode, equip_init_args['time_res'])

    def test_simulate(self):
        results = self.equipment.simulate(duration=dt.timedelta(hours=1))
        self.assertEqual(len(results), 60)
        self.assertIn('Test Equipment Electric Power (kW)', results.columns)
        self.assertIn('Test Equipment Mode', results.columns)
        self.assertNotIn('Test Equipment Gas Power (therms/hour)', results.columns)

        modes = (['On'] * 5 + ['Off'] * 5) * 6
        self.assertListEqual(results['Test Equipment Mode'].values.tolist(), modes)

        powers = [min(i, 15) if m == 'On' else 0 for i, m in enumerate(modes)]
        self.assertListEqual(results['Test Equipment Electric Power (kW)'].values.tolist(), powers)

    def test_generate_results(self):
        self.equipment.update({}, {})

        # low verbosity
        results = self.equipment.generate_results(1)
        self.assertDictEqual(results, {})

        # high verbosity
        results = self.equipment.generate_results(9)
        self.assertDictEqual(results, {'Test Equipment Mode': 'On'})

    def test_run_zip(self):
        pf_multiplier = 0.48432210483785254
        self.equipment.electric_kw = 2
        self.equipment.run_zip(1)
        self.assertEqual(self.equipment.electric_kw, 2)
        self.assertAlmostEqual(self.equipment.reactive_kvar, pf_multiplier * 2)

        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.electric_kw, 2)
        self.assertAlmostEqual(self.equipment.reactive_kvar, pf_multiplier * 2)

        self.equipment.zip_data['pf'] = -0.9
        self.equipment.zip_data['pf_mult'] = -self.equipment.zip_data['pf_mult']
        self.equipment.run_zip(1)
        self.assertEqual(self.equipment.electric_kw, 2)
        self.assertAlmostEqual(self.equipment.reactive_kvar, - pf_multiplier * 2)

        self.equipment.electric_kw = 2
        self.equipment.zip_data.update({'pf': 1, 'pf_mult': 0, 'Zp': 0, 'Ip': 1, 'Pp': 0})
        self.equipment.run_zip(1.1)
        self.assertEqual(self.equipment.electric_kw, 2.2)
        self.assertEqual(self.equipment.reactive_kvar, 0)

        self.equipment.electric_kw = 2
        self.equipment.zip_data.update({'pf': 1, 'Zp': 1, 'Ip': 0, 'Pp': 0})
        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.electric_kw, 2.42)
        self.assertEqual(self.equipment.reactive_kvar, 0)

        self.equipment.electric_kw = 2
        self.equipment.zip_data.update({'pf': 0.9, 'pf_mult': pf_multiplier, 'Zq': 1, 'Iq': 0, 'Pq': 0})
        self.equipment.run_zip(1.1)
        self.assertAlmostEqual(self.equipment.reactive_kvar, pf_multiplier * 2.42)


if __name__ == '__main__':
    unittest.main()
