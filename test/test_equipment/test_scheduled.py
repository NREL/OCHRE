import unittest
import pandas as pd
import numpy as np

from ochre.Equipment import ScheduledLoad
from test.test_equipment import equip_init_args

init_args = equip_init_args.copy()
init_args.update({
    'MELs electric power (W)': 100,
    'MELs convective gainfrac': 0.2,
    'MELs radiative gainfrac': 0.3,
    'MELs latent gainfrac': 0.4,
})

file_init_args = init_args.copy()
file_init_args.update({
    'equipment_schedule_file': 'widget_schedule.csv',
    'val_col': 'widget_power',
    'schedule_scale_factor': 2,
})

times = pd.date_range(equip_init_args['start_time'], periods=60, freq=equip_init_args['time_res'])
eq_schedule = pd.DataFrame({'plug_loads': np.arange(1, 7, 0.1)}, index=times)
schedule_init_args = init_args.copy()
schedule_init_args['schedule'] = eq_schedule


class ScheduledLoadTestCase(unittest.TestCase):
    """
    Test Case to test schedule-based Equipment.
    """

    def setUp(self):
        # print(schedule_init_args['schedule'].iloc[0])
        self.equipment = ScheduledLoad(name='MELs', **schedule_init_args)

    def test_init(self):
        self.assertIsNone(self.equipment.schedule)
        self.assertEqual(self.equipment.schedule_name, 'plug_loads')
        self.assertEqual(self.equipment.sensible_gain_fraction, 0.5)

    def test_parse_control_signal(self):
        mode = self.equipment.parse_control_signal({'plug_loads': 100}, {'Load Fraction': 1})
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(self.equipment.p_set_point, 0.1)

        mode = self.equipment.parse_control_signal({'plug_loads': 200}, {'Load Fraction': 0.5})
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(self.equipment.p_set_point, 0.2 * 0.5)

        mode = self.equipment.parse_control_signal({'plug_loads': 200}, {'Load Fraction': 0})
        self.assertEqual(mode, 0)
        self.assertAlmostEqual(self.equipment.p_set_point, 0)

    def test_run_internal_control(self):
        mode = self.equipment.run_internal_control({'plug_loads': 100})
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(self.equipment.p_set_point, 0.1)

        mode = self.equipment.run_internal_control({'plug_loads': 0})
        self.assertEqual(mode, 0)
        self.assertAlmostEqual(self.equipment.p_set_point, 0)

    def test_calculate_power_and_heat(self):
        self.equipment.on_frac = 'O1
        self.equipment.p_set_point = 2
        self.equipment.calculate_power_and_heat({})
        self.assertAlmostEqual(self.equipment.sensible_gain, 1000)
        self.assertAlmostEqual(self.equipment.latent_gain, 800)
        self.assertAlmostEqual(self.equipment.electric_kw, 2)

    def test_generate_results(self):
        results = self.equipment.generate_results(3)
        self.assertEqual(len(results), 0)

        results = self.equipment.generate_results(6)
        self.assertEqual(len(results), 3)


class ScheduleFileLoadTestCase(unittest.TestCase):
    """
    Test Case to test schedule-based Equipment.
    """

    def setUp(self):
        # print(schedule_init_args['schedule'].iloc[0])
        self.equipment = ScheduledLoad(name='Widget', **file_init_args)

    def test_init(self):
        self.assertIsNotNone(self.equipment.schedule)
        self.assertEqual(len(self.equipment.schedule), 1440)
        self.assertIsNotNone(self.equipment.schedule_iterable)

    def test_reset_time(self, start_time=None):
        for _ in range(5):
            next(self.equipment.schedule_iterable)
        self.equipment.reset_time()
        p = next(self.equipment.schedule_iterable)
        self.assertEqual(p, 0.2)

    def test_run_internal_control(self):
        mode = self.equipment.run_internal_control({})
        self.assertEqual(mode, 1)
        self.assertAlmostEqual(self.equipment.p_set_point, 0.2)


if __name__ == '__main__':
    unittest.main()
