import os
import unittest
import datetime as dt

from ochre.utils import default_input_path
from ochre.Equipment import PV
from test.test_equipment import equip_init_args

init_args = equip_init_args.copy()
init_args.update({
    'capacity': 10,
    'tilt': 20,
    'azimuth': 180,
    'time_res': dt.timedelta(minutes=15),
})

scheduled_init_args = equip_init_args.copy()
scheduled_init_args.update({
    'time_res': dt.timedelta(minutes=15),
    'equipment_schedule_file': 'test_pv_schedule.csv',
})


class PVTestCase(unittest.TestCase):
    """
    Test Case to test PV Equipment.
    """
    def setUp(self):
        self.pv = PV(**init_args)

        # run until noon so power is not zero
        for _ in range(4 * 12):
            self.pv.update(1, {}, {})

    def test_init(self):
        self.assertEqual(len(self.pv.schedule), 4 * 24)
        self.assertAlmostEqual(self.pv.schedule[init_args['start_time'] + dt.timedelta(hours=12)], -8.16, places=2)
        self.assertAlmostEqual(self.pv.inverter_min_pf_factor, 0.75, places=2)

    def test_parse_control_signal(self):
        mode = self.pv.parse_control_signal({}, {'P Setpoint': -5, 'Q Setpoint': 1})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -5)
        self.assertAlmostEqual(self.pv.q_set_point, 1)

        mode = self.pv.parse_control_signal({}, {'P Setpoint': -20, 'Q Setpoint': 1})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -8.22, places=2)
        self.assertAlmostEqual(self.pv.q_set_point, 1)

        # test PV curtailment
        mode = self.pv.parse_control_signal({}, {'P Curtailment (kW)': 1})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -7.25, places=2)

        # test PV curtailment
        mode = self.pv.parse_control_signal({}, {'P Curtailment (%)': 50})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -4.13, places=2)

        # test power factor
        mode = self.pv.parse_control_signal({}, {'Power Factor': -0.95})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -6.37, places=2)
        self.assertAlmostEqual(self.pv.q_set_point, 2.09, places=2)

        # test priority
        self.pv.parse_control_signal({}, {'Priority': 'CPF'})
        self.assertEqual(self.pv.inverter_priority, 'CPF')

    def test_update_internal_control(self):
        mode = self.pv.update_internal_control({})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -8.16, places=2)
        self.assertAlmostEqual(self.pv.q_set_point, 0)

    def test_calculate_power_and_heat(self):
        self.pv.p_set_point = -5
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat({})
        self.assertEqual(self.pv.electric_kw, -5)
        self.assertEqual(self.pv.reactive_kvar, 1)

        # var priority
        self.pv.p_set_point = -12
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat({})
        self.assertAlmostEqual(self.pv.electric_kw, -(10 ** 2 - 1 ** 2) ** 0.5)
        self.assertEqual(self.pv.reactive_kvar, 1)

        # 0.8 min PF, var priority
        self.pv.p_set_point = -11
        self.pv.q_set_point = 11
        self.pv.calculate_power_and_heat({})
        self.assertAlmostEqual(self.pv.electric_kw, -8)
        self.assertAlmostEqual(self.pv.reactive_kvar, 6, places=2)

        # watt priority
        self.pv.inverter_priority = 'Watt'
        self.pv.p_set_point = -12
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat({})
        self.assertAlmostEqual(self.pv.electric_kw, -10)
        self.assertAlmostEqual(self.pv.reactive_kvar, 0)

        # CPF priority
        self.pv.inverter_priority = 'CPF'
        self.pv.p_set_point = -12
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat({})
        self.assertAlmostEqual(self.pv.electric_kw, -9.97, places=2)
        self.assertAlmostEqual(self.pv.reactive_kvar, 0.83, places=2)

    def test_generate_results(self):
        results = self.pv.generate_results(6)
        self.assertEqual(len(results), 3)


class ScheduledPVTestCase(unittest.TestCase):
    """
    Test Case to test Scheduled PV Equipment.
    """
    def setUp(self):
        self.pv = PV(**scheduled_init_args)

        # run until noon so power is not zero
        for _ in range(4 * 12):
            self.pv.update(1, {}, {})

    def test_init(self):
        self.assertEqual(len(self.pv.schedule), 4 * 24)
        self.assertAlmostEqual(self.pv.schedule[init_args['start_time'] + dt.timedelta(hours=12)], -2.01, places=2)

    def test_update_internal_control(self):
        mode = self.pv.update_internal_control({})
        self.assertEqual(mode, 'On')
        self.assertAlmostEqual(self.pv.p_set_point, -2.01, places=2)
        self.assertAlmostEqual(self.pv.q_set_point, 0)


if __name__ == '__main__':
    unittest.main()
