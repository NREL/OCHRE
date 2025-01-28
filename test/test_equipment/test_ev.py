import unittest
import numpy as np
import datetime as dt
import pandas as pd

from ochre.Equipment import ElectricVehicle, ScheduledEV
from test.test_equipment import equip_init_args

times = pd.date_range(equip_init_args['start_time'], equip_init_args['start_time'] + equip_init_args['duration'],
                      freq=equip_init_args['time_res'], inclusive='left')
init_args = equip_init_args.copy()
init_args.update({
    'vehicle_type': 'PHEV',
    'charging_level': 'Level 0',
    'range': 20,
    'schedule': pd.DataFrame({'ambient_dry_bulb': [15] * 1440}, index=times),
})

schedule_init_args = equip_init_args.copy()
schedule_init_args.update({
    'vehicle_num': 'Vehicle 8',
})


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

        self.assertEqual(len(self.ev.all_events), 1)
        self.assertIn('start_time', self.ev.all_events.columns)
        self.assertIn('end_time', self.ev.all_events.columns)
        self.assertIn('end_soc', self.ev.all_events.columns)
        self.assertEqual(self.ev.all_events.loc[0, 'start_time'], dt.datetime(2019, 4, 1, 17, 12))
        self.assertEqual(self.ev.all_events.loc[0, 'start_soc'], 0.588)

        self.assertIn(self.ev.temps_by_day.min(), range(-20, 45, 5))
        self.assertIn(self.ev.temps_by_day.max(), range(-20, 45, 5))

    def test_generate_all_events(self):
        probabilities = {(15, True): np.arange(0, 5)}
        event_data = pd.DataFrame({'day_id': np.arange(0, 5),
                                   'start_time': np.arange(200, 1200, 200),
                                   'duration': np.arange(400, 1400, 200),
                                   'start_soc': [0] * 5,
                                   'weekday': [True] * 5,
                                   'temperature': [15] * 5,
                                   }).set_index('day_id')
        self.ev.temps_by_day = pd.concat([self.ev.temps_by_day] * 5)
        self.ev.temps_by_day.index = pd.date_range(self.ev.temps_by_day.index[0], freq=dt.timedelta(days=1), periods=5)

        # test with overlap
        self.ev.generate_events(probabilities, event_data, None)

    def test_update_external_control(self):
        start = self.ev.event_start
        end = self.ev.event_end
        one_min = dt.timedelta(minutes=1)

        # test outside of event
        self.ev.update_external_control({}, {'Delay': False})
        self.assertEqual(self.ev.event_start, start)
        self.assertEqual(self.ev.event_end, end)
        self.assertEqual(self.ev.p_setpoint, None)

        self.ev.update_external_control({}, {'Delay': True})
        self.assertEqual(self.ev.event_start, start + one_min)
        self.assertEqual(self.ev.event_end, end)

        self.ev.update_external_control({}, {'Delay': 2})
        self.assertEqual(self.ev.event_start, start + one_min * 3)
        self.assertEqual(self.ev.event_end, end)

        self.ev.update_external_control({}, {'Delay': 10000})
        self.assertEqual(self.ev.event_start, end)
        self.assertEqual(self.ev.event_end, end)

        # setpoint control
        self.ev.update_external_control({}, {'P Setpoint': 1})
        self.assertEqual(self.ev.p_setpoint, None)

        # setpoint with part load enabled
        self.ev.event_start = self.ev.current_time
        self.ev.update_external_control({}, {'P Setpoint': 1})
        self.assertEqual(self.ev.p_setpoint, 1.4)

        self.ev.enable_part_load = True
        self.ev.update_external_control({}, {'P Setpoint': 1})
        self.assertEqual(self.ev.p_setpoint, 1)

        # SOC rate control
        self.ev.event_start = self.ev.current_time
        self.ev.update_external_control({}, {'SOC Rate': 0.2})
        self.assertAlmostEqual(self.ev.p_setpoint, 1.444, places=3)

    def test_update_internal_control(self):
        # test outside of event
        mode = self.ev.update_internal_control({})
        self.assertEqual(mode, 'Off')
        self.assertIsNone(self.ev.p_setpoint)

        # test event start
        self.ev.current_time = self.ev.event_start + dt.timedelta(minutes=2)
        self.soc = 0.5
        mode = self.ev.update_internal_control({})
        self.assertEqual(mode, 'On')
        self.assertIsNone(self.ev.p_setpoint)
        self.assertNotEqual(self.ev.soc, 0.5)

        # test event end with unmet load
        self.ev.current_time = self.ev.event_end + dt.timedelta(minutes=2)
        self.ev.soc = 0.1
        mode = self.ev.update_internal_control({})
        self.assertEqual(mode, 'Off')
        self.assertGreater(self.ev.unmet_load, 0)

    def test_calculate_power_and_heat(self):
        self.ev.mode = 'Off'
        self.ev.calculate_power_and_heat({})
        self.assertEqual(self.ev.electric_kw, 0)

        self.ev.mode = 'On'
        self.ev.soc = 0.5
        self.ev.calculate_power_and_heat({})
        self.ev.update_model({})
        self.assertAlmostEqual(self.ev.electric_kw, 1.4)
        self.assertAlmostEqual(self.ev.soc, 0.503, places=3)

        self.ev.soc = 0.999
        self.ev.calculate_power_and_heat({})
        self.ev.update_model({})
        self.assertLess(self.ev.electric_kw, 1.4)
        self.assertAlmostEqual(self.ev.soc, 1)

        self.ev.soc = 0.5
        self.ev.p_setpoint = 1
        self.ev.calculate_power_and_heat({})
        self.ev.update_model({})
        self.assertAlmostEqual(self.ev.electric_kw, 1)
        self.assertAlmostEqual(self.ev.soc, 0.502, places=3)

    def test_generate_results(self):
        results = self.ev.generate_results(6)
        self.assertEqual(len(results), 5)
        self.assertIn('EV SOC (-)', results)

    def test_simulate(self):
        np.random.seed(1)
        results = self.ev.simulate(duration=dt.timedelta(days=1))

        self.assertEqual(self.ev.current_time, self.ev.start_time + dt.timedelta(days=1))

        self.assertEqual(results['EV Electric Power (kW)'].max(), 1.4)
        self.assertEqual(results['EV Electric Power (kW)'].min(), 0)
        self.assertAlmostEqual(results['EV Electric Power (kW)'].mean(), 0.12, places=2)
        self.assertEqual(results['EV Unmet Load (kWh)'].mean(), 0)

        self.assertEqual(results['EV SOC (-)'].max(), 1)
        self.assertEqual(results['EV SOC (-)'].min(), 0)
        self.assertAlmostEqual((results['EV Mode'] == 'On').mean(), 0.28, places=2)

        self.assertDictEqual(self.ev.mode_cycles, {'On': 1, 'Off': 0})


class ScheduledEVTestCase(unittest.TestCase):
    """
    Test Case to test Scheduled EV Equipment.
    """

    def setUp(self):
        self.ev = ScheduledEV(**schedule_init_args)

    def test_init(self):
        self.assertEqual(len(self.ev.schedule), 1440)
        self.assertAlmostEqual(self.ev.schedule.iloc[0], 1.920)
        self.assertAlmostEqual(self.ev.schedule.iloc[10], 0)
        self.assertIsNotNone(self.ev.schedule_iterable)


if __name__ == '__main__':
    unittest.main()
