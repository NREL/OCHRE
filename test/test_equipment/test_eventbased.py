import unittest
import numpy as np
import datetime as dt

from ochre.Equipment import DailyLoad
from test.test_equipment import equip_init_args

init_args = equip_init_args.copy()
init_args.update({
    'max_power': 2,
    'event_duration': dt.timedelta(minutes=10),
    'equipment_pdf_file': 'widget_pdf.csv',
})


class EventBasedLoadTestCase(unittest.TestCase):
    """
    Test Case to test event-based Equipment. Uses DailyLoad class for testing
    """

    def setUp(self):
        np.random.seed(1)
        self.e = DailyLoad(name='Event Widget', **init_args)

    def test_init(self):
        self.assertEqual(len(self.e.all_events), 2)
        self.assertGreater(self.e.event_start, self.e.current_time)
        self.assertEqual(self.e.event_start.date(), self.e.current_time.date())
        self.assertEqual(self.e.event_end - self.e.event_start, init_args['event_duration'])

    def test_update_external_control(self):
        first_event_start = self.e.event_start

        mode = self.e.update_external_control({}, {'nothing': 0})
        self.assertEqual(self.e.event_start, first_event_start)
        self.assertEqual(mode, 'Off')

        mode = self.e.update_external_control({}, {'Delay': True})
        self.assertEqual(self.e.event_start, first_event_start + equip_init_args['time_res'])
        self.assertEqual(mode, 'Off')

        mode = self.e.update_external_control({}, {'Delay': 2})
        self.assertEqual(self.e.event_start, first_event_start + 3 * equip_init_args['time_res'])
        self.assertEqual(mode, 'Off')

        # negative delay - start immediately
        mode = self.e.update_external_control({}, {'Delay': self.e.current_time - self.e.event_start})
        self.assertEqual(self.e.event_start, self.e.current_time)
        self.assertEqual(mode, 'On')

    def test_update_internal_control(self):
        first_event_start = self.e.event_start
        current_index = self.e.event_index

        mode = self.e.update_internal_control({})
        self.assertEqual(mode, 'Off')
        self.assertEqual(self.e.event_index, current_index)

        self.e.current_time = self.e.event_start
        mode = self.e.update_internal_control({})
        self.assertEqual(mode, 'On')
        self.assertEqual(self.e.event_index, current_index)

        self.e.current_time = self.e.event_end
        mode = self.e.update_internal_control({})
        self.assertEqual(mode, 'Off')
        self.assertEqual(self.e.event_index, current_index + 1)
        self.assertNotEqual(self.e.event_start, first_event_start)

    def test_calculate_power_and_heat(self):
        self.e.mode = 'On'
        self.e.calculate_power_and_heat({})
        self.assertEqual(self.e.electric_kw, init_args['max_power'])
        self.assertEqual(self.e.sensible_gain, 0)
        self.assertEqual(self.e.latent_gain, 0)


if __name__ == '__main__':
    unittest.main()
