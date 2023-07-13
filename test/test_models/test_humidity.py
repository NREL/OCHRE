import unittest
import datetime as dt

from ochre.Models import HumidityModel
from test.test_models.test_envelope import update_args1

humidity_init_args = {
    'time_res': dt.timedelta(minutes=1),
    't_indoor': 20,
    'building volume (m^3)': 100,
    'initial_schedule': update_args1.copy(),
}

humidity_update_args = {
    't_indoor': 20,
    'schedule': {
        # 'ambient_dry_bulb': 30,
        # 'ambient_humidity': 0.005,
        'ambient_pressure': 101,
    }
}


class HumidityTestCase(unittest.TestCase):
    """
    Test Case to test the Humidity Model class.
    """

    def setUp(self):
        self.humidity = HumidityModel(**humidity_init_args)

    def test_initialize(self):
        self.assertEqual(self.humidity.volume, 100)
        self.assertAlmostEqual(self.humidity.rh, 0.48, places=2)

        self.assertAlmostEqual(self.humidity.density, 1.20, places=2)
        self.assertAlmostEqual(self.humidity.w, 0.007, places=4)
        self.assertAlmostEqual(self.humidity.wet_bulb, 13.5, places=1)

    def test_update_humidity(self):
        self.humidity.latent_gains = -10000

        self.humidity.update_humidity(**humidity_update_args)
        self.assertAlmostEqual(self.humidity.rh, 0.47, places=2)
        self.assertAlmostEqual(self.humidity.w, 0.0069, places=4)
        self.assertAlmostEqual(self.humidity.density, 1.20, places=2)
        self.assertAlmostEqual(self.humidity.wet_bulb, 13.4, places=1)


if __name__ == '__main__':
    unittest.main()
