import unittest
import os
import pandas as pd
import datetime as dt

from ochre import FileIO
from ochre.utils import default_input_path, SCHEDULE_COLUMNS

resample_args1 = {'start_time': dt.datetime(2019, 1, 1),
                  'duration': dt.timedelta(days=31),
                  'time_res': dt.timedelta(minutes=5),
                  'initialization_time': dt.timedelta(days=1),
                  }
resample_args2 = {'start_time': dt.datetime(2019, 12, 29),
                  'duration': dt.timedelta(days=3),
                  'time_res': dt.timedelta(minutes=1),
                  }


class InputFileTestCase(unittest.TestCase):
    """
    Test Case to test all properties file functions in FileIO.py
    """

    def test_get_rc_params(self):
        # Correctly parses envelope parameters
        fake_properties = {'Nothing': None,
                           'R_EW_ext': 1,
                           'R_EW1': 1,
                           'R_EW2': 0.5,
                           'R_EW3': 1,
                           'C_EW1': 1,
                           'C_EW2': 0.5,
                           'C_LIV': 100,
                           }
        check = {'R_EW_ext': 1,
                 'R_EW1': 1,
                 'R_EW2': 0.5,
                 'R_EW3': 1,
                 'C_EW1': 1000,
                 'C_EW2': 500.0,
                 'C_LIV': 100000,
                 }

        rc_params = FileIO.get_boundary_rc_values(**fake_properties)
        self.assertDictEqual(rc_params, check)

    def test_convert_hpxml_element(self):
        # FileIO.convert_hpxml_element()
        pass

    def test_import_properties_from_hpxml(self):
        # Load properties file, note: includes a garage
        hpxml_file = os.path.join(default_input_path, 'Input Files', 'L120AC.xml')
        properties = FileIO.import_hpxml(hpxml_file)
        self.assertEqual(properties['country'], 'USA')

    def test_import_properties(self):
        # Load properties file, note: includes a garage
        hpxml_file = os.path.join(default_input_path, 'Input Files', 'sample_beopt_house.properties')
        properties = FileIO.import_properties(hpxml_file)
        self.assertEqual(properties['country'], 'USA')

    def test_import_zip_model(self):
        result = FileIO.import_zip_model()
        self.assertEqual(result['Refrigerator']['pf'], 0.8)


class TimeSeriesFileTestCase(unittest.TestCase):
    """
    Test Case to test all time-series file functions in FileIO.py
    """

    def test_import_generic(self):
        # loads water draw profile
        water_file = os.path.join(default_input_path, 'Water Draw Profiles', 'DHW_2bed_unit0_1min.csv')
        result = FileIO.import_time_series(water_file, fillna=0, **resample_args1)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result['Showers'].iloc[0], 0)
        self.assertAlmostEqual(result['Sinks'].iloc[114], 0.1755248)

    def test_import_schedule(self):
        # test with ResStock schedule file, and with weekday/weekend HVAC setpoints
        properties.update({
            'weekday heating setpoint': [68] * 12 + [70] * 12,
            'weekend heating setpoint': [69] * 12 + [71] * 12,
            'weekday cooling setpoint': [76] * 12 + [78] * 12,
            'weekend cooling setpoint': [75] * 12 + [77] * 12,
        })
        schedule_input_file = os.path.join(default_input_path, 'Input Files', 'occupant_schedule_test.csv')
        result = FileIO.import_occupancy_schedule(schedule_input_file, **properties)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertTupleEqual(result.shape, (31 * 24 * 12, 21))
        self.assertTrue(all([col in SCHEDULE_COLUMNS for col in result.columns]))
        self.assertAlmostEqual(result['HVAC Heating Setpoint (C)'][0], 20, places=1)

    def test_import_weather(self):
        sam_properties = {
            'latitude': 20,
            'longitude': -100,
            'timezone': -7,
            'house_name': 'SAM_test'
        }
        sam_file = FileIO.default_sam_weather_file.format('SAM_test')
        args = resample_args1.copy()
        args.update(sam_properties)
        if os.path.exists(sam_file):
            os.remove(sam_file)

        # test epw
        epw_file = os.path.join(default_input_path, 'Weather', 'CA_RIVERSIDE_MUNI_722869_12.epw')
        result = FileIO.import_weather(epw_file, create_sam_file=True, **args)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.loc[dt.datetime(2019, 1, 1, 0, 10), 'GHI (W/m^2)'], 0)
        noon = dt.datetime(2019, 1, 1, 12, 30)
        self.assertAlmostEqual(result.loc[noon, 'GHI (W/m^2)'], 640, places=0)
        self.assertAlmostEqual(result.loc[noon, 'Ambient Dry Bulb (C)'], 29.5, places=2)
        self.assertAlmostEqual(result.loc[noon, 'Ambient Relative Humidity (-)'], 0.06, places=2)
        self.assertAlmostEqual(result.loc[noon, 'Ambient Pressure (kPa)'], 99.0, places=2)

        # check that SAM file exists
        self.assertTrue(os.path.exists(sam_file))

        # test csv
        nsrdb_file = os.path.join(default_input_path, 'Weather', 'FortCollins_NSRDB.csv')
        result = FileIO.import_weather(nsrdb_file, **resample_args1)
        self.assertIsInstance(result.index, pd.DatetimeIndex)
        self.assertEqual(result.loc[dt.datetime(2019, 1, 1, 0, 10), 'GHI (W/m^2)'], 0)
        noon = dt.datetime(2019, 1, 1, 12, 15)
        self.assertAlmostEqual(result.loc[noon, 'GHI (W/m^2)'], 344, places=0)
        self.assertAlmostEqual(result.loc[noon, 'Ambient Dry Bulb (C)'], 1.85, places=2)
        self.assertAlmostEqual(result.loc[noon, 'Ambient Relative Humidity (-)'], 0.368, places=2)
        self.assertAlmostEqual(result.loc[noon, 'Ambient Pressure (kPa)'], 83.25, places=2)

    def test_calculate_plane_irradiance(self):
        data = {
            'azimuth': 0,  # Noon, due south
            'zenith': 20,  # in degrees
            'apparent_zenith': 20,  # in degrees
            'dni_extraterrestrial': 1400,  # in W/m^2
            'airmass': 1,
            'GHI (W/m^2)': 900,
            'DNI (W/m^2)': 1000,
            'DHI (W/m^2)': 300,
        }
        data = pd.DataFrame(data, index=[0])

        result = FileIO.calculate_plane_irradiance(data, tilt=0, panel_azimuth=0, separate=True)
        self.assertAlmostEqual(result['poa_global'][0], 1240, places=0)
        self.assertAlmostEqual(result['poa_direct'][0], 940, places=0)
        self.assertAlmostEqual(result['poa_diffuse'][0], 300, places=0)

        result = FileIO.calculate_plane_irradiance(data, tilt=20, panel_azimuth=0)
        self.assertAlmostEqual(result[0], 1330, places=-1)

        result = FileIO.calculate_plane_irradiance(data, tilt=20, panel_azimuth=30)
        self.assertAlmostEqual(result[0], 1311, places=-1)

        # test SHGC and separate columns
        result = FileIO.calculate_plane_irradiance(data, tilt=20, panel_azimuth=0, window_shgc=0.5, window_u=2,
                                                   separate=True)
        self.assertListEqual(result.columns.to_list(),
                             ['poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse', 'poa_ground_diffuse'])
        self.assertAlmostEqual(result['poa_global'][0], 1290, places=0)
        self.assertAlmostEqual(result['poa_direct'][0], 1008, places=0)
        self.assertAlmostEqual(result['poa_diffuse'][0], 282, places=0)

    def test_resample(self):
        times_1 = pd.date_range(resample_args1['start_time'], resample_args1['start_time'] + resample_args1['duration'],
                                freq=resample_args1['time_res'], inclusive='left')
        times_2 = pd.date_range(resample_args2['start_time'], resample_args2['start_time'] + resample_args2['duration'],
                                freq=resample_args2['time_res'], inclusive='left')

        # test clipping times, upsampling
        times_december = pd.date_range(dt.datetime(2019, 12, 1),
                                       resample_args2['start_time'] + resample_args2['duration'],
                                       freq=dt.timedelta(minutes=2), inclusive='left')
        n = 28 * 24 * 60 / 2
        data_december = pd.DataFrame({'A': range(len(times_december))}, index=times_december)
        result = FileIO.set_annual_index(data_december, annual_input=False, **resample_args2)
        self.assertTrue(result.index.equals(times_2))
        self.assertListEqual(result['A'][:4].values.tolist(), [n, n, n + 1, n + 1])

        # test with interpolation
        result = FileIO.set_annual_index(data_december, annual_input=False, interpolate=True, **resample_args2)
        self.assertTrue(result.index.equals(times_2))
        self.assertListEqual(result['A'][:4].values.tolist(), [n, n + 0.5, n + 1, n + 1.5])

        # test with full year input, downsampling
        times_annual = pd.date_range(dt.datetime(2019, 1, 1), dt.datetime(2020, 1, 1),
                                     freq=dt.timedelta(minutes=1), inclusive='left')
        data = pd.DataFrame({'A': range(len(times_annual))}, index=times_annual)
        result = FileIO.set_annual_index(data, **resample_args1)
        self.assertTrue(result.index.equals(times_1))
        self.assertEqual(result['A'][-1], 31 * 24 * 60 - 3)

        # test with offset
        offset = dt.timedelta(minutes=2)
        result = FileIO.set_annual_index(data, offset=offset, **resample_args1)
        self.assertTrue(result.index.equals(times_1))
        self.assertListEqual(result['A'][1:4].values.tolist(), [5, 10, 15])

        # test at end of year
        result = FileIO.set_annual_index(data, **resample_args2)
        self.assertTrue(result.index.equals(times_2))
        self.assertEqual(result['A'][-1], 365 * 24 * 60 - 1)

        # test with annual_output=True
        times_5min = pd.date_range(dt.datetime(2019, 1, 1), dt.datetime(2020, 1, 1),
                                   freq=dt.timedelta(minutes=5), inclusive='left')
        result = FileIO.set_annual_index(data, annual_output=True, **resample_args1)
        self.assertTrue(result.index.equals(times_5min))
        self.assertEqual(result['A'][-1], 365 * 24 * 60 - 3)

        # test repeat_years
        resample_args = resample_args1.copy()
        resample_args['duration'] += dt.timedelta(days=365)
        result = FileIO.set_annual_index(data, repeat_years=True, **resample_args)
        self.assertEqual(len(result), 396 * 24 * 12)
        self.assertEqual(result['A'].iloc[2], 12)
        self.assertEqual(result['A'].iloc[365 * 24 * 12 - 1], 525597)
        self.assertEqual(result['A'].iloc[365 * 24 * 12], 2)
        self.assertEqual(result['A'][-1], 31 * 24 * 60 - 3)

        # test preserve_sum, upsampling and downsampling
        result = FileIO.set_annual_index(data_december, annual_input=False, preserve_sum=True, **resample_args2)
        self.assertListEqual(result['A'][:4].values.tolist(), [n / 2, n / 2, (n + 1) / 2, (n + 1) / 2])

        result = FileIO.set_annual_index(data, preserve_sum=True, **resample_args1)
        self.assertListEqual(result['A'][:4].values.tolist(), [10, 35, 60, 85])

    def test_import_all_time_series(self):
        # Load properties file, note: includes a garage
        hpxml_file = os.path.join(default_input_path, 'Input Files', 'sample_beopt_house.properties')
        properties = FileIO.import_properties_from_beopt(hpxml_file)

        # Testing absolute and relative paths
        schedule_input_file = 'test_case_schedule.properties'
        # weather_file = os.path.join(default_input_path, 'Weather', 'CA_RIVERSIDE_MUNI_722869_12.epw')
        weather_file = os.path.join(default_input_path, 'Weather', 'CO_FORT-COLLINS-LOVELAND-AP_724769S_18.epw')
        water_draw_file = 'DHW_2bed_unit0_1min.csv'

        # Test with water draw file
        args = resample_args2.copy()
        args.update(properties)
        result = FileIO.import_schedule(None, schedule_input_file, water_draw_file, weather_file=weather_file, **args)
        self.assertEqual(len(result), 3 * 1440)
        self.assertEqual(result.index[1], dt.datetime(2019, 12, 29, 0, 1))
        cols_to_check = ['ambient_pressure', 'solar_RF', 'GHI (W/m^2)', 'Horizontal Irradiance - poa_global (W/m^2)',
                         'ground_temperature', 'plug_loads']
        for col in cols_to_check:
            self.assertIn(col, result.columns)

        # Test with initialization time
        args = resample_args1.copy()
        args.update(properties)
        result = FileIO.import_schedule(None, schedule_input_file, weather_file=weather_file, **args)
        self.assertEqual(len(result), 31 * 1440 // 5)
        self.assertEqual(result.index[1], dt.datetime(2019, 1, 1, 0, 5))


if __name__ == '__main__':
    unittest.main()
