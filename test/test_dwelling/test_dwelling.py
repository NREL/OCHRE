import unittest
import os
import datetime as dt
import time

from ochre import Dwelling
from ochre.Dwelling import add_equipment_from_properties
from test import test_output_path

dwelling_args = {
    'name': 'test_dwelling',

    # Timing parameters
    'start_time': dt.datetime(2019, 5, 5, 12, 0, 0),  # May 5, 12:00PM
    'time_res': dt.timedelta(minutes=15),
    'duration': dt.timedelta(days=1),
    'ext_time_res': dt.timedelta(hours=1),

    # Input and Output Files
    'output_path': test_output_path,
    'hpxml_file': 'sample_beopt_house.properties',
    # 'schedule_input_file': 'test_case_schedule.properties',
    'schedule_input_file': 'occupant_schedule_test.csv',
    'water_draw_file': 'DHW_2bed_unit0_1min.csv',
    'weather_file': 'CA_RIVERSIDE_MUNI_722869_12.epw',
    'verbosity': 9,  # verbosity of results file (0-9); 8: include envelope; 9: include water heater
    'metrics_verbosity': 9,  # verbosity of results file (0-9)
}

test_equipment = {'Air Source Heat Pump': {'use_ideal_mode': True},
                  }


class DwellingTestCase(unittest.TestCase):
    """
    Test Case to test the Dwelling class.
    """

    def setUp(self):
        self.dwelling = Dwelling(**dwelling_args)

    def tearDown(self):
        out_file = os.path.join(test_output_path, 'test dwelling.csv')
        if os.path.exists(out_file):
            os.remove(out_file)

    def test_init(self):
        self.assertEqual(self.dwelling.name, 'test dwelling')
        self.assertEqual(self.dwelling.current_time, dt.datetime(2019, 5, 5, 12, 0))
        self.assertTrue(os.path.exists(test_output_path))
        self.assertEqual(self.dwelling.equipment, [])

        self.assertEqual(len(self.dwelling.schedule), 96)
        self.assertTrue(self.dwelling.schedule.notna().all().all())

    def test_add_equipment_from_properties(self):
        properties = FileIO.import_properties_from_beopt(**dwelling_args)
        house_args = {**properties, **dwelling_args}

        # test with no original equipment
        equipment = add_equipment_from_properties({}, **house_args)
        self.assertGreater(len(equipment), 0)
        self.assertIn('ASHP Heater', equipment)
        self.assertIn('ASHP Cooler', equipment)
        self.assertIn('Electric Resistance Water Heater', equipment)
        self.assertIn('Lighting', equipment)
        self.assertDictEqual(equipment['Lighting'], {})

        # test with HVAC and water heater included
        equipment = add_equipment_from_properties({'Gas Furnace': {}, 'Gas Tankless Water Heater': {}}, **house_args)
        self.assertGreater(len(equipment), 0)
        self.assertIn('Gas Furnace', equipment)
        self.assertIn('ASHP Cooler', equipment)
        self.assertIn('Gas Tankless Water Heater', equipment)

    def test_update(self):
        start = self.dwelling.current_time

        results = self.dwelling.update()

        self.assertEqual(self.dwelling.current_time, start + dwelling_args['time_res'])
        self.assertEqual(self.dwelling.total_p_kw, 0)
        self.assertEqual(len(self.dwelling.results), 1)

        self.assertEqual(results['Total Electric Power (kW)'], 0)
        self.assertEqual(results['Total Reactive Power (kVAR)'], 0)

    def test_compile_results(self):
        result = self.dwelling.generate_results()

        self.assertEqual(len(result), 90)
        self.assertEqual(result['Voltage (-)'], 1)
        self.assertEqual(result['Total Electric Power (kW)'], 0)
        self.assertEqual(result['Total Electric Energy (kWh)'], 0)

    def test_export_results(self):
        self.dwelling.results = [{'A': 1}]
        self.dwelling.export_results()

        with open(self.dwelling.results_file, 'r') as f:
            data = f.read()
        self.assertEqual(data, 'A\n1\n')

        self.dwelling.results = [{'A': 3}]
        self.dwelling.export_results()

        with open(self.dwelling.results_file, 'r') as f:
            data = f.read()
        self.assertEqual(data, 'A\n1\n3\n')

    def test_initialize(self):
        save = self.dwelling.save_results

        self.dwelling.initialize(dt.timedelta(days=1))
        self.assertEqual(self.dwelling.current_time, dt.datetime(2019, 5, 5, 12, 0))
        self.assertEqual(self.dwelling.save_results, save)

    def test_simulate(self):
        self.assertFalse(os.path.exists(self.dwelling.results_file))
        df, metrics, hourly = self.dwelling.simulate()

        self.assertEqual(self.dwelling.current_time, self.dwelling.start_time + dwelling_args['duration'])
        self.assertGreater(os.stat(self.dwelling.results_file).st_size, 0)

        # check time series outputs
        self.assertEqual(len(df), 96)
        self.assertTrue((df['Total Electric Power (kW)'] == 0).all())
        self.assertAlmostEqual(df['Temperature - Indoor (C)'].mean(), 23.0, places=0)
        self.assertAlmostEqual(df['Temperature - Indoor (C)'].std(), 1.5, places=0)

        # check hourly outputs
        self.assertEqual(len(hourly), 24)
        self.assertTrue((hourly['Total Electric Power (kW)'] == 0).all())
        self.assertAlmostEqual(hourly['Temperature - Indoor (C)'].mean(), 23.0, places=0)
        self.assertAlmostEqual(hourly['Temperature - Indoor (C)'].std(), 1.5, places=0)

        # check output metrics
        self.assertEqual(len(metrics), 17)
        self.assertEqual(metrics['Total Electric Energy (kWh)'], 0)
        self.assertAlmostEqual(metrics['Average Temperature - Indoor (C)'], 23.0, places=0)


class DwellingWithEquipmentTestCase(unittest.TestCase):
    """
    Test Case to test the Dwelling class with some equipment
    """

    def setUp(self):
        self.dwelling = Dwelling(initialization_time=dt.timedelta(hours=1), **dwelling_args.copy())

    def test_init(self):
        self.assertEqual(len(self.dwelling.equipment), 11)
        self.assertEqual(len(self.dwelling.equipment_by_end_use['HVAC Heating']), 1)
        self.assertEqual(len(self.dwelling.equipment_by_end_use['HVAC Cooling']), 1)
        self.assertEqual(len(self.dwelling.equipment_by_end_use['Water Heating']), 1)
        self.assertEqual(len(self.dwelling.equipment_by_end_use['PV']), 0)
        self.assertEqual(len(self.dwelling.equipment_by_end_use['Battery']), 0)

        # check that ideal equipment is last
        self.assertEqual(self.dwelling.equipment[-1].name, 'ASHP Cooler')

    def test_update(self):
        results = self.dwelling.update()
        self.assertAlmostEqual(self.dwelling.results[-1]['Total Electric Power (kW)'], 1.85, places=1)
        self.assertAlmostEqual(results['Total Electric Power (kW)'], 1.85, places=1)
        self.assertAlmostEqual(results['Total Reactive Power (kVAR)'], 0.3, places=1)
        self.assertAlmostEqual(results['Lighting Electric Power (kW)'], 0.1, places=1)
        self.assertAlmostEqual(results['Temperature - Indoor (C)'], 22.2, places=1)
        self.assertEqual(results['HVAC Heating Mode'], 'Off')
        self.assertEqual(results['HVAC Cooling Mode'], 'On')
        self.assertEqual(results['Water Heating Mode'], 'Upper On')
        for e in self.dwelling.equipment:
            self.assertEquals(e.current_time, self.dwelling.current_time)

        # test with outage
        results = self.dwelling.update(voltage=0)
        self.assertAlmostEqual(results['Total Electric Power (kW)'], 0, places=2)
        self.assertAlmostEqual(results['Total Reactive Power (kVAR)'], 0, places=2)
        self.assertAlmostEqual(results['Temperature - Indoor (C)'], 22.2, places=1)
        self.assertAlmostEqual(self.dwelling.envelope.indoor_zone.temperature, 23.1, places=1)
        self.assertEqual(results['HVAC Cooling Mode'], 'Off')
        for e in self.dwelling.equipment:
            self.assertEquals(e.current_time, self.dwelling.current_time)

    def test_update_external(self):
        control = {'HVAC Heating': {'Duty Cycle': 0.3}, 'Load Fractions': {'Lighting': 0, 'Exterior Lighting': 0}}
        results = self.dwelling.update(control_signal=control)
        self.assertEqual(results['HVAC Heating Mode'], 'HP On')
        self.assertAlmostEqual(results['Total Electric Power (kW)'], 12, places=0)
        self.assertAlmostEqual(results['Total Reactive Power (kVAR)'], 0.3, places=1)
        self.assertEqual(results['Lighting Electric Power (kW)'], 0)

    def test_simulate(self):
        t0 = time.time()
        df, metrics, hourly = self.dwelling.simulate()
        t_sim = time.time() - t0

        # check speed of simulation
        self.assertLess(t_sim, 0.5)

        # check time series outputs
        self.assertEqual(len(df), 96)
        # self.assertEqual(len(df.columns), 159)
        self.assertTrue((df['Total Electric Power (kW)'] > 0).all())
        self.assertAlmostEqual(df['Water Heating Delivered (W)'].mean(), 0.27, places=1)

        # check time series outputs
        self.assertEqual(len(hourly), 24)
        self.assertTrue((hourly['Total Electric Power (kW)'] > 0).all())

        # check output metrics
        # self.assertEqual(len(metrics), 70)
        self.assertAlmostEqual(metrics['Total Electric Energy (kWh)'], 21, places=0)
        self.assertAlmostEqual(metrics['HVAC Cooling Electric Energy (kWh)'], 4, places=0)
        self.assertAlmostEqual(metrics['Water Heating Electric Energy (kWh)'], 6.5, places=0)
        self.assertAlmostEqual(metrics['Total HVAC Cooling Delivered (kWh)'], 16.6, places=0)
        self.assertAlmostEqual(metrics['Unmet Cooling Load (C-hours)'], 0, places=1)
        # self.assertAlmostEqual(metrics['HVAC Cooling Cycles'], 1, places=0)
        self.assertAlmostEqual(metrics['Total Hot Water Delivered (gal/day)'], 42, places=0)
        self.assertAlmostEqual(metrics['Total Hot Water Unmet Demand (kWh)'], 0, places=0)
        # self.assertAlmostEqual(metrics['HVAC Cooling Cycles'], 3, places=-1)


if __name__ == '__main__':
    unittest.main()
