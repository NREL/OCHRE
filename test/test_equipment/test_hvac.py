import unittest

from ochre.Equipment.HVAC import *
from ochre.Models import Envelope
from test.test_models.test_envelope import init_args as env_init_args
from test.test_equipment import equip_init_args

envelope = Envelope(**{**env_init_args, 'initial_temp_setpoint': 20})

init_args = equip_init_args.copy()
init_args.update({
    # TODO
    'heating capacity (W)': [5000],
    'cooling capacity (W)': [5000],
    'heating airflow rate (cfm)': [250],
    'cooling airflow rate (cfm)': [250],
    'heating EIR': [0.5],
    'cooling EIR': [0.5],
    'cooling SHR': 0.9,
    'heating duct dse': 0.5,
    'cooling duct dse': 0.5,
    'heating fan power (W/cfm)': 0.3,
    'cooling fan power (W/cfm)': 0.3,
    'supplemental heater cut in temp (C)': -17,
    'supplemental heating capacity (W)': 1000,
    'setpoint_ramp_rate': None,
    'envelope_model': envelope,
})

update_args_heat = {
    'to_envelope': {'H_LIV_latent': 0, **{name: 0 for name in envelope.input_names}},
    'heating_setpoint': 21,
    'cooling_setpoint': 23,
    'ambient_dry_bulb': 15,
    'ambient_humidity': 0.004,
    'ambient_pressure': 101,
    'ground_temperature': 10,
    'sky_temperature': 0,
    'occupants': 0,
    'wind_speed': 0,
    'ventilation_rate': 0,
    'solar_WD': 20, 'solar_GW1': 100,
}

update_args_cool = update_args_heat.copy()
update_args_cool.update({
    'heating_setpoint': 17,
    'cooling_setpoint': 19,
    'ambient_dry_bulb': 25,
    'solar_WD': 0, 'solar_GW1': 0,
})

# indoor temperature within deadband
update_args_inside = update_args_heat.copy()
update_args_inside.update({
    'heating_setpoint': 19.9,
    'cooling_setpoint': 20.1,
    'ambient_dry_bulb': 19,
    'solar_WD': 0, 'solar_GW1': 0,
})
init_args['initial_schedule'] = update_args_inside


class HVACTestCase(unittest.TestCase):
    """
    Test Case to test HVAC Equipment.
    """

    def setUp(self):
        self.hvac = Heater(**init_args)

    def test_init(self):
        self.assertTrue(self.hvac.is_heater)
        self.assertFalse(self.hvac.use_ideal_mode)
        self.assertAlmostEqual(self.hvac.fan_power_max, 75, places=1)

        self.assertIsNone(self.hvac.Ao_list)
        self.assertIsNotNone(self.hvac.thermal_model)
        self.assertEqual(self.hvac.zone, envelope.indoor_zone)

    def test_parse_control_signal(self):
        # test with load fraction
        mode = self.hvac.parse_control_signal(update_args_heat, {'Load Fraction': 0})
        self.assertEqual(mode, 'Off')

        with self.assertRaises(Exception):
            self.hvac.parse_control_signal(update_args_heat, {'Load Fraction': 0.5})

        # test with setpoint and deadband
        self.hvac.mode = 'Off'
        mode = self.hvac.parse_control_signal(update_args_inside, {'Setpoint': 18})
        self.assertEqual(mode, 'Off')
        self.assertEqual(self.hvac.temp_setpoint, 18)
        self.assertEqual(self.hvac.temp_deadband, 1)

        mode = self.hvac.parse_control_signal(update_args_inside, {'Deadband': 2})
        self.assertEqual(mode, None)
        self.assertEqual(self.hvac.temp_setpoint, 19.9)
        self.assertEqual(self.hvac.temp_deadband, 2)

        mode = self.hvac.parse_control_signal(update_args_inside, {'Setpoint': 22})
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.temp_setpoint, 22)
        self.assertEqual(self.hvac.temp_deadband, 2)

    def test_run_internal_control(self):
        mode = self.hvac.run_internal_control(update_args_heat)
        self.assertEqual(mode, 'On')

    def test_update_setpoint(self):
        self.hvac.update_setpoint(update_args_heat)
        self.assertEqual(self.hvac.temp_setpoint, 21)

        # test with ramp rate
        self.hvac.setpoint_ramp_rate = 0.5
        self.hvac.update_setpoint(update_args_cool)
        self.assertEqual(self.hvac.temp_setpoint, 20.5)

    def test_run_thermostat_control(self):
        mode = self.hvac.run_thermostat_control(update_args_cool)
        self.assertEqual(mode, 'Off')

        mode = self.hvac.run_thermostat_control(update_args_heat)
        self.assertEqual(mode, 'On')

        mode = self.hvac.run_thermostat_control(update_args_inside)
        self.assertEqual(mode, None)

    def test_update_capacity(self):
        capacity = self.hvac.update_capacity(update_args_heat)
        self.assertEqual(capacity, self.hvac.capacity_list[0])

    def test_update_eir(self):
        eir = self.hvac.update_eir(update_args_heat)
        self.assertEqual(eir, self.hvac.eir_list[0])

    def test_update_fan_power(self):
        power = self.hvac.update_fan_power()
        self.assertEqual(power, self.hvac.fan_power_list[0])

    def test_calculate_power_and_heat(self):
        self.hvac.mode = 'Off'
        self.hvac.calculate_power_and_heat(update_args_heat)
        self.assertDictEqual(self.hvac.sensible_gain, {envelope.indoor_zone.h_idx: 0})
        self.assertEqual(self.hvac.latent_gain, 0)
        self.assertEqual(self.hvac.capacity, 0)
        # self.assertEqual(self.hvac.eir, 0)
        self.assertEqual(self.hvac.electric_kw, 0)

        self.hvac.mode = 'On'
        self.hvac.calculate_power_and_heat(update_args_heat)
        self.assertEqual(len(self.hvac.sensible_gain), 1)
        self.assertAlmostEqual(self.hvac.sensible_gain[envelope.indoor_zone.h_idx], 2540, places=-1)
        self.assertEqual(self.hvac.latent_gain, 0)
        self.assertEqual(self.hvac.capacity, self.hvac.capacity_list[0])
        self.assertEqual(self.hvac.eir, self.hvac.eir_list[0])
        self.assertAlmostEqual(self.hvac.electric_kw, 2.6, places=1)  # includes fan power

        # test with basement fraction and duct losses
        new_args = {'basement airflow ratio': 0.5,
                    'duct location': 'unfinishedattic',
                    **init_args}
        tmp = Heater(**new_args)
        self.hvac.zone_fractions = tmp.zone_fractions
        self.hvac.mode = 'On'
        self.hvac.calculate_power_and_heat(update_args_heat)
        self.assertEqual(len(self.hvac.sensible_gain), 3)
        self.assertAlmostEqual(self.hvac.sensible_gain[envelope.indoor_zone.h_idx], 1270, places=-1)
        self.assertAlmostEqual(self.hvac.sensible_gain[envelope.zones['FND'].h_idx], 1270, places=-1)
        self.assertAlmostEqual(self.hvac.sensible_gain[envelope.zones['ATC'].h_idx], 2540, places=-1)
        self.assertEqual(self.hvac.latent_gain, 0)
        self.assertEqual(self.hvac.capacity, self.hvac.capacity_list[0])
        self.assertEqual(self.hvac.eir, self.hvac.eir_list[0])
        self.assertAlmostEqual(self.hvac.electric_kw, 2.6, places=1)  # includes fan power

    def test_generate_results(self):
        self.hvac.mode = 'On'
        self.hvac.electric_kw = 2.5
        results = self.hvac.generate_results(2)
        self.assertEqual(len(results), 0)

        results = self.hvac.generate_results(4)
        self.assertEqual(len(results), 1)

        results = self.hvac.generate_results(6)
        self.assertEqual(len(results), 9)
        self.assertAlmostEqual(results['HVAC Heating COP (-)'], 2.0)
        self.assertEqual(results['HVAC Heating Mode'], 'On')


class IdealHeaterTestCase(unittest.TestCase):

    def setUp(self):
        self.hvac = Heater(use_ideal_mode=True, **init_args)

    def test_run_internal_control(self):
        mode = self.hvac.run_internal_control(update_args_heat)
        self.assertEqual(mode, 'On')

        mode = self.hvac.run_internal_control(update_args_inside)
        self.assertEqual(mode, 'On')

        mode = self.hvac.run_internal_control(update_args_cool)
        self.assertEqual(mode, 'Off')

    def test_solve_ideal_capacity(self):
        self.hvac.temp_setpoint = 19
        capacity = self.hvac.solve_ideal_capacity()
        self.assertAlmostEqual(capacity, -6000, places=-2)

        self.hvac.temp_setpoint = 20
        capacity = self.hvac.solve_ideal_capacity()
        self.assertAlmostEqual(capacity, 700, places=-2)

        self.hvac.temp_setpoint = 21
        capacity = self.hvac.solve_ideal_capacity()
        self.assertAlmostEqual(capacity, 7400, places=-2)

    def test_update_capacity(self):
        # test with capacity max
        self.hvac.temp_setpoint = 21
        capacity = self.hvac.update_capacity(update_args_heat)
        self.assertEqual(capacity, 5000)

        # test with capacity min
        self.hvac.temp_setpoint = 20
        capacity = self.hvac.update_capacity(update_args_heat)
        self.assertAlmostEqual(capacity, 700, places=-2)

        self.hvac.capacity_min = 1000
        capacity = self.hvac.update_capacity(update_args_heat)
        self.assertEqual(capacity, 0)


class IdealCoolerTestCase(unittest.TestCase):

    def setUp(self):
        self.hvac = Cooler(use_ideal_mode=True, **init_args)

    def test_solve_ideal_capacity(self):
        self.hvac.temp_setpoint = 21
        capacity = self.hvac.solve_ideal_capacity()
        self.assertAlmostEqual(capacity, -8500, places=-2)

        self.hvac.temp_setpoint = 20
        capacity = self.hvac.solve_ideal_capacity()
        self.assertAlmostEqual(capacity, -800, places=-2)

        self.hvac.temp_setpoint = 19
        capacity = self.hvac.solve_ideal_capacity()
        self.assertAlmostEqual(capacity, 6900, places=-2)


class GasFurnaceTestCase(unittest.TestCase):
    def setUp(self):
        self.hvac = GasFurnace(**init_args)

    def test_update(self):
        # test gas consumption
        self.hvac.update(1, update_args_heat, {})
        self.assertEqual(self.hvac.mode, 'On')
        self.assertAlmostEqual(self.hvac.delivered_heat * self.hvac.duct_dse, 2540, places=-1)
        self.assertAlmostEqual(self.hvac.electric_kw, 0.075, places=3)
        self.assertAlmostEqual(self.hvac.gas_therms_per_hour, 0.085, places=3)

        self.hvac.update(1, update_args_cool, {})
        self.assertEqual(self.hvac.mode, 'Off')
        self.assertEqual(self.hvac.electric_kw, 0)
        self.assertAlmostEqual(self.hvac.gas_therms_per_hour, 0)


class GasBoilerTestCase(unittest.TestCase):
    def setUp(self):
        self.hvac = GasBoiler(**init_args)

    def test_init(self):
        self.assertEqual(self.hvac.condensing, True)

    def test_update(self):
        # test when on
        self.hvac.update(1, update_args_heat, {})
        self.assertAlmostEqual(self.hvac.eir, self.hvac.eir_list[0], places=1)
        self.assertAlmostEqual(self.hvac.eir, 0.51, places=2)


class DynamicHVACTestCase(unittest.TestCase):
    """
    Test Case to test single-speed, dynamic (non-ideal) HVAC Equipment.
    """

    def setUp(self):
        self.hvac = AirConditioner(**init_args)

    def test_init(self):
        self.assertAlmostEqual(self.hvac.Ao_list, 0.495, places=3)
        self.assertAlmostEqual(self.hvac.biquad_params[0]['eir_t'][0], -0.30428)
        self.assertEqual(self.hvac.use_ideal_mode, False)

    def test_calculate_biquadratic_param(self):
        self.hvac.mode = 'On'
        self.hvac.update_shr(update_args_cool)

        result = self.hvac.calculate_biquadratic_param(update_args_cool, 'cap', 0)
        self.assertAlmostEqual(result, 4930, places=-1)

        result = self.hvac.calculate_biquadratic_param(update_args_inside, 'cap', 0)
        self.assertAlmostEqual(result, 5100, places=-1)

        result = self.hvac.calculate_biquadratic_param(update_args_heat, 'cap', 0)
        self.assertAlmostEqual(result, 5120, places=-1)

        result = self.hvac.calculate_biquadratic_param(update_args_cool, 'eir', 0)
        self.assertAlmostEqual(result, 0.40, places=2)

    def test_update_shr(self):
        self.hvac.mode = 'On'
        self.hvac.speed_idx = 0
        shr = self.hvac.update_shr(update_args_cool)
        self.assertAlmostEqual(shr, 0.95, places=2)
        self.assertAlmostEqual(self.hvac.coil_input_db, 20.5, places=1)
        self.assertAlmostEqual(self.hvac.coil_input_wb, 13.7, places=1)

    def test_update(self):
        self.hvac.update(1, update_args_cool, {})
        self.assertEqual(self.hvac.mode, 'On')
        self.assertAlmostEqual(self.hvac.capacity, 4930, places=-1)
        self.assertAlmostEqual(self.hvac.delivered_heat * self.hvac.duct_dse, -2110, places=-1)
        self.assertAlmostEqual(self.hvac.sensible_gain[envelope.indoor_zone.h_idx], -2110, places=-1)
        self.assertAlmostEqual(self.hvac.latent_gain, -120, places=-1)
        self.assertAlmostEqual(self.hvac.electric_kw, 2.42, places=2)
        self.assertAlmostEqual(self.hvac.fan_power, 464, places=0)

    def test_generate_results(self):
        results = self.hvac.generate_results(4)
        self.assertEqual(len(results), 1)

        results = self.hvac.generate_results(6)
        self.assertEqual(len(results), 10)


class TwoSpeedHVACTestCase(unittest.TestCase):

    def setUp(self):
        args = init_args.copy()
        args.update({
            'cooling number of speeds': 2,
            **{key: [val[0] / x for x in range(2, 0, -1)] for key, val in init_args.items() if isinstance(val, list)}
        })
        self.hvac = AirConditioner(**args)

    def test_init(self):
        self.assertAlmostEqual(self.hvac.Ao_list, 0.495, places=3)
        self.assertEqual(self.hvac.n_speeds, 2)
        self.assertListEqual(self.hvac.capacity_list, [2500, 5000])
        self.assertEqual(self.hvac.min_time_in_speed[0], dt.timedelta(minutes=5))

    def test_parse_control_signal(self):
        # test disable speeds
        mode = self.hvac.parse_control_signal(update_args_cool, {'Disable Speed 1': 1})
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 1)
        self.assertListEqual(list(self.hvac.disable_speeds), [True, False])

        mode = self.hvac.parse_control_signal(update_args_cool, {'Disable Speed 1': 0, 'Disable Speed 2': 1})
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertListEqual(list(self.hvac.disable_speeds), [False, True])

    def test_run_two_speed_control(self):
        self.hvac.mode = 'Off'

        # Time control
        mode = self.hvac.run_two_speed_control(update_args_cool)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertEqual(self.hvac.time_in_speed, dt.timedelta(minutes=1))

        # Test min time in speed
        self.hvac.mode = 'On'
        mode = self.hvac.run_two_speed_control(update_args_cool)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertEqual(self.hvac.time_in_speed, dt.timedelta(minutes=2))

        # test with no temp change
        self.hvac.temp_indoor_prev = 20
        self.hvac.time_in_speed = dt.timedelta(minutes=5)
        mode = self.hvac.run_two_speed_control(update_args_cool)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertEqual(self.hvac.time_in_speed, dt.timedelta(minutes=6))

        # test with temp change
        self.hvac.temp_indoor_prev = 19
        mode = self.hvac.run_two_speed_control(update_args_cool)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 1)
        self.assertEqual(self.hvac.time_in_speed, dt.timedelta(minutes=1))

        # test disable speed
        self.hvac.disable_speeds = np.array([False, True])
        mode = self.hvac.run_two_speed_control(update_args_cool)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertEqual(self.hvac.time_in_speed, dt.timedelta(minutes=1))

        self.hvac.speed_idx = 1
        self.hvac.disable_speeds = np.array([False, False])
        mode = self.hvac.run_two_speed_control(update_args_heat)
        self.assertEqual(mode, 'Off')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertEqual(self.hvac.time_in_speed, dt.timedelta(0))

        # test setpoint based control
        self.hvac.control_type = 'Setpoint'

        self.hvac.mode = 'On'
        mode = self.hvac.run_two_speed_control(update_args_inside)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)

        self.hvac.time_in_speed = dt.timedelta(minutes=5)
        update_args = update_args_cool.copy()
        update_args['cooling_setpoint'] = 18.5
        mode = self.hvac.run_two_speed_control(update_args)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 1)

        self.hvac.time_in_speed = dt.timedelta(minutes=5)
        update_args['cooling_setpoint'] = 19.5
        mode = self.hvac.run_two_speed_control(update_args)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 1)

        update_args['cooling_setpoint'] = 20.1
        mode = self.hvac.run_two_speed_control(update_args)
        self.assertEqual(mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)

        # TODO: test Time2 controls

    def test_calculate_biquadratic_param(self):
        self.hvac.mode = 'On'
        self.hvac.update_shr(update_args_cool)

        param = self.hvac.calculate_biquadratic_param(update_args_cool, 'cap', 0)
        self.assertAlmostEqual(param, 2540, places=-1)

        param = self.hvac.calculate_biquadratic_param(update_args_cool, 'cap', 1)
        self.assertAlmostEqual(param, 4830, places=-1)

        param = self.hvac.calculate_biquadratic_param(update_args_cool, 'eir', 0)
        self.assertAlmostEqual(param, 0.19, places=2)

        param = self.hvac.calculate_biquadratic_param(update_args_cool, 'eir', 1)
        self.assertAlmostEqual(param, 0.41, places=2)

    def test_update(self):
        self.hvac.update(1, update_args_cool, {})
        self.assertEqual(self.hvac.mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 0)
        self.assertAlmostEqual(self.hvac.delivered_heat * self.hvac.duct_dse, -1150, places=-1)
        self.assertAlmostEqual(self.hvac.latent_gain, 0, places=-1)
        self.assertAlmostEqual(self.hvac.electric_kw, 0.72, places=2)

        # check that the mode stays low for 5 minutes
        for _ in range(4):
            self.hvac.update(1, update_args_cool, {})
            self.assertEqual(self.hvac.mode, 'On')
            self.assertEqual(self.hvac.speed_idx, 0)
            self.hvac.temp_indoor_prev = 19.9

        # check for update to high speed
        self.hvac.update(1, update_args_cool, {})
        self.assertEqual(self.hvac.mode, 'On')
        self.assertEqual(self.hvac.speed_idx, 1)


class VariableSpeedHVACTestCase(unittest.TestCase):

    def setUp(self):
        args = init_args.copy()
        args.update({
            'cooling number of speeds': 4,
            **{key: [val[0] / x for x in range(4, 0, -1)] for key, val in init_args.items() if isinstance(val, list)}
        })
        self.hvac = AirConditioner(**args)

    def test_init(self):
        self.assertAlmostEqual(self.hvac.Ao_list, 0.495, places=3)
        self.assertEqual(self.hvac.n_speeds, 4)
        self.assertEqual(self.hvac.use_ideal_mode, True)
        self.assertAlmostEqual(self.hvac.capacity_list[0], 5000 / 4, places=3)

    def test_update_capacity(self):
        # Capacity should follow ideal update
        self.hvac.temp_setpoint = 19.8
        capacity = self.hvac.update_capacity(update_args_inside)
        self.assertAlmostEqual(capacity, 760, places=-1)

        # test max capacity
        self.hvac.temp_setpoint = update_args_cool['cooling_setpoint']
        capacity = self.hvac.update_capacity(update_args_cool)
        self.assertAlmostEqual(self.hvac.capacity_max, 5660, places=-1)
        self.assertAlmostEqual(capacity, 5660, places=-1)

        # test with disabled speeds
        self.hvac.disable_speeds = np.array([False, True, True, True])
        capacity = self.hvac.update_capacity(update_args_cool)
        self.assertAlmostEqual(self.hvac.capacity_max, 1450, places=-1)
        self.assertAlmostEqual(capacity, 1450, places=-1)

        # test off
        self.hvac.temp_setpoint = update_args_heat['cooling_setpoint']
        capacity = self.hvac.update_capacity(update_args_heat)
        self.assertAlmostEqual(capacity, 0, places=-1)

    def test_update_eir(self):
        # EIR should follow dynamic update
        self.hvac.capacity = 1000
        eir = self.hvac.update_eir(update_args_cool)
        self.assertAlmostEqual(eir, 0.07, places=2)
        self.assertAlmostEqual(self.hvac.speed_idx, -0.31, places=2)

        self.hvac.capacity = 0
        eir = self.hvac.update_eir(update_args_cool)
        self.assertAlmostEqual(eir, 0.09, places=2)
        self.assertAlmostEqual(self.hvac.speed_idx, -1, places=2)

        self.hvac.capacity = 3000
        eir = self.hvac.update_eir(update_args_cool)
        self.assertAlmostEqual(eir, 0.18, places=2)
        self.assertAlmostEqual(self.hvac.speed_idx, 2.1, places=1)

        self.hvac.capacity = 6000
        eir = self.hvac.update_eir(update_args_cool)
        self.assertAlmostEqual(eir, 0.34, places=2)
        self.assertAlmostEqual(self.hvac.speed_idx, 3, places=2)

    def test_update(self):
        self.hvac.update(1, update_args_cool, {})
        self.assertEqual(self.hvac.mode, 'On')
        self.assertAlmostEqual(self.hvac.delivered_heat * self.hvac.duct_dse, -2390, places=-1)
        self.assertAlmostEqual(self.hvac.electric_kw, 1.91, places=2)

        self.hvac.update(1, update_args_heat, {})
        self.assertEqual(self.hvac.mode, 'Off')
        self.assertAlmostEqual(self.hvac.delivered_heat * self.hvac.duct_dse, 0, places=-1)
        self.assertAlmostEqual(self.hvac.electric_kw, 0, places=3)

    def test_generate_results(self):
        results = self.hvac.generate_results(6)
        self.assertEqual(len(results), 10)


class RoomACTestCase(unittest.TestCase):
    def setUp(self):
        self.hvac = RoomAC(**{'cooling conditioned space fraction': 0.5, **init_args})

    def test_update(self):
        # test space fraction
        self.hvac.update(1, update_args_cool, {})
        self.assertEqual(self.hvac.mode, 'On')
        self.assertAlmostEqual(self.hvac.capacity, 4450, places=-1)
        self.assertAlmostEqual(self.hvac.eir, 0.62, places=2)
        self.assertAlmostEqual(self.hvac.delivered_heat * self.hvac.duct_dse, -942, places=-1)
        self.assertAlmostEqual(self.hvac.sensible_gain[envelope.indoor_zone.h_idx], -1880, places=-1)
        self.assertAlmostEqual(self.hvac.electric_kw, 1.6, places=1)


class HeatPumpHeaterTestCase(unittest.TestCase):
    def setUp(self):
        self.hvac = HeatPumpHeater(**init_args)

    def test_update_capacity(self):
        self.hvac.mode = 'On'
        self.hvac.update_capacity(update_args_heat)
        self.assertFalse(self.hvac.defrost)

        # test defrost
        update_args = update_args_heat.copy()
        update_args['ambient_dry_bulb'] = -5
        update_args['ambient_humidity'] = 0.001
        self.hvac.update_capacity(update_args)
        self.assertTrue(self.hvac.defrost)
        self.assertAlmostEqual(self.hvac.power_defrost, 0.032, places=3)
        self.assertAlmostEqual(self.hvac.defrost_power_mult, 1.09, places=2)

    def test_update_eir(self):
        # test defrost
        self.hvac.mode = 'On'
        update_args = update_args_heat.copy()
        update_args['ambient_dry_bulb'] = -5
        update_args['ambient_humidity'] = 0.001
        self.hvac.capacity = self.hvac.update_capacity(update_args)
        eir = self.hvac.update_eir(update_args)
        self.assertTrue(self.hvac.defrost)
        self.assertAlmostEqual(eir, 0.66, places=2)


class ASHPHeaterTestCase(unittest.TestCase):
    def setUp(self):
        self.hvac = ASHPHeater(**init_args)

    def test_run_internal_control(self):
        self.hvac.mode = 'Off'
        mode = self.hvac.run_internal_control(update_args_heat)
        self.assertEqual(mode, 'HP On')

        self.hvac.mode = 'HP and ER On'
        mode = self.hvac.run_internal_control(update_args_heat)
        self.assertEqual(mode, 'HP and ER On')

        self.hvac.mode = 'HP and ER On'
        mode = self.hvac.run_internal_control(update_args_inside)
        self.assertEqual(mode, 'HP On')

        self.hvac.mode = 'HP On'
        mode = self.hvac.run_internal_control(update_args_inside)
        self.assertEqual(mode, 'HP On')

        self.hvac.mode = 'HP and ER On'
        mode = self.hvac.run_internal_control(update_args_cool)
        self.assertEqual(mode, 'Off')

        # test with cold indoor temperature
        update_args = update_args_heat.copy()
        self.hvac.zone.temperature = 16
        mode = self.hvac.run_internal_control(update_args)
        self.assertEqual(mode, 'HP and ER On')

        # test with cold outdoor temperature
        self.hvac.zone.temperature = 18
        update_args['ambient_dry_bulb'] = -20
        mode = self.hvac.run_internal_control(update_args)
        self.assertEqual(mode, 'ER On')

        # reset zone temperature
        self.hvac.zone.temperature = 20

    def test_run_er_thermostat_control(self):
        mode = self.hvac.run_er_thermostat_control(update_args_heat)
        self.assertEqual(mode, 'Off')

        update_args = update_args_heat.copy()
        self.hvac.zone.temperature = 19
        mode = self.hvac.run_er_thermostat_control(update_args)
        self.assertEqual(mode, None)

        # test with cold indoor temperature
        self.hvac.zone.temperature = 16
        mode = self.hvac.run_er_thermostat_control(update_args)
        self.assertEqual(mode, 'On')

        # reset zone temperature
        self.hvac.zone.temperature = 20

    def test_update_capacity(self):
        self.hvac.mode = 'HP On'
        hp_cap = self.hvac.update_capacity(update_args_heat)
        self.assertAlmostEqual(hp_cap, 5870, places=-1)

        self.hvac.mode = 'ER On'
        er_cap = self.hvac.update_capacity(update_args_heat)
        self.assertEqual(er_cap, self.hvac.er_capacity_rated)

        self.hvac.mode = 'HP and ER On'
        tot_cap = self.hvac.update_capacity(update_args_heat)
        self.assertEqual(tot_cap, hp_cap + er_cap)

    def test_update_eir(self):
        self.hvac.mode = 'HP On'
        hp_eir = self.hvac.update_eir(update_args_heat)
        self.assertAlmostEqual(hp_eir, 0.5, places=1)

        self.hvac.mode = 'ER On'
        er_eir = self.hvac.update_eir(update_args_heat)
        self.assertEqual(er_eir, self.hvac.er_eir_rated)

        self.hvac.mode = 'HP and ER On'
        self.hvac.capacity = self.hvac.update_capacity(update_args_heat)
        tot_eir = self.hvac.update_eir(update_args_heat)
        self.assertAlmostEqual(tot_eir, 0.5, places=1)
        self.assertTrue(hp_eir < tot_eir < er_eir)


class VariableSpeedASHPHeaterTestCase(unittest.TestCase):
    def setUp(self):
        self.hvac = ASHPHeater(use_ideal_mode=True, **init_args)

    def test_run_internal_control(self):
        mode = self.hvac.run_internal_control(update_args_heat)
        self.assertEqual(mode, 'HP and ER On')

        mode = self.hvac.run_internal_control(update_args_inside)
        self.assertEqual(mode, 'HP On')

        mode = self.hvac.run_internal_control(update_args_cool)
        self.assertEqual(mode, 'Off')

        # test with cold indoor temperature
        update_args = update_args_heat.copy()
        update_args['heating_setpoint'] = 22
        mode = self.hvac.run_internal_control(update_args)
        self.assertEqual(mode, 'HP and ER On')

        # test with cold outdoor temperature
        update_args['ambient_dry_bulb'] = -20
        mode = self.hvac.run_internal_control(update_args)
        self.assertEqual(mode, 'ER On')

    def test_update_capacity(self):
        self.hvac.mode = 'HP On'
        self.hvac.temp_setpoint = 20.5
        hp_cap = self.hvac.update_capacity(update_args_heat)
        self.assertAlmostEqual(hp_cap, 4010, places=-1)

        self.hvac.mode = 'ER On'
        er_cap = self.hvac.update_capacity(update_args_heat)
        self.assertAlmostEqual(er_cap, 1000, places=-1)
        self.assertAlmostEqual(self.hvac.er_capacity, 1000, places=-1)

        self.hvac.mode = 'HP and ER On'
        self.hvac.temp_setpoint = 21
        tot_cap = self.hvac.update_capacity(update_args_heat)
        self.assertAlmostEqual(tot_cap, 6863, places=-1)
        self.assertAlmostEqual(self.hvac.er_capacity, 1000, places=-1)


if __name__ == '__main__':
    unittest.main()
