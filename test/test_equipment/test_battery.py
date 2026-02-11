import unittest
import datetime as dt
import pandas as pd

from ochre.Equipment import Battery
from ochre.Equipment.Battery import BatteryThermalModel
from test.test_equipment import equip_init_args


# Basic init args for Battery
init_args = equip_init_args.copy()
init_args.update(
    {
        "soc_init": 0.5,
        "capacity": 5,  # kW
        "capacity_kwh": 10,  # kWh
    }
)


class BatteryTestCase(unittest.TestCase):
    """
    Test Case to test Battery Equipment without thermal model.
    """

    def setUp(self):
        self.battery = Battery(**init_args)

    def test_init(self):
        """Test basic battery initialization"""
        self.assertAlmostEqual(self.battery.capacity, 5)
        self.assertAlmostEqual(self.battery.capacity_kwh, 10)
        self.assertAlmostEqual(self.battery.soc, 0.5)
        self.assertEqual(self.battery.mode, "Off")
        # Default: no thermal model, yes degradation
        self.assertIsNone(self.battery.thermal_model)
        self.assertIsNotNone(self.battery.degradation_states)
        # Check efficiency type defaults to advanced
        self.assertEqual(self.battery.efficiency_type, "advanced")
        # Check internal resistance is calculated for advanced mode
        self.assertIsNotNone(self.battery.r_internal)
        self.assertIsNotNone(self.battery.n_series)

    def test_init_constant_efficiency(self):
        """Test battery with constant efficiency mode"""
        args = init_args.copy()
        args["efficiency_type"] = "constant"
        battery = Battery(**args)
        self.assertEqual(battery.efficiency_type, "constant")
        self.assertIsNone(battery.r_internal)
        self.assertIsNone(battery.n_series)

    def test_update_external_control_p_setpoint(self):
        """Test P Setpoint control signal"""
        mode = self.battery.update_external_control({"P Setpoint": 2})
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.battery.power_setpoint, 2)

        mode = self.battery.update_external_control({"P Setpoint": -2})
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.battery.power_setpoint, -2)

        mode = self.battery.update_external_control({"P Setpoint": 0})
        self.assertEqual(mode, "Off")
        self.assertAlmostEqual(self.battery.power_setpoint, 0)

    def test_update_external_control_soc_target(self):
        """Test SOC target control signal"""
        self.battery.soc = 0.5
        mode = self.battery.update_external_control({"SOC": 0.6})
        self.assertEqual(mode, "On")
        self.assertGreater(self.battery.power_setpoint, 0)  # charging

        mode = self.battery.update_external_control({"SOC": 0.4})
        self.assertEqual(mode, "On")
        self.assertLess(self.battery.power_setpoint, 0)  # discharging

    def test_update_external_control_soc_limits(self):
        """Test Min SOC and Max SOC control signals"""
        self.battery.update_external_control({"Min SOC": 0.2})
        self.assertAlmostEqual(self.battery.soc_min_ctrl, 0.2)

        self.battery.update_external_control({"Max SOC": 0.9})
        self.assertAlmostEqual(self.battery.soc_max_ctrl, 0.9)

    def test_update_external_control_self_consumption(self):
        """Test Self Consumption Mode control signal"""
        self.battery.update_external_control({"Self Consumption Mode": True})
        self.assertTrue(self.battery.self_consumption_mode)

        self.battery.update_external_control({"Self Consumption Mode": False})
        self.assertFalse(self.battery.self_consumption_mode)

    def test_update_external_control_import_export_limits(self):
        """Test import and export limit control signals"""
        self.battery.update_external_control({"Max Import Limit": 3})
        self.assertEqual(self.battery.import_limit, 3)

        self.battery.update_external_control({"Max Export Limit": 2})
        self.assertEqual(self.battery.export_limit, 2)

    def test_update_internal_control_schedule(self):
        """Test schedule-based internal control"""
        self.battery.self_consumption_mode = False
        # No schedule power, should be off
        self.battery.current_schedule = {}
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.battery.power_setpoint, 0)

        # With schedule power
        self.battery.current_schedule = {"Battery Electric Power (kW)": 3}
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertEqual(self.battery.power_setpoint, 3)

    def test_update_internal_control_self_consumption(self):
        """Test self-consumption mode internal control"""
        self.battery.self_consumption_mode = True
        self.battery.soc = 0.5

        # House consuming power, battery should discharge
        self.battery.current_schedule = {"net_power": 2}
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.battery.power_setpoint, -2)

        # House exporting power, battery should charge
        self.battery.current_schedule = {"net_power": -2}
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.battery.power_setpoint, 2)

    def test_update_internal_control_solar_only(self):
        """Test charge_solar_only mode"""
        self.battery.self_consumption_mode = True
        self.battery.charge_solar_only = True
        self.battery.soc = 0.5

        # Net export but no PV, should not charge
        self.battery.current_schedule = {"net_power": -2}
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "Off")
        self.assertEqual(self.battery.power_setpoint, 0)

        # Net export with PV
        self.battery.current_schedule = {"net_power": -2, "pv_power": -3}
        mode = self.battery.update_internal_control()
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.battery.power_setpoint, 2)

    def test_get_power_limits_at_max_soc(self):
        """Test power limits when SOC is at max"""
        self.battery.soc = self.battery.soc_max
        p_min, p_max = self.battery.get_power_limits()
        self.assertEqual(p_max, 0)  # Can't charge
        self.assertAlmostEqual(p_min, -5)  # Can discharge at full power

    def test_get_power_limits_at_min_soc(self):
        """Test power limits when SOC is at min"""
        self.battery.soc = self.battery.soc_min
        p_min, p_max = self.battery.get_power_limits()
        self.assertEqual(p_min, 0)  # Can't discharge
        self.assertAlmostEqual(p_max, 5)  # Can charge at full power

    def test_get_power_limits_mid_soc(self):
        """Test power limits at mid SOC"""
        self.battery.soc = 0.5
        p_min, p_max = self.battery.get_power_limits()
        self.assertAlmostEqual(p_min, -5)
        self.assertAlmostEqual(p_max, 5)

    def test_calculate_efficiency_advanced(self):
        """Test advanced efficiency calculation (internal resistance model)"""
        self.battery.efficiency_type = "advanced"
        # Efficiency should depend on power
        eff_low = self.battery.calculate_efficiency(1)
        eff_high = self.battery.calculate_efficiency(5)
        # Higher power = more losses = lower efficiency
        self.assertGreater(eff_low, eff_high)
        self.assertGreater(eff_low, 0.9)
        self.assertLess(eff_high, 1.0)

    def test_calculate_efficiency_constant(self):
        """Test constant efficiency mode"""
        args = init_args.copy()
        args["efficiency_type"] = "constant"
        battery = Battery(**args)
        # Should return rated efficiency regardless of power
        # Note: constant efficiency still includes inverter efficiency
        eff = battery.calculate_efficiency(1)
        eff2 = battery.calculate_efficiency(5)
        # In constant mode, efficiency should be the same at different power levels
        self.assertAlmostEqual(eff, eff2)

    def test_calculate_power_and_heat_charging(self):
        """Test power and heat calculation during charging"""
        self.battery.soc = 0.5
        self.battery.power_setpoint = 3
        self.battery.mode = "On"
        self.battery.calculate_power_and_heat()
        self.assertGreater(self.battery.electric_kw, 0)
        self.assertGreater(self.battery.next_soc, self.battery.soc)

    def test_calculate_power_and_heat_discharging(self):
        """Test power and heat calculation during discharging"""
        self.battery.soc = 0.5
        self.battery.power_setpoint = -3
        self.battery.mode = "On"
        self.battery.calculate_power_and_heat()
        self.assertLess(self.battery.electric_kw, 0)
        self.assertLess(self.battery.next_soc, self.battery.soc)

    def test_calculate_power_and_heat_off(self):
        """Test power and heat when battery is off"""
        self.battery.mode = "Off"
        self.battery.power_setpoint = 0
        self.battery.calculate_power_and_heat()
        self.assertEqual(self.battery.electric_kw, 0)

    def test_get_kwh_remaining(self):
        """Test remaining energy calculation"""
        self.battery.soc = 0.5
        self.battery.capacity_kwh = 10

        # Discharge capacity
        kwh_discharge = self.battery.get_kwh_remaining(discharge=True, include_efficiency=False)
        expected = (0.5 - self.battery.soc_min) * 10
        self.assertAlmostEqual(kwh_discharge, expected, places=1)

        # Charge capacity
        kwh_charge = self.battery.get_kwh_remaining(discharge=False, include_efficiency=False)
        expected = (self.battery.soc_max - 0.5) * 10
        self.assertAlmostEqual(kwh_charge, expected, places=1)

    def test_get_setpoint_from_soc(self):
        """Test calculating power setpoint to achieve target SOC"""
        self.battery.soc = 0.5
        # Target higher SOC should give positive (charging) setpoint
        setpoint = self.battery.get_setpoint_from_soc(0.6)
        self.assertGreater(setpoint, 0)

        # Target lower SOC should give negative (discharging) setpoint
        setpoint = self.battery.get_setpoint_from_soc(0.4)
        self.assertLess(setpoint, 0)

    def test_generate_results(self):
        """Test result generation"""
        self.battery.verbosity = 6
        results = self.battery.generate_results()
        self.assertIn("Time", results)
        self.assertIn("Battery Electric Power (kW)", results)
        self.assertIn("Battery SOC (-)", results)

    def test_reset_time(self):
        """Test reset_time resets degradation states"""
        self.battery.degradation_states = (1, 2, 3)
        self.battery.capacity_kwh = 8
        self.battery.reset_time()
        self.assertEqual(self.battery.degradation_states, (0, 0, 0))
        self.assertEqual(self.battery.capacity_kwh, self.battery.capacity_rated)

    def test_simulate(self):
        """Test running a short simulation"""
        args = init_args.copy()
        args["duration"] = dt.timedelta(hours=1)
        battery = Battery(**args)

        # Set a simple schedule
        battery.schedule["Battery Electric Power (kW)"] = 1
        battery.reset_time()

        df = battery.simulate()
        self.assertIsNotNone(df)
        assert df is not None  # For type checker
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("Battery Electric Power (kW)", df.columns)

    def test_make_equivalent_battery_model(self):
        """Test EBM parameter generation"""
        self.battery.soc = 0.5
        ebm = self.battery.make_equivalent_battery_model()
        self.assertIsInstance(ebm, dict)
        # Check for actual EBM keys (not 'capacity')
        self.assertIn("Battery EBM Energy (kWh)", ebm)
        self.assertIn("Battery EBM Max Power (kW)", ebm)
        self.assertIn("Battery EBM Efficiency (-)", ebm)


class BatteryDegradationTestCase(unittest.TestCase):
    """
    Test Case for battery degradation model.
    """

    def test_degradation_enabled(self):
        """Test that degradation is enabled by default"""
        args = init_args.copy()
        args["enable_degradation"] = True
        battery = Battery(**args)
        self.assertIsNotNone(battery.degradation_states)
        self.assertEqual(battery.degradation_states, (0, 0, 0))

    def test_degradation_disabled(self):
        """Test disabling degradation model"""
        args = init_args.copy()
        args["enable_degradation"] = False
        battery = Battery(**args)
        self.assertIsNone(battery.degradation_states)

    def test_calculate_degradation(self):
        """Test degradation calculation (runs at end of day)"""
        args = init_args.copy()
        args["enable_degradation"] = True
        battery = Battery(**args)

        # Add some cycling data
        battery.degradation_data = [(0.3, 0.7), (0.4, 0.8), (0.2, 0.6)]
        initial_capacity = battery.capacity_kwh_nominal  # noqa: F841

        # Run degradation calculation
        battery.calculate_degradation()

        # Degradation states should be updated (may or may not reduce capacity
        # depending on cycles - just verify it runs without error)
        self.assertIsNotNone(battery.degradation_states)


class BatteryThermalModelTestCase(unittest.TestCase):
    """
    Test Case for battery with thermal model.
    """

    def test_init_with_thermal_model(self):
        """Test battery initialization with thermal model"""
        args = init_args.copy()
        args["enable_thermal_model"] = True
        args["Initial Battery Temperature (C)"] = 25
        battery = Battery(**args)
        self.assertIsNotNone(battery.thermal_model)
        self.assertIsInstance(battery.thermal_model, BatteryThermalModel)
        assert battery.thermal_model is not None  # For type checker
        self.assertAlmostEqual(battery.thermal_model.states[0], 25)

    def test_thermal_model_standalone(self):
        """Test BatteryThermalModel directly"""
        thermal_model = BatteryThermalModel(
            resistance=0.5,
            capacitance=90000,
            start_time=dt.datetime(2020, 1, 1),
            duration=dt.timedelta(hours=1),
            time_res=dt.timedelta(minutes=5),
            verbosity=7,
            save_results=False,
        )
        thermal_model.states[0] = 25
        self.assertEqual(thermal_model.name, "Battery Temperature")

        results = thermal_model.generate_results()
        self.assertIn("Battery Temperature (C)", results)

    def test_thermal_model_results(self):
        """Test that thermal model is properly initialized and produces results"""
        args = init_args.copy()
        args["enable_thermal_model"] = True
        args["Initial Battery Temperature (C)"] = 25
        args["verbosity"] = 7
        battery = Battery(**args)

        # Add Zone Temperature to schedule (required for thermal model)
        battery.schedule["Zone Temperature (C)"] = 20
        battery.reset_time()

        # Verify thermal model exists and has correct initial state
        self.assertIsNotNone(battery.thermal_model)
        assert battery.thermal_model is not None  # For type checker
        self.assertAlmostEqual(battery.thermal_model.states[0], 25)

        # Run one time step
        battery.update()

        # Thermal model results come from the sub-simulator
        thermal_results = battery.thermal_model.generate_results()
        self.assertIn("Battery Temperature (C)", thermal_results)


if __name__ == "__main__":
    unittest.main()
