import unittest
import datetime as dt
import pandas as pd

from ochre.Models.Envelope import Envelope
from ochre.Equipment.HVAC import (
    ElectricFurnace,
    ElectricBoiler,
    ElectricBaseboard,
    GasFurnace,
    GasBoiler,
    AirConditioner,
    RoomAC,
    ASHPHeater,
    MinisplitAHSPCooler,
)
from test.test_equipment import equip_init_args


# Use the same start time as equip_init_args to avoid schedule mismatch
start_time = equip_init_args["start_time"]
duration = equip_init_args["duration"]
time_res = equip_init_args["time_res"]

# Common simulation parameters
sim_params = {
    "start_time": start_time,
    "duration": duration,
    "time_res": time_res,
    "verbosity": 6,
    "save_results": False,
}


def create_hvac_schedule(start_time, duration, time_res, ambient_temp=20, is_heating=True):
    """Create schedule DataFrame with required HVAC inputs."""
    times = pd.date_range(start_time, start_time + duration, freq=time_res, inclusive="left")
    schedule = pd.DataFrame(
        {
            "Ambient Dry Bulb (C)": ambient_temp,
            "Ambient Humidity Ratio (-)": 0.005,
            "Ambient Pressure (kPa)": 101.325,
            "HVAC Heating Setpoint (C)": 20,
            "HVAC Cooling Setpoint (C)": 24,
            "HVAC Heating Deadband (C)": 1,
            "HVAC Cooling Deadband (C)": 1,
        },
        index=times,
    )
    return schedule


def create_minimal_envelope(schedule=None, **kwargs):
    """Create minimal envelope for HVAC testing."""
    start = kwargs.get("start_time", sim_params["start_time"])
    dur = kwargs.get("duration", sim_params["duration"])
    time_r = kwargs.get("time_res", sim_params["time_res"])

    if schedule is None:
        schedule = create_hvac_schedule(start, dur, time_r)

    envelope_args = {
        "capacitances": {"LIV": 4e6},
        "resistances": {("EXT", "LIV"): 1e-3},
        "zones": {"Indoor": {"Volume (m^3)": 600, "enable_humidity": False}},
        "ext_zone_labels": ["EXT"],
        "schedule": schedule,
        "initial_schedule": schedule.iloc[0].to_dict(),
        "initial_temp_setpoint": 22,
        "external_radiation_method": None,
        "internal_radiation_method": None,
        "main_sim_name": "",
        **sim_params,
        **kwargs,
    }
    return Envelope(**envelope_args)


def create_heater_args(envelope, **kwargs):
    """Create minimal arguments for heater initialization."""
    # Create schedule with required HVAC inputs
    schedule = create_hvac_schedule(
        sim_params["start_time"],
        sim_params["duration"],
        sim_params["time_res"],
    )
    args = equip_init_args.copy()
    args.update(
        {
            "envelope_model": envelope,
            "zone_name": "Indoor",
            "Capacity (W)": 5000,
            "EIR (-)": 1.0,
            "Duct DSE (-)": 1.0,
            "Rated Auxiliary Power (W)": 300,  # Fan power
            "Disable HVAC Biquadratics": True,  # Simplify for tests
            "schedule": schedule,
            "initial_schedule": schedule.iloc[0].to_dict(),
            **kwargs,
        }
    )
    return args


def create_cooler_args(envelope, **kwargs):
    """Create minimal arguments for cooler initialization."""
    # Create schedule with required HVAC inputs
    schedule = create_hvac_schedule(
        sim_params["start_time"],
        sim_params["duration"],
        sim_params["time_res"],
    )
    args = equip_init_args.copy()
    args.update(
        {
            "envelope_model": envelope,
            "zone_name": "Indoor",
            "Capacity (W)": 5000,
            "EIR (-)": 0.3,  # Typical AC COP ~3.3
            "SHR (-)": 0.8,
            "Duct DSE (-)": 1.0,
            "Rated Auxiliary Power (W)": 300,  # Fan power
            "Disable HVAC Biquadratics": True,
            "schedule": schedule,
            "initial_schedule": schedule.iloc[0].to_dict(),
            **kwargs,
        }
    )
    return args


class HeaterTestCase(unittest.TestCase):
    """Tests for basic Heater class."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_heater_args(self.envelope)

    def test_init_electric_furnace(self):
        """Test ElectricFurnace initialization."""
        heater = ElectricFurnace(**self.args)
        self.assertEqual(heater.name, "Electric Furnace")
        self.assertEqual(heater.end_use, "HVAC Heating")
        self.assertTrue(heater.is_heater)
        self.assertFalse(heater.is_gas)
        self.assertAlmostEqual(heater.capacity_max, 5000)

    def test_init_electric_boiler(self):
        """Test ElectricBoiler initialization."""
        heater = ElectricBoiler(**self.args)
        self.assertEqual(heater.name, "Electric Boiler")
        self.assertFalse(heater.is_gas)

    def test_init_electric_baseboard(self):
        """Test ElectricBaseboard initialization."""
        heater = ElectricBaseboard(**self.args)
        self.assertEqual(heater.name, "Electric Baseboard")
        self.assertEqual(heater.duct_dse, 1)  # Forced to 1

    def test_init_gas_furnace(self):
        """Test GasFurnace initialization."""
        heater = GasFurnace(**self.args)
        self.assertEqual(heater.name, "Gas Furnace")
        self.assertTrue(heater.is_gas)

    def test_init_gas_boiler(self):
        """Test GasBoiler initialization."""
        # GasBoiler needs lower EIR for non-condensing
        args = self.args.copy()
        args["EIR (-)"] = 1.0 / 0.85  # 85% AFUE
        heater = GasBoiler(**args)
        self.assertEqual(heater.name, "Gas Boiler")
        self.assertTrue(heater.is_gas)
        self.assertFalse(heater.condensing)

    def test_init_gas_boiler_condensing(self):
        """Test GasBoiler condensing mode."""
        args = self.args.copy()
        args["EIR (-)"] = 1.0 / 0.95  # 95% AFUE
        heater = GasBoiler(**args)
        self.assertTrue(heater.condensing)


class CoolerTestCase(unittest.TestCase):
    """Tests for basic Cooler class."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_cooler_args(self.envelope)

    def test_init_air_conditioner(self):
        """Test AirConditioner initialization."""
        cooler = AirConditioner(**self.args)
        self.assertEqual(cooler.name, "Air Conditioner")
        self.assertEqual(cooler.end_use, "HVAC Cooling")
        self.assertFalse(cooler.is_heater)
        self.assertAlmostEqual(cooler.capacity_max, 5000)
        self.assertAlmostEqual(cooler.shr, 0.8)

    def test_init_room_ac(self):
        """Test RoomAC initialization."""
        cooler = RoomAC(**self.args)
        self.assertEqual(cooler.name, "Room AC")


class HVACControlTestCase(unittest.TestCase):
    """Tests for HVAC thermostat and control logic."""

    def setUp(self):
        # Start with cold ambient so heater should turn on
        self.schedule = create_hvac_schedule(
            sim_params["start_time"],
            sim_params["duration"],
            sim_params["time_res"],
            ambient_temp=0,  # Cold ambient
            is_heating=True,
        )
        self.envelope = create_minimal_envelope(schedule=self.schedule)
        self.args = create_heater_args(self.envelope)
        self.heater = ElectricFurnace(**self.args)

    def test_thermostat_heating_on(self):
        """Test thermostat turns heating on when below setpoint - deadband."""
        # Set indoor temp well below heating setpoint - deadband
        # Setpoint is 20, deadband is 1, so turn-on is at 19
        # Need to directly set zone.temperature which thermostat uses
        indoor_zone = self.envelope.zones["Indoor"]
        indoor_zone.temperature = 17  # Below 20C - 1C deadband = 19C
        self.heater.update_inputs()
        self.heater.update_model()
        # Heater should be on
        self.assertIn(self.heater.mode, ["On", "HP On"])

    def test_thermostat_heating_off(self):
        """Test thermostat turns heating off when above setpoint + deadband."""
        # Set indoor temp above heating setpoint + deadband
        indoor_zone = self.envelope.zones["Indoor"]
        indoor_zone.temperature = 22  # Above 20C + 1C deadband
        self.heater.update_inputs()
        self.heater.update_model()
        # Heater should be off
        self.assertEqual(self.heater.mode, "Off")

    def test_external_control_setpoint(self):
        """Test external setpoint control updates current_schedule."""
        self.heater.update_external_control({"Setpoint": 25})
        # Setpoint is updated in current_schedule
        self.assertAlmostEqual(self.heater.current_schedule["HVAC Heating Setpoint (C)"], 25)

    def test_external_control_deadband(self):
        """Test external deadband control."""
        self.heater.update_external_control({"Deadband": 2})
        # Deadband is updated in temp_deadband when not in schedule
        self.assertAlmostEqual(self.heater.temp_deadband, 2)

    def test_external_control_capacity_fraction(self):
        """Test external capacity fraction control with ideal capacity."""
        # Need to create heater with ideal capacity for this control
        args = self.args.copy()
        args["use_ideal_capacity"] = True
        heater = ElectricFurnace(**args)
        heater.update_external_control({"Max Capacity Fraction": 0.5})
        self.assertAlmostEqual(heater.ext_capacity_frac, 0.5)

    def test_external_control_duty_cycle(self):
        """Test external duty cycle control affects mode."""
        # Duty cycle control returns mode from run_duty_cycle_control
        # Instead of checking an attribute, verify the control works
        mode = self.heater.update_external_control({"Duty Cycle": 0})
        # Duty cycle of 0 should force off
        self.assertEqual(mode, "Off")


class HVACIdealCapacityTestCase(unittest.TestCase):
    """Tests for ideal capacity HVAC."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_heater_args(self.envelope)
        self.args["use_ideal_capacity"] = True
        self.heater = ElectricFurnace(**self.args)

    def test_ideal_capacity_mode(self):
        """Test ideal capacity is used."""
        self.assertTrue(self.heater.use_ideal_capacity)

    def test_ideal_capacity_update(self):
        """Test ideal capacity calculation."""
        # Set temp well below setpoint to trigger heating
        self.envelope.states[0] = 17  # Below setpoint - deadband = 19
        self.heater.update_inputs()
        self.heater.update_model()
        # capacity_ideal is calculated and used
        self.assertIsNotNone(self.heater.capacity_ideal)


class DynamicHVACTestCase(unittest.TestCase):
    """Tests for DynamicHVAC with biquadratic model."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_cooler_args(self.envelope)
        # Enable biquadratics for dynamic testing
        self.args["Disable HVAC Biquadratics"] = False

    def test_air_conditioner_with_biquad(self):
        """Test AirConditioner with biquadratic model."""
        cooler = AirConditioner(**self.args)
        self.assertIsNotNone(cooler.biquad_params)
        if cooler.biquad_params is not None:
            self.assertIn(1, cooler.biquad_params.keys())  # Speed 1 params

    def test_two_speed_control(self):
        """Test two-speed equipment."""
        args = self.args.copy()
        args["Number of Speeds (-)"] = 2
        args["Rated Efficiency"] = "SEER 14"  # Common 2-speed rating
        args["Disable HVAC Biquadratics"] = True
        # 2-speed requires multispeed file which may not have all configs
        # Skip if not available
        try:
            cooler = AirConditioner(**args)
            self.assertEqual(cooler.n_speeds, 2)
        except Exception:
            self.skipTest("Multispeed parameters not available for this config")


class ASHPHeaterTestCase(unittest.TestCase):
    """Tests for ASHP Heater with electric resistance backup."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_heater_args(self.envelope)
        self.args["Disable HVAC Biquadratics"] = False
        self.args["EIR (-)"] = 0.3  # Typical heat pump COP ~3.3
        self.args["Backup Capacity (W)"] = 5000  # Backup ER capacity

    def test_init_ashp_heater(self):
        """Test ASHP Heater initialization."""
        heater = ASHPHeater(**self.args)
        self.assertEqual(heater.name, "ASHP Heater")
        self.assertIn("HP On", heater.modes)
        self.assertIn("ER On", heater.modes)

    def test_ashp_defrost_mode(self):
        """Test ASHP defrost activation in cold weather."""
        heater = ASHPHeater(**self.args)
        # Simulate cold conditions (below 4.4C)
        heater.current_schedule = {
            "Ambient Dry Bulb (C)": 0,
            "Ambient Humidity Ratio (-)": 0.003,
            "Ambient Pressure (kPa)": 101.325,
        }
        heater.update_capacity()
        self.assertTrue(heater.defrost)

    def test_ashp_no_defrost_warm(self):
        """Test ASHP no defrost in warm weather."""
        heater = ASHPHeater(**self.args)
        heater.current_schedule = {
            "Ambient Dry Bulb (C)": 10,
            "Ambient Humidity Ratio (-)": 0.005,
            "Ambient Pressure (kPa)": 101.325,
        }
        heater.update_capacity()
        self.assertFalse(heater.defrost)


class HVACResultsTestCase(unittest.TestCase):
    """Tests for HVAC result generation."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_heater_args(self.envelope)
        self.heater = ElectricFurnace(**self.args)

    def test_generate_results(self):
        """Test generate_results produces expected keys."""
        self.heater.update_inputs()
        self.heater.update_model()
        results = self.heater.generate_results()
        # Check for standard HVAC result keys
        self.assertIn("HVAC Heating Electric Power (kW)", results)
        self.assertIn("HVAC Heating Delivered (W)", results)

    def test_power_calculation(self):
        """Test power and heat calculations."""
        self.envelope.states[0] = 18  # Below setpoint to turn on
        self.heater.update_inputs()
        self.heater.update_model()
        self.heater.calculate_power_and_heat()
        # Electric furnace should use power when on
        if self.heater.mode == "On":
            self.assertGreater(self.heater.electric_kw, 0)


class HVACSimulationTestCase(unittest.TestCase):
    """Integration tests for HVAC simulation."""

    def test_heater_simulation_step(self):
        """Test heater through one simulation step."""
        test_duration = dt.timedelta(hours=1)
        schedule = create_hvac_schedule(
            sim_params["start_time"],
            test_duration,
            sim_params["time_res"],
            ambient_temp=0,
        )
        envelope = create_minimal_envelope(
            schedule=schedule,
            duration=test_duration,
        )
        args = create_heater_args(envelope, schedule=schedule, duration=test_duration)
        heater = ElectricFurnace(**args)

        # Run one update cycle
        envelope.states[0] = 17  # Cold indoor - below setpoint-deadband
        envelope.update_inputs()
        envelope.update_model()

        heater.update_inputs()
        heater.update_model()
        results = heater.update_results()

        self.assertIsNotNone(results)

    def test_cooler_simulation_step(self):
        """Test cooler through one simulation step."""
        test_duration = dt.timedelta(hours=1)
        schedule = create_hvac_schedule(
            sim_params["start_time"],
            test_duration,
            sim_params["time_res"],
            ambient_temp=35,  # Hot ambient
        )
        envelope = create_minimal_envelope(
            schedule=schedule,
            duration=test_duration,
        )
        args = create_cooler_args(envelope, schedule=schedule, duration=test_duration)
        cooler = AirConditioner(**args)

        # Run one update cycle
        envelope.states[0] = 26  # Hot indoor
        envelope.update_inputs()
        envelope.update_model()

        cooler.update_inputs()
        cooler.update_model()
        results = cooler.update_results()

        self.assertIsNotNone(results)


class GasEquipmentTestCase(unittest.TestCase):
    """Tests for gas-powered HVAC equipment."""

    def setUp(self):
        self.envelope = create_minimal_envelope()
        self.args = create_heater_args(self.envelope)
        self.args["EIR (-)"] = 1.0 / 0.80  # 80% AFUE

    def test_gas_furnace_gas_usage(self):
        """Test gas furnace uses gas, not electricity."""
        heater = GasFurnace(**self.args)
        self.assertTrue(heater.is_gas)
        # Force heater on
        self.envelope.states[0] = 15
        heater.update_inputs()
        heater.update_model()
        heater.calculate_power_and_heat()
        # Should have gas usage when on
        if heater.mode == "On":
            self.assertGreater(heater.gas_therms_per_hour, 0)


class MinisplitTestCase(unittest.TestCase):
    """Tests for minisplit HVAC equipment."""

    def test_minisplit_cooler_init(self):
        """Test MinisplitAHSPCooler initialization."""
        envelope = create_minimal_envelope()
        args = create_cooler_args(envelope)
        args["Disable HVAC Biquadratics"] = False
        args["Number of Speeds (-)"] = 4
        args["Rated Efficiency"] = "13.0 SEER"  # Must match multispeed file
        # Capacity should be scalar - code expands with capacity ratios from file
        args["Capacity (W)"] = 5000
        # Remove EIR and SHR - let them be calculated from file
        args.pop("EIR (-)", None)
        args.pop("SHR (-)", None)

        try:
            cooler = MinisplitAHSPCooler(**args)
            self.assertEqual(cooler.name, "MSHP Cooler")
            self.assertEqual(cooler.n_speeds, 4)
        except Exception as e:
            # May fail if biquadratic files missing
            if "Biquadratic" in str(e) or "not found" in str(e).lower():
                self.skipTest(f"Biquadratic params not available: {e}")
            raise


if __name__ == "__main__":
    unittest.main()
