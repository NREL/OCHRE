import unittest
import datetime as dt
import numpy as np
import pandas as pd

from ochre.Models.Envelope import Envelope, ExteriorZone
from ochre.utils import OCHREException


# Common simulation parameters
sim_params = {
    "start_time": dt.datetime(2020, 1, 1),
    "duration": dt.timedelta(hours=24),
    "time_res": dt.timedelta(minutes=5),
    "verbosity": 6,
    "save_results": False,
}


def create_minimal_schedule(start_time, duration, time_res, ambient_temp=20):
    """Create minimal schedule DataFrame for envelope testing"""
    times = pd.date_range(start_time, start_time + duration, freq=time_res, inclusive="left")
    return pd.DataFrame(
        {
            "Ambient Dry Bulb (C)": ambient_temp,
            "HVAC Heating Setpoint (C)": 20,
            "HVAC Cooling Setpoint (C)": 24,
            "HVAC Heating Deadband (C)": 1,
            "HVAC Cooling Deadband (C)": 1,
            "Ambient Humidity Ratio (-)": 0.005,
        },
        index=times,
    )


def create_minimal_envelope(schedule=None, **kwargs):
    """Create minimal single-zone envelope for testing"""
    start_time = kwargs.get("start_time", sim_params["start_time"])
    duration = kwargs.get("duration", sim_params["duration"])
    time_res = kwargs.get("time_res", sim_params["time_res"])

    if schedule is None:
        schedule = create_minimal_schedule(start_time, duration, time_res)

    envelope_args = {
        "capacitances": {"LIV": 4e6},  # ~1 hour time constant
        "resistances": {("EXT", "LIV"): 1e-3},  # 1kW per degree C
        "zones": {"Indoor": {"Volume (m^3)": 600}},
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


class EnvelopeInitTestCase(unittest.TestCase):
    """Tests for Envelope initialization"""

    def test_init_minimal(self):
        """Test minimal envelope initialization"""
        envelope = create_minimal_envelope()
        self.assertEqual(envelope.name, "Envelope")
        self.assertIsNotNone(envelope.zones)
        self.assertIn("Indoor", envelope.zones)
        self.assertIsNotNone(envelope.indoor_zone)

    def test_init_with_capacitances_resistances(self):
        """Test initialization with direct capacitances and resistances"""
        envelope = create_minimal_envelope()
        # Check state and input names are created correctly
        self.assertIn("T_LIV", envelope.state_names)
        self.assertIn("T_EXT", envelope.input_names)
        self.assertIn("H_LIV", envelope.input_names)

    def test_init_zones(self):
        """Test zone creation"""
        envelope = create_minimal_envelope()
        self.assertEqual(len(envelope.zones), 1)
        indoor_zone = envelope.zones["Indoor"]
        self.assertEqual(indoor_zone.name, "Indoor")
        self.assertEqual(indoor_zone.label, "LIV")

    def test_init_ext_zones(self):
        """Test external zone creation"""
        envelope = create_minimal_envelope()
        self.assertIn("Outdoor", envelope.ext_zones)
        outdoor = envelope.ext_zones["Outdoor"]
        self.assertIsInstance(outdoor, ExteriorZone)
        self.assertEqual(outdoor.label, "EXT")

    def test_init_multiple_ext_zones(self):
        """Test with multiple external zones"""
        schedule = create_minimal_schedule(sim_params["start_time"], sim_params["duration"], sim_params["time_res"])
        # Add ground temperature to schedule
        schedule["Ground Temperature (C)"] = 15

        envelope = Envelope(
            capacitances={"LIV": 4e6, "FND": 1e6},
            resistances={("EXT", "LIV"): 1e-3, ("GND", "FND"): 2e-3, ("LIV", "FND"): 5e-3},
            zones={
                "Indoor": {"Volume (m^3)": 600},
                "Foundation": {"Volume (m^3)": 100},
            },
            ext_zone_labels=["EXT", "GND"],
            schedule=schedule,
            initial_schedule=schedule.iloc[0].to_dict(),
            initial_temp_setpoint=22,
            external_radiation_method=None,
            internal_radiation_method=None,
            main_sim_name="",
            **sim_params,
        )
        self.assertIn("Outdoor", envelope.ext_zones)
        self.assertIn("Ground", envelope.ext_zones)


class EnvelopeZoneTestCase(unittest.TestCase):
    """Tests for zone-related functionality"""

    def setUp(self):
        self.envelope = create_minimal_envelope()

    def test_get_zone_temperature_indoor(self):
        """Test getting indoor zone temperature"""
        temp = self.envelope.get_zone_temperature("Indoor")
        self.assertIsInstance(temp, (int, float, np.floating))
        # Should be close to initial setpoint
        self.assertAlmostEqual(temp, 22, places=0)

    def test_get_zone_temperature_outdoor(self):
        """Test getting outdoor zone temperature"""
        # Need to run update first to get ambient temperature into ext zone
        self.envelope.update_inputs()
        temp = self.envelope.get_zone_temperature("Outdoor")
        # Should match schedule ambient temperature
        self.assertAlmostEqual(temp, 20, places=1)

    def test_get_zone_temperature_invalid(self):
        """Test getting temperature for invalid zone"""
        with self.assertRaises(OCHREException):
            self.envelope.get_zone_temperature("NonexistentZone")

    def test_indoor_zone_reference(self):
        """Test indoor_zone reference is correct"""
        self.assertEqual(self.envelope.indoor_zone, self.envelope.zones["Indoor"])


class EnvelopeUpdateTestCase(unittest.TestCase):
    """Tests for envelope update cycle"""

    def setUp(self):
        self.envelope = create_minimal_envelope()

    def test_update_inputs(self):
        """Test schedule input handling"""
        schedule_inputs = {
            "Ambient Dry Bulb (C)": 25,
            "HVAC Heating Setpoint (C)": 20,
            "HVAC Cooling Setpoint (C)": 24,
        }
        self.envelope.update_inputs(schedule_inputs)
        # Check that external temperature is updated in ext_zones
        # Note: inputs_init has the temperature, not inputs (which is set by update_model)
        self.assertAlmostEqual(self.envelope.ext_zones["Outdoor"].temperature, 25)

    def test_update_model(self):
        """Test model state update"""
        initial_temp = self.envelope.states[0]
        # Update with warmer ambient
        schedule_inputs = {
            "Ambient Dry Bulb (C)": 30,
            "HVAC Heating Setpoint (C)": 20,
            "HVAC Cooling Setpoint (C)": 24,
        }
        self.envelope.update_inputs(schedule_inputs)
        self.envelope.update_model()
        # Verify external temperature was set correctly in inputs_init
        ext_idx = self.envelope.input_names.index("T_EXT")
        self.assertAlmostEqual(self.envelope.inputs_init[ext_idx], 30)
        # next_states should show temperature increase (states updated in update_results)
        self.assertGreater(self.envelope.next_states[0], initial_temp)

    def test_update_results(self):
        """Test result update cycle"""
        self.envelope.update_inputs()
        self.envelope.update_model()
        results = self.envelope.update_results()
        self.assertIsNotNone(results)

    def test_state_transition(self):
        """Test temperature changes over multiple steps"""
        initial_temp = self.envelope.states[0]

        # Run several update cycles with warm ambient
        for _ in range(10):
            self.envelope.update(schedule_inputs={"Ambient Dry Bulb (C)": 30})

        final_temp = self.envelope.states[0]
        # Temperature should have increased towards 30C
        self.assertGreater(final_temp, initial_temp)
        self.assertLess(final_temp, 30)


class EnvelopeInfiltrationTestCase(unittest.TestCase):
    """Tests for infiltration calculations"""

    def test_no_infiltration(self):
        """Test envelope with no infiltration"""
        envelope = create_minimal_envelope()
        # By default, minimal envelope has no infiltration method
        self.assertIsNone(envelope.indoor_zone.infiltration_method)

    def test_linearize_infiltration(self):
        """Test linearized infiltration mode"""
        envelope = create_minimal_envelope(linearize_infiltration=True)
        self.assertTrue(envelope.linearize_infiltration)


class EnvelopeRadiationTestCase(unittest.TestCase):
    """Tests for radiation methods"""

    def test_external_radiation_none(self):
        """Test with external radiation disabled"""
        envelope = create_minimal_envelope(external_radiation_method=None)
        self.assertFalse(envelope.run_external_rad)

    def test_internal_radiation_none(self):
        """Test with internal radiation disabled"""
        envelope = create_minimal_envelope(internal_radiation_method=None)
        self.assertFalse(envelope.run_internal_rad)

    def test_internal_radiation_linear(self):
        """Test with linear internal radiation"""
        envelope = create_minimal_envelope(internal_radiation_method="linear")
        self.assertFalse(envelope.run_internal_rad)
        self.assertTrue(envelope.linearize_int_radiation)


class EnvelopeSolverTestCase(unittest.TestCase):
    """Tests for solve_for_input (used by HVAC)"""

    def setUp(self):
        self.envelope = create_minimal_envelope()

    def test_solve_for_input(self):
        """Test solving for a single input to achieve desired state"""
        current_temp = self.envelope.states[0]
        # Solve for heat input to maintain current temperature
        u_desired = self.envelope.solve_for_input("T_LIV", "H_LIV", current_temp)
        self.assertIsInstance(u_desired, (int, float, np.floating))


class EnvelopeResultsTestCase(unittest.TestCase):
    """Tests for result generation at various verbosity levels"""

    def test_generate_results_verbosity_3(self):
        """Test results at verbosity 3"""
        # main_sim_name=None required for Time to be in results
        envelope = create_minimal_envelope(verbosity=3, main_sim_name=None)
        results = envelope.generate_results()
        self.assertIn("Time", results)
        self.assertIn("Temperature - Indoor (C)", results)

    def test_generate_results_verbosity_5(self):
        """Test results at verbosity 5"""
        envelope = create_minimal_envelope(verbosity=5)
        results = envelope.generate_results()
        self.assertIn("Temperature - Indoor (C)", results)

    def test_generate_results_verbosity_6(self):
        """Test results at verbosity 6"""
        envelope = create_minimal_envelope(verbosity=6)
        results = envelope.generate_results()
        self.assertIn("Temperature - Indoor (C)", results)


class EnvelopeHumidityTestCase(unittest.TestCase):
    """Tests for humidity model integration"""

    def test_humidity_enabled(self):
        """Test envelope with humidity model enabled"""
        schedule = create_minimal_schedule(sim_params["start_time"], sim_params["duration"], sim_params["time_res"])
        schedule["Ambient Humidity Ratio (-)"] = 0.01

        envelope = create_minimal_envelope(
            schedule=schedule,
            enable_humidity=True,
        )
        # Indoor zone should have a humidity model
        self.assertIsNotNone(envelope.indoor_zone.humidity)

    def test_humidity_disabled_via_zone(self):
        """Test envelope with humidity model disabled via zone args"""
        # Humidity is enabled by default for Indoor zones when humidity ratio is in schedule.
        # To disable it, pass enable_humidity=False in the zone args.
        envelope = create_minimal_envelope(
            zones={"Indoor": {"Volume (m^3)": 600, "enable_humidity": False}},
        )
        self.assertIsNone(envelope.indoor_zone.humidity)


class EnvelopeEBMTestCase(unittest.TestCase):
    """Tests for equivalent battery model parameters"""

    def test_get_ebm_parameters(self):
        """Test EBM parameter generation - currently not implemented"""
        envelope = create_minimal_envelope()
        ebm = envelope.get_ebm_parameters()
        # Note: get_ebm_parameters is not yet implemented (returns None)
        self.assertIsNone(ebm)


class EnvelopeSimulationTestCase(unittest.TestCase):
    """Integration tests - run simulation"""

    def test_simulate(self):
        """Test running a short simulation"""
        # main_sim_name=None is required for main_simulator=True
        envelope = create_minimal_envelope(
            duration=dt.timedelta(hours=1),
            main_sim_name=None,
        )
        df = envelope.simulate()
        self.assertIsNotNone(df)
        assert df is not None  # For type checker
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_simulate_with_humidity(self):
        """Test simulation with humidity model"""
        schedule = create_minimal_schedule(sim_params["start_time"], dt.timedelta(hours=1), sim_params["time_res"])
        schedule["Ambient Humidity Ratio (-)"] = 0.01

        envelope = create_minimal_envelope(
            schedule=schedule,
            duration=dt.timedelta(hours=1),
            enable_humidity=True,
            main_sim_name=None,
        )
        df = envelope.simulate()
        self.assertIsNotNone(df)
        assert df is not None
        self.assertGreater(len(df), 0)


class ZoneTestCase(unittest.TestCase):
    """Tests for Zone class"""

    def test_zone_creation(self):
        """Test Zone is created correctly via Envelope"""
        envelope = create_minimal_envelope()
        zone = envelope.indoor_zone
        self.assertEqual(zone.name, "Indoor")
        self.assertEqual(zone.label, "LIV")
        self.assertEqual(zone.volume, 600)


class ExteriorZoneTestCase(unittest.TestCase):
    """Tests for ExteriorZone class"""

    def test_exterior_zone_init(self):
        """Test ExteriorZone initialization"""
        ext_zone = ExteriorZone("Outdoor", "EXT")
        self.assertEqual(ext_zone.name, "Outdoor")
        self.assertEqual(ext_zone.label, "EXT")


if __name__ == "__main__":
    unittest.main()
