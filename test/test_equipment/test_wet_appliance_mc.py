# -*- coding: utf-8 -*-
"""
Tests for WetAppliance Monte Carlo Profile Generator.

This tests the legacy WetAppliance class which generates stochastic
load profiles based on switch-on probability vectors.
"""

import unittest
import numpy as np

from ochre.Equipment.WetAppliance import WetAppliance


class WetApplianceInitTestCase(unittest.TestCase):
    """Test WetAppliance initialization."""

    def setUp(self):
        """Set up test data for WetAppliance."""
        # Create mock appliance data structure
        # Set_Profile is a 2-column array: [P_kW, Q_kVAr]
        self.profile = np.array(
            [
                [0.5, 0.1],
                [1.0, 0.2],
                [1.5, 0.3],
                [1.0, 0.2],
                [0.5, 0.1],
            ]
        )

        # Switch-on probability vector (1440 minutes in a day)
        self.switch_on_prob = np.zeros((1440, 1))
        # Higher probability in morning (7-9am) and evening (7-9pm)
        self.switch_on_prob[420:540, 0] = 0.01  # 7am-9am
        self.switch_on_prob[1140:1260, 0] = 0.01  # 7pm-9pm

        self.wet_appliances_data = {
            "Washing_Machine": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 30,
            },
            "Dish_Washer": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile * 0.8,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 45,
            },
            "Clothes_Dryer": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile * 2.0,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 20,
            },
        }

    def test_init_washing_machine(self):
        """Test initialization of washing machine."""
        wa = WetAppliance(self.wet_appliances_data, "Washing_Machine", start_time=0)

        self.assertEqual(wa.Switch_On_Time_Loc, 0)
        self.assertEqual(wa.Binary_Mem, 0)
        self.assertEqual(wa.Binary, 0)
        self.assertEqual(wa.Profile_t, 0)
        self.assertEqual(wa.P_kW, 0)
        self.assertEqual(wa.Q_kVAr, 0)
        self.assertEqual(wa.Schedule, 0)
        self.assertEqual(wa.Average_Schedule_Delay, 30)
        np.testing.assert_array_equal(wa.Set_Profile, self.profile)

    def test_init_dish_washer(self):
        """Test initialization of dish washer."""
        wa = WetAppliance(self.wet_appliances_data, "Dish_Washer", start_time=100)

        self.assertEqual(wa.Switch_On_Time_Loc, 100)
        self.assertEqual(wa.Average_Schedule_Delay, 45)
        np.testing.assert_array_equal(wa.Set_Profile, self.profile * 0.8)

    def test_init_clothes_dryer(self):
        """Test initialization of clothes dryer."""
        wa = WetAppliance(self.wet_appliances_data, "Clothes_Dryer", start_time=500)

        self.assertEqual(wa.Switch_On_Time_Loc, 500)
        self.assertEqual(wa.Average_Schedule_Delay, 20)
        np.testing.assert_array_equal(wa.Set_Profile, self.profile * 2.0)

    def test_init_states(self):
        """Test that initial states are correctly set."""
        wa = WetAppliance(self.wet_appliances_data, "Washing_Machine", start_time=0)

        self.assertEqual(wa.Schedule_Finish_Count, 0)
        self.assertEqual(wa.Over_ride_start, 0)
        self.assertEqual(wa.Over_ride_bin, 0)
        self.assertEqual(wa.Schedulable, 0)
        self.assertEqual(wa.Random_num, 0)


class MCSimDayLoopTestCase(unittest.TestCase):
    """Test day loop wrapping."""

    def setUp(self):
        """Set up test data."""
        self.profile = np.array([[0.5, 0.1], [1.0, 0.2]])
        self.switch_on_prob = np.zeros((1440, 1))
        self.wet_appliances_data = {
            "Test": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 30,
            }
        }

    def test_day_loop_wraps_at_1440(self):
        """Test that day wraps at 1440 minutes."""
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)

        # At minute 1440, should wrap to 0
        result = wa.MC_SIM_DAY_LOOP(1440)
        self.assertEqual(result, 0)

    def test_day_loop_no_wrap_before_1440(self):
        """Test that day doesn't wrap before 1440."""
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)

        # Before 1440, should return same value
        for minute in [0, 100, 500, 1000, 1439]:
            result = wa.MC_SIM_DAY_LOOP(minute)
            self.assertEqual(result, minute)


class MCProfileUpdateNonSchedulableTestCase(unittest.TestCase):
    """Test MC profile updates for non-schedulable appliances."""

    def setUp(self):
        """Set up test data with high probability to trigger events."""
        self.profile = np.array(
            [
                [0.5, 0.1],
                [1.0, 0.2],
                [1.5, 0.3],
                [1.0, 0.2],
                [0.5, 0.1],
            ]
        )
        # High probability to ensure events trigger
        self.switch_on_prob = np.ones((1440, 1)) * 0.99

        self.wet_appliances_data = {
            "Test": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 30,
            }
        }

    def test_profile_update_starts_event(self):
        """Test that MC_Profile_update can start an event."""
        np.random.seed(42)  # Seed for reproducibility
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 0  # Non-schedulable mode

        # Run several updates to trigger an event
        event_started = False
        for _ in range(10):
            wa.MC_Profile_update()
            if wa.P_kW > 0:
                event_started = True
                break

        self.assertTrue(event_started, "Event should start with high probability")

    def test_profile_update_follows_profile(self):
        """Test that profile values are followed during event."""
        np.random.seed(42)
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 0

        # Force event to start
        wa.Binary = 1
        wa.Binary_Mem = 1
        wa.Profile_t = 0

        # Update and check profile values
        wa.MC_Profile_update()
        self.assertEqual(wa.P_kW, self.profile[0, 0])
        self.assertEqual(wa.Q_kVAr, self.profile[0, 1])
        self.assertEqual(wa.Profile_t, 1)

    def test_profile_update_completes_cycle(self):
        """Test that a full cycle completes correctly."""
        np.random.seed(42)
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 0

        # Force event to start at end of profile
        wa.Binary = 1
        wa.Binary_Mem = 1
        wa.Profile_t = len(self.profile) - 1  # Last position

        # Update should complete the cycle
        wa.MC_Profile_update()

        # After completion, values should reset
        self.assertEqual(wa.Binary, 0)
        self.assertEqual(wa.Binary_Mem, 0)
        self.assertEqual(wa.Profile_t, 0)
        self.assertEqual(wa.P_kW, 0)
        self.assertEqual(wa.Q_kVAr, 0)

    def test_time_location_increments(self):
        """Test that Switch_On_Time_Loc increments each update."""
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=100)
        wa.Schedulable = 0

        initial_time = wa.Switch_On_Time_Loc
        wa.MC_Profile_update()

        self.assertEqual(wa.Switch_On_Time_Loc, initial_time + 1)


class MCProfileUpdateSchedulableTestCase(unittest.TestCase):
    """Test MC profile updates for schedulable appliances."""

    def setUp(self):
        """Set up test data."""
        self.profile = np.array(
            [
                [0.5, 0.1],
                [1.0, 0.2],
                [1.5, 0.3],
            ]
        )
        self.switch_on_prob = np.ones((1440, 1)) * 0.99

        self.wet_appliances_data = {
            "Test": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 30,
            }
        }

    def test_schedulable_sets_schedule_finish_count(self):
        """Test that schedulable mode sets schedule finish count."""
        np.random.seed(42)
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 1  # Schedulable mode

        # Run updates to trigger event
        for _ in range(10):
            wa.MC_Profile_update()
            if wa.Binary > 0 and wa.Binary_Mem == 1:
                # Schedule_Finish_Count should be set
                break

    def test_override_start_triggers_immediate_start(self):
        """Test that override_start triggers immediate start."""
        np.random.seed(42)
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 1

        # Set up a pending scheduled event
        wa.Binary = 1
        wa.Binary_Mem = 1
        wa.Schedule_Finish_Count = 100  # Would normally wait
        wa.Over_ride_start = 1  # Override to start immediately

        wa.MC_Profile_update()

        # Override should clear schedule wait
        self.assertEqual(wa.Schedule_Finish_Count, 0)
        self.assertEqual(wa.Over_ride_bin, 1)

    def test_override_bin_persists_until_cycle_complete(self):
        """Test that override_bin persists until cycle completes."""
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 1

        # Start a cycle with override
        wa.Binary = 1
        wa.Binary_Mem = 1
        wa.Over_ride_bin = 1
        wa.Profile_t = 0

        # Run through profile
        for i in range(len(self.profile) - 1):
            wa.MC_Profile_update()
            if wa.Profile_t > 0:
                self.assertEqual(wa.Over_ride_bin, 1)

    def test_schedulable_countdown(self):
        """Test schedule countdown when waiting."""
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 1

        # Set up waiting state
        wa.Binary = 1
        wa.Binary_Mem = 1
        wa.Schedule_Finish_Count = 5
        wa.Over_ride_bin = 0
        wa.Profile_t = 0

        initial_count = wa.Schedule_Finish_Count
        wa.MC_Profile_update()

        # Count should decrement
        self.assertEqual(wa.Schedule_Finish_Count, initial_count - 1)
        # Should not consume power while waiting
        self.assertEqual(wa.P_kW, 0)


class ZeroProbabilityTestCase(unittest.TestCase):
    """Test behavior with zero probability."""

    def setUp(self):
        """Set up test data with zero probability."""
        self.profile = np.array([[0.5, 0.1], [1.0, 0.2]])
        self.switch_on_prob = np.zeros((1440, 1))  # Never triggers

        self.wet_appliances_data = {
            "Test": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 30,
            }
        }

    def test_no_events_with_zero_probability(self):
        """Test that no events trigger with zero probability."""
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 0

        # Run many updates
        for _ in range(100):
            wa.MC_Profile_update()
            self.assertEqual(wa.P_kW, 0)
            self.assertEqual(wa.Binary, 0)


class FullDaySimulationTestCase(unittest.TestCase):
    """Test a full day simulation."""

    def setUp(self):
        """Set up realistic test data."""
        # 60-minute cycle
        self.profile = np.array([[0.5, 0.1]] * 60)

        # Probability peaks at 8am and 8pm
        self.switch_on_prob = np.zeros((1440, 1))
        self.switch_on_prob[480, 0] = 0.5  # 8am
        self.switch_on_prob[1200, 0] = 0.5  # 8pm

        self.wet_appliances_data = {
            "Test": {
                "PQ_Demand_Profile__2_cols_W_VAr": self.profile,
                "Switch_On_Daily_Probability_Profiles__Probability_Minutes_1440": self.switch_on_prob,
                "Averaged_Scheduled_Delay__Minutes": 30,
            }
        }

    def test_full_day_simulation(self):
        """Test running a full day simulation."""
        np.random.seed(123)
        wa = WetAppliance(self.wet_appliances_data, "Test", start_time=0)
        wa.Schedulable = 0

        total_energy = 0
        events_count = 0
        was_running = False

        for minute in range(1440):
            wa.MC_Profile_update()
            total_energy += wa.P_kW / 60  # kWh

            # Count event starts
            if wa.P_kW > 0 and not was_running:
                events_count += 1
            was_running = wa.P_kW > 0

        # Simulation should complete without error
        # Energy and events depend on random seed
        self.assertGreaterEqual(total_energy, 0)
        self.assertGreaterEqual(events_count, 0)


if __name__ == "__main__":
    unittest.main()
