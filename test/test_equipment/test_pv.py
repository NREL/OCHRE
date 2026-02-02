import unittest
import datetime as dt
import pandas as pd
import numpy as np

from ochre.Equipment import PV
from test.test_equipment import equip_init_args, start_time, duration


def create_pv_schedule():
    """Create a synthetic PV schedule for testing."""
    times = pd.date_range(start_time, start_time + duration, freq=dt.timedelta(minutes=15), inclusive="left")
    # Generate power: negative during sun hours (6am-6pm), zero at night
    hours = (times - start_time).total_seconds() / 3600
    pv_power = np.where(
        (hours >= 6) & (hours <= 18),
        -10 * np.sin(np.pi * (hours - 6) / 12),  # Peak at noon = -10 kW
        0,
    )
    return pd.DataFrame({"PV (kW)": pv_power}, index=times)


init_args = equip_init_args.copy()
init_args.update(
    {
        "capacity": 10,
        "tilt": 20,
        "azimuth": 180,
        "time_res": dt.timedelta(minutes=15),
        "schedule": create_pv_schedule(),
    }
)


class PVTestCase(unittest.TestCase):
    """
    Test Case to test PV Equipment.
    """

    def setUp(self):
        self.pv = PV(**init_args)

        # run until noon so power is not zero (48 * 15min = 12 hours)
        for _ in range(48):
            self.pv.update()

    def test_init(self):
        self.assertEqual(len(self.pv.schedule), 96)  # 96 points at 15-min resolution
        # Check noon power is approximately -10 kW (peak)
        noon_power = self.pv.schedule.loc[start_time + dt.timedelta(hours=12), "Power (kW)"]
        self.assertAlmostEqual(noon_power, -10, places=1)
        self.assertAlmostEqual(self.pv.inverter_min_pf_factor, 0.75, places=2)

    def test_update_external_control(self):
        # update_external_control now takes only one argument (control_signal)
        mode = self.pv.update_external_control({"P Setpoint": -5, "Q Setpoint": 1})
        self.assertEqual(mode, "On")
        self.assertAlmostEqual(self.pv.p_set_point, -5)
        self.assertAlmostEqual(self.pv.q_set_point, 1)

        # test setpoint larger than available power - should be clipped
        mode = self.pv.update_external_control({"P Setpoint": -20, "Q Setpoint": 1})
        self.assertEqual(mode, "On")
        # p_set_point should be max of scheduled power and setpoint
        self.assertAlmostEqual(self.pv.p_set_point, self.pv.electric_kw, places=2)
        self.assertAlmostEqual(self.pv.q_set_point, 1)

        # test PV curtailment in kW
        mode = self.pv.update_external_control({"P Curtailment (kW)": 1})
        self.assertEqual(mode, "On")
        # Curtailment reduces generation
        self.assertGreater(self.pv.p_set_point, -10)

        # test PV curtailment in %
        mode = self.pv.update_external_control({"P Curtailment (%)": 50})
        self.assertEqual(mode, "On")
        # 50% curtailment should roughly halve the power
        self.assertGreater(self.pv.p_set_point, -6)

        # test power factor
        mode = self.pv.update_external_control({"Power Factor": -0.95})
        self.assertEqual(mode, "On")
        # Power factor affects Q setpoint
        self.assertGreater(self.pv.q_set_point, 0)

        # test priority
        self.pv.update_external_control({"Priority": "CPF"})
        self.assertEqual(self.pv.inverter_priority, "CPF")

    def test_update_internal_control(self):
        # update_internal_control takes no args now
        mode = self.pv.update_internal_control()
        self.assertEqual(mode, "On")
        # Should set to max scheduled power (around -10 kW at noon)
        self.assertLess(self.pv.p_set_point, -5)
        self.assertAlmostEqual(self.pv.q_set_point, 0)

    def test_calculate_power_and_heat(self):
        # calculate_power_and_heat takes no args now
        self.pv.p_set_point = -5
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat()
        self.assertEqual(self.pv.electric_kw, -5)
        self.assertEqual(self.pv.reactive_kvar, 1)

        # var priority - when exceeding inverter capacity
        self.pv.p_set_point = -12
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat()
        self.assertAlmostEqual(self.pv.electric_kw, -((10**2 - 1**2) ** 0.5))
        self.assertEqual(self.pv.reactive_kvar, 1)

        # 0.8 min PF, var priority
        self.pv.p_set_point = -11
        self.pv.q_set_point = 11
        self.pv.calculate_power_and_heat()
        self.assertAlmostEqual(self.pv.electric_kw, -8)
        self.assertAlmostEqual(self.pv.reactive_kvar, 6, places=2)

        # watt priority
        self.pv.inverter_priority = "Watt"
        self.pv.p_set_point = -12
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat()
        self.assertAlmostEqual(self.pv.electric_kw, -10)
        self.assertAlmostEqual(self.pv.reactive_kvar, 0)

        # CPF priority
        self.pv.inverter_priority = "CPF"
        self.pv.p_set_point = -12
        self.pv.q_set_point = 1
        self.pv.calculate_power_and_heat()
        self.assertAlmostEqual(self.pv.electric_kw, -9.97, places=2)
        self.assertAlmostEqual(self.pv.reactive_kvar, 0.83, places=2)

    def test_generate_results(self):
        # generate_results takes no args now, uses self.verbosity
        self.pv.verbosity = 6
        results = self.pv.generate_results()
        self.assertIn("Time", results)
        self.assertIn("PV P Setpoint (kW)", results)


if __name__ == "__main__":
    unittest.main()
