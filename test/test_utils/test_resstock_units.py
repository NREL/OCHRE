"""Unit tests verifying pint-based conversions used in resstock.py."""

import unittest

from ochre.utils.units import convert
from ochre.utils.resstock import convert_units


class TestPintConversions(unittest.TestCase):
    """Verify pint produces correct conversion factors for all unit pairs."""

    def test_kw_to_kbtu_hr(self):
        result = convert(1, "kW", "kBtu/hr")
        self.assertAlmostEqual(result, 3.412141633, places=5)

    def test_therm_to_kbtu(self):
        result = convert(1, "therm", "kBtu")
        self.assertAlmostEqual(result, 100.0, places=5)

    def test_therm_to_kwh(self):
        result = convert(1, "therm", "kWh")
        self.assertAlmostEqual(result, 29.3071, places=3)

    def test_degc_to_degf_freezing(self):
        result = convert(0, "degC", "degF")
        self.assertAlmostEqual(result, 32.0, places=5)

    def test_degc_to_degf_boiling(self):
        result = convert(100, "degC", "degF")
        self.assertAlmostEqual(result, 212.0, places=5)

    def test_degc_to_degf_negative(self):
        result = convert(-40, "degC", "degF")
        self.assertAlmostEqual(result, -40.0, places=5)

    def test_delta_degc_to_delta_degf(self):
        # Used in resstock.py for pandas-compatible temperature conversion
        result = convert(1, "delta_degC", "delta_degF")
        self.assertAlmostEqual(result, 1.8, places=5)

    def test_m3s_to_cfm(self):
        result = convert(1, "m^3/s", "cubic_feet/min")
        self.assertAlmostEqual(result, 2118.88, places=1)

    def test_kwh_to_mbtu(self):
        result = convert(1, "kWh", "MBtu")
        self.assertAlmostEqual(result, 0.003412141633, places=9)

    def test_kbtu_to_mbtu(self):
        result = convert(1, "kBtu", "MBtu")
        self.assertAlmostEqual(result, 0.001, places=9)


class TestConvertUnits(unittest.TestCase):
    """Integration tests for the convert_units function."""

    def test_kw_to_kwh(self):
        # 1 kW for 0.25 hours = 0.25 kWh
        self.assertAlmostEqual(convert_units(1.0, "kW", "kWh", 0.25), 0.25)

    def test_w_to_kwh(self):
        # 1000 W for 1 hour = 1 kWh
        self.assertAlmostEqual(convert_units(1000.0, "W", "kWh", 1.0), 1.0)

    def test_w_to_kbtu(self):
        # 1000 W for 1 hour = 1 kWh = 3.412 kBtu
        result = convert_units(1000.0, "W", "kBtu", 1.0)
        self.assertAlmostEqual(result, 3.412141, places=3)

    def test_therms_per_hour_to_kbtu(self):
        # 1 therm/hour for 1 hour = 100 kBtu
        self.assertAlmostEqual(convert_units(1.0, "therms/hour", "kBtu", 1.0), 100.0, places=3)

    def test_therms_per_hour_to_kwh(self):
        result = convert_units(1.0, "therms/hour", "kWh", 1.0)
        self.assertAlmostEqual(result, 29.3071, places=3)

    def test_c_to_f(self):
        self.assertAlmostEqual(convert_units(0, "C", "F"), 32.0, places=5)
        self.assertAlmostEqual(convert_units(100, "C", "F"), 212.0, places=5)

    def test_m3s_to_cfm(self):
        result = convert_units(1.0, "m^3/s", "cfm")
        self.assertAlmostEqual(result, 2118.88, places=1)

    def test_kwh_to_mbtu(self):
        result = convert_units(1.0, "kWh", "MBtu")
        self.assertAlmostEqual(result, 0.003412, places=5)

    def test_kbtu_to_mbtu(self):
        self.assertAlmostEqual(convert_units(1.0, "kBtu", "MBtu"), 0.001, places=9)

    def test_same_unit_passthrough(self):
        self.assertEqual(convert_units(42.0, "kW", "kW"), 42.0)

    def test_empty_unit_passthrough(self):
        self.assertEqual(convert_units(42.0, "", "kW"), 42.0)
        self.assertEqual(convert_units(42.0, "kW", ""), 42.0)


if __name__ == "__main__":
    unittest.main()
