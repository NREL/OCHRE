import math
from pint import UnitRegistry

# Uses Pint package to convert units
ureg = UnitRegistry()
Q_ = ureg.Quantity

# Add "unitless" units to convert percentages
ureg.define('unitless = [no_unit]')
ureg.define('percent = unitless / 100 = percentage')


def convert(value, old_unit, new_unit):
    if value is None:
        return None
    return (Q_(value, old_unit)).to(Q_(new_unit)).magnitude


def pitch2deg(pitch):
    radian = math.atan(pitch / 12)
    return convert(radian, 'rad', 'deg')


# Useful conversions for faster parsing
degC_to_K = convert(0, 'degC', 'K')
kwh_to_therms = convert(1, 'kWh', 'therms')
cfm_to_m3s = convert(1, 'cubic_feet/min', 'm^3/s')
