# $Date: 2015-02-02 12:16:03 -0700 (Mon, 02 Feb 2015) $
# $Rev: 5061 $
# $Author: jmaguire $
# $HeadURL: https://cbr.nrel.gov/beopt2/svn/trunk/Modeling/units.py $

"""
There is a function for each unit convention with a docstring more
explicitly explaining the conversion performed.

This can be used like so:
import units
y = units.ft2m(x)
"""

import os
import csv
import math


class FuelConversionsDict(dict):
    '''
    A FuelConversionsDict instance is used to perform Btu <-> gal
    unit conversions for liquid fuels like oil and propane.
    '''

    def __init__(self):
        # Determine gal/Btu conversions for fuel types
        super().__init__()
        self._fuel_conversions = {}

        fueltypescsv = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'FuelTypes.csv')
        with open(fueltypescsv, "rb") as ifile:
            reader = csv.DictReader(ifile)

            for ft in reader:
                try:
                    self[ft['Fuel Name'].lower()] = float(ft['Btu/gal Conversion Factor'])
                except:
                    self[ft['Fuel Name'].lower()] = None


# Conversion factors are variables in the form of [unit converting
# from]2[unit converting to] Since '/' and '^' are operators, quotients
# are denoted with '_' and powers are ignored. e.g. a conversion from
# lbm/ft^3 to kg/m^3 is written as 'lbm_ft32kg_m3'

# Angle
def rad2deg(rad):
    """radians -> degrees"""
    return math.degrees(rad)


def deg2rad(deg):
    """degrees -> radians"""
    return math.radians(deg)


# Trig functions in degrees
def sin_deg(x):
    '''Compute sine of x where x is in degrees.'''
    return math.sin(deg2rad(x))


def cos_deg(x):
    '''Compute cosine of x where x is in degrees.'''
    return math.cos(deg2rad(x))


def tan_deg(x):
    '''Compute tangent of x where x is in degrees.'''
    return math.tan(deg2rad(x))


def asin_deg(x):
    '''Compute inverse sine of x in degrees.'''
    return rad2deg(math.asin(x))


def acos_deg(x):
    '''Compute inverse cosine of x in degrees.'''
    return rad2deg(math.acos(x))


def atan_deg(x):
    '''Compute inverse tangent of x in degrees.'''
    return rad2deg(math.atan(x))


# Time
def yr2day(yr):
    """yr -> day"""
    return yr * 365.


def day2yr(day):
    """day -> yr"""
    return day / 365


def hr2min(hr):
    """hr -> min"""
    return hr * 60.


def min2sec(mins):
    """min -> sec"""
    return mins * 60.


def hr2sec(hr):
    """hr -> sec"""
    return min2sec(hr2min(hr))


def day2hr(day):
    """day -> hr"""
    return day * 24.


def yr2hr(yr):
    """year -> hour"""
    return day2hr(yr2day(yr))


# Distance
def ft2m(ft):
    """feet -> meters"""
    return 0.3048 * ft


def m2ft(meters):
    """meters -> feet"""
    return meters / 0.3048


def in2ft(inches):
    """inches -> feet"""
    return inches / 12.


def ft2in(feet):
    """feet -> inches"""
    return feet * 12.


def ft22in2(ftsq):
    """feet -> inches"""
    return ftsq * 144.


def in2m(inches):
    """inches -> meters"""
    return inches * 0.0254


def m2in(meters):
    """meters -> inches"""
    return meters / 0.0254


def mm2m(millimeters):
    """millimeters -> meters"""
    return millimeters / 1000


# Area
def ft22cm2(ftsq):
    """ft^2 -> cm^2"""
    return 929.0304 * ftsq


def cm22ft2(cmsq):
    """cm^2 -> ft^2"""
    return cmsq / 929.0304


def ft22m2(ftsq):
    """ft^2 -> m^2"""
    return ftsq * 0.09290304


def m22ft2(msq):
    """m^2 -> ft^2"""
    return msq / 0.09290304


# Volume
def ft32gal(ft3):
    """ft^3 -> gal"""
    return 7.4805195 * ft3


def ft32liter(ft3):
    """ft^3 -> liter"""
    return 28.316846 * ft3


def gal2ft3(gal):
    """gal -> ft^3"""
    return 0.13368056 * gal


def gal2in3(gal):
    """gal -> in^3"""
    return 231. * gal


def in32gal(in3):
    """in^3 -> gal"""
    return 0.0043290043 * in3


def gal2m3(gal):
    """gal -> m^3"""
    return 0.0037854118 * gal


def m32gal(m3):
    """m^3 -> gal"""
    return 264.17205 * m3


def ft32m3(ft3):
    """ft^3 -> m^3"""
    return 0.028316847 * ft3


def m32ft3(m3):
    """m^3 -> ft^3"""
    return 35.314667 * m3


def pint2liter(pints):
    """pint -> liter"""
    return 0.47317647 * pints


def liter2pint(liters):
    """liter -> pint"""
    return 2.1133764 * liters


# Velocity
def mph2m_s(mph):
    """mph -> m/s"""
    return 0.44704 * mph


def knots2m_s(knots):
    """knots -> m/s"""
    return 0.51444444 * knots


def m_s2knots(m_s):
    """m/s -> knots"""
    return 1.9438445 * m_s


# Volume flow rate
def gpm2m3_s(gpm):
    """gpm -> m^3/s"""
    return 6.3090196e-05 * gpm


def gpm2m3_h(gpm):
    """gpm -> m^3/hr"""
    return 0.2271247056 * gpm


def gpm2cfm(gpm):
    """gpm -> cfm"""
    return 0.1337 * gpm


def cfm2gpm(cfm):
    """cfm -> gpm"""
    return cfm / 0.1337


def cfm2m3_s(cfm):
    """cfm -> m^3/s"""
    return 0.00047194744 * cfm


def cfm2ft3_h(cfm):
    """cfm -> ft^3/h"""
    return 60 * cfm


def m3_hr2cfm(m3_hr):
    """m^3/hr -> cfm"""
    return 0.58857778 * m3_hr


def m3_hr2gpm(m3_hr):
    """m^3/hr -> gpm"""
    return cfm2gpm(m3_hr2cfm(m3_hr))


def m3_s2cfm(m3_s):
    """m^3/s -> cfm"""
    return 2118.88 * m3_s


# Mass
def lbm2kg(lbm):
    """lbm -> kg"""
    return 0.45359237 * lbm


def kg2lbm(kg):
    """kg -> lbm"""
    return kg / 0.45359237


# Mass flow rate
def lbm_min2kg_hr(lbm_min):
    """lbm/min -> kg/hr"""
    return 27.215542 * lbm_min


def lbm_min2kg_s(lbm_min):
    """lbm/min -> kg/s"""
    return lbm_min2kg_hr(lbm_min) / 3600.  # /hr2min/min2sec


# Humidity ratio
def lbm_lbm2grains(lbm_lbm):
    """lbm/lbm -> grains"""
    return lbm_lbm * 7000.


# Temperature
def deltaF2C(degF):
    """delta degF -> delta degC"""
    return R2K(degF)


def deltaC2F(degC):
    """delta degC -> delta degF"""
    return K2R(degC)


def R2K(degR):
    """degR -> degK"""
    return degR / 1.8


def K2R(degK):
    """degK -> degR"""
    return degK * 1.8


def F2C(degF):
    """degF -> degC"""
    return (degF - 32.) / 1.8


def C2F(degC):
    """degC -> degF"""
    return 1.8 * degC + 32.


def F2R(degF):
    """degF -> degR"""
    return degF + 459.67


def R2F(degR):
    """degR -> degF"""
    return degR - 459.67


def C2K(degC):
    """degC -> degK"""
    return degC + 273.15


def K2C(degK):
    """degK -> degC"""
    return degK - 273.15


# Thermal Conductivity
def Btu_hftR2W_mK(x):
    """Btu/(hr-ft-R) -> W/(m-K)"""
    return 1.731 * x


def Btu_hftR2kJ_hmK(x):
    """Btu/(hr-ft-R) -> kJ/(h-m-K)"""
    return 6.231 * x


def Btuin_hft2R2W_mK(x):
    """Btu-in/(h-ft^2-R) -> W/(m-K)"""
    return Btu_hftR2W_mK(in2ft(x))


def Btuin_hft2R2kJ_hmK(x):
    """Btu-in/(hr-ft^2-R) -> kJ/(h-m-K)"""
    return Btu_hftR2kJ_hmK(in2ft(x))


# U-value (Thermal Conductance)
def Btu_hft2F2kJ_hm2C(x):
    """Btu/(h-ft^2-F) -> kJ/(h-m^2-C)"""
    return 20.44 * x


def Btu_hft2F2W_m2K(x):
    """Btu/(h-ft^2-F) -> W/(m^2-K)"""
    return 5.678 * x


def W_m2K2Btu_hft2F(x):
    """W/(m^2-K) -> Btu/(h-ft^2-F)"""
    return x / 5.678


# UA
def Btu_hF2W_K(x):
    """Btu/(h-F) -> W/K"""
    return 0.5275 * x


# Density
def lbm_ft32kg_m3(x):
    """lbm/ft^3 -> kg/m^3"""
    return 16.02 * x


def kg_m32lbm_ft3(x):
    """kg/m^3 -> lbm/ft^3"""
    return x / 16.02


def lbm_ft32inH2O_mph2(x):  # yep, we use it
    """lbm/ft^3 -> inH2O/mph^2"""
    return 0.01285 * x


# Dynamic viscosity of Fluid (mu)
def lbm_hft2Ns_m2(x):
    """lbm/ft-h -> N-s/m^2"""
    return 0.00041337887 * x


# Enthalpy
def J_kg2Btu_lb(x):
    """J/kg -> Btu/lb"""
    return x / 2326.0


def Btu_lb2J_kg(x):
    """Btu/lb -> J/kg"""
    return x * 2326.0


# Specific Heat
def Btu_lbR2J_kgK(x):  # By mass
    """Btu/(lbm-R) -> J/(kg-K)"""
    return 4187. * x


def Btu_lbR2kJ_kgK(x):  # By mass
    """Btu/(lbm-R) -> kJ/(kg-K)"""
    return 4.187 * x


def Btu_ft3F2J_m3K(x):  # By volume
    """Btu/(ft^3-F) -> J/(m^3-K)"""
    return 67100. * x


# R-value (Thermal Resistance)
def hft2F_Btu2m2K_W(x):
    """h-ft^2-F/Btu (aka R-value) -> m^2-K/W"""
    return 0.1761 * x


def hft2F_Btu2hm2K_kJ(x):
    """h-ft^2-F/Btu (aka R-value) -> h-m^2-K/kJ"""
    return 0.04892 * x


# Energy
def Btu2kWh(x):
    """Btu -> kWh"""
    return x / kWh2Btu(1)


def MBtu2kWh(x):
    """MBtu -> kWh"""
    return x / kWh2MBtu(1)


def Btu2Wh(x):
    """Btu -> Wh"""
    return Btu2kWh(x) * 1000


def therms2kWh(x):
    """therms -> kWh"""
    return 29.3 * x


def kWh2therms(x):
    """kWh -> therms"""
    return x / 29.3


def MBtu2therms(x):
    """MBtu -> therms"""
    return kWh2therms(MBtu2kWh(x))


def W2kBtu_h(x):
    """W -> kBtu/h"""
    return Wh2Btu(x) / 1000


def kWh2Btu(x):
    """kWh -> Btu"""
    return 3412. * x


def kWh2MBtu(x):
    """kWh -> MBtu"""
    return Btu2MBtu(kWh2Btu(x))


def Btu2MBtu(x):
    """Btu -> MBtu"""
    return 1.e-6 * x


def MBtu2Btu(x):
    """MBtu -> Btu"""
    return 1.e6 * x


def Btu2therm(x):
    """Btu -> therm"""
    return 1.e-5 * x


def kWh2kJ(x):
    """kWh -> kJ or Wh -> J"""
    return 3600. * x


def therms2kJ(x):
    """therms -> kJ"""
    return 105506. * x


def therms2MBtu(x):
    """therms -> MBtu"""
    return kWh2MBtu(therms2kWh(x))


def Wh2kJ(x):
    """Wh -> kJ"""
    return 3.6 * x


def J2kWh(x):
    """J -> kWh"""
    return x / 1000. / 3600.


def kJ2kWh(x):
    """kJ -> kWh or J -> Wh"""
    return x / 3600.


def J2Btu(x):
    """J -> Btu"""
    return x * 3412. / 1000. / 3600.


def J2kBtu(x):
    """J -> Btu"""
    return J2Btu(x) / 1000


def J2MBtu(x):
    """J -> MBtu"""
    return J2Btu(x) / 1000000


def GJ2MBtu(x):
    """GJ -> MBtu"""
    return J2MBtu(x * 1000000000)


def Btu2J(x):
    """Btu -> J"""
    return x / J2Btu(1.)


def MBtu2J(x):
    """MBtu -> J"""
    return Btu2J(MBtu2Btu(x))


def kJ2Btu(x):
    """kJ -> Btu"""
    return J2Btu(1000. * x)


def Wh2Btu(x):
    """Wh -> Btu"""
    return kWh2Btu(x) / 1000


def Btu2kJ(x):
    """Btu -> kJ"""
    return x / kJ2Btu(1.)


def gal2Btu(x, fueltype, fuel_conversions_dict=None):
    """gal (fuel) -> Btu

    Note: If this function is being called repeatedly, it will be faster to pass
    in a FuelConversionDict object rather than having a new one created each time
    on the fly.
    """

    if fuel_conversions_dict is None:
        fuel_conversions_dict = FuelConversionsDict()

    if fueltype.lower() in [ft.lower() for ft in fuel_conversions_dict.keys()]:
        factor = fuel_conversions_dict[fueltype.lower()]
        return x * factor
    raise ValueError("Unexpected fuel type: " + fueltype)


def Btu2gal(x, fueltype, fuel_conversions_dict=None):
    """Btu -> gal (fuel)

    Note: If this function is being called repeatedly, it will be faster to pass
    in a FuelConversionDict object rather than having a new one created each time
    on the fly.
    """

    if fuel_conversions_dict is None:
        fuel_conversions_dict = FuelConversionsDict()

    if fueltype.lower() in [ft.lower() for ft in fuel_conversions_dict.keys()]:
        factor = fuel_conversions_dict[fueltype.lower()]
        return x / factor
    raise ValueError("Unexpected fuel type: " + fueltype)


def m32Btu(x, fueltype, fuel_conversions_dict=None):
    """m3 (fuel) -> Btu"""
    return gal2Btu(m32gal(x), fueltype, fuel_conversions_dict)


def Btu2m3(x, fueltype, fuel_conversions_dict=None):
    """Btu -> m3 (fuel)"""
    return gal2m3(Btu2gal(x, fueltype, fuel_conversions_dict))


# Power
def Btu_h2W(x):
    """Btu/h -> W"""
    return 0.2931 * x


def Btu_h2kW(x):
    """Btu/h -> W"""
    return 0.0002931 * x


def kW2W(x):
    """kW -> W"""
    return 1000. * x


def W2kW(x):
    """W -> kW"""
    return x / 1000


def kW2Btu_h(x):
    """kW -> Btu/h"""
    return 3412. * x


def kW2kBtu_h(x):
    """kW -> kBtu/h"""
    return Btu_h2kBtu_h(kW2Btu_h(x))


def W2Btu_h(x):
    """W -> Btu/h"""
    return Wh2Btu(x)


def Btu_h2kJ_h(x):
    """Btu/h -> kJ/h"""
    return Btu2kJ(x)


def kJ_h2W(x):
    """kJ/h -> W"""
    return x / 3.6


def Ton2W(x):
    """Ton (of cooling) -> W"""
    return 3517.2 * x


def Ton2Btu_h(x):
    """Ton (of cooling) -> Btu/h"""
    return 12000. * x


def Ton2kBtu_h(x):
    """Ton (of cooling) -> Btu/h"""
    return 12. * x


def Btu_h2Ton(x):
    """Btu/h -> Ton (of cooling)"""
    return x / Ton2Btu_h(1.)


def W2Ton(x):
    """W -> Ton (of cooling)"""
    return x / Ton2W(1.)


def Btu_h2MBtu_h(x):
    """Btu/h -> MBtu/h"""
    return 1.e-6 * x


def Btu_h2kBtu_h(x):
    """Btu/h -> kBtu/h"""
    return 1.e-3 * x


def kBtu_h2MBtu_h(x):
    """kBtu/h -> MBtu/h"""
    return 1.e-3 * x


def kBtu_h2Btu_h(x):
    """kBtu/h -> Btu/h"""
    return 1000. * x


def kBtu_h2W(x):
    """kBtu/h -> W"""
    return Btu_h2W(kBtu_h2Btu_h(x))


def kBtu_h2kW(x):
    """kBtu/h -> kW"""
    return Btu_h2kW(kBtu_h2Btu_h(x))


# Power Flux
def W_m22Btu_ft2(x):
    """W/m^2 -> Btu/h/ft^2"""
    # the units indicated in the function name are wrong!
    # I put the correct units in the docstring.
    return kWh2Btu(x) / 1000. * ft22m2(1)


# Pressure
def lbm_fts22inH2O(x):
    """lbm/(ft-s^2) -> inH2O"""
    return 0.005974 * x


def atm2Btu_ft3(x):
    """atm -> Btu/ft^3"""
    return 2.719 * x


def atm2Pa(x):
    """atm -> Btu/ft^3"""
    return 101325. * x


def inH2O2Pa(x):
    """inH2O -> Pa"""
    return 249.1 * x


def Pa2inH2O(x):
    """Pa -> inH2O"""
    return x / 249.1


def inHg2atm(x):
    """inHg -> atm"""
    return 0.0334172218 * x


def Pa2atm(x):
    """Pa -> atm"""
    return x / 101325.


def Pa2kPa(x):
    '''Pa -> kPa'''
    return x / 1000


def psi2Btu_ft3(x):
    '''psi -> Btu/ft3'''
    return x * 0.185


def psi2kPa(x):
    '''psi -> kPa'''
    return x * 6.895


def kPa2psi(x):
    '''kPa -> psi'''
    return x / 6.895


def atm2psi(x):
    '''atm -> psi'''
    return x * 14.692


def atm2kPa(x):
    '''atm -> kPa'''
    return x * 101.325


# Stack Coefficient
def inH2O_R2Pa_K(x):
    """inH2O/R -> Pa/K"""
    return 448.4 * x


def ft2_s2R2L2_s2cm4K(x):
    """ft^2/(s^2-R) -> L^2/(s^2-cm^4-K)"""
    return 0.001672 * x


# Wind Coefficient
def inH2O_mph2Pas2_m2(x):
    """inH2O/mph^2 -> Pa-s^2/m^2"""
    return 1246. * x


def _2L2s2_s2cm4m2(x):
    """I don't know what this means. I just copied it directly out of Global.bmi"""
    return 0.01 * x


# Amount
def mol2lbmol(x):
    """mol -> lbmol"""
    return x / 453.592
