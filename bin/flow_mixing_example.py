import numpy as np
from ochre.utils import convert

# Code to implement flow-rate mixing for water heaters
# based on model from https://www.mdpi.com/1996-1073/14/9/2611

# constants (come from water tank model)
tank_volume = None
tank_nodes = None
tank_height = 1.22  # FIXME: should be ['Tank Height (m)']
vol_fractions = np.ones(tank_nodes) / tank_nodes  # uniform volume fractions
water_density_liters = 1  # kg/L
water_dynamic_viscosity = 0.0005568  # kg/(m-s), at 49 C
water_kinematic_viscosity = water_dynamic_viscosity / (water_density_liters * 1000)  # m^2/s
water_cp = 4.183  # kJ/kg-K
g = 9.81  # m/s^2, gravitational constant
water_beta = 0.000298  # 1/C, thermal expansion coefficient at 49 C
water_c = water_cp * water_density_liters * 1000  # heat capacity with useful units: J/K-L

#Dip tube parameters
dip_tube_diffuser = "Nonperforated" # "Nonperforated", "Helical", "Slit-perforated"
dip_tube_od = 0.05  # m, outer diameter of the dip tube
dip_tube_id = 0.04  # m, inner diameter of the dip tube
dip_tube_th = 0.002  # m, thickness of the blocked section of the dip tube
dip_tube_h = 0.95 * convert(tank_height, 'ft', 'm') #Assumed
def calculate_a_b(): #a and b from https://www.sciencedirect.com/science/article/abs/pii/S0735193320303663
    # Calculate a and b from paper
    pass

def calculate_heat_transfers(flow_rate, tank_temps: np.ndarray,dip_tube_diffuser,dip_tube_od,dip_tube_id,dip_tube_th):
    # Calculate Reynolds number (Re) and Richardson number (Ri)
    if dip_tube_diffuser == "Nonperforated":
        d_hydraulic = dip_tube_od
    elif dip_tube_diffuser == "Helical":
        d_hydraulic = 4 * ((np.pi * dip_tube_od ** 2) / (4 - dip_tube_th * dip_tube_od)) / (np.pi * dip_tube_od - 2 * dip_tube_th + 2 * dip_tube_od)
    elif dip_tube_diffuser == "Slit-perforated":
        d_hydraulic = 4 * ((np.pi * dip_tube_id ** 2 / 4) - 3 * (dip_tube_od - dip_tube_id) * dip_tube_th) / (np.pi * dip_tube_od)

    Re = (water_density_liters * 1000) * flow_rate * d_hydraulic / water_dynamic_viscosity  # dimensionless
    Gr = g * water_beta * abs(T_mains - T_tank) * d_hydraulic ** 3 / water_kinematic_viscosity ** 2  # dimensionless
    Ri = Gr / Re ** 2  # dimensionless
    water_density_liters

    #Calculate eddy diffusivity factor

    # Calculate Courant number (is this necessary?)

    

    # get temperature differences
    delta_t = tank_temps[1:] - tank_temps[:-1]

    # Get temperature change based on flow rate mixing (2nd term in eqn 5)
    t_change = None

    # Calculate change in heat per node based on flow rate mixing
    h_change = t_change * vol_fractions * tank_volume * water_c  # in J
    return h_change
