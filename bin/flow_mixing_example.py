import numpy as np

# Code to implement flow-rate mixing for water heaters
# based on model from https://www.mdpi.com/1996-1073/14/9/2611

# constants (come from water tank model)
tank_volume = None
tank_nodes = None
vol_fractions = np.ones(tank_nodes) / tank_nodes  # uniform volume fractions
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_c = water_cp * water_density_liters * 1000  # heat capacity with useful units: J/K-L


def calculate_a_b():
    # Calculate a and b from paper
    pass

def calculate_heat_transfers(flow_rate, tank_temps: np.ndarray):
    # Calculate Reynolds number (Re) and Richardson number (Ri)

    # Calculate Courant number (is this necessary?)

    # get temperature differences
    delta_t = tank_temps[1:] - tank_temps[:-1]

    # Get temperature change based on flow rate mixing (2nd term in eqn 5)
    t_change = None

    # Calculate change in heat per node based on flow rate mixing
    h_change = t_change * vol_fractions * tank_volume * water_c  # in J
    return h_change
