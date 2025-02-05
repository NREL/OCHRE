import numpy as np

from ochre.Models import StratifiedWaterModel

# PCM properties from manufacturer, same units as water properties
PCM_PROPERTIES = {
    "t_m1": 50,  # C
    "t_m2": 55,  # C
    "h_fus": 226,  # J/g
    "h_conv": 1000,  # W/K, accounts for surface area
    "solid": {
        "pcm_density": 0.904,  # g/m**3 -> this should be g/cm**3
        "pcm_cp": 1.9,  # J/g-C
        # "pcm_conductivity": 0.28,  # W/m-C, not used
        # "pcm_c": 1717.6,  # J/m**3-C, not used
    },
    "liquid": {
        "pcm_density": 0.829,  # g/m**3 -> this should be g/cm**3
        "pcm_cp": 2.2,  # J/g-C
        # "pcm_conductivity": 0.16,  # W/m-C, not used
        # "pcm_c": 1823.8,  # J/m**3-C, not used
    },
}

def calculate_interpolation_data(t_m1, t_m2, h_fus, solid, liquid, t_max=100, **kwargs):
    temps = np.array([0, t_m1, t_m2, t_max])  # in C

    h_1 = solid["pcm_cp"] * t_m1
    h_2 = h_1 + h_fus
    h_max = h_2 + liquid["pcm_cp"] * (t_max - t_m2)
    enthalpies = np.array([0, h_1, h_2, h_max])  # in J/g

    return temps, enthalpies


class TankWithPCM(StratifiedWaterModel):
    """
    Water Tank Model with Phase Change Material

    Defaults to a 13 node tank with 12 water nodes and 1 PCM node.
    """

    name = "Water Tank with PCM"

    def __init__(self, pcm_water_node=5, pcm_vol_fraction=0.5, **kwargs):
        
        # PCM node data
        self.pcm_water_node = pcm_water_node  # node number, from the top
        self.t_pcm_wh_idx = self.pcm_water_node - 1
        self.pcm_vol_fraction = pcm_vol_fraction
        self.pcm_mass = None  # in g
        
        super().__init__(**kwargs)

        # Bounds check for pcm_vol_fraction for stability
        if not (6.582730627258115e-08 <= self.pcm_vol_fraction <= 0.9999999999999725):
            raise ValueError(f"pcm_vol_fraction {pcm_vol_fraction} must be between 6.582730627258115e-08 and 0.9999999999999725 to ensure stability.")
        if self.pcm_vol_fraction < 0.01 or self.pcm_vol_fraction > 0.99:
            self.warn(f"pcm_vol_fraction {pcm_vol_fraction} is outside the recommended range (0.01 to 0.99). Results may be inaccurate.")
        
        self.key_temp, self.key_enthalpy = calculate_interpolation_data(**PCM_PROPERTIES)
        self.key_enthalpy *= self.pcm_mass  # in J

        # PCM state and input indices
        self.t_pcm_idx = self.state_names.index("T_PCM")
        self.h_pcm_idx = self.input_names.index("H_PCM")
        assert self.state_names.index(f"T_WH{self.pcm_water_node}") == self.t_pcm_wh_idx
        self.h_pcm_wh_idx = self.input_names.index(f"H_WH{self.pcm_water_node}")

        # PCM results variables
        self.pcm_heat_to_water = None  # in W
        t_pcm = self.states[self.t_pcm_idx]  # PCM temperature, in C
        self.enthalpy_pcm = np.interp(t_pcm, self.key_temp, self.key_enthalpy)  # PCM enthalpy, in J

    def load_rc_data(self, **kwargs):
        rc_params = super().load_rc_data(**kwargs)
#
        # Add PCM capacitance, default to solid state for now
        pcm_node_vol_fraction = self.vol_fractions[self.t_pcm_wh_idx]
        pcm_volume = self.volume * pcm_node_vol_fraction * self.pcm_vol_fraction
        self.pcm_mass = PCM_PROPERTIES["solid"]["pcm_density"] * pcm_volume / 1e3  # in g
        rc_params["C_PCM"] = PCM_PROPERTIES["solid"]["pcm_cp"] * self.pcm_mass  # in J/K
        # rc_params["C_PCM"] = PCM_PROPERTIES["solid"]["pcm_c"] * pcm_volume

        # Reduce water volume and vol_fractions from PCM volume
        self.volume -= pcm_volume
        self.vol_fractions[self.t_pcm_wh_idx] *= 1 - self.pcm_vol_fraction
        self.vol_fractions = self.vol_fractions / sum(self.vol_fractions)

        # Reduce water node capacitance
        rc_params[f"C_WH{self.pcm_water_node}"] *= 1 - self.pcm_vol_fraction

        # Add water-PCM resistance, default to solid state for now
        rc_params[f"R_PCM_WH{self.pcm_water_node}"] = 1 / PCM_PROPERTIES["h_conv"]
#
        return rc_params

    def get_pcm_heat_xfer(self):
        # if convection coefficient changes by phase, add heat transfer here

        # # calculate heat transfer (pcm to water)
        # t_water = self.states[self.t_pcm_wh_idx]
        # t_pcm = self.states[self.t_pcm_idx]
        # h_pcm = PCM_PROPERTIES["h_conv"] * (t_pcm - t_water)  # in W
        # return h_pcm
        return 0 # keep zero for now to prevent double counting pcm heat transfer


    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)

        # get heat injections from PCM
        self.pcm_heat_to_water = self.get_pcm_heat_xfer()

        # add PCM heat to inputs       
        self.inputs_init = np.append(self.inputs_init, -self.pcm_heat_to_water) # fix heat flow direction
        # self.inputs_init[self.h_pcm_idx] = self.pcm_heat_to_water
        self.inputs_init[self.h_pcm_wh_idx] += self.pcm_heat_to_water # fix heat flow direction

    def update_model(self, control_signal=None):
        super().update_model(control_signal)

        # calculate new PCM enthalpy based on linear temperature change
        delta_t = self.next_states[self.t_pcm_idx] - self.states[self.t_pcm_idx]
        q_pcm = delta_t * self.capacitances[self.t_pcm_idx]
        self.enthalpy_pcm += q_pcm

        # update PCM temperature in new_states
        t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
        self.next_states[self.t_pcm_idx] = t_pcm

    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 6:
            results["Water Tank PCM Temperature (C)"] = self.states[self.t_pcm_idx]
            results["Water Tank PCM Water Temperature (C)"] = self.states[self.t_pcm_wh_idx]
            results["Water Tank PCM Enthalpy (J)"] = self.enthalpy_pcm
            results["Water Tank PCM Heat Injected (W)"] = self.pcm_heat_to_water
        return results
