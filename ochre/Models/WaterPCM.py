import numpy as np
import os
import math

from ochre.Models import StratifiedWaterModel





# TODO: priorities: multi-node pcm properties
# TODO: interpolated cp values for non-linear enthalpies
# TODO: find out differences in draw profiles


# PCM properties from manufacturer, same units as water properties
DEFAULT_PCM_PROPERTIES = {
    "t_m1": 50,  # C
    "t_m2": 55,  # C
    "h_fus": 226,  # J/g
    "h": 600,  # W/m^2K
    "sa_ratio": 15, # m^2/m^3 of total pcm volume
    "h_conv": 100,  # W/K, accounts for surface area (ha)
    "setpoint_temp": 60,
    "solid": {
        "pcm_density": 0.904,  # g/cm**3
        "pcm_cp": 1.20,  # J/g-C # adjusted by real measurements average from 0-45c
        "pcm_conductivity": 0.28,  # W/m-C, not used
        # "pcm_c": 1717.6,  # J/m**3-C, not used
    },
    "liquid": {
        "pcm_density": 0.829,  # g/cm**3
        "pcm_cp": 1.33,  # J/g-C # adjusted by real measurements average from 55-100c
        "pcm_conductivity": 0.16,  # W/m-C, not used
        # "pcm_c": 1823.8,  # J/m**3-C, not used
    },
    "enthalpy_lut": np.loadtxt(os.path.join(os.path.dirname(__file__), "cp_h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1)
}

def calculate_interpolation_data(pcm_properties):
    # TODO: add in the enthalpy intorpolation values based on current temperature
    # short timesteps can use previous temperatures
    # investigate the timestep bounds where this starts getting wonky
    # temps = np.array([0, t_m1, t_m2, t_max])   # in C
    temps = pcm_properties['enthalpy_lut'][:,0]  # in C 

    specific_heats = pcm_properties['enthalpy_lut'][:,1]           # in J/g
    # adjusted enthalpy values from real data
    # [0, 76, 267, 326]
    # enthalpies = np.array([0.0, 76.0, 267.0, 326.0])          # in J/g
    enthalpies = pcm_properties["enthalpy_lut"][:,2]           # in J/g
    return temps, specific_heats, enthalpies




def get_pcm_enthalpy(t_pcm, pcm_properties):
    '''look up the enthalpy of the PCM using an interpolated LUT'''
    # col 1 : t_pcm (C), col 2 : cp (J/(gC)), col 3 : enthalpy (J/kg)
    # must be in ascending sorted order by t_pcm
    if t_pcm < pcm_properties["enthalpy_lut"][0,0] or t_pcm > pcm_properties["enthalpy_lut"][-1,0]:
        raise ValueError(f"t_pcm {t_pcm} is outside the range of the LUT [{pcm_properties['enthalpy_lut'][0,0]} to {pcm_properties['enthalpy_lut'][-1,0]}]")
    
    lut = pcm_properties["enthalpy_lut"]
    idx = (lut[:,0] <= t_pcm).nonzero()[0][-1]
    
    t_low, t_high = lut[idx:idx+2, 0]
    h_low, h_high = lut[idx:idx+2, 2]
    
    return h_low + (t_pcm - t_low) * (h_high - h_low) / (t_high - t_low)

class TankWithPCM(StratifiedWaterModel):
    """
    Water Tank Model with Phase Change Material

    Defaults to a 13 node tank with 12 water nodes and 1 PCM node.
    """

    name = "Water Tank with PCM"

    def __init__(self, pcm_properties=DEFAULT_PCM_PROPERTIES, pcm_water_node=5, pcm_vol_fraction=0.5, **kwargs):
        
        # PCM node data
        self.pcm_water_node = pcm_water_node  # node number, from the top
        self.t_pcm_wh_idx = self.pcm_water_node - 1
        self.pcm_vol_fraction = pcm_vol_fraction
        self.pcm_mass = None  # in g
        
        super().__init__(**kwargs)
        
        self.pcm_properties = pcm_properties

        # Bounds check for pcm_vol_fraction for stability
        if not (6.582730627258115e-08 <= self.pcm_vol_fraction <= 0.9999999999999725):
            raise ValueError(f"pcm_vol_fraction {pcm_vol_fraction} must be between (6.582730627258115e-08 and 0.9999999999999725) to ensure stability.")
        if self.pcm_vol_fraction < 0.01 or self.pcm_vol_fraction > 0.99:
            self.warn(f"pcm_vol_fraction {pcm_vol_fraction} is outside the recommended range (0.01 to 0.99). Results may be inaccurate.")
        
        self.key_temp, self.specific_heats, self.key_enthalpy = calculate_interpolation_data(**self.pcm_properties)
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
        self.pcm_mass = self.pcm_porperties["solid"]["pcm_density"] * pcm_volume / 1e3  # in g
        rc_params["C_PCM"] = self.pcm_porperties["solid"]["pcm_cp"] * self.pcm_mass  # in J/K
        # rc_params["C_PCM"] = PCM_PROPERTIES["solid"]["pcm_c"] * pcm_volume

        # Reduce water volume and vol_fractions from PCM volume
        self.volume -= pcm_volume
        self.vol_fractions[self.t_pcm_wh_idx] *= 1 - self.pcm_vol_fraction
        self.vol_fractions = self.vol_fractions / sum(self.vol_fractions)

        # Reduce water node capacitance
        rc_params[f"C_WH{self.pcm_water_node}"] *= 1 - self.pcm_vol_fraction

        # Add water-PCM resistance, default to solid state for now rc_params[f"R_PCM_WH{self.pcm_water_node}"] = 1 / PCM_PROPERTIES["h_conv"]
       
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
    
    
    

    
    
class TankWithMultiPCM(StratifiedWaterModel):
    """
    Water Tank Model with Phase Change Material in Multiple Nodes
    
    Input a dictionary of {pcm_node: pcm_vol_fractions} to select the nodes and vol_fractions the pcm will be in

    Defaults to a 15 node tank with 12 water nodes and 3 PCM node.
    """

    name = "Water Tank with Multi PCM"

    def __init__(self, pcm_properties: dict=DEFAULT_PCM_PROPERTIES, pcm_node_vol_fractions: dict[int:float]={4:0.5, 5:0.5, 6:0.5}, **kwargs):
        
        # PCM node data
        self.pcm_node_vol_fractions = pcm_node_vol_fractions  # node number, from the top
        self.pcm_water_nodes = list(self.pcm_node_vol_fractions.keys())
        self.t_pcm_wh_idx = [node -1 for node in self.pcm_water_nodes]
        self.pcm_vol_fraction = list(self.pcm_node_vol_fractions.values())
        self.pcm_mass = None  # in g
        self.external_nodes = ['AMB']
        self.pcm_heat_to_water_rc_network = None
        self.enthalpy_pcm = None
        self.pcm_properties = pcm_properties
        self.pcm_properties['enthalpy_lut'] = np.loadtxt(os.path.join(os.path.dirname(__file__), "cp_h-T_data_shifted_120F.csv"), delimiter=",", skiprows=1)
        
        super().__init__(**kwargs)

        # Bounds check for pcm_vol_fraction for stability
        # for node, vol_fraction in self.pcm_node_vol_fractions.items():
        #     if not (6.582730627258115e-08 <= vol_fraction <= 0.9999999999999725):
        #         raise ValueError(f"pcm_node: {node} vol_fraction {vol_fraction} must be between 6.582730627258115e-08 and 0.9999999999999725 to ensure stability.")
        #     if vol_fraction < 0.01 or vol_fraction> 0.85:
        #         self.warn(f"pcm_node: {node} pcm_vol_fraction {vol_fraction} is outside the recommended range (0.01 to 0.99). Results may be inaccurate.")
        self.h = self.pcm_properties['h']
        self.ha = self.pcm_properties['h_conv']
        self.conductivity= self.pcm_properties['solid']['pcm_conductivity']
        self.key_temp, self.key_specific_heats, self.key_enthalpy = calculate_interpolation_data(self.pcm_properties)
        self.key_enthalpy *= self.pcm_mass  # in J
        # self.time_res = datetime.timedelta(seconds=5)

        # PCM state and input indices
        self.t_pcm_idx = [i for i, name in enumerate(self.state_names) if "T_PCM" in name]
        self.h_pcm_idx = [i for i, name in enumerate(self.input_names) if "H_PCM" in name]
        assert [self.state_names.index(f"T_WH{node}") for node in self.pcm_water_nodes] == self.t_pcm_wh_idx
        self.h_pcm_wh_idx = [self.input_names.index(f"H_WH{node}") for node in self.pcm_water_nodes]
        
        # Iteration variables
        self.iter = 0
        self.max_iter = 600
        self.epsilon = 1e-3
        self.step_num = 0

        # PCM results variables
        self.pcm_heat_to_water = None  # in W
        t_pcm = self.states[self.t_pcm_idx]  # PCM temperature, in C
        self.enthalpy_pcm = np.interp(t_pcm, self.key_temp, self.key_enthalpy) # PCM enthalpy, in J

    def load_rc_data(self, **kwargs):
        rc_params = super().load_rc_data(**kwargs)
        self.rc_params = rc_params
        
        # Create a dictionary to store PCM mass, capacitance, and resitances for each node
        self.pcm_mass_dict = {}
        self.pcm_node_properties = {}
        c_pcm_dict = {}
        r_wh_pcm_dict = {}
        r_pcm_pcm_dict = {}
        r_pcm_amb_dict = {}
        total_pcm_volume = 0.0
        total_pcm_mass = 0.0
        original_volume = self.volume  # Keep the original volume for proper PCM volume computation
        
        # parameters to calcualte pcm-pcm conductivity resistance
        # water heater height 4ft (static)
        # backout radius from volume and height
        
        self.tank_height = 4 * 0.3048 # in m
        start_temp = kwargs.get('Start Temperature (C)', 51.666666666666686)
        self.volume_m3 = self.volume / 1e3 # convert liters to m3
        self.tank_radius = math.sqrt(self.volume_m3 / (math.pi * self.tank_height))
        effective_area_ratio = 0.8
        length = self.tank_height / 12 # length on 1 cell
        water_heater_cross_area = math.pi * self.tank_radius**2
        effective_area = effective_area_ratio * water_heater_cross_area

        # Loop over each PCM node and apply the modifications
        for i, node in enumerate(self.pcm_water_nodes):
            # Convert the 1-indexed node number to a 0-indexed index
            idx = node - 1
            
            # Get the PCM volume fraction for this node
            pcm_frac = self.pcm_node_vol_fractions[node]
            
            # Calculate the water volume present in this node (before PCM extraction)
            # volume is in Liters
            node_volume = original_volume * self.vol_fractions[idx]
            
            # Compute the PCM volume in this node
            pcm_volume = node_volume * pcm_frac
            total_pcm_volume += pcm_volume
            
            # Compute the PCM mass in this node (in grams) and store it
            pcm_mass = self.pcm_properties["solid"]["pcm_density"] * pcm_volume * 1e3 # ensure units are in g
            self.pcm_mass_dict[node] = pcm_mass
            total_pcm_mass += pcm_mass
            
            # Add PCM capacitance for this node (in J/K)
            pcm_cp = np.interp(start_temp, self.pcm_properties['enthalpy_lut'][:,0], self.pcm_properties['enthalpy_lut'][:,1])
            c_pcm_dict[f"C_PCM{node}"] = pcm_cp * pcm_mass
            # c_pcm_dict[f"C_PCM{node}"] = 1e-3
            
            # Reduce the water node capacitance for this node by the PCM fraction
            rc_params[f"C_WH{node}"] *= (1 - pcm_frac)
            
            # Update the water node’s volume fraction to remove the PCM volume fraction
            self.vol_fractions[idx] *= (1 - pcm_frac)
            
            # Add the water-PCM thermal resistance for this node (in K/W)
            ha = self.pcm_properties["h"] * pcm_volume * 1e-3 * self.pcm_properties["sa_ratio"] # in W/K
            r_wh_pcm_dict[f"R_PCM{node}_WH{node}"] = 1 / ha
            
            # use pure thermal conductivity for this
            if i < len(self.pcm_water_nodes) - 1:
                next_pcm_node = self.pcm_water_nodes[i + 1]
                if next_pcm_node - node == 1:  # Check if sequential
                    r_pcm_pcm_dict[f"R_PCM{node}_PCM{next_pcm_node}"] = length/(self.pcm_properties['solid']['pcm_conductivity'] * effective_area * pcm_frac) # L/(kA)
                    
            # add PCM-AMB thermal resistance for node (in K/W) Should be arbitrarily high
            r_pcm_amb_dict[f"R_PCM{node}_AMB"] = 200000
            self.pcm_node_properties[node] = {"volume[L]": pcm_volume, "ha": ha, "sa_ratio": self.pcm_properties["sa_ratio"], "h": self.pcm_properties["h"]}
            print(f"Node {node} pcm mass: {(pcm_mass)/1000:.3e} kg")        
        # extend the rc_params dictionary with the thermal capacitance and resitances
        rc_params.update(c_pcm_dict)
        rc_params.update(r_wh_pcm_dict)
        rc_params.update(r_pcm_amb_dict)
        rc_params.update(r_pcm_pcm_dict)
        
        # Subtract the total PCM volume from the global water volume
        # TODO - add in option to do external pcm volumes
        self.volume -= total_pcm_volume
        self.pcm_mass = total_pcm_mass
        print(f"Total PCM mass: {(self.pcm_mass/1000):.3e} kg")
        
        # Normalize the water volume fractions so that they sum to 1
        self.vol_fractions = self.vol_fractions / np.sum(self.vol_fractions)
        
        self.rc_params = rc_params
        return rc_params
    
    def update_rc_network(self, t_pcm, **kwargs):
        '''Get the dynamic specific heat for each of the pcm nodes and update the capacitance in the rc_network'''
        
        pcm_specific_heats = np.interp(t_pcm, self.key_temp, self.key_specific_heats)
        

        # Loop over each PCM node and apply the modifications
        for i, node in enumerate(self.pcm_water_nodes):
            
            pcm_node_mass = self.pcm_mass_dict[node]
            pcm_specific_heat = pcm_specific_heats[i]

            
            # Update the capacitance for this node (in J/K)
            self.rc_params[f"C_PCM{node}"] = pcm_specific_heat * pcm_node_mass
            # self.rc_params[f"C_PCM{node}"] = 1e-3
        return self.rc_params
        
    def update_state_space_model(self, **kwargs):
        
        pcm_temps = self.next_states[self.t_pcm_idx]
        dynamic_rc_params = self.update_rc_network(pcm_temps)
        all_cap = {name.upper().split('_')[1:][0]: val 
               for name, val in dynamic_rc_params.items() if name[0] == 'C'}
        all_res = {tuple(name.upper().split('_')[1:]): val 
                for name, val in dynamic_rc_params.items() if name[0] == 'R'}
        
        # You may need to re-identify internal/external nodes if not stored already
        internal_nodes = [node for node in all_cap.keys()]
        
        external_nodes = [node for node in self.external_nodes]  # assuming these were stored
        
        # Recompute the state-space matrices
        A_c, B_c = self.create_rc_matrices(all_cap, all_res, internal_nodes, external_nodes)
        
        # Update the model’s matrices
        self.A = A_c
        self.B = B_c
        self.capacitances = np.array(list(all_cap.values()))
        
        # Define state and input names
        state_names = ['T_' + node for node in internal_nodes]
        input_names = ['T_' + node for node in external_nodes] + ['H_' + node for node in internal_nodes]
        
        outputs = None
        matrices=(A_c, B_c)
        
        # if unused_inputs is not None:
        #     good_input_idx = [i for (i, name) in enumerate(input_names) if name not in unused_inputs]
        #     B_c = B_c[:, good_input_idx]
        #     input_names = [name for name in input_names if name not in unused_inputs]

        # initialize states based on matrices
        
        # Define states
        self.nx = len(self.states)
        # if isinstance(self.states, dict):
        #     self.states = np.array(list(self.states.values()), dtype=float)
        # else:
        #     self.states = np.zeros(self.nx, dtype=float)

        # Define inputs
        self.nu = len(input_names)
        # if isinstance(input_names, dict):
        #     self.inputs = np.array(list(input_names.values()), dtype=float)
        # else:
        #     self.inputs = np.zeros(self.nu, dtype=float)
        self.input_names = list(input_names)
        self.use_schedule_for_inputs = all([col in self.input_names for col in self.schedule.columns])
        self.inputs_init = self.inputs  # for saving values from update_inputs step
        
        # Define outputs
        if outputs is None:
            self.ny = self.nx
            self.output_names = self.state_names.copy()
        else:
            self.ny = len(outputs)
            self.output_names = outputs
        # self.outputs = np.zeros(self.ny, dtype=float)
        # self.next_outputs = self.outputs  # for saving outputs of next time step

        # Define continuous-time matrices
        self.A_c, self.B_c, self.C, self.D = self.create_matrices(matrices)

        # Update output values
        self.outputs = self.C.dot(self.states) + self.D.dot(self.inputs)

        # Reduce model order (i.e. number of states)
        self.reduced = False
        self.transformation_matrix = None
        if 'reduced_states' in kwargs or 'reduced_min_accuracy' in kwargs:
            self.reduce_model(update_discrete=False, **kwargs)

        # Create A, B discrete matrices
        self.A, self.B = self.to_discrete()


    def get_pcm_heat_xfer(self):
        # if convection coefficient changes by phase, add heat transfer here


        return np.array([0] * len(self.t_pcm_idx))

        # # calculate heat transfer (pcm to water)
        # t_water = self.states[self.t_pcm_wh_idx]
        # t_pcm = self.states[self.t_pcm_idx]
        # h_pcm = PCM_PROPERTIES["h_conv"] * (t_pcm - t_water)  # in W
        # return h_pcml
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

    # def update_model(self, control_signal=None):
    #     super().update_model(control_signal)

    #     # calculate new PCM enthalpy based on linear temperature change
    #     delta_t = self.next_states[self.t_pcm_idx] - self.states[self.t_pcm_idx] 
    #     q_pcm = delta_t * self.capacitances[self.t_pcm_idx] # J
    #     self.pcm_heat_to_water_rc_network = -q_pcm/self.time_res.total_seconds()
    #     self.enthalpy_pcm += q_pcm

    #     # update PCM temperature in new_states
    #     t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
    #     self.next_states[self.t_pcm_idx] = t_pcm
    #     self.update_state_space_model()
        
    def update_model(self, control_signal=None, epsilon=None, max_iter=None, iter_count=0,
                    original_states=None, original_inputs=None, original_inputs_init=None):
        # Use provided convergence criteria or fall back to instance attributes.
        # if epsilon is None:
        #     epsilon = self.epsilon
        # if max_iter is None:
        #     max_iter = self.max_iter

        # # On the first call, store copies of the original states, inputs, and inputs_init.
        # if original_states is None:
        #     original_states = self.states.copy()
        # if original_inputs is None:
        #     original_inputs = self.inputs.copy()
        # if original_inputs_init is None:
        #     original_inputs_init = self.inputs_init.copy()

        # Call the parent's update_model function.
        super().update_model(control_signal)
        
        # Only use temperatures output from state space model and enthalpies nothing else
        enthalpy_state = np.interp(self.states[self.t_pcm_idx], self.key_temp, self.key_enthalpy)
        enthalpy_next_state = np.interp(self.next_states[self.t_pcm_idx], self.key_temp, self.key_enthalpy)
        q_pcm = enthalpy_next_state - enthalpy_state
        self.pcm_heat_to_water_rc_network = -q_pcm / self.time_res.total_seconds()
        self.enthalpy_pcm = enthalpy_next_state
        
        

        # Update the PCM temperature using interpolation from enthalpy to temperature.
        # t_pcm = np.interp(self.enthalpy_pcm, self.key_enthalpy, self.key_temp)
        # t_pcm = self.next_states[self.t_pcm_idx]
        # self.next_states[self.t_pcm_idx] = t_pcm

        # Call the state space model update (which may modify states, inputs, and inputs_init).
        self.update_state_space_model()

        # Reset the states and inputs to their original values so they stay constant between iterations.
        # self.states = original_states.copy()
        # self.inputs = original_inputs.copy()
        # self.inputs_init = original_inputs_init.copy()

        # # Check convergence: if the maximum change is smaller than epsilon or max iterations reached, stop.
        # if np.max(np.abs(delta_t)) < epsilon or iter_count > max_iter:
        #     self.step_num += 1
        #     print(f"Convergence reached after {iter_count + 1} iterations: ΔT = {np.max(np.abs(delta_t)):.6f} [{self.step_num}/{self.sim_times.size}]")
        #     return
        # else:
        #     # Update the current state for the next iteration.
        #     self.states[self.t_pcm_idx] = t_pcm

        #     # Recursively call update_model with the original copies maintained.
        #     self.update_model(control_signal, epsilon, max_iter, iter_count + 1,
        #                     original_states, original_inputs, original_inputs_init)
        
    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 6:
            
            for i, idx in enumerate(self.t_pcm_idx):
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Temperature (C)"] = self.states[idx]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Water Temperature (C)"] = self.states[self.t_pcm_wh_idx[i]]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Enthalpy (J)"] = self.enthalpy_pcm[i]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Heat Injected (W)"] = self.pcm_heat_to_water_rc_network[i]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} Capacitance (J/K)"] = self.capacitances[idx]
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} h (W/m^2K)"] = self.pcm_node_properties[self.t_pcm_wh_idx[i]+1]['h']
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} volume (L)"] = self.pcm_node_properties[self.t_pcm_wh_idx[i]+1]['volume[L]']
                results[f"Water Tank PCM{self.t_pcm_wh_idx[i]+1} sa_ratio"] = self.pcm_node_properties[self.t_pcm_wh_idx[i]+1]['sa_ratio']
                
            results['Total PCM Enthalpy (J)'] = self.enthalpy_pcm.sum()
            results['Total PCM Heat Injected (W)'] = self.pcm_heat_to_water_rc_network.sum()
            results['PCM Mass (kg)'] = self.pcm_mass * 1e-3
            results['Water Volume (L)'] = self.volume

            
        return results