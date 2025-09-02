import numpy as np

from ochre.Models import RCModel, ModelException
from ochre.utils import convert

# Water Constants
# TODO: add any constants from flow_mixing_example.py here
water_density = 1000  # kg/m^3
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_conductivity = 0.6406  # W/m-K
water_c = water_cp * water_density_liters * 1000  # heat capacity with useful units: J/K-L


class StratifiedWaterModel(RCModel):
    """
    Stratified Water Tank RC Thermal Model

    - Partitions a water tank into n nodes (12 by default).
    - Nodes can have different volumes, but are equal volume by default.
    - Node 1 is at the top of the tank (at outlet).
    - State names are [T_WH1, T_WH2, ...], length n
    - Input names are [T_AMB, H_WH1, H_WH2, ...], length n+1
    - The model can accept 2 additional inputs for water draw:
      - draw: volume of water to deliver
      - draw_tempered: volume to deliver at setpoint temperature.
      If tank temperature is higher than setpoint, the model assumes mixing with water mains.
    - The model considers the following effects on temperature at each time step:
      - Internal (node-to-node) conduction
      - External (node-to-ambient) conduction
      - Heat injections due to water heater
      - Heat injections due to water draw (calculated before the state-space update)
      - Heat transfer due to inversion mixing (assumes fast mixing, calculated after the state-space update)
    - At each time step, the model calculates:
      - The internal states (temperatures)
      - The heat delivered to the load (relative to mains temperature)
      - The heat lost to ambient air
    """
    name = 'Water Tank'
    optional_inputs = [
        'Water Heating (L/min)',
        'Clothes Washer (L/min)',
        'Dishwasher (L/min)',
        'Mains Temperature (C)',
        'Zone Temperature (C)',
    ]

    def __init__(self, water_nodes=12, water_vol_fractions=None, include_flow_mixing=False, **kwargs):
        if water_vol_fractions is None:
            self.n_nodes = water_nodes
            self.vol_fractions = np.ones(self.n_nodes) / self.n_nodes
        else:
            self.n_nodes = len(water_vol_fractions)
            self.vol_fractions = np.array(water_vol_fractions) / sum(water_vol_fractions)

        self.volume = kwargs['Tank Volume (L)']  # in L

        super().__init__(external_nodes=['AMB'], **kwargs)
        self.next_states = self.states  # for holding state info for next time step

        self.t_amb_idx = self.input_names.index('T_AMB')
        assert self.t_amb_idx == 0  # should always be first
        self.t_1_idx = self.state_names.index('T_WH1')
        self.h_1_idx = self.input_names.index('H_WH1')
        
        if include_flow_mixing:
            # TODO: may need to modify based on what parameters are needed
            self.flow_mixing_params = self.calculate_flow_mixing_params(**kwargs)
        else:
            self.flow_mixing_params = None

        # key variables for results
        self.draw_total = 0  # in L
        self.h_delivered = 0  # heat delivered in outlet water, in W
        self.h_injections = 0  # heat from water heater, in W
        self.h_loss = 0  # conduction heat loss from tank, in W
        self.h_unmet_load = 0  # unmet load from outlet temperature, fixtures only, in W
        self.mains_temp = 0  # water mains temperature, in C
        self.outlet_temp = 0  # temperature of outlet water, in C

        # mixed temperature (i.e. target temperature) setpoint for fixtures - Sink/Shower/Bath (SSB)
        self.tempered_draw_temp = kwargs.get('Mixed Delivery Temperature (C)', convert(105, 'degF', 'degC'))
        # Removing target temperature for clothes washers
        # self.washer_draw_temp = kwargs.get('Clothes Washer Delivery Temperature (C)', convert(92.5, 'degF', 'degC'))

    def load_rc_data(self, **kwargs):
        # Get properties from input file
        h = kwargs['Tank Height (m)']  # in m
        top_area = self.volume / h / 1000  # in m^2
        r = (top_area / np.pi) ** 0.5

        if 'Heat Transfer Coefficient (W/m^2/K)' in kwargs:
            u = kwargs['Heat Transfer Coefficient (W/m^2/K)']
        elif 'UA (W/K)' in kwargs:
            ua = kwargs['UA (W/K)']
            total_area = 2 * top_area + 2 * np.pi * r * h
            u = ua / total_area
        else:
            raise ModelException('Missing heat transfer coefficient (UA) for {}'.format(self.name))

        # calculate general RC parameters for whole tank
        c_water_tot = self.volume * water_c  # Heat capacity of water (J/K)
        r_int = (h / self.n_nodes) / water_conductivity / top_area  # R between nodes (K/W)
        r_side_tot = 1 / u / (2 * np.pi * r * h)  # R from side of tank (K/W)
        r_top = 1 / u / top_area  # R from top/bottom of tank (K/W)

        # Capacitance per node
        rc_params = {'C_WH' + str(i + 1): c_water_tot * frac for i, frac in enumerate(self.vol_fractions)}

        # Resistance to exterior from side, top, and bottom
        rc_params.update({'R_WH{}_AMB'.format(i + 1): r_side_tot / frac for i, frac in enumerate(self.vol_fractions)})
        rc_params['R_WH1_AMB'] = self.par(rc_params['R_WH1_AMB'], r_top)
        rc_params['R_WH{}_AMB'.format(self.n_nodes)] = self.par(rc_params['R_WH{}_AMB'.format(self.n_nodes)], r_top)

        # Resistance between nodes
        if self.n_nodes > 1:
            rc_params.update({'R_WH{}_WH{}'.format(i + 1, i + 2): r_int for i in range(self.n_nodes - 1)})

        return rc_params

    @staticmethod
    def initialize_state(state_names, input_names, A_c, B_c, **kwargs):
        t_init = kwargs.get('Initial Temperature (C)')
        if t_init is None:
            t_max = kwargs.get('Setpoint Temperature (C)', convert(125, 'degF', 'degC'))
            t_db = kwargs.get('Deadband Temperature (C)', convert(10, 'degR', 'K'))
            # temp = t_max - np.random.rand(1) * t_db

            # set initial temperature close to top of deadband
            t_init = t_max - t_db / 10

        # Return states as a dictionary
        return {name: t_init for name in state_names}

    @staticmethod
    def calculate_flow_mixing_params(**kwargs):
        # TODO: copy initialization code from flow_mixing_example.py here
        pass

    def update_water_draw(self):
        heats_to_model = np.zeros(self.nx)
        self.mains_temp = self.current_schedule.get('Mains Temperature (C)')
        self.outlet_temp = self.states[self.t_1_idx]  # initial outlet temp, for estimating draw volume

        # Note: removing target draw temperature for clothes washers, not implemented in ResStock
        draw_tempered = self.current_schedule.get('Water Heating (L/min)', 0)
        draw_hot = (self.current_schedule.get('Clothes Washer (L/min)', 0)
                    + self.current_schedule.get('Dishwasher (L/min)', 0))
        # draw_cw = self.current_schedule.get('Clothes Washer (L/min)', 0)
        # draw_hot = self.current_schedule.get('Dishwasher (L/min)', 0)
        if not (draw_tempered + draw_hot):
            # No water draw
            self.draw_total = 0
            self.h_delivered = 0
            self.h_unmet_load = 0
            return heats_to_model

        if self.mains_temp is None:
            raise ModelException('Mains temperature required when water draw exists')

        # calculate total draw volume from tempered draw volume(s)
        # for tempered draw, assume outlet temperature == T1, slightly off if the water draw is very large
        self.draw_total = draw_hot
        if draw_tempered:
            if self.outlet_temp <= self.tempered_draw_temp:
                self.draw_total += draw_tempered
            else:
                vol_ratio = (self.tempered_draw_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
                self.draw_total += draw_tempered * vol_ratio
        # if draw_cw:
        #     if self.outlet_temp <= self.washer_draw_temp:
        #         self.draw_total += draw_cw
        #     else:
        #         vol_ratio = (self.washer_draw_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
        #         self.draw_total += draw_cw * vol_ratio

        t_s = self.time_res.total_seconds()
        draw_liters = self.draw_total * t_s / 60  # in liters
        draw_fraction = draw_liters / self.volume  # unitless

        if self.n_nodes == 2 and draw_fraction < self.vol_fractions[1]:
            # Use empirical factor for determining water flow by node
            flow_fraction = 0.95  # Totally empirical factor based on detailed lab validation
            if draw_fraction > self.vol_fractions[0]:
                # outlet temp is volume-weighted average of lower and upper temps
                self.outlet_temp = (self.states[0] * self.vol_fractions[0] +
                                    self.states[1] * (draw_fraction - self.vol_fractions[0])) / draw_fraction
            q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J

            # q_to_mains_upper = self.state_capacitances[0] * (self.x[0] - self.mains_temp)
            q_to_mains_lower = self.capacitances[1] * (self.states[1] - self.mains_temp)
            if q_delivered * flow_fraction > q_to_mains_lower:
                # If you'd fully cool the bottom node to mains, set bottom node to mains and cool top node
                q_nodes = np.array([q_to_mains_lower - q_delivered, -q_to_mains_lower])
            else:
                q_nodes = np.array([-q_delivered * (1 - flow_fraction), -q_delivered * flow_fraction])

        else:
            if draw_fraction < min(self.vol_fractions):
                # water draw is smaller than all node volumes
                q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J
                # all volume transfers are from the node directly below
                q_nodes = draw_liters * water_c * np.diff(self.states, append=self.mains_temp)  # in J
            else:
                # calculate volume transfers to/from each node, including q_delivered
                vols_pre = np.append(self.vol_fractions, draw_fraction).cumsum()
                vols_post = np.insert(self.vol_fractions, 0, draw_fraction).cumsum()
                temps = np.append(self.states, self.mains_temp)

                # update outlet temp as a weighted average of temps, by volume
                vols_delivered = np.diff(vols_pre.clip(max=draw_fraction), prepend=0)
                self.outlet_temp = np.dot(temps, vols_delivered) / draw_fraction
                q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J

                # calculate heat in/out of each node (in J)
                q_nodes = []
                for i in range(self.n_nodes):
                    t_start = temps[i]
                    vols_delivered = np.diff(vols_pre.clip(min=vols_post[i], max=vols_post[i + 1]),
                                             prepend=vols_post[i])
                    t_end = np.dot(temps, vols_delivered) / self.vol_fractions[i]
                    q_nodes.append((t_end - t_start) * self.capacitances[i])
                q_nodes = np.array(q_nodes)

        # convert heat transfer from J to W
        self.h_delivered = q_delivered / t_s
        heats_to_model += q_nodes / t_s

        if self.flow_mixing_params is not None:
            # TODO: update heats_to_model using code from
            # flow_mixing_example.py:calculate_heat_transfers
            # - can use self.states for the tank temperatures
            # - can use self.draw_total for the flow rate
            pass

        # calculate unmet loads, fixtures only, in W
        self.h_unmet_load = max(draw_tempered / 60 * water_c * (self.tempered_draw_temp - self.outlet_temp), 0)  # in W

        return heats_to_model

    def update_inputs(self, schedule_inputs=None):
        # Note: self.inputs_init are not updated here, only self.current_schedule
        super().update_inputs(schedule_inputs)

        # get zone temperature from schedule
        t_zone = self.current_schedule['Zone Temperature (C)']

        # update heat injections from water draw
        # FUTURE: revise CW and DW when event based schedules are added
        heats_to_model = self.update_water_draw()

        # update water tank model
        self.inputs_init = np.concatenate(([t_zone], heats_to_model))

    def run_inversion_mixing_rule(self):
        # Inversion Mixing Rule
        # See https://energyplus.net/sites/all/modules/custom/nrel_custom/pdfs/pdfs_v9.1.0/EngineeringReference.pdf
        #     p. 1528
        # Starting from the top, check for mixing at each node

        init_states = self.next_states.copy()
        for node_idx in range(self.n_nodes - 1):
            current_temp = self.next_states[node_idx]

            # new temp is the max of any possible mixings
            heats = self.next_states * self.vol_fractions  # note: excluding c_p and volume factors
            heat_sums = heats[node_idx:].cumsum()
            vol_sums = self.vol_fractions[node_idx:].cumsum()
            new_temp = (heat_sums / vol_sums).max()

            # Allow inversion mixing if a significant difference in temperature exists
            if new_temp > current_temp + 0.001:  # small computational errors are possible
                # print('Inversion mixing occuring at node {}. Temperature raises from {} to {}'.format(
                #     node + 1, self.x[node], new_temp))

                # calculate heat transfer, update temperatures of current node and node below
                q = (new_temp - current_temp) * self.vol_fractions[node_idx]
                self.next_states[node_idx] = new_temp
                self.next_states[node_idx + 1] -= q / self.vol_fractions[node_idx + 1]

                if not any(np.diff(self.next_states) > 0.1):
                    # no more inversions
                    return

            elif new_temp < current_temp - 0.001:  # small computational errors are possible:
                msg = 'Error in inversion mixing algorithm. ' \
                      'New temperature ({}) less than previous ({}) at node {}.'
                raise ModelException(msg.format(new_temp, self.next_states[node_idx], node_idx + 1))

        # check final heat to ensure no losses from mixing
        heat_check = np.dot(self.next_states - init_states, self.capacitances)  # in J
        if not abs(heat_check) < 1:
            raise ModelException(
                'Large error ({}) in water heater inversion mixing algorithm.'
                'Final state temperatures are: {}'.format(heat_check, self.next_states))

    def update_model(self, control_signal=None):
        if control_signal is not None:
            # control signal must be heat injections from water heater, by node
            assert isinstance(control_signal, np.ndarray) and len(control_signal) == self.nx
            self.h_injections = sum(control_signal)
            control_signal = self.inputs_init + np.insert(control_signal, 0, 0)  # adds heat injections in inputs_init
        else:
            self.h_injections = 0

        super().update_model(control_signal)

        q_change = (self.next_states - self.states).dot(self.capacitances)  # in J
        h_change = q_change / self.time_res.total_seconds()

        # calculate heat loss, in W
        self.h_loss = self.h_injections - h_change - self.h_delivered
        if abs(self.h_loss) > 1000:
            raise ModelException('Error in calculating heat loss for {} model'.format(self.name))

        # If any temperatures are inverted, run inversion mixing algorithm
        delta_t = 0.1 if self.high_res else 0.01
        if any(np.diff(self.next_states) > delta_t):
            self.run_inversion_mixing_rule()

    def update_results(self):
        current_results = super().update_results()

        # check that states are within reasonable range
        # Note: default max temp on water heater model is 60C (140F). Temps may exceed that slightly
        if max(self.states) > 62 or min(self.states) < self.mains_temp - 10:
            if max(self.states) > 65 or min(self.states) < self.mains_temp - 15:
                raise ModelException(f'Water temperatures are outside acceptable range: {self.states}')
            else:
                self.warn(f'Water temperatures are outside acceptable range: {self.states}')

        return current_results

    def generate_results(self):
        # Note: most results are included in Dwelling/WH. Only inputs and states are saved to self.results
        results = super().generate_results()

        if self.verbosity >= 3:
            results['Hot Water Unmet Demand (kW)'] = self.h_unmet_load / 1000
            results['Hot Water Outlet Temperature (C)'] = self.outlet_temp
        if self.verbosity >= 4:
            results['Hot Water Delivered (L/min)'] = self.draw_total
            results['Hot Water Delivered (W)'] = self.h_delivered
        if self.verbosity >= 7:
            results['Hot Water Heat Injected (W)'] = self.h_injections
            results['Hot Water Heat Loss (W)'] = self.h_loss
            results['Hot Water Average Temperature (C)'] = self.states.dot(self.vol_fractions)
            results['Hot Water Maximum Temperature (C)'] = self.states.max()
            results['Hot Water Minimum Temperature (C)'] = self.states.min()
            results['Hot Water Mains Temperature (C)'] = self.mains_temp
        return results


class OneNodeWaterModel(StratifiedWaterModel):
    """
    1-node Water Tank Model
    """

    def __init__(self, **kwargs):
        kwargs.pop('water_nodes', None)
        super().__init__(water_nodes=1, **kwargs)


class TwoNodeWaterModel(StratifiedWaterModel):
    """
    2-node Water Tank Model

    - Partitions tank into 2 nodes
    - Top node is 1/3 of volume, Bottom node is 2/3
    """

    def __init__(self, **kwargs):
        kwargs.pop('water_nodes', None)
        super().__init__(water_nodes=2, water_vol_fractions=[1 / 3, 2 / 3], **kwargs)


class IdealWaterModel(OneNodeWaterModel):
    """
    Ideal water tank with near-perfect insulation. Used for TanklessWaterHeater. Modeled as 1-node tank.
    """

    def load_rc_data(self, **kwargs):
        # ignore RC parameters from the properties file
        self.volume = 1000
        return {'R_WH1_AMB': 1e6,
                'C_WH1': self.volume * water_c}

    @staticmethod
    def initialize_state(state_names, input_names, A_c, B_c, **kwargs):
        # set temperature to upper threshold
        t_max = kwargs.get('Setpoint Temperature (C)', convert(125, 'degF', 'degC'))

        # Return states as a dictionary
        return {name: t_max for name in state_names}
