import numpy as np

from ochre.Models import RCModel, ModelException
from ochre import Units

# Water Constants
water_density = 1000  # kg/m^3
water_density_liters = 1  # kg/L
water_cp = 4.183  # kJ/kg-K
water_conductivity = 0.6406  # W/m-K
water_c = water_cp * water_density_liters * 1000  # heat capacity with useful units: J/K-L

# Water draw types
WATER_DRAWS = ['Showers', 'Sinks', 'Baths', 'CW', 'DW']


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

    def __init__(self, water_nodes=12, water_vol_fractions=None, **kwargs):
        if water_vol_fractions is None:
            self.n_nodes = water_nodes
            self.vol_fractions = np.ones(self.n_nodes) / self.n_nodes
        else:
            self.n_nodes = len(water_vol_fractions)
            self.vol_fractions = np.array(water_vol_fractions) / sum(water_vol_fractions)

        self.volume = None  # in L
        super().__init__(**kwargs)
        self.t_amb_idx = self.input_names.index('T_AMB')
        self.t_1_idx = self.state_names.index('T_WH1')
        self.h_1_idx = self.input_names.index('H_WH1')
        self.next_states = self.states  # for holding state info for next time step

        # key variables for results
        self.draw_total = 0  # in L
        self.h_delivered = 0  # heat delivered in outlet water, in W
        self.h_injections = 0  # heat from water heater, in W
        self.h_loss = 0  # conduction heat loss from tank, in W
        self.h_unmet_shower = 0  # unmet load from lower outlet temperature, showers only, in W
        self.mains_temp = 0  # water mains temperature, in C
        self.outlet_temp = 0  # temperature of outlet water, in C

        # mixed temperature setpoints for tempered draws - separate for Sink/Shower/Bath (SSB) and for CW
        self.tempered_draw_temp = Units.F2C(kwargs.get('mixed delivery temperature (F)', 110))
        self.cw_draw_temp = Units.F2C(kwargs.get('clothes washer delivery temperature (F)', 92.5))

    def load_rc_data(self, **kwargs):
        # Get properties from input file
        r = kwargs['tank radius (m)'] if 'tank radius (m)' in kwargs else kwargs['Tank Radius (m)']  # in m
        h = kwargs['tank height (m)'] if 'tank height (m)' in kwargs else kwargs['Tank Height (m)']  # in m
        top_area = np.pi * r ** 2
        self.volume = top_area * h * 1000  # in L

        if 'Heat Transfer Coefficient (W/m^2/K)' in kwargs:
            u = kwargs['Heat Transfer Coefficient (W/m^2/K)']
        elif 'UA (W/K)' in kwargs:
            ua = kwargs['UA (W/K)']
            total_area = 2 * top_area + 2 * np.pi * r
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

    def load_initial_state(self, **kwargs):
        t_max = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        t_db = Units.deltaF2C(kwargs.get('deadband temperature (F)', 10))
        # temp = t_max - np.random.rand(1) * t_db

        # set initial temperature close to top of deadband
        temp = t_max - t_db / 10
        return super().load_initial_state(initial_states=temp)

    def update_water_draw(self, schedule, heats_to_model=None):
        if heats_to_model is None:
            heats_to_model = np.zeros(len(self.states))
        self.mains_temp = schedule.get('mains_temperature')
        self.outlet_temp = self.states[self.t_1_idx]  # initial outlet temp, for estimating draw volume

        draw_tempered = schedule.get('Sinks', 0) + schedule.get('Showers', 0) + schedule.get('Baths', 0)
        draw_cw = schedule.get('CW', 0)
        draw_hot = schedule.get('DW', 0)
        if not (draw_tempered + draw_cw + draw_hot):
            # No water draw
            self.draw_total = 0
            self.h_delivered = 0
            self.h_unmet_shower = 0
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
        if draw_cw:
            if self.outlet_temp <= self.cw_draw_temp:
                self.draw_total += draw_cw
            else:
                vol_ratio = (self.cw_draw_temp - self.mains_temp) / (self.outlet_temp - self.mains_temp)
                self.draw_total += draw_cw * vol_ratio

        t_s = self.time_res.total_seconds()
        draw_liters = self.draw_total * t_s / 60  # in liters
        draw_fraction = draw_liters / self.volume  # unitless

        if self.n_nodes == 2 and draw_fraction < self.vol_fractions[1]:
            # Use empirical factor for determining water flow by node
            flow_fraction = 0.95  # Totally empirical factor based on detailed lab validation
            if draw_fraction > self.vol_fractions[0]:
                # outlet temp is volume-weighted average of lower and upper temps
                self.outlet_temp = (self.states[0] * self.vol_fractions[0] + self.states[0] * (
                        draw_fraction - self.vol_fractions[0])) / draw_fraction
            q_delivered = draw_liters * water_c * (self.outlet_temp - self.mains_temp)  # in J

            # q_to_mains_upper = self.state_capacitances[0] * (self.x[0] - self.mains_temp)
            q_to_mains_lower = self.state_capacitances[1] * (self.states[1] - self.mains_temp)
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
                    q_nodes.append((t_end - t_start) * self.state_capacitances[i])
                q_nodes = np.array(q_nodes)

        # convert heat transfer from J to W
        self.h_delivered = q_delivered / t_s
        heats_to_model += q_nodes / t_s

        # calculate unmet loads, in W
        shower_draw = schedule.get('Showers', 0) / 60  # in L/sec
        self.h_unmet_shower = max(shower_draw * water_c * (self.tempered_draw_temp - self.outlet_temp), 0)

        return heats_to_model

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
        heat_check = np.dot(self.next_states - init_states, self.state_capacitances)  # in J
        if not abs(heat_check) < 1:
            raise ModelException(
                'Large error ({}) in water heater inversion mixing algorithm.'
                'Final state temperatures are: {}'.format(heat_check, self.next_states))

    def update(self, heats_to_model=None, schedule=None, check_bounds=True, **kwargs):
        # Note: heats_to_model is NOT self.inputs; it only includes the 'H_' heats, not the ambient temperature
        if heats_to_model is None:
            heats_to_model = np.zeros(self.n_nodes)
        self.h_injections = sum(heats_to_model)

        # update heat injections from water draw
        # FUTURE: revise CW and DW when event based schedules are added
        heats_to_model = self.update_water_draw(schedule, heats_to_model)

        # update water tank model
        # TODO: update with WH location
        water_inputs = np.insert(heats_to_model, self.t_amb_idx, schedule['Indoor'])
        self.next_states = super().update(water_inputs, return_states=True)

        q_change = np.dot(self.next_states - self.states, self.state_capacitances)  # in J
        h_change = q_change / self.time_res.total_seconds()

        # calculate heat loss, in W
        self.h_loss = self.h_injections - h_change - self.h_delivered
        if abs(self.h_loss) > 1000:
            raise ModelException('Error in calculating heat loss for {} model'.format(self.name))

        # If any temperatures are inverted, run inversion mixing algorithm
        delta_t = 0.1 if self.high_res else 0.01
        if any(np.diff(self.next_states) > delta_t):
            self.run_inversion_mixing_rule()

        # check that states are within reasonable range
        # Note: default max temp on water heater model is 60C (140F). Temps may exceed that slightly
        if check_bounds:
            if max(self.next_states) > 62 or min(self.next_states) < self.mains_temp - 3:
                print('WARNING: Water temperatures are outside acceptable range: {}'.format(self.next_states))
            if max(self.next_states) > 65 or min(self.next_states) < self.mains_temp - 5:
                raise ModelException('Water temperatures are outside acceptable range: {}'.format(self.next_states))

        # return the heat loss for envelope model
        return self.h_loss

    def generate_results(self, verbosity, to_ext=False):
        if to_ext:
            return {}
        else:
            results = {}
            if verbosity >= 3:
                results.update({'Hot Water Delivered (L/min)': self.draw_total,
                                'Hot Water Outlet Temperature (C)': self.outlet_temp,
                                'Hot Water Delivered (kW)': self.h_delivered / 1000,
                                'Hot Water Unmet Demand, Showers (kW)': self.h_unmet_shower / 1000,
                                })
            if verbosity >= 6:
                results.update({'Hot Water Heat Injected (kW)': self.h_injections / 1000,
                                'Hot Water Heat Loss (kW)': self.h_loss / 1000,
                                'Hot Water Average Temperature (C)': sum(self.states * self.vol_fractions),
                                'Hot Water Maximum Temperature (C)': max(self.states),
                                'Hot Water Minimum Temperature (C)': min(self.states),
                                'Hot Water Mains Temperature (C)': self.mains_temp,
                                })
            if verbosity >= 9:
                results.update(self.get_states())
                results.update(self.get_inputs())
            return results


class OneNodeWaterModel(StratifiedWaterModel):
    """
    1-node Water Tank Model
    """
    def __init__(self, water_nodes=None, **kwargs):
        super().__init__(water_nodes=1, **kwargs)


class TwoNodeWaterModel(StratifiedWaterModel):
    """
    2-node Water Tank Model

    - Partitions tank into 2 nodes
    - Top node is 1/3 of volume, Bottom node is 2/3
    """
    def __init__(self, water_nodes=None, **kwargs):
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

    def load_initial_state(self, **kwargs):
        # set temperature to upper threshold
        t_max = Units.F2C(kwargs.get('setpoint temperature (F)', 125))
        return RCModel.load_initial_state(self, initial_states=t_max)
