# -*- coding: utf-8 -*-

import numpy as np
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
import rainflow

from . import Generator
from ochre import Units
from ochre.Models import RCModel

CONTROL_TYPES = ['Schedule', 'Self-Consumption', 'Off']


class BatteryThermalModel(RCModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.capacitance = kwargs['rc_params']['C_BATT']  # in J/K
        self.next_states = self.states


class Battery(Generator):
    name = 'Battery'
    end_use = 'Battery'
    allow_consumption = True
    is_gas = False

    def __init__(self, enable_degradation=True, **kwargs):
        super().__init__(**kwargs)

        # Battery electrical parameters
        self.capacity_kwh = self.parameters['capacity_kwh']  # in kWh, instantaneous capacity
        self.soc = self.parameters['soc_init']  # Initial State of Charge
        self.soc_max = self.parameters['soc_max']
        self.soc_min = self.parameters['soc_min']
        self.discharge_pct = self.parameters['discharge_pct']  # Self-discharge rate (% per day)
        self.n_series = self.parameters['n_series']  # Number of cells in series
        self.r_cell = self.parameters['r_cell']  # Cell resistance in ohms (for voltage calculation only)
        self.initial_voltage = self.parameters['initial_voltage']  # Initial open circuit voltage

        # Create thermal model, for now, assume battery is in conditioned space
        self.t_idx = 0
        if self.zone is not None:
            self.thermal_model = BatteryThermalModel(rc_params={'R_BATT_AMB': self.parameters['thermal_r'],
                                                                'C_BATT': self.parameters['thermal_c']},
                                                     initial_states=kwargs['initial_schedule']['Indoor'], **kwargs)
            # check that it is a 1 node model
            assert len(self.thermal_model.states) == 1
        else:
            self.thermal_model = None

        # Degradation model
        self.capacity_kwh_nominal = self.capacity_kwh  # starts at rated capacity, reduces from degradation
        self.degradation_data = []
        if enable_degradation:
            self.degradation_states = (0, 0, 0)
            self.degradation_curves = self.initialize_parameters('degradation_curves.csv', name_col='SOC', val_col=None)
        else:
            self.degradation_states = None
            self.degradation_curves = None

    def reset_time(self):
        super().reset_time()

        # reset degradation states and capacity
        if self.degradation_states is not None:
            self.degradation_states = (0, 0, 0)
            self.capacity_kwh = self.parameters['capacity_kwh']
            self.capacity_kwh_nominal = self.capacity_kwh

    def update_internal_control(self, schedule):
        super().update_internal_control(schedule)

        if (self.control_type == 'Self-Consumption' and self.parameters.get('charge_from_solar', False) and
                self.power_setpoint > 0):
            # Force charge from solar
            if 'pv_power' not in schedule:
                self.warn('Cannot run Self-Consumption control without PV power')
                self.power_setpoint = 0
            else:
                pv_power = schedule.get('pv_power', 0)
                self.power_setpoint = min(self.power_setpoint, -pv_power)

        # Update setpoint if SOC limits are reached
        if self.power_setpoint > 0 and self.soc >= self.soc_max:
            self.power_setpoint = 0
        if self.power_setpoint < 0 and self.soc <= self.soc_min:
            self.power_setpoint = 0

        return 'On' if self.power_setpoint != 0 else 'Off'

    def get_power_limits(self):
        # returns min (discharge) and max (charge) output power limits based on capacity and SOC
        max_discharge, max_charge = super().get_power_limits()

        hours = self.time_res.total_seconds() / 3600
        max_charge = min((self.soc_max - self.soc) * self.capacity_kwh / hours / self.efficiency_charge, max_charge)
        max_discharge = min((self.soc - self.soc_min) * self.capacity_kwh / hours * self.efficiency, -max_discharge)
        return -max_discharge, max_charge

    def calculate_power_and_heat(self, schedule):
        # run degradation algorithm once per day
        if self.degradation_states is not None and self.current_time.time() == dt.time(0, 0):
            self.calculate_degradation()

        # update instantaneous capacity (d0) based on battery temperature
        t_ref = 298.15  # K, = 25 degC
        R = 8.31446  # J / K / mol
        e_ad1 = 4126  # J / mol
        e_ad2 = 9.752e6 ** 0.5  # J / mol  # TODO: causing error in d0 calculation
        if self.thermal_model is not None:
            t_batt = Units.C2K(self.thermal_model.states[self.t_idx])
            self.capacity_kwh = self.capacity_kwh_nominal * np.exp(-e_ad1 / R * (1 / t_batt - 1 / t_ref) -
                                                                   (e_ad2 / R) ** 2 * (1 / t_batt - 1 / t_ref) ** 2)
        else:
            self.capacity_kwh = self.capacity_kwh_nominal

        super().calculate_power_and_heat(schedule)

        if self.thermal_model is not None:
            t_batt = self.thermal_model.states[self.t_idx]
            inputs = {'T_AMB': schedule['Indoor'],
                      'H_BATT': self.sensible_gain}
            self.thermal_model.next_states = self.thermal_model.update(inputs, return_states=True)

            # recalculate sensible gains to space using heat conservation
            delta_t = self.thermal_model.next_states[self.t_idx] - t_batt  # change in battery temperature
            delta_h = self.thermal_model.capacitance * delta_t / self.time_res.total_seconds()  # in W
            self.sensible_gain -= delta_h
        else:
            t_batt = 25  # reference temp

        # append SOC and temperature to degradation data
        if self.degradation_states is not None:
            self.degradation_data.append((self.soc, t_batt))

    def calculate_degradation(self):
        # Calculates battery capacity degradation using Li-limited capacity (Q_Li) due to aging
        # for details, see section 3.A in https://ieeexplore.ieee.org/document/7963578

        if len(self.degradation_data) <= 1:
            return

        t_ref = 298.15  # K, = 25 degC
        v_ref = 3.7  # V
        u_ref = 0.08  # V
        F = 96485  # 96485.33  # A s / mol
        R = 8.314  # 8.31446  # J / K / mol

        e_ab1 = 35392  # J / mol
        e_ab2 = -42800  # J / mol
        e_ab3 = 42800  # J / mol
        b0 = 1  # 1.07  # unitless  # TODO: Matlab value varies
        b1_ref = 3.503e-3  # days ^ -0.5
        b2_ref = 1.541e-5  # cycles ^ -1
        b3_ref = 2.805e-2  # unitless
        alpha_b1 = -1  # unitless
        alpha_b3 = 0.0066  # unitless
        beta_b1 = 2.157  # unitless
        gamma_b1 = 2.472  # unitless
        tau_b3 = 5  # days
        theta = -0.135  # unitless  # TODO: might be positive? using value from Matlab, not paper

        # run rainflow algorithm
        df = pd.DataFrame(self.degradation_data, columns=['soc', 'temp'])
        df['soc'] = df['soc'].clip(0, 1)
        df['temp'] = Units.C2K(df['temp'])
        cycles = rainflow.extract_cycles(df['soc'])
        cycles = pd.DataFrame(cycles, columns=['dsoc', 'avg_soc', 'ncycle', 'start', 'end'])
        cycles['avg_temp'] = [df['temp'].iloc[start: end].mean() for start, end in zip(cycles['start'], cycles['end'])]
        deg_time = len(df) * self.time_res.total_seconds() / 3600 / 24  # days since last degradation update
        max_dod = cycles['dsoc'].max()

        # interpolate SOC to get v_oc and u_neg
        v_oc = interp1d(self.degradation_curves.index, self.degradation_curves['V_oc'])(df['soc'])
        u_neg = interp1d(self.degradation_curves.index, self.degradation_curves['U_neg'])(df['soc'])

        # t_batt = Units.C2K(self.thermal_model.states[self.t_idx])  # in K
        # time = (self.start_time - self.current_time).total_seconds() / 3600 / 24  # days since start of simulation

        # Tafel and Arrhenius equations
        b1_tfl = np.exp(alpha_b1 * F / R * (u_neg / df['temp'] - u_ref / t_ref))
        b3_tfl = np.exp(alpha_b3 * F / R * (v_oc / df['temp'] - v_ref / t_ref))
        b1_arr = np.exp(-e_ab1 / R * (1 / df['temp'] - 1 / t_ref))
        b2_arr = np.exp(-e_ab2 / R * (1 / cycles['avg_temp'] - 1 / t_ref))
        b3_arr = np.exp(-e_ab3 / R * (1 / df['temp'] - 1 / t_ref))

        # degradation rates
        b1 = b1_ref * ((b1_tfl * b1_arr).mean() * np.exp(gamma_b1 * max_dod ** beta_b1))
        b2 = b2_ref * ((b2_arr * cycles['ncycle']) ** 2).sum() ** 0.5 / deg_time if max_dod > 0 else 0
        b3 = b3_ref * (b3_tfl * b3_arr).mean() * (1 - theta * max_dod)

        # update degradation states
        # q_li = d0 * (b0 - b1 * time ** 0.5 - b2 * charge_cycles - b3 * (1 - np.exp(-time / tau_b3)))
        q1, q2, q3 = self.degradation_states  # previous states
        q1 += deg_time * b1 * 0.5 * (q1 / b1) ** -1 if q1 != 0 else b1 * deg_time ** 0.5
        q2 += deg_time * b2
        q3 += deg_time / tau_b3 * max(b3 - q3, 0)
        q3 = min(q3, b3)
        self.degradation_states = q1, q2, q3

        # update nominal capacity due to degradation
        self.capacity_kwh_nominal = self.parameters['capacity_kwh'] * (b0 - sum(self.degradation_states))

        # reset degradation data
        self.degradation_data = []

    def generate_results(self, verbosity, to_ext=False):
        results = super().generate_results(verbosity, to_ext)
        if to_ext:
            results[self.name].update({'SOC': self.soc})
            return results
        else:
            if verbosity >= 3:
                results.update({self.name + ' SOC (-)': self.soc})
            if verbosity >= 6 and self.thermal_model is not None:
                results.update({self.name + ' Temperature (C)': self.thermal_model.states[self.t_idx],
                                })
            if verbosity >= 9 and self.degradation_states is not None:
                results.update({self.name + ' Nominal Capacity (kWh)': self.capacity_kwh_nominal,
                                self.name + ' Actual Capacity (kWh)': self.capacity_kwh,
                                self.name + ' Degradation State Q1': self.degradation_states[0],
                                self.name + ' Degradation State Q2': self.degradation_states[1],
                                self.name + ' Degradation State Q3': self.degradation_states[2],
                                })
        return results

    def update_model(self, schedule):
        super().update_model(schedule)

        # update SOC
        hours = self.time_res.total_seconds() / 3600
        self_discharge = self.discharge_pct / 100 * hours / 24
        self.soc += self.power_input * hours / self.capacity_kwh - self_discharge

        # check with upper and lower bound of usable SOC
        assert self.soc_max + 0.001 >= self.soc >= self.soc_min - 0.001  # small computational errors possible

        # thermal model update is already done. Set states to next_states
        if self.thermal_model is not None:
            self.thermal_model.states = self.thermal_model.next_states

    def get_kwh_remaining(self, discharge=True, include_efficiency=True):
        # returns the remaining SOC, in units of kWh. Option for remaining charging/discharging
        # if include_efficiency: return kWh AC (incorporating efficiency). Otherwise, return kWh DC
        if discharge:
            kwh_rem = (self.soc - self.soc_min) * self.capacity_kwh
            if include_efficiency:
                kwh_rem *= self.efficiency
        else:
            kwh_rem = (self.soc_max - self.soc) * self.capacity_kwh
            if include_efficiency:
                kwh_rem /= self.efficiency_charge

        return kwh_rem
