# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:28:35 2019

@author: rchintal, xjin, mblonsky
"""

import numpy as np
import datetime as dt
import pandas as pd
from scipy.interpolate import interp1d
import rainflow

from ochre.utils.units import convert, degC_to_K
from ochre.Models import OneNodeRCModel
from ochre.Equipment import Generator

CONTROL_TYPES = ['Schedule', 'Self-Consumption', 'Off']


class BatteryThermalModel(OneNodeRCModel):
    name = 'Battery Temperature'
    int_name = 'BATT'
    ext_name = 'AMB'

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 6:
            results[f'{self.name} (C)'] = self.states[0]

        return results


class Battery(Generator):
    name = 'Battery'
    end_use = 'Battery'
    allow_consumption = True
    is_gas = False
    optional_inputs = ['net_power', 'pv_power']

    def __init__(self, enable_degradation=True, efficiency_type='advanced', enable_thermal_model=None, **kwargs):
        if enable_thermal_model is None:
            enable_thermal_model = kwargs.get('zone_name') is not None
        
        # add zone temperature to schedule
        if enable_thermal_model:
            optional_inputs = self.optional_inputs + ['Zone Temperature (C)']
        else:
            optional_inputs = None  # use the default

        # Create Generator model
        self.degradation_states = None
        super().__init__(efficiency_type=efficiency_type, optional_inputs=optional_inputs, **kwargs)

        # Battery electrical parameters
        # Note: Parameter values taken from SAM Li-NMC defaults, version 2020.2.29
        self.capacity_kwh = self.parameters['capacity_kwh']  # in kWh, instantaneous capacity
        self.soc = self.parameters['soc_init']  # Initial State of Charge
        self.next_soc = self.soc
        self.soc_max = self.parameters['soc_max']
        self.soc_min = self.parameters['soc_min']
        self.discharge_rate = convert(self.parameters['discharge_pct'], 'percent/day',
                                      'unitless/hour')  # Self-discharge rate (1/hour)
        self.efficiency_inverter = self.parameters['efficiency_inverter']  # Inverter efficiency, unitless

        # Pack efficiency depends on pack internal resistance
        self.efficiency_internal = 1  # varies with power output, unitless
        if self.efficiency_type == 'advanced':
            # voltage and resistance depends on # of cells in parallel/series
            capacity_cell = self.parameters['ah_cell'] * self.parameters['v_cell'] / 1000  # in kWh per cell
            n_cells_tot = self.capacity_kwh / capacity_cell  # not necessarily an integer

            # usually 14 cells in series to achieve typical 50.6V pack voltage
            # TODO: update with data from https://github.com/NREL/PyChargeModel/blob/main/ElectricVehicles.py
            n_series_by_voltage = self.parameters.get('initial_voltage', 50.4) / self.parameters['v_cell']
            self.n_series = self.parameters.get('n_series', n_series_by_voltage)  # Number of cells in series
            n_parallel = n_cells_tot / self.n_series

            self.r_internal = self.parameters['r_cell'] * self.n_series / n_parallel  # internal resistance, in ohms
        else:
            self.n_series = None
            self.r_internal = None

        # Create thermal model if zone is specified
        self.t_idx = 0  # index for battery temperature (based on states)
        if enable_thermal_model:
            self.thermal_model = BatteryThermalModel(self.parameters['thermal_r'], self.parameters['thermal_c'],
                                                     **kwargs)
            self.sub_simulators.append(self.thermal_model)
            assert len(self.thermal_model.states) == 1

            if 'Initial Battery Temperature (C)' in kwargs:
                self.thermal_model.states[0] = kwargs['Initial Battery Temperature (C)']
            elif self.zone:
                self.thermal_model.states[0] = self.zone.temperature
            else:
                raise Exception('Must specify "Initial Battery Temperature (C)"')
        else:
            self.thermal_model = None

        # Degradation model
        self.capacity_kwh_nominal = self.capacity_kwh  # starts at rated capacity, reduces from degradation
        self.degradation_data = []
        if enable_degradation:
            self.degradation_states = (0, 0, 0)

        # Curves for degradation and efficiency using internal resistance
        # TODO: update with data from https://github.com/NREL/PyChargeModel/blob/main/ElectricVehicles.py
        df_curves = self.initialize_parameters('degradation_curves.csv', name_col='SOC', value_col=None)
        self.voc_curve = interp1d(df_curves.index, df_curves['V_oc'], bounds_error=False,
                                  fill_value=(df_curves['V_oc'].iloc[0], df_curves['V_oc'].iloc[-1]))
        self.uneg_curve = interp1d(df_curves.index, df_curves['U_neg'], bounds_error=False,
                                   fill_value=(df_curves['U_neg'].iloc[0], df_curves['U_neg'].iloc[-1]))

    def update_inputs(self, schedule_inputs=None):
        # Add zone temperature to schedule inputs for water tank
        if not self.main_simulator and self.thermal_model:
            schedule_inputs['Zone Temperture (C)'] = schedule_inputs[f'{self.zone_name} Temperature (C)']
    
        super().update_inputs(schedule_inputs)

    def reset_time(self, start_time=None, **kwargs):
        super().reset_time(start_time, **kwargs)

        # reset degradation states and capacity
        if self.degradation_states is not None:
            self.degradation_states = (0, 0, 0)
            self.capacity_kwh = self.parameters['capacity_kwh']
            self.capacity_kwh_nominal = self.capacity_kwh

    def update_external_control(self, control_signal):
        # Options for external control signals:
        # - P Setpoint: Directly sets power setpoint, in kW
        #   - Note: still subject to SOC limits and charge/discharge limits
        # - SOC Rate: Solves for power setpoint to achieve desired SOC, same sign as P Setpoint, in 1/hour
        # - Control Type: Sets the control type to one of CONTROL_TYPES
        # - Parameters: Update control parameters, including:
        #   - Schedule: charge/discharge start times
        #   - Schedule: charge/discharge powers
        #   - Self-Consumption: charge/discharge offsets, in kW
        #   - Self-Consumption: charge type (from any solar or from net power)

        super().update_external_control(control_signal)

        if 'SOC Rate' in control_signal:
            power_dc = control_signal['SOC Rate'] * self.capacity_kwh  # in kW, DC
            efficiency = self.calculate_efficiency(power_dc, is_output_power=False)
            self.power_setpoint = power_dc / efficiency if power_dc > 0 else power_dc * efficiency

        return 'On' if self.power_setpoint != 0 else 'Off'

    def update_internal_control(self):
        super().update_internal_control()

        if (self.control_type == 'Self-Consumption' and self.parameters.get('charge_from_solar', False) and
                self.power_setpoint > 0):
            # Force charge from solar
            pv_power = self.current_schedule.get('pv_power')
            if pv_power is not None:
                self.power_setpoint = min(self.power_setpoint, -pv_power)
            else:
                self.warn('Cannot run Self-Consumption control without PV power')
                self.power_setpoint = 0

        # Update setpoint if SOC limits are reached
        if self.power_setpoint > 0 and self.soc >= self.soc_max:
            self.power_setpoint = 0
        if self.power_setpoint < 0 and self.soc <= self.soc_min:
            self.power_setpoint = 0

        return 'On' if self.power_setpoint != 0 else 'Off'

    def get_kwh_remaining(self, discharge=True, include_efficiency=True, max_power=None):
        # returns the remaining SOC, in units of kWh. Option for remaining charging/discharging
        # if include_efficiency: return kWh AC (incorporating efficiency). Otherwise, return kWh DC

        if include_efficiency:
            if max_power is None:
                # if max_power not specified, uses the battery limit
                max_discharge, max_charge = self.get_power_limits()
                max_power = max_discharge if discharge else max_charge
            efficiency = self.calculate_efficiency(max_power)
        else:
            efficiency = 1

        if discharge:
            return (self.soc - self.soc_min) * self.capacity_kwh * efficiency
        else:
            return (self.soc_max - self.soc) * self.capacity_kwh / efficiency

    def get_power_limits(self):
        # returns min (discharge) and max (charge) output power limits based on capacity and SOC
        max_discharge, max_charge = super().get_power_limits()
        hours = self.time_res.total_seconds() / 3600

        # update max charge based on charging efficiency
        max_charge_dc = self.get_kwh_remaining(discharge=False, include_efficiency=False) / hours  # in kW
        efficiency = self.calculate_efficiency(min(max_charge_dc, max_charge), is_output_power=False)
        max_charge = min(max_charge_dc / efficiency, max_charge)

        # update max discharge based on discharging efficiency
        max_discharge_dc = self.get_kwh_remaining(discharge=True, include_efficiency=False) / hours  # in kW
        efficiency = self.calculate_efficiency(-min(max_discharge_dc, -max_discharge * 1.1), is_output_power=False)
        max_discharge = -min(max_discharge_dc * efficiency, -max_discharge)

        return max_discharge, max_charge

    def calculate_efficiency(self, electric_kw=None, is_output_power=True):
        if electric_kw is None:
            electric_kw = self.electric_kw

        if self.efficiency_type == 'advanced':
            # determine total cell voltage based on V_oc and output power
            # Note: if power > 0, battery is charging and v > voc; when discharging, v < voc
            voc = float(self.voc_curve(self.soc)) * self.n_series
            if is_output_power:
                electric_kw *= self.efficiency_inverter
                v = voc / 2 + np.sqrt((voc / 2) ** 2 + (electric_kw * 1000) * self.r_internal)  # V = V_oc + P*R/V
            else:
                v = voc + (electric_kw * 1000 / voc) * self.r_internal  # V = V_oc + I*R = V_oc + P/V_oc * R

            if electric_kw <= 0:
                # discharging - efficiency is p_out / p_in = v_out / v_in
                self.efficiency_internal = v / voc
            else:
                # charging - efficiency is p_in / p_out = v_in / v_out
                self.efficiency_internal = voc / v
        elif self.efficiency_type == 'constant':
            if electric_kw < 0:
                self.efficiency_internal = self.efficiency_rated
            else:
                self.efficiency_internal = self.parameters.get('efficiency_charge', self.efficiency_rated)
        else:
            self.efficiency_internal = super().calculate_efficiency(electric_kw, is_output_power)

        return self.efficiency_internal * self.efficiency_inverter

    def calculate_power_and_heat(self):
        # run degradation algorithm once per day
        if self.degradation_states is not None and self.current_time.time() == dt.time(0, 0):
            self.calculate_degradation()

        # update instantaneous capacity (d0) based on battery temperature
        d0_ref = 1.001
        t_ref = 298.15  # K, = 25 degC
        R = 8.31446  # J / K / mol
        e_ad1 = 4126  # J / mol
        e_ad2 = 9.752e6  # J / mol
        if self.thermal_model is not None:
            t_batt = self.thermal_model.states[self.t_idx] + degC_to_K
            d0 = d0_ref * np.exp(-e_ad1 / R * (1 / t_batt - 1 / t_ref) +
                                 -e_ad2 / R * (1 / t_batt - 1 / t_ref) ** 2)
            self.capacity_kwh = self.capacity_kwh_nominal * d0
        else:
            self.capacity_kwh = self.capacity_kwh_nominal

        super().calculate_power_and_heat()

        # update SOC for next time step
        # FUTURE: non-zero self-discharge could cause SOC limit issues
        hours = self.time_res.total_seconds() / 3600
        self_discharge = self.discharge_rate * hours
        self.next_soc = self.soc + self.power_input * hours / self.capacity_kwh - self_discharge

        # check with upper and lower bound of usable SOC
        assert self.soc_max + 0.001 >= self.next_soc >= self.soc_min - 0.001  # small computational errors possible

        # append SOC and temperature to degradation data
        t_batt = self.thermal_model.states[self.t_idx] if self.thermal_model is not None else 25
        if self.degradation_states is not None:
            self.degradation_data.append((self.soc, t_batt))

        if self.thermal_model is not None:
            # TODO: add battery node to envelope model and incorporate into sensible gains
            # delta_t = self.thermal_model.next_states[self.t_idx] - t_batt  # change in battery temperature
            # delta_h = self.thermal_model.capacitance * delta_t / self.time_res.total_seconds()  # in W
            # self.sensible_gain -= delta_h

            # calculate inputs to thermal model
            p_internal = self.power_input * self.efficiency_internal if self.electric_kw < 0 else \
                self.power_input / self.efficiency_internal
            h_batt = p_internal - self.power_input
            return {
                'T_AMB': self.current_schedule['Zone Temperature (C)'],
                'H_BATT': h_batt}
        else:
            return None

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
        b3_ref = -2.805e-2  # unitless, negative means increasing capacity
        alpha_b1 = -1  # unitless
        alpha_b3 = 0.0066  # unitless
        beta_b1 = 2.157  # unitless
        gamma_b1 = 2.472  # unitless
        tau_b3 = 5  # days
        theta = -0.135  # unitless  # TODO: might be positive? using value from Matlab, not paper

        # run rainflow algorithm
        df = pd.DataFrame(self.degradation_data, columns=['soc', 'temp'])
        df['soc'] = df['soc'].clip(0, 1)
        df['temp'] = df['temp'] + degC_to_K
        cycles = rainflow.extract_cycles(df['soc'])
        cycles = pd.DataFrame(cycles, columns=['dsoc', 'avg_soc', 'ncycle', 'start', 'end'])
        cycles['avg_temp'] = [df['temp'].iloc[start: end].mean() for start, end in zip(cycles['start'], cycles['end'])]
        deg_time = len(df) * self.time_res.total_seconds() / 3600 / 24  # days since last degradation update
        max_dod = cycles['dsoc'].max()

        # interpolate SOC to get v_oc and u_neg
        v_oc = self.voc_curve(df['soc'])
        u_neg = self.uneg_curve(df['soc'])

        # t_batt = convert(self.thermal_model.states[self.t_idx], 'degC', 'K')  # in K
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
        q1 += deg_time * b1 * 0.5 * max(q1 / b1, 1) ** -1
        q2 += deg_time * b2
        q3 += deg_time / tau_b3 * min(b3 - q3, 0)  # q3 always decreasing, always negative
        q3 = max(q3, b3)
        self.degradation_states = q1, q2, q3

        # raise warning/error if degradation is too high
        if sum(self.degradation_states) >= 1:
            raise Exception('{} degraded beyond useful life.'.format(self.name))
        elif sum(self.degradation_states) >= 0.7:
            self.warn('Degraded beyond useful life.')

        # update nominal capacity due to degradation
        self.capacity_kwh_nominal = self.parameters['capacity_kwh'] * (b0 - sum(self.degradation_states))

        # reset degradation data
        self.degradation_data.clear()

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 3:
            results[f'{self.end_use} SOC (-)'] = self.soc
        if self.verbosity >= 6:
            results[f'{self.end_use} Energy to Discharge (kWh)'] = self.get_kwh_remaining()
        if self.verbosity >= 9 and self.degradation_states is not None:
            results[f'{self.end_use} Nominal Capacity (kWh)'] = self.capacity_kwh_nominal
            results[f'{self.end_use} Actual Capacity (kWh)'] = self.capacity_kwh
            results[f'{self.end_use} Degradation State Q1'] = self.degradation_states[0]
            results[f'{self.end_use} Degradation State Q2'] = self.degradation_states[1]
            results[f'{self.end_use} Degradation State Q3'] = self.degradation_states[2]
        if self.save_ebm_results:
            results.update(self.make_equivalent_battery_model())

        return results

    def update_results(self):
        current_results = super().update_results()

        # Update next time step SOC
        self.soc = self.next_soc

        return current_results

    def make_equivalent_battery_model(self):
        # returns a dictionary of equivalent battery model parameters
        max_discharge, max_charge = self.get_power_limits()  # accounts for SOC and power limits
        return {
            f'{self.end_use} EBM Energy (kWh)': self.soc * self.capacity_kwh,
            f'{self.end_use} EBM Min Energy (kWh)': 0,
            f'{self.end_use} EBM Max Energy (kWh)': self.capacity_kwh,
            f'{self.end_use} EBM Max Power (kW)': max_charge,
            f'{self.end_use} EBM Efficiency (-)': self.calculate_efficiency(max_charge),
            f'{self.end_use} EBM Baseline Power (kW)': self.discharge_rate * self.capacity_kwh,
            f'{self.end_use} EBM Max Discharge Power (kW)': max_discharge,
            f'{self.end_use} EBM Discharge Efficiency (-)': self.calculate_efficiency(max_discharge),
        }
