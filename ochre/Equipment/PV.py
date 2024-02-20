import numpy as np
import pandas as pd
import datetime as dt
from PySAM.PySSC import PySSC

from ochre.utils import OCHREException
from ochre.utils.schedule import default_sam_weather_file
from ochre.Equipment import ScheduledLoad


class PV(ScheduledLoad):
    """
    PV System implemented using SAM or from an external schedule file.

    If using SAM, the PV capacity must be specified. Tilt and orientation can be specified, but will default to the
    roof pitch and south.

    If using an external schedule, the external schedule file must be specified.
    """
    name = 'PV'
    end_use = 'PV'
    zone_name = None

    def __init__(self, use_sam=None, capacity=None, tilt=None, orientation=None, include_inverter=True, **kwargs):
        if use_sam is None:
            use_sam = 'equipment_schedule_file' not in kwargs

        if use_sam:
            # Create PV schedule using SAM - requires capacity, tilt, orientation, and inverter efficiency
            if capacity is None:
                raise OCHREException('Must specify {} capacity (in kW) when using SAM'.format(self.name))
            if tilt is None:
                tilt = kwargs['roof pitch']
            if orientation is None:
                orientation = kwargs.get('building orientation', 0) % 360
                if 90 < orientation <= 270:
                    # use back roof when closer to due south (orientation always within 90 deg of south)
                    orientation = (orientation + 180) % 360

            inverter_efficiency = kwargs.get('inverter_efficiency') if include_inverter else 100  # in %
            schedule = run_sam(tilt=tilt, orientation=orientation, inv_efficiency=inverter_efficiency, **kwargs)

            # Note: defining schedule as positive=consuming power
            schedule = schedule['Power_AC'] if include_inverter else schedule['Power_DC']
            schedule *= capacity  # normalize with capacity, in kW
            schedule = schedule.to_frame(self.name + ' (kW)')
            kwargs['schedule'] = schedule

        super().__init__(**kwargs)

        # check that schedule is negative
        if abs(self.schedule[self.electric_name].min()) < abs(self.schedule[self.electric_name].max()):
            self.warn('Schedule should be negative (i.e. generating power).',
                      'Reversing schedule so that PV power is negative/generating')
            self.schedule = -self.schedule
            self.reset_time()

        self.capacity = capacity if capacity is not None else -self.schedule[self.electric_name].min()  # in kW, DC
        self.q_set_point = 0  # in kW, positive = consuming power

        # Inverter constraints, if included; if not, assume no limits and 100% efficiency
        self.include_inverter = include_inverter
        self.inverter_priority = kwargs.get('inverter_priority', 'Var')
        self.inverter_capacity = kwargs.get('inverter_capacity', self.capacity)  # in kVA, AC
        self.inverter_min_pf = kwargs.get('inverter_min_pf', 0.8)
        if self.inverter_min_pf is not None:
            self.inverter_min_pf_factor = ((1 / self.inverter_min_pf ** 2) - 1) ** 0.5
        else:
            self.inverter_min_pf_factor = None

    def update_external_control(self, control_signal):
        # External PV control options:
        # - P/Q Setpoint: set P and Q directly from external controller (assumes positive = consuming)
        # - P Curtailment: set P by specifying curtailment in kW or %
        # - Power Factor: set Q based on P setpoint (internal or external) and given power factor
        # - Priority: set inverter_priority to 'Watt' or 'Var'

        self.update_internal_control()

        # Update P from external control
        if 'P Setpoint' in control_signal:
            p_set = control_signal['P Setpoint']
            if p_set > 0:
                self.warn('Setpoint should be negative (i.e. generating power). Reversing sign to be negative.')
                p_set *= -1
            # if p_set < self.p_set_point - 0.1:
            #     # Print warning if setpoint is significantly larger than max power
            #     self.warn('Setpoint ({}) is larger than max power ({})'.format(p_set, self.p_set_point))
            self.p_set_point = max(self.p_set_point, p_set)
        elif 'P Curtailment (kW)' in control_signal:
            p_curt = min(max(control_signal['P Curtailment (kW)'], 0), -self.p_set_point)
            self.p_set_point += p_curt
        elif 'P Curtailment (%)' in control_signal:
            pct_curt = min(max(control_signal['P Curtailment (%)'], 0), 100)
            self.p_set_point *= 1 - pct_curt / 100

        # Update Q from external control
        if 'Q Setpoint' in control_signal:
            self.q_set_point = control_signal['Q Setpoint']
        elif 'Power Factor' in control_signal:
            # Note: power factor should be negative for generating P/consuming Q
            pf = control_signal['Power Factor']
            self.q_set_point = ((1 / pf ** 2) - 1) ** 0.5 * self.p_set_point * (pf / abs(pf))

        if 'Priority' in control_signal:
            priority = control_signal['Priority']
            if priority in ['Watt', 'Var', 'CPF']:
                self.inverter_priority = priority
            else:
                self.warn(f'Invalid priority type: {priority}')

        return 'On' if self.p_set_point != 0 else 'Off'

    def update_internal_control(self):
        # Set to maximum P, Q=0
        super().update_internal_control()
        self.p_set_point = min(self.p_set_point, 0)
        self.q_set_point = 0
        return 'On' if self.p_set_point < 0 else 'Off'

    def calculate_power_and_heat(self):
        super().calculate_power_and_heat()

        # determine power from set point
        p = self.electric_kw
        q = self.q_set_point
        s = (p ** 2 + q ** 2) ** 0.5
        if self.include_inverter and s > self.inverter_capacity:
            if self.inverter_priority == 'Watt':
                p = -min(-p, self.inverter_capacity)  # Note: P <= 0
                max_q_capacity = (self.inverter_capacity ** 2 - p ** 2) ** 0.5
                if self.inverter_min_pf is not None:
                    max_q_pf = self.inverter_min_pf_factor * -p
                    q = min(abs(q), max_q_capacity, max_q_pf)
                else:
                    q = min(abs(q), max_q_capacity)
                q = q if self.q_set_point >= 0 else -q
            elif self.inverter_priority == 'Var':
                if self.inverter_min_pf is not None:
                    max_q_capacity = self.inverter_min_pf_factor * self.inverter_min_pf * self.inverter_capacity
                    max_q_pf = self.inverter_min_pf_factor * -p
                    q = min(abs(q), max_q_capacity, max_q_pf)
                else:
                    max_q_capacity = self.inverter_capacity
                    q = min(abs(q), max_q_capacity)
                q = q if self.q_set_point >= 0 else -q
                max_p_capacity = (self.inverter_capacity ** 2 - q ** 2) ** 0.5
                p = -min(-p, max_p_capacity)
            elif self.inverter_priority == 'CPF':
                # Reduce P and Q by the same ratio
                kva_ratio = s / self.inverter_capacity
                p /= kva_ratio
                q /= kva_ratio
            else:
                raise OCHREException('Unknown {} inverter priority mode: {}'.format(self.name, self.inverter_priority))

        # Set powers. Negative = generating power
        self.electric_kw = p
        self.reactive_kvar = q

    def generate_results(self):
        results = super().generate_results()
        if self.verbosity >= 6:
            results[f'{self.end_use} P Setpoint (kW)'] = self.p_set_point
            results[f'{self.end_use} Q Setpoint (kW)'] = self.q_set_point
        return results


# FUTURE: try pvlib package directly, might be faster and easier to implement than SAM
def run_sam(tilt=0, orientation=0, capacity=5, strings=2, modules=20, inv_efficiency=None,
            per_kw=True, sam_weather_file=None, start_time=None, duration=None, **kwargs):
    """
    Runs the System Advisory Model (SAM) PV model with standard parameters. Adjustable parameters include weather data;
    panel capacity, tilt, and orientation; number of panels and strings; and inverter efficiency. This code was
    generated using the following parameters from SAM:
     - Photovoltaic (detailed), no financial model
     - PV Module: SunPower SPR-X20-250-BLK, default parameters
     - PV Inverter: SolarEdge SE5000 (277V), default parameters
     - Default parameters for PV Albedo and Radiation, System Design, Shading, Layout, and Losses

    It has been verified that the default capacity and number of modules and strings gives reasonable outputs. However,
    adjusting these parameters may lead to unexpected results, as the PV module and inverter models are fixed. If the
    actual system set up is unknown, we recommend leaving these parameters as is, setting per_kw=True, and multiplying
    the resulting PV schedule by the expected capacity.

    :param tilt: PV array tilt angle, in degrees (0 = horizontal)
    :param orientation: PV array orientation angle, in degrees (0=south, west-of-south=positive)
    :param capacity: PV system capacity, in kW
    :param strings: Number of strings
    :param modules: number of modules
    :param inv_efficiency: inverter efficiency, in %, defaults to the efficiency of the SolarEdge SE5000 (98.1%)
    :param per_kw: boolean, if True (default), return the PV schedule as a fraction of the PV capacity.
    :param sam_weather_file: file name of the weather file, must be in a SAM readable format.
    :param start_time: starting time of the simulation
    :param duration: duration of the simulation
    :param kwargs: not used
    :return: a pandas DataFrame of the PV schedule with 3 columns:
     - Irradiance: solar irradiance on the PV array, in W/m^2
     - Power_AC: AC electrical power, in kW (negative for generating)
     - Power_DC: DC electrical power, in kW (negative for generating)
    """
    if sam_weather_file is None:
        sam_weather_file = default_sam_weather_file.format(kwargs['main_sim_name'])

    # Initialize SAM
    ssc = PySSC()
    # print('Running PySAM (SSC Version = {})'.format(ssc.version()))
    # print('SSC Build Information = ', ssc.build_info().decode("utf - 8"))
    ssc.module_exec_set_print(0)

    data = ssc.data_create()

    # Enter necessary data
    ssc.data_set_string(data, b'solar_resource_file', sam_weather_file.encode())
    ssc.data_set_number(data, b'system_capacity', capacity)
    albedo = [0.2] * 12

    ssc.data_set_array(data, b'albedo', albedo)
    ssc.data_set_number(data, b'inverter_count', 1)
    ssc.data_set_number(data, b'subarray1_nstrings', strings)
    ssc.data_set_number(data, b'subarray1_modules_per_string', modules // strings)
    ssc.data_set_number(data, b'subarray1_tilt', tilt)
    ssc.data_set_number(data, b'subarray1_azimuth', (orientation + 180) % 360)
    ssc.data_set_number(data, b'subarray1_track_mode', 0)
    ssc.data_set_number(data, b'subarray1_shade_mode', 0)
    subarray1_soiling = [5] * 12
    ssc.data_set_array(data, b'subarray1_soiling', subarray1_soiling)
    ssc.data_set_number(data, b'subarray1_rear_irradiance_loss', 0)
    ssc.data_set_number(data, b'subarray1_mismatch_loss', 2)
    ssc.data_set_number(data, b'subarray1_diodeconn_loss', 0.5)
    ssc.data_set_number(data, b'subarray1_dcwiring_loss', 2)
    ssc.data_set_number(data, b'subarray1_tracking_loss', 0)
    ssc.data_set_number(data, b'subarray1_nameplate_loss', 0)
    ssc.data_set_number(data, b'dcoptimizer_loss', 0)
    ssc.data_set_number(data, b'acwiring_loss', 1)
    ssc.data_set_number(data, b'transmission_loss', 0)
    ssc.data_set_number(data, b'subarray1_mod_orient', 0)
    ssc.data_set_number(data, b'subarray1_nmodx', 7)
    ssc.data_set_number(data, b'subarray1_nmody', 2)
    ssc.data_set_number(data, b'subarray2_enable', 0)
    ssc.data_set_number(data, b'subarray3_enable', 0)
    ssc.data_set_number(data, b'subarray4_enable', 0)
    # * key module variables
    ssc.data_set_number(data, b'module_model', 1)
    ssc.data_set_number(data, b'cec_area', 1.2439999580383301)
    ssc.data_set_number(data, b'cec_a_ref', 1.9386559724807739)
    ssc.data_set_number(data, b'cec_adjust', 4.3963689804077148)
    ssc.data_set_number(data, b'cec_alpha_sc', 0.00082499999552965164)
    ssc.data_set_number(data, b'cec_beta_oc', -0.14820599555969238)
    ssc.data_set_number(data, b'cec_gamma_r', -0.38999998569488525)
    ssc.data_set_number(data, b'cec_i_l_ref', 6.2045078277587891)
    ssc.data_set_number(data, b'cec_i_mp_ref', 5.8400001525878906)
    ssc.data_set_number(data, b'cec_i_o_ref', 2.3781549646217925e-11)
    ssc.data_set_number(data, b'cec_i_sc_ref', 6.1999998092651367)
    ssc.data_set_number(data, b'cec_n_s', 72)
    ssc.data_set_number(data, b'cec_r_s', 0.36243200302124023)
    ssc.data_set_number(data, b'cec_r_sh_ref', 498.47784423828125)
    ssc.data_set_number(data, b'cec_t_noct', 44.5)
    ssc.data_set_number(data, b'cec_v_mp_ref', 42.799999237060547)
    ssc.data_set_number(data, b'cec_v_oc_ref', 50.930000305175781)
    # * end module variables

    ssc.data_set_number(data, b'cec_temp_corr_mode', 0)
    ssc.data_set_number(data, b'cec_is_bifacial', 0)
    ssc.data_set_number(data, b'cec_bifacial_transmission_factor', 0.013000000268220901)
    ssc.data_set_number(data, b'cec_bifaciality', 0.64999997615814209)
    ssc.data_set_number(data, b'cec_bifacial_ground_clearance_height', 1)
    ssc.data_set_number(data, b'cec_standoff', 6)
    ssc.data_set_number(data, b'cec_height', 0)
    ssc.data_set_number(data, b'cec_transient_thermal_model_unit_mass', 11.091900000000001)
    ssc.data_set_number(data, b'inverter_model', 0)

    # * key inverter variables
    ssc.data_set_number(data, b'mppt_low_inverter', 405)
    ssc.data_set_number(data, b'mppt_hi_inverter', 480)
    ssc.data_set_number(data, b'inv_num_mppt', 1)
    ssc.data_set_number(data, b'inv_snl_c0', -1.9196679659216898e-06)
    ssc.data_set_number(data, b'inv_snl_c1', 2.4000000848900527e-05)
    ssc.data_set_number(data, b'inv_snl_c2', 0.0054540000855922699)
    ssc.data_set_number(data, b'inv_snl_c3', 0.0030330000445246696)
    ssc.data_set_number(data, b'inv_snl_paco', 5010)
    ssc.data_set_number(data, b'inv_snl_pdco', 5116.28173828125)
    ssc.data_set_number(data, b'inv_snl_pnt', 1.503000020980835)
    ssc.data_set_number(data, b'inv_snl_pso', 10.168439865112305)
    ssc.data_set_number(data, b'inv_snl_vdco', 425)
    ssc.data_set_number(data, b'inv_snl_vdcmax', 480)
    # * end inverter variable

    inv_tdc_cec_db = [[1, 52.799999237060547, -0.020999999716877937]]
    ssc.data_set_matrix(data, b'inv_tdc_cec_db', inv_tdc_cec_db)
    ssc.data_set_number(data, b'adjust:constant', 0)
    ssc.data_set_number(data, b'dc_adjust:constant', 0)
    inv_efficiency = inv_efficiency if inv_efficiency is not None else 98.071479797363281
    ssc.data_set_number(data, b'inv_snl_eff_cec', inv_efficiency)  # key variable
    module = ssc.module_create(b'pvsamv1')

    # run model
    print('Running annual PV model using SAM...')
    ssc.module_exec_set_print(0)
    if ssc.module_exec(module, data) == 0:
        print('pvsamv1 simulation error')
        idx = 1
        msg = ssc.module_log(module, 0)
        while msg is not None:
            print('	: ' + msg.decode("utf - 8"))
            msg = ssc.module_log(module, idx)
            idx = idx + 1
        SystemExit("Simulation Error")
    ssc.module_free(module)

    # Collect timeseries data
    # For details, see https://pysam-docs.readthedocs.io/en/latest/modules/Pvsamv1.html#outputs-group
    irr = ssc.data_get_array(data, b'subarray1_poa_nom')  # Nominal Front Total Irradiance W/m^2
    # irr = ssc.data_get_array(data, b'subarray1_poa_eff')  # Includes reflection
    # irr = ssc.data_get_array(data, b'poa_nom')  # Nominal Front Total Irradiance kW
    # irr = ssc.data_get_array(data, b'poa_eff')
    # irr = ssc.data_get_array(data, b'inv_total_loss')  # Inverter total power losses, kW
    power_dc = ssc.data_get_array(data, b'dc_net')  # DC system power, kW
    power_ac = ssc.data_get_array(data, b'gen')  # AC system power, kW
    if per_kw:
        power_dc = np.array(power_dc) / capacity
        power_ac = np.array(power_ac) / capacity
    ssc.data_free(data)

    # return dataframe of results
    df_input = pd.read_csv(sam_weather_file, skiprows=2)
    df_input = df_input.astype(int)
    time_vals = df_input.iloc[:, :5].values
    times = [dt.datetime(*data) for data in time_vals]

    # Note: Irradiance is positive, power is negative (for generation)
    df = pd.DataFrame({'Irradiance': irr,
                       'Power_AC': -power_ac,  # in kW or per unit
                       'Power_DC': -power_dc,  # in kW or per unit
                       }, index=times)

    # resample to times in range
    if start_time is not None:
        df = df.loc[df.index >= start_time]
    if duration is not None:
        if kwargs.get('initialization_time') is not None:
            # update duration to include duration of initialization
            duration = max(duration, kwargs['initialization_time'])
        df = df.loc[df.index < start_time + duration]
    return df
