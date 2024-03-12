import pandas as pd
import PySAM.Pvwattsv8 as pvwatts

from ochre.utils import OCHREException
from ochre.Equipment import ScheduledLoad


# FUTURE: try pvlib package directly, might be faster and easier to implement than SAM
def run_sam(
    capacity,
    tilt,
    azimuth,
    weather,
    location,
    inv_capacity=None,
    inv_efficiency=None,
):
    """
    Runs the System Advisory Model (SAM) PVWatts model. Adjustable parameters include panel capacity, tilt, and azimuth;
    weather and location data, and inverter capacity and efficiency.

    :param capacity: PV system capacity, in kW
    :param tilt: PV array tilt angle, in degrees (0 = horizontal)
    :param azimuth: PV array azimuth angle, in degrees (0=south, west-of-south=positive)
    :param weather: Pandas DataFrame of time series weather data in OCHRE schedule format
    :param location: dict of location data including timezone, elevation, latitude, and longitude
    :param inv_capacity: inverter capacity, in kW, defaults to `capacity`
    :param inv_efficiency: inverter efficiency, in %, uses PVWatts default (96%)
    :return: a Pandas Series of the PV AC power, using the same index as `weather`
    """
    if capacity is None:
        raise OCHREException('Must specify PV capacity (in kW) when using SAM')

    # get weather and location data
    time = weather.index
    solar_resource_data = {
        'tz': location['timezone'],
        'elev': location['altitude'],
        'lat': location['latitude'],
        'lon': location['longitude'],
        'year': tuple(time.year),
        'month': tuple(time.month),
        'day': tuple(time.day),
        'hour': tuple(time.hour),
        'minute': tuple(time.minute),
        'dn': tuple(weather['DNI (W/m^2)']),  # direct normal irradiance
        'df': tuple(weather['DHI (W/m^2)']),  # diffuse irradiance
        'gh': tuple(weather['GHI (W/m^2)']),  # global horizontal irradiance
        'wspd': tuple(weather['Wind Speed (m/s)']),  # windspeed
        'tdry': tuple(weather['Ambient Dry Bulb (C)'])  # dry bulb temperature
    }
    # create an instance of the Pvwattsv8 module
    system_model = pvwatts.default('PVWattsNone')
    
    # update system parameters
    system_model.value('system_capacity', capacity)
    system_model.value('tilt', tilt)
    system_model.value('azimuth', (azimuth + 180) % 360)  # SAM convention is south=180
    if inv_capacity is not None:
        system_model.value('dc_ac_ratio', capacity / inv_capacity)
    if inv_efficiency is not None:
        system_model.value('inv_eff', inv_efficiency)
    system_model.SolarResource.assign({'solar_resource_data': solar_resource_data})

    # run the modules in the correct order
    system_model.execute()

    # get results, make negative for generation
    ac = pd.Series(system_model.Outputs.ac, index=time) / 1000  # in kW
    # dc = pd.Series(system_model.Outputs.dc, index=time) / 1000  # in kW
    return ac


class PV(ScheduledLoad):
    """
    PV System implemented using SAM or from an external schedule.

    If using SAM, the PV capacity must be specified. Tilt and azimuth can be specified, but will default to the
    angle of the most southern facing roof.

    If not using SAM, an external schedule must be specified as a DataFrame (via `schedule`) or as a file (as
    `equipment_schedule_file`)
    """
    name = 'PV'
    end_use = 'PV'
    zone_name = None

    def __init__(self,
                 capacity=None,
                 tilt=None,
                 azimuth=None,
                 envelope_model=None,
                 inverter_capacity=None,
                 inverter_efficiency=None,
                 inverter_priority='Var',
                 inverter_min_pf=0.8,
                 **kwargs):
        self.capacity = capacity  # in kW, DC
        self.tilt = tilt
        self.azimuth = azimuth

        # get tilt and azimuth inputs to run SAM
        if (tilt is None or azimuth is None):
            if envelope_model is None:
                raise OCHREException('Must specify PV tilt and azimuth, or provide an envelope_model with a roof.')
            roofs = [bd.ext_surface for bd in envelope_model.boundaries if 'Roof' in bd.name]
            if not roofs:
                raise OCHREException('No roofs in envelope model. Must specify PV tilt and azimuth')
            # Use roof closest to south with preference to west (0-45 degrees)
            roof_data = pd.DataFrame([[bd.tilt, az] for bd in roofs for az in bd.azimuths], columns=['Tilt', 'Az'])
            best_idx = (roof_data['Az'] - 185).abs().idxmax()
            self.tilt = roof_data.loc[best_idx, 'Tilt']
            self.azimuth = roof_data.loc[best_idx, 'Az']

        # Inverter constraints
        self.inverter_capacity = inverter_capacity or self.capacity  # in kVA, AC
        self.inverter_efficiency = inverter_efficiency
        self.inverter_priority = inverter_priority
        self.inverter_min_pf = inverter_min_pf
        if self.inverter_min_pf is not None:
            self.inverter_min_pf_factor = ((1 / self.inverter_min_pf ** 2) - 1) ** 0.5
        else:
            self.inverter_min_pf_factor = None

        super().__init__(envelope_model=envelope_model, **kwargs)

        self.q_set_point = 0  # in kW, positive = consuming power
        if self.capacity is None:
            self.capacity = -self.schedule[self.electric_name].min()
        if self.inverter_capacity is None:
            self.inverter_capacity = self.capacity

        # check that schedule is negative
        if self.schedule[self.electric_name].abs().min() < self.schedule[self.electric_name].abs().max():
            self.warn('Schedule should be negative (i.e. generating power).',
                      'Reversing schedule so that PV power is negative/generating')
            self.schedule = self.schedule * -1
            self.reset_time()

    def initialize_schedule(self, schedule=None, equipment_schedule_file=None, location=None, **kwargs):
        if (schedule is None or self.name + ' (kW)' not in schedule) and equipment_schedule_file is None:
            self.print('Running SAM')
            schedule = run_sam(self.capacity, self.tilt, self.azimuth, schedule, location,
                               self.inverter_capacity, self.inverter_efficiency)
            schedule = schedule.to_frame(self.name + ' (kW)')

        return super().initialize_schedule(schedule, equipment_schedule_file, **kwargs)

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
        p = self.electric_kw  # updated from super().calculate_power_and_heat
        q = self.q_set_point
        s = (p ** 2 + q ** 2) ** 0.5
        if s > self.inverter_capacity:
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
