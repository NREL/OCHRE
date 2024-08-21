import datetime as dt
import pandas as pd

from ochre import Dwelling, CreateFigures
from bin.run_dwelling import dwelling_args

# Test script to run single Dwelling with constant external control signal

dwelling_args.update(
    {
        "time_res": dt.timedelta(minutes=10),
        "ext_time_res": dt.timedelta(minutes=60),  # for duty cycle control only
        "Equipment": {
            "EV": {
                'vehicle_type': 'BEV',
                'charging_level': 'Level 2',
                'mileage': 150,
            },
        },
    }
)

example_control_signal = {
    "HVAC Heating": {
        "Setpoint": 19,
        #  'Max Capacity Fraction': 0.8,
        "Max ER Capacity Fraction": 0.5,
    },  # in C
    "HVAC Cooling": {"Setpoint": 22},  # in C
    "Water Heating": {"Setpoint": 50},  # in C
    "PV": {"P Setpoint": -1.1, "Q Setpoint": 0.5},  # in kW, kVAR
    "Battery": {
        # 'P Setpoint': -1.0,  # in kW
        # "SOC": 0.8,
        "Self Consumption Mode": True,
        "Max Import Limit": 1,  # in kW
        "Max Export Limit": 1,  # in kW
        # 'Min SOC': 0.2,
        # 'Max SOC': 0.8,
    },
    "EV": {
        # "Delay": True,
        "Max Power": 6,
        "Max SOC": 0.7,
        # "P Setpoint": 5,
        # "SOC": 0.6,
        # "SOC Rate": 0.02,
    },
}


def run_with_schedule_control():
    # Create Dwelling model (same as above)
    dwelling = Dwelling(name="Test House with Controller", **dwelling_args)

    # Get HVAC heater setpoints
    heater = dwelling.get_equipment_by_end_use('HVAC Heating')
    setpoints = heater.schedule['HVAC Heating Setpoint (C)']

    # Reduce heating setpoint by 1C from 5-9PM
    peak_times = setpoints.between_time(dt.time(17, 0, 0), dt.time(21, 0, 0), inclusive='left').index
    setpoints.loc[peak_times] -= 1
    heater.reset_time()  # resets the schedule

    # Run simulation
    dwelling.simulate()


def run_constant_control_signal(control_signal=None):
    # Initialization
    dwelling = Dwelling(name='OCHRE with Controller', **dwelling_args)

    # Simulation
    for t in dwelling.sim_times:
        # assert dwelling.current_time == t
        house_status = dwelling.update(control_signal=control_signal)

    df, _, _ = dwelling.finalize()

    df["EV Electric Power (kW)"].plot()
    df["EV SOC (-)"].plot()
    CreateFigures.plt.show()


def run_with_cooptimization():
    # Initialization
    dwelling = Dwelling(name="OCHRE with Controller", **dwelling_args)

    # Simulation
    for t in dwelling.sim_times:
        # set up simulation time step (only run once)
        dwelling.update_inputs()

        # simulate 1 time step with different controls
        control_signal = {}
        converged = False
        while not converged:
            # run model and get results without advancing time
            dwelling.update_model(control_signal)
            house_status = dwelling.generate_results()

            # check house_status for convergence
            if True:
                converged = True
            else:
                # change control if necessary
                control_signal = {}

        # complete time step and get results
        house_status = dwelling.update_results()
    
    # Finalize simulation
    return dwelling.finalize()


def get_hvac_controls(hour_of_day, occupancy, heating_setpoint, **unused_inputs):
    # Use some of the controller_inputs to determine setpoints (or other control signals)
    if 14 <= hour_of_day < 20:  # 2PM-8PM
        heating_setpoint -= 1  # reduce setpoint by 1 degree C
        # if occupancy > 0:
        #     heating_setpoint -= 1  # reduce setpoint by 1 degree C
        # else:
        #     heating_setpoint -= 2  # reduce setpoint by 2 degrees C

    return {
            'HVAC Heating': {
                'Capacity': 1000,
                # 'Setpoint': heating_setpoint,
                #  'Deadband': 2,
                # 'Load Fraction': 0,  # Set to 0 for force heater off
                # 'Duty Cycle': 0.5,  # Sets fraction of on-time explicitly
            },
            # 'HVAC Cooling': {...},
        }


def run_with_hvac_controller():
    # Initialization
    dwelling = Dwelling(name="OCHRE with Controller", **dwelling_args)
    heater = dwelling.get_equipment_by_end_use('HVAC Heating')
    cooler = dwelling.get_equipment_by_end_use('HVAC Cooling')

    # Change initial parameters if necessary (note, best to change both heater and cooler setpoints to prevent overlap)
    # heater.schedule['HVAC Heating Setpoint (C)'] = 20  # Override original HVAC setpoint schedule, can be time-varying
    # heater.schedule['HVAC Heating Deadband (C)'] = 2  # Override original HVAC deadband, can be time-varying
    # heater.reset_time()

    # Simulation
    controller_inputs = {}
    for t in dwelling.sim_times:
        assert dwelling.current_time == t

        # get control inputs from schedule
        controller_inputs.update({
            'current_time': t,
            'hour_of_day': t.hour,
            'outdoor_temp': dwelling.envelope.schedule.loc[t, 'Ambient Dry Bulb (C)'],
            'occupancy': dwelling.envelope.schedule.loc[t, 'Occupancy (Persons)'],
            # Original setpoints for current time
            'heating_setpoint': heater.schedule.loc[t, 'HVAC Heating Setpoint (C)'],  
            'cooling_setpoint': cooler.schedule.loc[t, 'HVAC Cooling Setpoint (C)'],
        })

        control_signal = get_hvac_controls(**controller_inputs)
        house_status = dwelling.update(control_signal=control_signal)

        # get control inputs from house status (note this will be used in the next time step)
        controller_inputs.update({
            'indoor_temp': house_status['Temperature - Indoor (C)'],
            # 'net_heat_gains': house_status['Net Sensible Heat Gain - Indoor (W)'],
        })

    return dwelling.finalize()


def run_controls_from_file(control_file):
    # Load external control file
    # Note: will need a MultiIndex, or some other method to convert to a dict of dicts
    df_ext = pd.read_csv(control_file, index_col='Time', parse_dates=True)

    # Initialization
    dwelling = Dwelling(name="OCHRE with Controller", **dwelling_args)

    # Simulation
    control_signal = None
    for t in dwelling.sim_times:
        assert dwelling.current_time == t
        if t in df_ext.index:
            control_signal = df_ext.loc[t].to_dict()  # May need a more complex process here

        dwelling.update(control_signal=control_signal)

    return dwelling.finalize()


if __name__ == '__main__':
    # run_with_schedule_control()
    run_constant_control_signal(example_control_signal)
    # run_with_hvac_controller()
    # run_controls_from_file(external_control_file='path/to/control_file.csv')