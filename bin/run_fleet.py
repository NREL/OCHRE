import datetime as dt
import numpy as np
import pandas as pd
# import cProfile

from ochre import ElectricResistanceWaterHeater
from ochre import CreateFigures

from bin.run_dwelling import dwelling_args


# Test script to run a fleet of water heaters


def run_water_heater_fleet(num_water_heaters=5):
    wh_names = [f'wh_{i + 1}' for i in range(num_water_heaters)]

    # Make water draw profile (constant flow rate with random water draw probability of 1%)
    times = pd.date_range(dwelling_args['start_time'], dwelling_args['start_time'] + dwelling_args['duration'],
                          freq=dwelling_args['time_res'])
    water_draw_magnitude = 12  # L/min
    withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=(len(times), num_water_heaters))
    withdraw_rate = pd.DataFrame(withdraw_rate, index=times, columns=wh_names)

    dwelling_args.update({
        'time_res': dt.timedelta(seconds=30),
        'output_path': None,
        'verbosity': 3,
    })

    equipment_args = {wh_name: {
        # Equipment parameters
        'name': wh_name,
        # 'water_nodes': 1,
        'Initial Temperature (C)': np.random.uniform(49, 49),
        'Setpoint Temperature (C)': np.random.randint(50, 52),
        'Deadband Temperature (C)': np.random.uniform(3, 3),
        'Capacity (W)': np.random.uniform(4800, 4800),
        'Efficiency (-)': np.random.uniform(0.99, 1),
        'Tank Volume (L)': np.random.uniform(227, 260),
        'Tank Height (m)': 1.22,
        'UA (W/K)': 2.17,
        'schedule': pd.DataFrame({
            'Water Heating (L/min)': withdraw_rate[wh_name],
            'Zone Temperature (C)': np.random.uniform(15, 18),
            'Mains Temperature (C)': np.random.uniform(5.6, 8.3),
        }, index=times),
        **dwelling_args,
    } for wh_name in wh_names}

    # Initialize equipment
    fleet = {wh_name: ElectricResistanceWaterHeater(**args) for wh_name, args in equipment_args.items()}

    # Simulate equipment
    all_data = {}
    for eq_name, eq in fleet.items():
        df = eq.simulate()
        all_data[eq_name] = df

    powers = pd.DataFrame({eq_name: df[f'Water Heating Electric Power (kW)'] for eq_name, df in all_data.items()})
    temps = pd.DataFrame({eq_name: df['Hot Water Outlet Temperature (C)'] for eq_name, df in all_data.items()})
    # temps.plot()
    # CreateFigures.plot_daily_profile(df, 'Battery Electric Power (kW)', plot_max=False, plot_min=False)
    # CreateFigures.plot_time_series_detailed((df['Battery SOC (-)'],))
    # CreateFigures.plt.show()


def run_fleet_controlled():
    # TODO: not working, convert to battery fleet
    # maybe add another example with EV fleet
    equipment_args = {
        # Equipment parameters
        # See defaults/Battery/default_parameters.csv for more options
        'capacity_kwh': 10,
        'control_type': 'Off',
        **dwelling_args,
    }

    # Initialize equipment
    equipment = ElectricResistanceWaterHeater(**equipment_args)

    # Simulate equipment
    for t in equipment.sim_times:
        assert equipment.current_time == t
        control_signal = {'P Setpoint': np.random.randint(-5, 5)}  # in kW
        equipment.update(control_signal, {})

    df = equipment.finalize()
    print(df.head())
    CreateFigures.plot_daily_profile(df, 'Battery Electric Power (kW)', plot_max=False, plot_min=False)
    CreateFigures.plot_time_series_detailed((df['Battery SOC (-)'],))
    CreateFigures.plt.show()


if __name__ == '__main__':
    run_water_heater_fleet()
    # run_fleet_controlled()
