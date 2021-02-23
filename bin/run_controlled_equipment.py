import os
import datetime as dt
import time
import pandas as pd

from ochre import Dwelling
from ochre.FileIO import default_output_path
# from ochre import CreateFigures
from bin.run_dwelling import dwelling_args

# Test script to run single equipment. Takes dwelling_args from run_dwelling
case_name = 'cycle100'

args = dwelling_args.copy()
args.update({
    'duration': dt.timedelta(days=365 * 5),
    'time_res': dt.timedelta(minutes=10),
    'assume_equipment': False,
    'initialization_time': None,
    'repeat_years': True,
    'verbosity': 9,
})

# input arguments for single equipment to test
equipment_dict = {
    # 'EV': {'vehicle_type': 'PHEV',
    #        'charging_level': 'Level1',
    #        'mileage': 20},
    # 'Electric Resistance Water Heater': {},
    'Battery': {'control_type': 'Off',
                'capacity_kwh': 10,
                'capacity_kw': 5,
                'soc_init': 0,
                'soc_min': 0,
                'soc_max': 1,
                'eta_charge': 1,
                'eta_discharge': 1,
                },
}

# optional dictionary to update the dwelling schedule
add_to_schedule = {
    'Indoor': 25,  # static indoor temp
}

if __name__ == '__main__':
    # Initialize dwelling, extract equipment
    dwelling = Dwelling('Test Equipment', equipment_dict, **args)

    equipment = dwelling.equipment[0]

    # Update schedule
    schedule = dwelling.schedule
    for key, val in add_to_schedule.items():
        schedule[key] = val

    # Simulate equipment
    t0 = time.time()

    # Simulation
    all_results = []
    schedule_data = dwelling.schedule.to_dict('records')
    for current_schedule in schedule_data:
        # external controller definition - charge, hold, discharge, hold; 2 hours each 3x per day
        h = equipment.current_time.hour
        if (h // 2) % 4 == 0:
            p = equipment.capacity
        elif (h // 2) % 4 == 2:
            p = -equipment.capacity
        else:
            p = 0
        # p = 0
        equip_from_ext = {'P Setpoint': p}

        # Update and save results
        equipment.update(1, current_schedule, equip_from_ext)

        results = {}
        if equipment.is_electric:
            results[equipment.name + ' Electric Power (kW)'] = equipment.electric_kw
            results[equipment.name + ' Reactive Power (kVAR)'] = equipment.reactive_kvar
        if equipment.is_gas:
            results[equipment.name + ' Gas Power (therms/hour)'] = equipment.gas_therms_per_hour
        results.update(equipment.generate_results(dwelling.verbosity))
        all_results.append(results)

        equipment.update_model(current_schedule)

    # save results and return data frame
    # TODO: try with not cycling, soc at 80%, 100%, 50%; cycling with 10-90 at 5kW and at 4kW
    df = pd.DataFrame(all_results, index=schedule.index)
    results_file = os.path.join(dwelling_args.get('output_path', default_output_path),
                                '{}_{}.csv'.format(equipment.name, case_name))
    df.to_csv(results_file)
    df_daily = df.resample('1D').mean()
    daily_file = os.path.join(dwelling_args.get('output_path', default_output_path),
                              '{}_{}_daily.csv'.format(equipment.name, case_name))
    df_daily.to_csv(daily_file)
    print('{} Simulation Complete, results saved to {}'.format(equipment.name, results_file))
    print('Battery Final Capacity:', df['Battery Nominal Capacity (kWh)'][-1])

    t1 = time.time()
    print('time to simulate: {}'.format(t1 - t0))

    # fig = CreateFigures.plot_daily(df, equipment.name + ' Electric Power (kW)', plot_singles=False, plot_max=False,
    #                                plot_min=False)
    # fig.show()
    #
    # # plot all powers and temperatures
    # power_cols = [col for col in df.columns if '(kW)' in col]
    # if power_cols:
    #     df_powers = df.loc[:, power_cols].rename(columns={col: col[:-5] for col in power_cols})
    #     fig = CreateFigures.plot_time_series(df_powers, 'Power (kW)', legend=True)
    #     fig.show()
    #
    # temp_cols = [col for col in df.columns if '(C)' in col]
    # if temp_cols:
    #     df_temps = df.loc[:, temp_cols].rename(columns={col: col[:-4] for col in temp_cols})
    #     fig = CreateFigures.plot_time_series(df_temps, 'Temperature (C)', legend=True)
    #     fig.show()
