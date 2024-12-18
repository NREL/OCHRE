# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:38:04 2024

@author: halgr
"""

import datetime as dt

from ochre import Dwelling, CreateFigures
from ochre.Models import TankWithPCM
from bin.run_dwelling import dwelling_args



pcm_water_node = 5
pcm_vol_fraction = 0.5

#UEF draw profiles
#LowUseUEF = 'LowUseL.csv'
#MediumUseUEF = 'MediumUseL.csv'
#HighUseUEF = 'HighUseL.csv'

no_pcm_title = "No PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)
with_pcm_title = "With PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)

load_profile = 'MediumUseL.csv'

dwelling_args.update(
    {
        "time_res": dt.timedelta(minutes=1),  # time resolution of the simulation
        "duration": dt.timedelta(days=2),  # duration of the simulation
        "verbosity": 9,
        "schedule_input_file": load_profile #changes the default load profile in run_dwelling.py for this code to call the UEF load_profile
        }
)

def add_pcm_model(dwelling_args):
    dwelling_args["Equipment"]["Water Heating"] = {
        "model_class": TankWithPCM,
        "Water Tank": {
            "pcm_water_node": pcm_water_node,
            "pcm_vol_fraction": pcm_vol_fraction,
        },
    }

    return dwelling_args


def run_water_heater(dwelling_args,plot_title,load_profile_in):
    # Create Dwelling from input files, see bin/run_dwelling.py
    dwelling = Dwelling(**dwelling_args)

    # Extract equipment by its end use and update simulation properties
    equipment = dwelling.get_equipment_by_end_use("Water Heating")
    equipment.main_simulator = True
    equipment.save_results = dwelling.save_results
    equipment.export_res = dwelling.export_res
    equipment.results_file = dwelling.results_file

    # If necessary, update equipment schedule
    equipment.model.schedule['Zone Temperature (C)'] = 19.722222 #from the UEF standard https://www.energy.gov/eere/buildings/articles/2014-06-27-issuance-test-procedures-residential-and-commercial-water
    #equipment.model.schedule['Water Use Schedule (L/min)'] = load_profile_in #converted the schedule files directly to L/min
    equipment.model.schedule['Mains Temperature (C)'] = 14.4444
    #TODO: 50% RH schedule? Will have some impact on HP performance, but not much
    equipment.reset_time()

    # Simulate equipment
    df = equipment.simulate()

    # print(df.head())
    CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))
    #CreateFigures.plot_time_series_detailed((df["Hot Water Delivered (L/min)"],))
    #CreateFigures.plot_time_series_detailed((df["Water Tank PCM Temperature (C)"],))
    CreateFigures.plt.title(plot_title)
    CreateFigures.plt.suptitle(load_profile_in)
    CreateFigures.plt.show()


if __name__ == '__main__':
    #run without PCM
    run_water_heater(dwelling_args,no_pcm_title,load_profile)

    #update to include PCM
    dwelling_args = add_pcm_model(dwelling_args)
    
    #Run with PCM
    run_water_heater(dwelling_args,with_pcm_title,load_profile)
