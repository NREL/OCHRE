import datetime as dt


from ochre import Dwelling, CreateFigures
from ochre.Models import TankWithPCM, TankWithMultiPCM
from bin.run_dwelling import dwelling_args
import matplotlib.pyplot as plt
import time
from itertools import chain, combinations




num_nodes = 12
vol_fract = 0.5

# pcm_vol_fractions = [{i: vol_fract for i in range(1, n + 1)} for n in range(1, num_nodes + 1)]
# pcm_vol_fractions = [
#                     {7: vol_fract},
#                     ]

# pcm_vol_fractions = [
#                     {12: vol_fract},
#                     ]

pcm_vol_fractions = [{node: vol_fract for node in range(1, num_nodes + 1)}]

# UEF draw profiles
# LowUseUEF = 'LowUseL.csv'
# MediumUseUEF = 'MediumUseL.csv'
# HighUseUEF = 'HighUseL.csv'

# no_pcm_title = "No PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)
# with_pcm_title = "With PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)

load_profile = "MediumUseL.csv"

def convert_dict_to_name(dict):
    return "".join([str(key) + "_" + str(value) + "_" for key, value in dict.items()])

dwelling_args.update(
    {
        "time_res": dt.timedelta(minutes=1),  # time resolution of the simulation
        "duration": dt.timedelta(days=2),  # duration of the simulation
        "verbosity": 9,
        "schedule_input_file": load_profile,  # changes the default load profile in run_dwelling.py for this code to call the UEF load_profile
        "output_path": "../OCHRE_output/OCHRE_results/results/all_nodes",
        "name": "pcm_none_default_water_heater",
    }
)


def add_pcm_model(dwelling_args, name, pcm_vol_fractions):
    dwelling_args["Equipment"]["Water Heating"] = {
        "model_class": TankWithMultiPCM,
        "water_nodes": 12,
        "Water Tank": {
            "pcm_node_vol_fractions": pcm_vol_fractions,
        },
    }

    dwelling_args["name"] = name
    dwelling_args["output_path"] = "../OCHRE_output/OCHRE_results/results/"
    return dwelling_args


def run_water_heater(dwelling_args, plot_title, load_profile_in, name):
    # Create Dwelling from input files, see bin/run_dwelling.py
    dwelling = Dwelling(**dwelling_args)

    # Extract equipment by its end use and update simulation properties
    equipment = dwelling.get_equipment_by_end_use("Water Heating")
    equipment.main_simulator = True
    equipment.save_results = dwelling.save_results
    # equipment.model.save_results = True
    # equipment.model.results_file = "test.csv"
    equipment.export_res = dwelling.export_res
    equipment.results_file = dwelling.results_file
    equipment.verbosity = 9
    equipment.model.verbosity = 9
    equipment.metrics_verbosity = 9

    # If necessary, update equipment schedule
    equipment.model.schedule["Zone Temperature (C)"] = (
        19.722222  # from the UEF standard https://www.energy.gov/eere/buildings/articles/2014-06-27-issuance-test-procedures-residential-and-commercial-water
    )
    # equipment.model.schedule['Water Use Schedule (L/min)'] = load_profile_in #converted the schedule files directly to L/min
    equipment.model.schedule["Mains Temperature (C)"] = 14.4444
    # TODO: 50% RH schedule? Will have some impact on HP performance, but not much
    equipment.reset_time()

    # Simulate equipment
    df = equipment.simulate()
    # print(df.columns)

    # # # print(df.head())
    CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))
    CreateFigures.plt.title(plot_title)
    CreateFigures.plt.suptitle(load_profile_in)

    # # print all water tank temperatures
    cols = [f"T_WH{i}" for i in range(1, 13)]
    if "With PCM" in plot_title:
        cols += ["T_PCM"]

    df[cols].plot()
    CreateFigures.plt.show()

    # calculate UEF
    Q_cons = (
        df["Water Heating Electric Power (kW)"].sum() * 1000
    )  # not sure if this is the correct term that I should be pulling
    Q_load = df[
        "Hot Water Delivered (W)"
    ].sum()  # not sure if this is the correct term that I should be pulling
    UEF = Q_load / Q_cons
    print(f"UEF = {UEF}")
    return UEF


if __name__ == "__main__":
    # Define initial bounds for binary search
    _start_time = time.perf_counter()

    # run without PCM
    # run_water_heater(dwelling_args,no_pcm_title,load_profile, no_pcm_title)

    # update to include PCM
    uef = []
    for i, pcm_vol_fraction in enumerate(pcm_vol_fractions):
        dwelling_args = add_pcm_model(
            dwelling_args, f"pcm_{convert_dict_to_name(pcm_vol_fraction)}", pcm_vol_fraction
        )

        # #Run with PCM
        uef_val = run_water_heater(
            dwelling_args,
            f"PCM_VOL_FRACTION:{convert_dict_to_name(pcm_vol_fraction)}",
            load_profile,
            f"PCM_VOL_FRACTION:{convert_dict_to_name(pcm_vol_fraction)}",
        )
        uef.append(uef_val)
