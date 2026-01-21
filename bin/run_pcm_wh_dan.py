import datetime as dt


from ochre import Dwelling, CreateFigures
from ochre.Models import TankWithPCM
from bin.run_dwelling import dwelling_args
import matplotlib.pyplot as plt
import time


pcm_water_node = 7
# pcm_vol_fraction = 6.59375e-8 # min fraction or thereabouts
# pcm_vol_fraction = 0.9999999999999725
# pcm_vol_fraction = 6.582730627258115e-08

pcm_vol_fractions = [
    6.582730627258115e-08,
    0.0001,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.99,
    0.999,
    0.9999,
    0.9999999999999725,
]

pcm_vol_fractions = [0.5]


# UEF draw profiles
# LowUseUEF = 'LowUseL.csv'
# MediumUseUEF = 'MediumUseL.csv'
# HighUseUEF = 'HighUseL.csv'

# no_pcm_title = "No PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)
# with_pcm_title = "With PCM, PCM Water Node:"+ str(pcm_water_node) +", PCM Vol Fraction:" +str(pcm_vol_fraction)

load_profile = "MediumUseL.csv"

dwelling_args.update(
    {
        "time_res": dt.timedelta(minutes=1),  # time resolution of the simulation
        "duration": dt.timedelta(days=2),  # duration of the simulation
        "verbosity": 9,
        "schedule_input_file": load_profile,  # changes the default load profile in run_dwelling.py for this code to call the UEF load_profile
        "output_path": "../OCHRE_output/OCHRE_results/results/",
        "name": "pcm_none_default_water_heater",
    }
)


def add_pcm_model(dwelling_args, name, pcm_vol_fraction):
    dwelling_args["Equipment"]["Water Heating"] = {
        "model_class": TankWithPCM,
        "water_nodes": 12,
        "Water Tank": {
            "pcm_water_node": pcm_water_node,
            "pcm_vol_fraction": pcm_vol_fraction,
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
    # CreateFigures.plot_time_series_detailed((df["Hot Water Outlet Temperature (C)"],))
    # CreateFigures.plt.title(plot_title)
    # CreateFigures.plt.suptitle(load_profile_in)

    # # print all water tank temperatures
    cols = [f"T_WH{i}" for i in range(1, 13)]
    if "With PCM" in plot_title:
        cols += ["T_PCM"]

    # df[cols].plot()
    # CreateFigures.plt.show()

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
            dwelling_args, f"pcm_{pcm_vol_fraction}", pcm_vol_fraction
        )

        # #Run with PCM
        uef_val = run_water_heater(
            dwelling_args,
            f"PCM_VOL_FRACTION:{pcm_vol_fraction}",
            load_profile,
            f"PCM_VOL_FRACTION:{pcm_vol_fraction}",
        )
        uef.append(uef_val)

# uef = [
#     0.9604064550862321,
#     0.9372060523311518,
#     0.9372248110213233,
#     0.9372466030232178,
#     0.9372942471509957,
#     0.9373411576821054,
#     0.9373929128440206,
#     0.9374401194642804,
#     0.9374761080657925,
#     0.9375125569208861,
#     0.9375628003756745,
#     0.937752258394644,
#     0.937778664503084,
#     0.9378047377363584,
#     0.9636448250902498,
# ]
plt.plot(pcm_vol_fractions, uef, marker="o", label="UEF Data")

# for x, y in zip(pcm_vol_fractions, uef):
#     plt.text(x, y, f"(UEF:{y:.2f})", fontsize=8, ha='right', va='bottom')

# Label the axes and add a title
plt.xlabel("PCM Volume Fraction")
plt.ylabel("UEF")
plt.title("UEF vs PCM Volume Fraction at Node 7")

print(f"Time taken: {(time.perf_counter() - _start_time):.3f} sec")
print(f"vol_fractions: {pcm_vol_fractions}")
print(f"UEF: {uef}")
# Display the plot
plt.legend()
plt.grid(True)
plt.show()
