# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
#     "buildstock-query @ git+https://github.com/NatLabRockies/buildstock-query.git@55b9d70a17fb62fb03e0e5f31c95cc739dbd0330",
# ]
# ///
"""Generate a minimal buildstock.csv that covers all key housing characteristics.
Update resstock_directory to point to your local ResStock copy. Output CSV will be saved to output_buildstock_csv path.
Usage:
    uv run generate_minimal_buildstock.py
"""
# %%
import pandas as pd
from buildstock_query.tools.upgrades_analyzer import UpgradesAnalyzer
import pathlib

root_dir = pathlib.Path(__file__).parent.parent.parent.parent
resstock_directory = root_dir / "resstock"  # Relative to this script
output_buildstock_csv = root_dir / "resstock" / "project_national" / "ochre_minimal_buildstock.csv"

project_national_dir = resstock_directory / "project_national"
yaml_file = project_national_dir / "sdr_upgrades_tmy3.yml"
buildstock_csv = resstock_directory / "resources" / "res_ochre_550K.csv"
opt_sat_file = project_national_dir / "resources" / "options_saturations.csv"
ua = UpgradesAnalyzer(
    buildstock=str(buildstock_csv),
    yaml_file=str(yaml_file),
    opt_sat_file=str(opt_sat_file)
)
report_df = ua.get_report()

df = ua.buildstock_df

# %%
heavily_electric_sfd_chars = {  # Large number of electric end uses\
    "hvac cooling type": "^None",
    "hvac heating type": "^None",
    "heating fuel": "Electricity",
    "hvac heating efficiency": {"ASHP, SEER 10, 6.2 HSPF", "ASHP, SEER 13, 7.7 HSPF", "ASHP, SEER 15, 8.5 HSPF"},
    "water heater fuel": "Electricity",
    "electric vehicle ownership": "Yes",
    "geometry building type recs": "Single-Family Detached",
    "geometry garage": "^None"
}

def is_heavily_electric(row):
    for col, val in heavily_electric_sfd_chars.items():
        if isinstance(val, set):
            if row[col] not in val:
                return False
        elif isinstance(val, str) and val.startswith("^"):
            if row[col] == val[1:]:
                return False
        elif row[col] != val:
            return False

    return True
# Add a column to the DataFrame indicating whether each building is heavily electric
# This ensures we will get at least one building that is heavily electric
df["is_heavily_electric"] = df.apply(is_heavily_electric, axis=1)

# %%
# deprioritize characteristics OCHRE doesn't model
# We still want at least one instance of these - just not too many 
soft_avoid_chars = [
    ("water heater fuel", "Solar Thermal"),
    ("heating fuel", "Wood"),
    ("heating fuel", "Propane"),
    ("heating fuel", "Fuel Oil"),
    ("vacancy status", "Vacant"),
    ("hvac cooling type", "Central AC")
]

chars_to_cover = ua.get_characteristics(min_cardinality=2, max_cardinality=20)

# This prevents too many fuel oil buildings
if "hvac secondary heating efficiency" in chars_to_cover:
    print("Removing 'HVAC Secondary Heating Efficiency' from characteristics to cover")
    chars_to_cover.remove("hvac secondary heating efficiency")  # Not modeled by OCHRE, often missing

minimal_bldgs = ua.get_minimal_representative_buildings(
    report_df,
    must_cover_chars=chars_to_cover,
    include_never_upgraded=True,
    verbose=True,
    soft_avoid_chars=soft_avoid_chars,
)
minimal_df = ua.buildstock_df_original.set_index("Building").loc[list(minimal_bldgs)]


minimal_df_all_cols = ua.buildstock_df.loc[list(minimal_bldgs)]
heavily_electric_bldgs = minimal_df_all_cols[minimal_df_all_cols["is_heavily_electric"]]
first_bldg = heavily_electric_bldgs.index[0]
heavily_electric_df = ua.buildstock_df_original.set_index("Building").loc[[heavily_electric_bldgs.index[0]],:]
forced_chars = {
    "Ceiling Fan": "Standard Efficiency",
    "Misc Freezer": "EF 12, National Average",
    "Misc Hot Tub Spa": "Electricity",
    "Misc Pool": "Has Pool",
    "Misc Pool Heater": "Electricity",
    "Misc Pool Pump": "1.0 HP Pump",
    "Misc Well Pump": "Typical Efficiency",
    "Clothes Dryer": "Electric",
    "Clothes Washer": "Standard",
    "Dishwasher": "318 Rated kWh",
    "Cooking Range": "Electric Resistance",
    "Refrigerator": "EF 15.9",
    "Hot Water Distribution": "R-2, Timer",
    "Mechanical Ventilation": "ERV, 72%",
    "PV System Size": "11.0 kWDC",
    "Has PV": "Yes",
    "PV Orientation":"East"
}
for col, val in forced_chars.items():
    heavily_electric_df[col] = val
heavily_electric_df.index = [f"9{first_bldg[1:]:0>6}"]
final_df = pd.concat([minimal_df, heavily_electric_df]).drop_duplicates()
final_df.to_csv(output_buildstock_csv, index_label="Building")
print(f"Generated {output_buildstock_csv} with {len(final_df)} buildings covering {len(chars_to_cover)} characteristics.")
uncovered_chars = set(ua.buildstock_df) - set(chars_to_cover)
print(f"{'Partially covered characteristics':<50} | {'N Unique':<10} | Example values")
print("-" * 80)
for char in uncovered_chars:
    if (ununique:=ua.buildstock_df[char].nunique()) > 1:
        example_values = ", ".join(map(lambda x: str(f'"{x}"'), ua.buildstock_df[char].unique()[:3]))
        print(f"{char:<50} | {ununique:<10} | {example_values}")
print("All other characteristics are fully covered - there is at least one building for every unique value of those characteristics.")
