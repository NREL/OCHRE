"""Compare OCHRE golden test results against EnergyPlus reference values.

This script should be run after generate_ochre_result.py because it reads
OCHRE simulation outputs from test/resstock_golden/ochre_result/.

Dynamically discovers all report_simulation_output columns with non-empty
values in both the OCHRE and EPlus results, and outputs one CSV file per
metric into test/resstock_golden/comparison/.

Usage:
    python test/resstock_golden/compare_ochre_and_eplus.py
"""

import csv
import math
import os
import sys

# Allow running from the OCHRE root directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from test.test_dwelling.resstock_test_utils import (
    COMPARISON_OUTPUT_PATH,
    GOLDEN_EPLUS_CSV,
    GOLDEN_NEW_RESULTS_CSV,
    GOLDEN_RESULTS_CSV,
    load_results_csv,
)

# Prefix to strip from column names for display
COL_PREFIX = "report_simulation_output."

# Building characteristics always shown for every metric.
SUMMARY_COLUMNS = [
    ("build_existing_model.state", "State"),
    ("build_existing_model.geometry_building_type_recs", "Building Type"),
    ("build_existing_model.vintage", "Vintage"),
    ("build_existing_model.geometry_floor_area", "Floor Area"),
]

# Extra characteristic columns by end-use keyword, appended after SUMMARY_COLUMNS.
# Keys are substrings matched against the metric column name (prefix-stripped).
# All matching patterns contribute columns (a metric can match multiple keys).
METRIC_EXTRA_COLUMNS = {
    "cooling": [
        ("build_existing_model.hvac_cooling_efficiency", "Cooling"),
        ("build_existing_model.hvac_heating_efficiency", "Heating"),
    ],
    "heating": [
        ("build_existing_model.hvac_heating_efficiency", "Heating"),
        ("build_existing_model.hvac_cooling_efficiency", "Cooling"),
    ],
    "hot_water": [
        ("build_existing_model.water_heater_efficiency", "Water Heating"),
    ],
    "ceiling_fan": [
        ("build_existing_model.ceiling_fan", "Ceiling Fan"),
    ],
    "clothes_dryer": [
        ("build_existing_model.clothes_dryer", "Clothes Dryer"),
        ("build_existing_model.clothes_dryer_usage_level", "Dryer Usage Level"),
    ],
    "clothes_washer": [
        ("build_existing_model.clothes_washer", "Clothes Washer"),
        ("build_existing_model.clothes_washer_usage_level", "Washer Usage Level"),
    ],
    "dishwasher": [
        ("build_existing_model.dishwasher", "Dishwasher"),
        ("build_existing_model.dishwasher_usage_level", "Dishwasher Usage Level"),
    ],
    "refrigerator": [
        ("build_existing_model.refrigerator", "Refrigerator"),
        ("build_existing_model.refrigerator_usage_level", "Refrigerator Usage Level"),
    ],
    "freezer": [
        ("build_existing_model.misc_freezer", "Freezer"),
    ],
    "range_oven": [
        ("build_existing_model.cooking_range", "Cooking Range"),
        ("build_existing_model.cooking_range_usage_level", "Cooking Usage Level"),
    ],
    "lighting": [
        ("build_existing_model.lighting", "Lighting"),
    ],
    "plug_loads": [
        ("build_existing_model.plug_loads", "Plug Loads"),
        ("build_existing_model.plug_load_diversity", "Plug Load Diversity"),
        ("build_existing_model.usage_level", "Usage Level"),
    ],
    "pv": [
        ("build_existing_model.pv_system_size", "PV Size"),
        ("build_existing_model.pv_orientation", "PV Orientation"),
    ],
    "electric_vehicle": [
        ("build_existing_model.electric_vehicle_charger", "EV Charger"),
        ("build_existing_model.electric_vehicle_battery", "EV Battery"),
    ],
    "pool_heater": [
        ("build_existing_model.misc_pool_heater", "Pool Heater"),
    ],
    "pool_pump": [
        ("build_existing_model.misc_pool_pump", "Pool Pump"),
    ],
    "permanent_spa": [
        ("build_existing_model.misc_hot_tub_spa", "Hot Tub/Spa"),
    ],
    "mech_vent": [
        ("build_existing_model.mechanical_ventilation", "Mech Vent"),
    ],
    "well_pump": [
        ("build_existing_model.misc_well_pump", "Well Pump"),
    ],
    "fireplace": [
        ("build_existing_model.misc_gas_fireplace", "Fireplace"),
    ],
    "grill": [
        ("build_existing_model.misc_gas_grill", "Grill"),
    ],
}

# All characteristic columns we might need (union of summary + all extras)
_ALL_CHAR_COLUMNS = list(SUMMARY_COLUMNS)
_seen = {col for col, _ in SUMMARY_COLUMNS}
for extras in METRIC_EXTRA_COLUMNS.values():
    for col, hdr in extras:
        if col not in _seen:
            _ALL_CHAR_COLUMNS.append((col, hdr))
            _seen.add(col)


def _get_columns_for_metric(metric_col):
    """Return SUMMARY_COLUMNS + any matching extra columns for a metric."""
    display_name = metric_col.removeprefix(COL_PREFIX)
    extras = []
    seen = {col for col, _ in SUMMARY_COLUMNS}
    for keyword, extra_cols in METRIC_EXTRA_COLUMNS.items():
        if keyword in display_name:
            for col, hdr in extra_cols:
                if col not in seen:
                    extras.append((col, hdr))
                    seen.add(col)
    return list(SUMMARY_COLUMNS) + extras


def _load_building_characteristics(csv_path):
    """Load building characteristics from results CSV for summary display."""
    chars = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            chars[bldg_name] = {}
            for csv_col, _ in _ALL_CHAR_COLUMNS:
                chars[bldg_name][csv_col] = row.get(csv_col, "")
    return chars


def _col_display(col):
    """Strip prefix from column name for display."""
    return col.removeprefix(COL_PREFIX)


def main():
    if not os.path.isfile(GOLDEN_EPLUS_CSV):
        print(f"EnergyPlus reference file not found: {GOLDEN_EPLUS_CSV}")
        sys.exit(1)

    eplus_results = load_results_csv(GOLDEN_EPLUS_CSV)
    ochre_csv = GOLDEN_NEW_RESULTS_CSV if os.path.isfile(GOLDEN_NEW_RESULTS_CSV) else GOLDEN_RESULTS_CSV
    ochre_results = load_results_csv(ochre_csv)
    print(f"  OCHRE results from: {ochre_csv}")
    building_chars = _load_building_characteristics(GOLDEN_EPLUS_CSV)

    # All buildings come from the EPlus reference (the complete roster)
    all_buildings = sorted(eplus_results.keys())

    # Buildings with no OCHRE results at all (simulation failed/crashed)
    failed_buildings = {bldg for bldg in all_buildings if not ochre_results.get(bldg)}

    # Discover columns with non-empty values in at least one building in both CSVs
    eplus_cols = {col for bldg in eplus_results.values() for col in bldg}
    ochre_cols = {col for bldg in ochre_results.values() for col in bldg}
    compared_cols = sorted(eplus_cols & ochre_cols)

    if not compared_cols:
        print("No common output columns found between OCHRE and EPlus results.")
        sys.exit(1)

    # Collect rows per metric: all buildings included, NA for missing values
    rows_per_metric = {}
    for col in compared_cols:
        rows_with_results = []
        rows_na = []

        for bldg_name in all_buildings:
            eplus_val = eplus_results.get(bldg_name, {}).get(col)
            ochre_val = ochre_results.get(bldg_name, {}).get(col)

            if ochre_val is not None and eplus_val is not None:
                if abs(eplus_val) > 1e-9:
                    pct = (ochre_val - eplus_val) / abs(eplus_val) * 100.0
                else:
                    pct = float("inf") if ochre_val > 0 else float("-inf") if ochre_val < 0 else 0.0
                rows_with_results.append((bldg_name, ochre_val, eplus_val, pct))
            else:
                rows_na.append((bldg_name, ochre_val, eplus_val, None))

        # Sort results by abs(%diff) descending, then append NA rows at the bottom
        rows_with_results.sort(key=lambda r: abs(r[3]) if math.isfinite(r[3]) else float("inf"), reverse=True)
        rows_per_metric[col] = rows_with_results + rows_na

    # Write CSV files
    os.makedirs(COMPARISON_OUTPUT_PATH, exist_ok=True)
    n_with_results = len(ochre_results)
    n_total = len(all_buildings)

    for col, rows in sorted(rows_per_metric.items()):
        if not rows:
            continue

        metric_columns = _get_columns_for_metric(col)
        metric_headers = [hdr for _, hdr in metric_columns]

        filename = _col_display(col) + ".csv"
        filepath = os.path.join(COMPARISON_OUTPUT_PATH, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Building"] + metric_headers + ["OCHRE", "EPlus", "%Diff"])

            for bldg_name, ochre_val, eplus_val, pct_diff in sorted(rows, key=lambda r: r[0]):
                bldg_chars = building_chars.get(bldg_name, {})
                char_vals = [bldg_chars.get(csv_col, "") for csv_col, _ in metric_columns]
                failed = bldg_name in failed_buildings
                ochre_str = f"{ochre_val:.4f}" if ochre_val is not None else ("FAILED" if failed else "NA")
                eplus_str = f"{eplus_val:.4f}" if eplus_val is not None else "NA"
                if pct_diff is not None and math.isfinite(pct_diff):
                    pct_str = f"{pct_diff:+.1f}%"
                elif pct_diff is not None:
                    pct_str = "inf"
                else:
                    pct_str = "FAILED" if failed else "NA"
                writer.writerow([bldg_name] + char_vals + [ochre_str, eplus_str, pct_str])

        print(f"  Written: {filepath}")

    # Print summary to stdout
    print()
    print("=" * 80)
    print("  EPlus Cross-Validation Summary")
    print(f"  ({n_with_results}/{n_total} buildings with OCHRE results)")
    print(f"  ({len(compared_cols)} metrics compared)")
    print("=" * 80)

    for col in compared_cols:
        rows = rows_per_metric.get(col)
        if not rows:
            continue

        metric_columns = _get_columns_for_metric(col)
        metric_headers = [hdr for _, hdr in metric_columns]

        # Compute dynamic column widths for characteristics
        char_widths = {}
        for csv_col, hdr in metric_columns:
            vals = [building_chars.get(r[0], {}).get(csv_col, "") for r in rows]
            char_widths[hdr] = max(len(hdr), max((len(v) for v in vals), default=0))

        char_hdr_parts = "  ".join(f"{hdr:<{char_widths[hdr]}}" for hdr in metric_headers)
        char_sep_parts = "  ".join("-" * char_widths[hdr] for hdr in metric_headers)

        print()
        print(f"  {_col_display(col)}")
        print(f"  {'Building':<16}{char_hdr_parts}  {'OCHRE':>12}  {'EPlus':>12}  {'%Diff':>10}")
        print(f"  {'-' * 16}{char_sep_parts}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

        for bldg_name, ochre_val, eplus_val, pct_diff in rows:
            bldg_chars = building_chars.get(bldg_name, {})
            char_vals = "  ".join(
                f"{bldg_chars.get(csv_col, ''):<{char_widths[hdr]}}" for csv_col, hdr in metric_columns
            )
            failed = bldg_name in failed_buildings
            na_label = "FAILED" if failed else "NA"
            ochre_str = f"{ochre_val:>12.3f}" if ochre_val is not None else f"{na_label:>12}"
            eplus_str = f"{eplus_val:>12.3f}" if eplus_val is not None else f"{'NA':>12}"
            if pct_diff is not None and math.isfinite(pct_diff):
                pct_str = f"{pct_diff:>+9.1f}%"
            elif pct_diff is not None:
                pct_str = f"{'inf':>10}"
            else:
                pct_str = f"{na_label:>10}"
            print(f"  {bldg_name:<16}{char_vals}  {ochre_str}  {eplus_str}  {pct_str}")

    print()


if __name__ == "__main__":
    main()
