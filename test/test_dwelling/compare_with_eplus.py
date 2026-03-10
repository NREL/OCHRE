"""Compare OCHRE golden test results against EnergyPlus reference values.

This script should be run after test_resstock_golden.py because it reads
OCHRE simulation outputs from test/resstock_golden/test_result/.

Outputs one CSV file per metric into test/resstock_golden/comparison/.
Buildings that failed OCHRE simulation appear with NA values at the bottom.

Usage:
    python test/test_dwelling/compare_with_eplus.py
"""

import csv
import math
import os
import sys

# Allow running from the OCHRE root directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from test import test_path

GOLDEN_TEST_RESULT_PATH = os.path.join(test_path, "resstock_golden", "test_result")
GOLDEN_RESULTS_CSV = os.path.join(test_path, "resstock_golden", "results_up00.csv")
GOLDEN_EPLUS_CSV = os.path.join(test_path, "resstock_golden", "results_up00_eplus.csv")
COMPARISON_OUTPUT_PATH = os.path.join(test_path, "resstock_golden", "comparison")

# Columns to compare between results_annual.csv and EnergyPlus reference.
# Each tuple: (results CSV column name, results_annual.csv metric name)
COLUMNS_TO_COMPARE = [
    ("report_simulation_output.fuel_use_electricity_total_m_btu", "Fuel Use: Electricity: Total (MBtu)"),
    ("report_simulation_output.fuel_use_natural_gas_total_m_btu", "Fuel Use: Natural Gas: Total (MBtu)"),
    ("report_simulation_output.end_use_electricity_heating_m_btu", "End Use: Electricity: Heating (MBtu)"),
    ("report_simulation_output.end_use_electricity_cooling_m_btu", "End Use: Electricity: Cooling (MBtu)"),
    ("report_simulation_output.end_use_electricity_hot_water_m_btu", "End Use: Electricity: Hot Water (MBtu)"),
    ("report_simulation_output.end_use_electricity_plug_loads_m_btu", "End Use: Electricity: Plug Loads (MBtu)"),
    ("report_simulation_output.load_heating_delivered_m_btu", "Load: Heating: Delivered (MBtu)"),
    ("report_simulation_output.load_cooling_delivered_m_btu", "Load: Cooling: Delivered (MBtu)"),
    ("report_simulation_output.load_hot_water_delivered_m_btu", "Load: Hot Water: Delivered (MBtu)"),
]

# Building characteristics to include in summary tables.
# Each tuple: (results_up00.csv column name, display header)
SUMMARY_COLUMNS = [
    ("build_existing_model.heating_fuel", "Heating Fuel"),
    ("build_existing_model.hvac_heating_type", "HVAC Heating"),
    ("build_existing_model.hvac_cooling_type", "HVAC Cooling"),
    ("build_existing_model.water_heater_fuel", "WH Fuel"),
]


def _read_results_annual(path):
    """Read results_annual.csv into a dict keyed by metric name."""
    results = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            results[row[0].strip()] = float(row[1])
    return results


def _load_expected_from_csv(csv_path):
    """Load expected values from a results CSV, keyed by bldg_name."""
    expected = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            expected[bldg_name] = {}
            for csv_col, annual_metric in COLUMNS_TO_COMPARE:
                val = row.get(csv_col, "")
                if val:
                    expected[bldg_name][annual_metric] = float(val)
    return expected


def _load_building_characteristics(csv_path):
    """Load building characteristics from results CSV for summary display."""
    chars = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            chars[bldg_name] = {}
            for csv_col, _ in SUMMARY_COLUMNS:
                chars[bldg_name][csv_col] = row.get(csv_col, "")
    return chars


def _sanitize_filename(metric):
    """Convert a metric name to a safe filename."""
    name = metric.replace(":", "").replace("(", "").replace(")", "")
    name = name.replace("  ", " ").strip().replace(" ", "_").lower()
    return name


def main():
    if not os.path.isfile(GOLDEN_EPLUS_CSV):
        print(f"EnergyPlus reference file not found: {GOLDEN_EPLUS_CSV}")
        sys.exit(1)

    expected_eplus = _load_expected_from_csv(GOLDEN_EPLUS_CSV)
    building_chars = _load_building_characteristics(GOLDEN_RESULTS_CSV)
    char_headers = [hdr for _, hdr in SUMMARY_COLUMNS]

    # All buildings come from the EPlus reference (the complete roster)
    all_buildings = sorted(expected_eplus.keys())

    # Load OCHRE results for buildings that have them
    ochre_results = {}
    for bldg_name in all_buildings:
        annual_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name, "results_annual.csv")
        if os.path.isfile(annual_path):
            ochre_results[bldg_name] = _read_results_annual(annual_path)

    if not ochre_results:
        print(f"No OCHRE test results found in {GOLDEN_TEST_RESULT_PATH}")
        print("Run the golden tests first:")
        print("  pytest test/test_dwelling/test_resstock_golden.py -v --tb=short")
        sys.exit(1)

    # Collect rows per metric: buildings with results first, then NA buildings
    rows_per_metric = {}
    for _, annual_metric in COLUMNS_TO_COMPARE:
        rows_with_results = []
        rows_na = []

        for bldg_name in all_buildings:
            eplus_val = expected_eplus.get(bldg_name, {}).get(annual_metric)
            if eplus_val is None:
                continue

            chars = {hdr: building_chars.get(bldg_name, {}).get(col, "") for col, hdr in SUMMARY_COLUMNS}
            actual = ochre_results.get(bldg_name)

            if actual is None:
                # Building failed OCHRE simulation
                rows_na.append((bldg_name, None, eplus_val, None, chars))
            else:
                actual_val = actual.get(annual_metric)
                if actual_val is None:
                    rows_na.append((bldg_name, None, eplus_val, None, chars))
                else:
                    if abs(eplus_val) > 1e-9:
                        pct = (actual_val - eplus_val) / abs(eplus_val) * 100.0
                    else:
                        pct = float("inf") if actual_val > 0 else float("-inf") if actual_val < 0 else 0.0
                    rows_with_results.append((bldg_name, actual_val, eplus_val, pct, chars))

        # Sort results by abs(%diff) descending, then append NA rows at the bottom
        rows_with_results.sort(key=lambda r: abs(r[3]) if math.isfinite(r[3]) else float("inf"), reverse=True)
        rows_per_metric[annual_metric] = rows_with_results + rows_na

    # Write CSV files
    os.makedirs(COMPARISON_OUTPUT_PATH, exist_ok=True)
    n_with_results = len(ochre_results)
    n_total = len(all_buildings)

    for metric, rows in sorted(rows_per_metric.items()):
        if not rows:
            continue

        filename = _sanitize_filename(metric) + ".csv"
        filepath = os.path.join(COMPARISON_OUTPUT_PATH, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Building"] + char_headers + ["OCHRE (MBtu)", "EPlus (MBtu)", "%Diff"])

            for bldg_name, ochre_val, eplus_val, pct_diff, chars in sorted(rows, key=lambda r: r[0]):
                char_vals = [chars.get(hdr, "") for hdr in char_headers]
                if ochre_val is None:
                    writer.writerow([bldg_name] + char_vals + ["NA", f"{eplus_val:.4f}", "NA"])
                elif pct_diff is not None and math.isfinite(pct_diff):
                    writer.writerow(
                        [bldg_name] + char_vals + [f"{ochre_val:.4f}", f"{eplus_val:.4f}", f"{pct_diff:+.1f}%"]
                    )
                else:
                    writer.writerow([bldg_name] + char_vals + [f"{ochre_val:.4f}", f"{eplus_val:.4f}", "inf"])

        print(f"  Written: {filepath}")

    # Print summary to stdout
    print()
    print("=" * 80)
    print("  EPlus Cross-Validation Summary")
    print(f"  ({n_with_results}/{n_total} buildings with OCHRE results)")
    print("=" * 80)

    for _, metric in COLUMNS_TO_COMPARE:
        rows = rows_per_metric.get(metric)
        if not rows:
            continue

        # Compute dynamic column widths for characteristics
        char_widths = {}
        for hdr in char_headers:
            vals = [r[4].get(hdr, "") for r in rows]
            char_widths[hdr] = max(len(hdr), max((len(v) for v in vals), default=0))

        char_hdr_parts = "  ".join(f"{hdr:<{char_widths[hdr]}}" for hdr in char_headers)
        char_sep_parts = "  ".join("-" * char_widths[hdr] for hdr in char_headers)

        print()
        print(f"  {metric}")
        print(f"  {'Building':<16}{char_hdr_parts}  {'OCHRE':>12}  {'EPlus':>12}  {'%Diff':>10}")
        print(f"  {'-' * 16}{char_sep_parts}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

        for bldg_name, ochre_val, eplus_val, pct_diff, chars in rows:
            char_vals = "  ".join(f"{chars.get(hdr, ''):<{char_widths[hdr]}}" for hdr in char_headers)
            if ochre_val is None:
                print(f"  {bldg_name:<16}{char_vals}  {'NA':>12}  {eplus_val:>12.3f}  {'NA':>10}")
            elif pct_diff is not None and math.isfinite(pct_diff):
                print(f"  {bldg_name:<16}{char_vals}  {ochre_val:>12.3f}  {eplus_val:>12.3f}  {pct_diff:>+9.1f}%")
            else:
                print(f"  {bldg_name:<16}{char_vals}  {ochre_val:>12.3f}  {eplus_val:>12.3f}  {'inf':>10}")

    print()


if __name__ == "__main__":
    main()
