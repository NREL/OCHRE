"""Update results_up00.csv with OCHRE simulation results from test_result/.

Run this script after test_resstock_golden.py when legitimate OCHRE changes
alter energy outputs. It reads results_annual.csv from each building's test
output and updates the corresponding energy columns in results_up00.csv.

Usage:
    python test/test_dwelling/update_resstock_golden_reference.py
"""

import csv
import os
import sys

# Allow running from the OCHRE root directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from test import test_path
from test.test_dwelling.resstock_test_utils import metric_to_column, read_results_annual

GOLDEN_TEST_RESULT_PATH = os.path.join(test_path, "resstock_golden", "test_result")
GOLDEN_RESULTS_CSV = os.path.join(test_path, "resstock_golden", "results_up00.csv")


def main():
    if not os.path.isfile(GOLDEN_RESULTS_CSV):
        print(f"Reference file not found: {GOLDEN_RESULTS_CSV}")
        sys.exit(1)

    # Read existing results_up00.csv preserving column order
    with open(GOLDEN_RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Index rows by building name
    row_index = {}
    for row in rows:
        bldg_id = int(row["building_id"])
        bldg_name = f"bldg{bldg_id:07d}"
        row_index[bldg_name] = row

    # Identify energy columns in results_up00.csv
    energy_columns = {
        col for col in fieldnames if col.startswith("report_simulation_output.") and col.endswith("_m_btu")
    }

    # Discover buildings with test results
    if not os.path.isdir(GOLDEN_TEST_RESULT_PATH):
        print(f"No test results found in {GOLDEN_TEST_RESULT_PATH}")
        print("Run the golden tests first:")
        print("  pytest test/test_dwelling/test_resstock_golden.py -v --tb=short")
        sys.exit(1)

    buildings_updated = 0
    values_changed = 0
    unmapped_metrics = set()

    for bldg_name in sorted(os.listdir(GOLDEN_TEST_RESULT_PATH)):
        annual_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name, "results_annual.csv")
        if not os.path.isfile(annual_path):
            continue

        if bldg_name not in row_index:
            print(f"  Warning: {bldg_name} has test results but is not in results_up00.csv")
            continue

        row = row_index[bldg_name]
        annual = read_results_annual(annual_path)
        building_changed = False

        for metric_name, value in annual.items():
            col_name = metric_to_column(metric_name)
            if col_name not in energy_columns:
                unmapped_metrics.add((metric_name, col_name))
                continue

            old_val = row.get(col_name, "")
            new_val = str(round(value, 3))
            if old_val != new_val:
                row[col_name] = new_val
                values_changed += 1
                building_changed = True

        if building_changed:
            buildings_updated += 1

    # Write updated CSV
    with open(GOLDEN_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated {GOLDEN_RESULTS_CSV}")
    print(f"  Buildings updated: {buildings_updated}")
    print(f"  Values changed: {values_changed}")

    if unmapped_metrics:
        print("\n  Metrics in results_annual.csv with no matching column in results_up00.csv:")
        for metric, col in sorted(unmapped_metrics):
            print(f"    {metric} -> {col}")


if __name__ == "__main__":
    main()
