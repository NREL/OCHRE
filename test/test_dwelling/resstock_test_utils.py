"""Shared utilities and constants for ResStock golden test scripts."""

import csv
import os

from ochre.utils.resstock import to_underscore_case
from test import test_path

# Paths shared across golden test scripts
GOLDEN_PATH = os.path.join(test_path, "resstock_golden")
GOLDEN_DATA_PATH = os.path.join(GOLDEN_PATH, "eplus_result")
GOLDEN_WEATHER_PATH = os.path.join(GOLDEN_PATH, "weather")
GOLDEN_RESULTS_CSV = os.path.join(GOLDEN_PATH, "ochre_result", "ochre_annual_result.csv")
GOLDEN_NEW_RESULTS_CSV = os.path.join(GOLDEN_PATH, "ochre_result", "ochre_annual_result_new.csv")
GOLDEN_EPLUS_CSV = os.path.join(GOLDEN_PATH, "eplus_result", "eplus_annual_result.csv")
GOLDEN_TEST_RESULT_PATH = os.path.join(GOLDEN_PATH, "ochre_result")
COMPARISON_OUTPUT_PATH = os.path.join(GOLDEN_PATH, "comparison")


def metric_to_column(metric_name):
    """Convert a results_annual.csv metric name to an annual result CSV column name.

    Example: "End Use: Electricity: Heating (MBtu)"
          -> "report_simulation_output.end_use_electricity_heating_m_btu"
    """
    return f"report_simulation_output.{to_underscore_case(metric_name)}"


def read_results_annual(path):
    """Read results_annual.csv into a dict mapping metric name to float value."""
    results = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            results[row[0].strip()] = float(row[1])
    return results


def load_results_csv(csv_path):
    """Load all numeric output metrics from a results CSV.

    Auto-discovers all report_simulation_output.* columns with non-empty
    numeric values. Returns {bldg_name: {column_name: float_value}}.
    """
    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        output_cols = sorted(
            col for col in reader.fieldnames if col.startswith("report_simulation_output.")
        )
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            results[bldg_name] = {}
            for col in output_cols:
                val = row.get(col, "")
                if val:
                    try:
                        results[bldg_name][col] = float(val)
                    except ValueError:
                        pass
    return results
