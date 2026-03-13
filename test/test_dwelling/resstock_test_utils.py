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
GOLDEN_EPLUS_CSV = os.path.join(GOLDEN_PATH, "eplus_result", "eplus_annual_result.csv")
GOLDEN_TEST_RESULT_PATH = os.path.join(GOLDEN_PATH, "ochre_result")
COMPARISON_OUTPUT_PATH = os.path.join(GOLDEN_PATH, "comparison")

# Metrics validated between OCHRE and EnergyPlus / golden reference.
RESSTOCK_METRICS = [
    "Fuel Use: Electricity: Total (MBtu)",
    "Fuel Use: Natural Gas: Total (MBtu)",
    "End Use: Electricity: Heating (MBtu)",
    "End Use: Electricity: Cooling (MBtu)",
    "End Use: Electricity: Hot Water (MBtu)",
    "End Use: Electricity: Plug Loads (MBtu)",
    "Load: Heating: Delivered (MBtu)",
    "Load: Cooling: Delivered (MBtu)",
    "Load: Hot Water: Delivered (MBtu)",
]


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


def load_expected_from_csv(csv_path, metrics):
    """Load expected values from a results CSV, keyed by bldg_name.

    *metrics* is a list of metric name strings (e.g. "Fuel Use: Electricity: Total (MBtu)").
    CSV column names are derived via metric_to_column().
    """
    expected = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            expected[bldg_name] = {}
            for metric in metrics:
                val = row.get(metric_to_column(metric), "")
                if val:
                    expected[bldg_name][metric] = float(val)
    return expected
