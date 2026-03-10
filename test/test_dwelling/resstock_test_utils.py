"""Shared utilities for ResStock golden test scripts."""

import csv

from ochre.utils.resstock import to_underscore_case


def metric_to_column(metric_name):
    """Convert a results_annual.csv metric name to a results_up00.csv column name.

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
