"""Shared utilities for ResStock golden test scripts."""

import csv


def read_results_annual(path):
    """Read results_annual.csv into a dict mapping metric name to float value."""
    results = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            results[row[0].strip()] = float(row[1])
    return results


def load_expected_from_csv(csv_path, columns):
    """Load expected values from a results CSV, keyed by bldg_name.

    *columns* is a list of (csv_column_name, annual_metric_name) tuples.
    """
    expected = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            expected[bldg_name] = {}
            for csv_col, annual_metric in columns:
                val = row.get(csv_col, "")
                if val:
                    expected[bldg_name][annual_metric] = float(val)
    return expected
