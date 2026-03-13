"""Golden tests: validate OCHRE ResStock results against reference values.

Tests read pre-computed results from test/resstock_golden/ochre_result/.
If ochre_result/ is missing, tests fail with instructions to generate it.

Generate results (re-run when OCHRE changes alter outputs):
    python test/resstock_golden/generate_ochre_result.py

Run all golden tests:
    pytest test/test_dwelling/test_resstock_golden.py -v --tb=short

Run a single building:
    pytest test/test_dwelling/test_resstock_golden.py -k bldg0108019 -v
"""

import os

import pytest

from test.test_dwelling.resstock_test_utils import (
    GOLDEN_DATA_PATH,
    GOLDEN_RESULTS_CSV,
    GOLDEN_TEST_RESULT_PATH,
    RESSTOCK_METRICS,
    load_expected_from_csv,
    read_results_annual,
)


# Tolerance for annual energy comparisons (MBtu)
ANNUAL_ATOL = 0.01


ALL_BUILDINGS = sorted(
    name
    for name in os.listdir(GOLDEN_DATA_PATH)
    if name.startswith("bldg") and os.path.isdir(os.path.join(GOLDEN_DATA_PATH, name))
)
EXPECTED_ANNUAL = load_expected_from_csv(GOLDEN_RESULTS_CSV, RESSTOCK_METRICS)


def _check_test_results_exist():
    """Raise a clear error if test_result/ hasn't been generated."""
    if not os.path.isdir(GOLDEN_TEST_RESULT_PATH):
        pytest.fail(
            f"Golden test results not found at {GOLDEN_TEST_RESULT_PATH}\n"
            "Run simulations first:\n"
            "    python test/resstock_golden/generate_ochre_result.py\n"
            "Re-run that script whenever OCHRE changes are expected to alter results.",
            pytrace=False,
        )


@pytest.mark.golden
@pytest.mark.parametrize("bldg_name", ALL_BUILDINGS)
def test_building_results(bldg_name):
    """Validate pre-computed OCHRE results for a single building."""
    _check_test_results_exist()

    output_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name)
    if not os.path.isdir(output_path):
        pytest.skip(f"No test results for {bldg_name} (vacant unit or not yet simulated)")

    # Check that timeseries and annual files exist
    ts_path = os.path.join(output_path, "results_timeseries.csv")
    annual_path = os.path.join(output_path, "results_annual.csv")

    assert os.path.isfile(ts_path), f"{bldg_name}: results_timeseries.csv not found in {output_path}"
    assert os.path.isfile(annual_path), f"{bldg_name}: results_annual.csv not found in {output_path}"

    actual = read_results_annual(annual_path)

    if bldg_name not in EXPECTED_ANNUAL:
        pytest.skip(f"{bldg_name} not in ochre_annual_result.csv reference")

    # Exact match against OCHRE reference results
    for metric, expected_val in EXPECTED_ANNUAL[bldg_name].items():
        actual_val = actual.get(metric)
        assert actual_val is not None, f"{bldg_name}: metric '{metric}' not found in results_annual.csv"
        assert abs(actual_val - expected_val) <= ANNUAL_ATOL, (
            f"{bldg_name}: {metric} = {actual_val}, expected {expected_val} (diff={abs(actual_val - expected_val):.4f})"
        )
