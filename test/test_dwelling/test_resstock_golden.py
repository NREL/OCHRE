"""Golden tests: validate OCHRE ResStock results against reference values.

Compares ochre_annual_result_new.csv (freshly generated) against the committed
ochre_annual_result.csv reference. All report_simulation_output columns with
non-empty numeric values are compared dynamically.

Generate results (re-run when OCHRE changes alter outputs):
    python test/resstock_golden/generate_ochre_result.py

Run all golden tests:
    pytest test/test_dwelling/test_resstock_golden.py -v --tb=short

Run a single building:
    pytest test/test_dwelling/test_resstock_golden.py -k bldg0108019 -v

If tests fail due to legitimate OCHRE changes, update the reference:
    cp test/resstock_golden/ochre_result/ochre_annual_result_new.csv \\
       test/resstock_golden/ochre_result/ochre_annual_result.csv
"""

import os

import pytest

from test.test_dwelling.resstock_test_utils import (
    GOLDEN_DATA_PATH,
    GOLDEN_NEW_RESULTS_CSV,
    GOLDEN_RESULTS_CSV,
    GOLDEN_TEST_RESULT_PATH,
    load_results_csv,
)


# Tolerance for annual energy comparisons (MBtu)
ANNUAL_ATOL = 0.01

# Buildings known to fail OCHRE simulation (update as bugs are fixed).
# When a bug is fixed and a building starts succeeding, remove it from this set
# so the test verifies it continues to work.
KNOWN_FAILURES = {
    "bldg0066501",
    "bldg0094522",
    "bldg0116109",
    "bldg0126235",
    "bldg0145426",
    "bldg0276627",
    "bldg0499827",
}


ALL_BUILDINGS = sorted(
    name
    for name in os.listdir(GOLDEN_DATA_PATH)
    if name.startswith("bldg") and os.path.isdir(os.path.join(GOLDEN_DATA_PATH, name))
)
EXPECTED_ANNUAL = load_results_csv(GOLDEN_RESULTS_CSV)
ACTUAL_ANNUAL = load_results_csv(GOLDEN_NEW_RESULTS_CSV) if os.path.isfile(GOLDEN_NEW_RESULTS_CSV) else {}


def _has_results(bldg_name):
    """Check if a building has simulation output files."""
    output_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name)
    return (
        os.path.isdir(output_path)
        and os.path.isfile(os.path.join(output_path, "results_annual.csv"))
        and os.path.isfile(os.path.join(output_path, "results_timeseries.csv"))
    )


def _check_test_results_exist():
    """Raise a clear error if no simulation results have been generated."""
    if not os.path.isfile(GOLDEN_NEW_RESULTS_CSV):
        pytest.fail(
            f"New results not found at {GOLDEN_NEW_RESULTS_CSV}\n"
            "Run simulations first:\n"
            "    python test/resstock_golden/generate_ochre_result.py\n"
            "Re-run that script whenever OCHRE changes are expected to alter results.",
            pytrace=False,
        )


@pytest.mark.golden
@pytest.mark.parametrize("bldg_name", ALL_BUILDINGS)
def test_building_results(bldg_name):
    """Validate OCHRE results against committed reference for a single building."""
    _check_test_results_exist()

    if bldg_name in KNOWN_FAILURES:
        if _has_results(bldg_name):
            pytest.fail(
                f"{bldg_name} is in KNOWN_FAILURES but simulation succeeded. "
                "Remove it from KNOWN_FAILURES if the bug is fixed."
            )
        pytest.xfail(f"{bldg_name} is a known simulation failure")

    # Check output files exist
    output_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name)
    ts_path = os.path.join(output_path, "results_timeseries.csv")
    annual_path = os.path.join(output_path, "results_annual.csv")
    assert os.path.isfile(ts_path), f"{bldg_name}: results_timeseries.csv not found in {output_path}"
    assert os.path.isfile(annual_path), f"{bldg_name}: results_annual.csv not found in {output_path}"

    if bldg_name not in EXPECTED_ANNUAL:
        pytest.skip(f"{bldg_name} not in reference ochre_annual_result.csv")
    if bldg_name not in ACTUAL_ANNUAL:
        pytest.fail(f"{bldg_name} not in ochre_annual_result_new.csv")

    expected = EXPECTED_ANNUAL[bldg_name]
    actual = ACTUAL_ANNUAL[bldg_name]

    # Compare all metrics present in either reference or new results
    all_cols = sorted(set(expected) | set(actual))
    for col in all_cols:
        exp = expected.get(col)
        act = actual.get(col)
        if exp is None and act is not None:
            pytest.fail(f"{bldg_name}: new metric '{col}' = {act} not in reference. Update reference if intentional.")
        if act is None and exp is not None:
            pytest.fail(f"{bldg_name}: metric '{col}' = {exp} missing from new results.")
        assert abs(act - exp) <= ANNUAL_ATOL, (
            f"{bldg_name}: {col} = {act}, expected {exp} (diff={abs(act - exp):.4f}). Update reference if intentional."
        )
