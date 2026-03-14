"""Golden tests: validate OCHRE ResStock results against reference values.

Tests read pre-computed results from test/resstock_golden/ochre_result/.
If ochre_result/ has no building subdirectories, tests fail with instructions
to generate results first.

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
    "bldg0288066",
    "bldg0451855",
    "bldg0464288",
    "bldg0475650",
    "bldg0499827",
    "bldg0501306",
    "bldg9064288",
}


ALL_BUILDINGS = sorted(
    name
    for name in os.listdir(GOLDEN_DATA_PATH)
    if name.startswith("bldg") and os.path.isdir(os.path.join(GOLDEN_DATA_PATH, name))
)
EXPECTED_ANNUAL = load_expected_from_csv(GOLDEN_RESULTS_CSV, RESSTOCK_METRICS)


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
    if not os.path.isdir(GOLDEN_TEST_RESULT_PATH):
        pytest.fail(
            f"Golden test results not found at {GOLDEN_TEST_RESULT_PATH}\n"
            "Run simulations first:\n"
            "    python test/resstock_golden/generate_ochre_result.py\n"
            "Re-run that script whenever OCHRE changes are expected to alter results.",
            pytrace=False,
        )
    # Check that at least some buildings have results
    has_any = any(
        os.path.isdir(os.path.join(GOLDEN_TEST_RESULT_PATH, name))
        for name in os.listdir(GOLDEN_TEST_RESULT_PATH)
        if name.startswith("bldg")
    )
    if not has_any:
        pytest.fail(
            "No building results found in ochre_result/.\n"
            "Run simulations first:\n"
            "    python test/resstock_golden/generate_ochre_result.py",
            pytrace=False,
        )


@pytest.mark.golden
@pytest.mark.parametrize("bldg_name", ALL_BUILDINGS)
def test_building_results(bldg_name):
    """Validate pre-computed OCHRE results for a single building."""
    _check_test_results_exist()

    if bldg_name in KNOWN_FAILURES:
        if _has_results(bldg_name):
            pytest.fail(
                f"{bldg_name} is in KNOWN_FAILURES but simulation succeeded. "
                "Remove it from KNOWN_FAILURES if the bug is fixed."
            )
        pytest.xfail(f"{bldg_name} is a known simulation failure")

    # For non-known-failure buildings, results must exist
    output_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name)
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
