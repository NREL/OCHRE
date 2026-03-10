"""Golden test suite: run OCHRE ResStock simulations on 69 diverse buildings.

Exercises OCHRE's full simulation pipeline with ResStock output mode across a
diverse set of residential buildings from a ResStock simulation. Serves as a
comprehensive smoke test ensuring OCHRE handles a wide variety of building
types, HVAC systems, and configurations.

Run all golden tests:
    pytest test/test_dwelling/test_resstock_golden.py -v --tb=short

Run a single building:
    pytest test/test_dwelling/test_resstock_golden.py -k bldg5219269 -v

After running the tests, compare OCHRE results against EnergyPlus:
    python test/test_dwelling/compare_with_eplus.py

"""

import os
import traceback
import warnings

import pytest

from ochre.cli import create_dwelling
from test import test_path
from test.test_dwelling.resstock_test_utils import load_expected_from_csv, read_results_annual


GOLDEN_DATA_PATH = os.path.join(test_path, "resstock_golden", "buildings")
GOLDEN_WEATHER_PATH = os.path.join(test_path, "resstock_golden", "weather")
GOLDEN_RESULTS_CSV = os.path.join(test_path, "resstock_golden", "results_up00.csv")
GOLDEN_TEST_RESULT_PATH = os.path.join(test_path, "resstock_golden", "test_result")


# Buildings known to fail (update as bugs are fixed).
KNOWN_FAILURES = {
    "bldg0203414",
    "bldg0999920",
    "bldg1138321",
    "bldg1532617",
    "bldg1638229",
    "bldg1785305",
    "bldg2164906",
    "bldg2543739",
    "bldg2557486",
    "bldg3145607",
    "bldg3800983",
    "bldg3873309",
    "bldg4484460",
}

METRICS_TO_VALIDATE = [
    "Fuel Use: Electricity: Total (MBtu)",
    "Fuel Use: Natural Gas: Total (MBtu)",
    "End Use: Electricity: Heating (MBtu)",
    "End Use: Electricity: Cooling (MBtu)",
    "End Use: Electricity: Plug Loads (MBtu)",
    "Load: Heating: Delivered (MBtu)",
    "Load: Cooling: Delivered (MBtu)",
    "Load: Hot Water: Delivered (MBtu)",
    "End Use: Electricity: Hot Water (MBtu)",
]


TIME_RES_MINUTES = 15
START_YEAR = 2007
INIT_DAYS = 1
VERBOSITY = 3

# Tolerance for annual energy comparisons (MBtu)
ANNUAL_ATOL = 0.01


ALL_BUILDINGS = sorted(
    name
    for name in os.listdir(GOLDEN_DATA_PATH)
    if name.startswith("bldg") and os.path.isdir(os.path.join(GOLDEN_DATA_PATH, name))
)
EXPECTED_ANNUAL = load_expected_from_csv(GOLDEN_RESULTS_CSV, METRICS_TO_VALIDATE)


@pytest.mark.golden
@pytest.mark.parametrize("bldg_name", ALL_BUILDINGS)
def test_building_simulation(bldg_name):
    """Run OCHRE ResStock simulation for a single building."""
    input_path = os.path.join(GOLDEN_DATA_PATH, bldg_name)
    output_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name)
    os.makedirs(output_path, exist_ok=True)

    # Extract numeric building ID for deterministic seeding
    bldg_id = int(bldg_name.replace("bldg", ""))

    try:
        dwelling = create_dwelling(
            input_path=input_path,
            hpxml_file="home.xml",
            hpxml_schedule_file="in.schedules.csv",
            weather_file_or_path=GOLDEN_WEATHER_PATH,
            output_path=output_path,
            output_format="resstock",
            duration=365,
            time_res=TIME_RES_MINUTES,
            start_year=START_YEAR,
            initialization_time=INIT_DAYS,
            verbosity=VERBOSITY,
            seed=bldg_id,
        )
        ts_df, annual_df, hourly_df = dwelling.simulate()
    except Exception as e:
        if bldg_name in KNOWN_FAILURES:
            tb = traceback.format_exc()
            warnings.warn(
                f"KNOWN FAILURE: {bldg_name}\n{tb}",
                stacklevel=1,
            )
            pytest.xfail(f"{type(e).__name__}: {str(e)[:120]}")
        else:
            raise
    if bldg_name in KNOWN_FAILURES:
        pytest.fail(
            f"{bldg_name} was supposed to fail but simulation succeeded."
            " If the bug is fixed, remove from KNOWN_FAILURES set."
        )

    assert ts_df is not None and len(ts_df) > 0
    assert os.path.isfile(os.path.join(output_path, "results_timeseries.csv"))

    annual_path = os.path.join(output_path, "results_annual.csv")
    assert os.path.isfile(annual_path)

    actual = read_results_annual(annual_path)

    # Exact match against OCHRE reference results
    for metric, expected_val in EXPECTED_ANNUAL[bldg_name].items():
        actual_val = actual.get(metric)
        assert actual_val is not None, f"{bldg_name}: metric '{metric}' not found in results_annual.csv"
        assert abs(actual_val - expected_val) <= ANNUAL_ATOL, (
            f"{bldg_name}: {metric} = {actual_val}, expected {expected_val} (diff={abs(actual_val - expected_val):.4f})"
        )
