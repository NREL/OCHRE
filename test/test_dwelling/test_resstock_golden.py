"""Golden test suite: run OCHRE ResStock simulations on 69 diverse buildings.

Exercises OCHRE's full simulation pipeline with ResStock output mode across a
diverse set of residential buildings from a ResStock simulation. Serves as a
comprehensive smoke test ensuring OCHRE handles a wide variety of building
types, HVAC systems, and configurations.

Run all golden tests:
    pytest test/test_dwelling/test_resstock_golden.py -v --tb=short

Run a single building:
    pytest test/test_dwelling/test_resstock_golden.py -k bldg5219269 -v

"""

import csv
import os
import traceback
import warnings

import pytest

from ochre.cli import create_dwelling
from test import test_path


GOLDEN_DATA_PATH = os.path.join(test_path, "resstock_golden", "buildings")
GOLDEN_WEATHER_PATH = os.path.join(test_path, "resstock_golden", "weather")
GOLDEN_RESULTS_CSV = os.path.join(test_path, "resstock_golden", "results_up00.csv")


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

# Columns to validate between results_up00.csv and results_annual.csv.
# Each tuple: (results_up00.csv column name, results_annual.csv metric name)
COLUMNS_TO_VALIDATE = [
    ("report_simulation_output.fuel_use_electricity_total_m_btu", "Fuel Use: Electricity: Total (MBtu)"),
    ("report_simulation_output.fuel_use_natural_gas_total_m_btu", "Fuel Use: Natural Gas: Total (MBtu)"),
    ("report_simulation_output.end_use_electricity_heating_m_btu", "End Use: Electricity: Heating (MBtu)"),
    ("report_simulation_output.end_use_electricity_cooling_m_btu", "End Use: Electricity: Cooling (MBtu)"),
    ("report_simulation_output.end_use_electricity_plug_loads_m_btu", "End Use: Electricity: Plug Loads (MBtu)"),
    ("report_simulation_output.load_heating_delivered_m_btu", "Load: Heating: Delivered (MBtu)"),
    ("report_simulation_output.load_cooling_delivered_m_btu", "Load: Cooling: Delivered (MBtu)"),
    ("report_simulation_output.load_hot_water_delivered_m_btu", "Load: Hot Water: Delivered (MBtu)"),
]


TIME_RES_MINUTES = 15
START_YEAR = 2007
INIT_DAYS = 1
VERBOSITY = 3

# Tolerance for annual energy comparisons (MBtu)
ANNUAL_ATOL = 0.01


def _read_results_annual(path):
    """Read results_annual.csv into a dict keyed by metric name."""
    results = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            results[row[0].strip()] = float(row[1])
    return results


def _load_golden_expected():
    """Load expected values from results_up00.csv, keyed by bldg_name."""
    expected = {}
    with open(GOLDEN_RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bldg_id = int(row["building_id"])
            bldg_name = f"bldg{bldg_id:07d}"
            expected[bldg_name] = {}
            for csv_col, annual_metric in COLUMNS_TO_VALIDATE:
                val = row.get(csv_col, "")
                if val:
                    expected[bldg_name][annual_metric] = float(val)
    return expected


def _discover_buildings(data_path):
    """Return sorted list of building directory names."""
    return sorted(
        name
        for name in os.listdir(data_path)
        if name.startswith("bldg") and os.path.isdir(os.path.join(data_path, name))
    )


DATA_AVAILABLE = os.path.isdir(GOLDEN_DATA_PATH) and os.path.isdir(GOLDEN_WEATHER_PATH)
ALL_BUILDINGS = _discover_buildings(GOLDEN_DATA_PATH) if DATA_AVAILABLE else []
EXPECTED_ANNUAL = _load_golden_expected() if DATA_AVAILABLE else {}


@pytest.mark.golden
@pytest.mark.skipif(not DATA_AVAILABLE, reason="Golden test data not available")
@pytest.mark.parametrize("bldg_name", ALL_BUILDINGS)
def test_building_simulation(bldg_name, tmp_path):
    """Run OCHRE ResStock simulation for a single building."""
    duration = 365
    input_path = os.path.join(GOLDEN_DATA_PATH, bldg_name)
    output_path = str(tmp_path / bldg_name)

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
            duration=duration,
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
        pytest.fail(f"{bldg_name} was supposed to fail but simulation succeeded."
                    " If the bug is fixed, remove from KNOWN_FAILURES set.")

    assert ts_df is not None and len(ts_df) > 0
    assert os.path.isfile(os.path.join(output_path, "results_timeseries.csv"))

    annual_path = os.path.join(output_path, "results_annual.csv")
    assert os.path.isfile(annual_path)

    actual = _read_results_annual(annual_path)
    for metric, expected_val in EXPECTED_ANNUAL[bldg_name].items():
        actual_val = actual.get(metric)
        assert actual_val is not None, (
            f"{bldg_name}: metric '{metric}' not found in results_annual.csv"
        )
        assert abs(actual_val - expected_val) <= ANNUAL_ATOL, (
            f"{bldg_name}: {metric} = {actual_val}, expected {expected_val} "
            f"(diff={abs(actual_val - expected_val):.4f})"
        )
