"""Run OCHRE ResStock simulations and update ochre_annual_result.csv.

Writes per-building results to test/resstock_golden/ochre_result/<bldg_name>/
and then updates ochre_result/ochre_annual_result.csv with the new annual values.
Re-run whenever OCHRE changes are expected to alter outputs.

Usage (from OCHRE root):
    python test/resstock_golden/generate_ochre_result.py              # all buildings
    python test/resstock_golden/generate_ochre_result.py bldg0108019  # one building
"""

import csv
import multiprocessing
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ochre.cli import create_dwelling
from test.test_dwelling.resstock_test_utils import (
    GOLDEN_EPLUS_CSV,
    GOLDEN_RESULTS_CSV,
    GOLDEN_TEST_RESULT_PATH,
    metric_to_column,
    read_results_annual,
)

HERE = Path(__file__).resolve().parent
BUILDINGS = HERE / "eplus_result"
WEATHER = HERE / "weather"
RESULTS = Path(GOLDEN_TEST_RESULT_PATH)

SIM_KWARGS = dict(
    output_format="resstock",
    duration=365,
    time_res=15,
    start_year=2007,
    initialization_time=1,
    verbosity=1,
)


def run_building(name):
    output = RESULTS / name
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        # Suppress OCHRE's print-based warnings and Python warnings
        with warnings.catch_warnings(), open(os.devnull, "w") as devnull:
            warnings.simplefilter("ignore")
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                dwelling = create_dwelling(
                    input_path=str(BUILDINGS / name),
                    weather_file_or_path=str(WEATHER),
                    output_path=str(output),
                    seed=int(name.removeprefix("bldg")),
                    **SIM_KWARGS,
                )
                dwelling.simulate()
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        return name, None, time.time() - t0
    except Exception:
        return name, traceback.format_exc(), time.time() - t0


def update_ochre_annual_result():
    """Update ochre_annual_result.csv by overlaying OCHRE results onto the EPlus baseline."""
    if not os.path.isfile(GOLDEN_EPLUS_CSV):
        print(f"\nSkipping ochre_annual_result.csv update: {GOLDEN_EPLUS_CSV} not found")
        return

    with open(GOLDEN_EPLUS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    row_index = {f"bldg{int(row['building_id']):07d}": row for row in rows}

    energy_columns = {
        col for col in fieldnames if col.startswith("report_simulation_output.") and col.endswith("_m_btu")
    }

    buildings_updated = 0
    values_changed = 0

    for bldg_name in sorted(os.listdir(GOLDEN_TEST_RESULT_PATH)):
        annual_path = os.path.join(GOLDEN_TEST_RESULT_PATH, bldg_name, "results_annual.csv")
        if not os.path.isfile(annual_path):
            continue

        if bldg_name not in row_index:
            continue

        row = row_index[bldg_name]
        annual = read_results_annual(annual_path)
        building_changed = False

        for metric_name, value in annual.items():
            col_name = metric_to_column(metric_name)
            if col_name not in energy_columns:
                continue

            old_val = row.get(col_name, "")
            new_val = str(round(value, 3))
            if old_val != new_val:
                row[col_name] = new_val
                values_changed += 1
                building_changed = True

        if building_changed:
            buildings_updated += 1

    with open(GOLDEN_RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nUpdated {GOLDEN_RESULTS_CSV}")
    print(f"  Buildings with OCHRE overlay: {buildings_updated}")
    print(f"  Values changed vs EPlus: {values_changed}")


def main():
    if len(sys.argv) > 1:
        buildings = sys.argv[1:]
    else:
        buildings = sorted(p.name for p in BUILDINGS.iterdir() if p.is_dir() and p.name.startswith("bldg"))

    # Clear old per-building simulation outputs, preserving ochre_annual_result.csv
    for entry in RESULTS.iterdir():
        if entry.is_dir() and entry.name.startswith("bldg"):
            shutil.rmtree(entry)

    n_workers = multiprocessing.cpu_count()
    print(f"Simulating {len(buildings)} buildings with {n_workers} workers\n")

    t0 = time.time()
    failed = []

    with multiprocessing.Pool(n_workers) as pool:
        for i, (name, error, elapsed) in enumerate(pool.imap_unordered(run_building, buildings), 1):
            status = "OK" if error is None else "FAILED"
            print(f"[{i}/{len(buildings)}] {name} {status} ({elapsed:.0f}s)")
            if error:
                print(error)
                failed.append(name)

    print(f"\n{len(buildings) - len(failed)} succeeded, {len(failed)} failed in {time.time() - t0:.0f}s")
    if failed:
        print("Failed:", " ".join(sorted(failed)))

    # Update ochre_annual_result.csv with the new simulation results
    update_ochre_annual_result()


if __name__ == "__main__":
    main()
