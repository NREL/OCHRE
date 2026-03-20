# ResStock Golden Test Dataset

## Overview

This directory contains a curated set of residential buildings used as golden test inputs
for OCHRE's ResStock simulation pipeline. The buildings are selected to cover a diverse
range of housing characteristics — HVAC types, fuels, climate zones, insulation levels,
water heaters, building geometries, and more — so that changes to OCHRE can be validated
against known-good results.

The golden tests serve two purposes:

1. **Regression testing** — verify that OCHRE produces the same annual energy results as a
   previous run (exact match within tolerance).
2. **Cross-validation** — compare OCHRE results against EnergyPlus for the same buildings
   to track where the two engines agree and diverge.

## What's in This Directory

### Data

| Path | Description |
|------|-------------|
| `eplus_result/` | One subdirectory per building containing `home.xml` (HPXML building definition) and `in.schedules.csv` (normalized schedule profiles). Also contains `eplus_annual_result.csv` with EnergyPlus annual energy results for all buildings. |
| `ochre_result/` | Contains the committed reference `ochre_annual_result.csv`. Per-building simulation outputs and `ochre_annual_result_new.csv` are written here by `generate_ochre_result.py` but are **gitignored**. |
| `weather/` | Shared EPW weather files referenced by the buildings' `home.xml` files. |
| `comparison/` | One CSV per metric comparing OCHRE vs EnergyPlus. Metrics are dynamically discovered — any `report_simulation_output` column with non-empty values in both OCHRE and EPlus is compared. Each CSV includes relevant building characteristics for that end use and shows "FAILED" for buildings where OCHRE simulation crashed. |

### Scripts

| Script | Purpose |
|--------|---------|
| `generate_ochre_result.py` | Run OCHRE simulations on all buildings and write `ochre_annual_result_new.csv`. |
| `compare_ochre_and_eplus.py` | Compare OCHRE results against EnergyPlus and write per-metric CSVs to `comparison/`. |
| `generate_minimal_buildstock.py` | Create a minimal buildstock CSV covering all key housing characteristics. Only needed when changing the building set. |
| `copy_eplus_result.py` | Extract building files, weather, and EnergyPlus results from a ResStock simulation output. Only needed when changing the building set. |

## How to Use the Golden Tests

All commands are run from the OCHRE root directory.

### Step 1: Generate OCHRE Results

```bash
python test/resstock_golden/generate_ochre_result.py              # all buildings
python test/resstock_golden/generate_ochre_result.py bldg0108019  # one building
```

This runs OCHRE on each building using all available CPU cores. For each building it writes
`results_annual.csv` and `results_timeseries.csv` to `ochre_result/<bldg>/`. After all
simulations complete, the script writes `ochre_result/ochre_annual_result_new.csv` with the
new annual values.

**When to run:** whenever OCHRE changes are expected to alter simulation outputs.

### Step 2: Run Golden Tests

```bash
pytest test/test_dwelling/test_resstock_golden.py -v --tb=short       # all buildings
pytest test/test_dwelling/test_resstock_golden.py -k bldg0108019 -v   # one building
```

The golden test compares `ochre_annual_result_new.csv` (freshly generated) against the
committed `ochre_annual_result.csv` reference. All `report_simulation_output` columns with
non-empty numeric values are compared dynamically — no hardcoded metric list is maintained.
The tolerance is **0.01 MBtu** per metric.

### Step 3: Compare with EnergyPlus (optional)

```bash
python test/resstock_golden/compare_ochre_and_eplus.py
```

Produces one CSV per metric in `comparison/`, showing each building's OCHRE value,
EnergyPlus value, and percent difference. Each CSV includes building characteristics
relevant to that end use (e.g., cooling efficiency for cooling metrics, water heater type
for hot water metrics). Buildings where OCHRE simulation crashed show "FAILED" instead of
"NA". These CSVs are committed so that changes are visible in PR diffs.

## Updating the OCHRE Reference

When OCHRE changes legitimately alter simulation outputs, the golden test reference needs
to be updated:

1. Run `generate_ochre_result.py` to produce new results.
2. Run the golden tests — they will fail because `ochre_annual_result_new.csv` differs from
   the committed reference.
3. Review the diffs to confirm the changes are expected.
4. Update the reference:
   ```bash
   cp test/resstock_golden/ochre_result/ochre_annual_result_new.csv \
      test/resstock_golden/ochre_result/ochre_annual_result.csv
   ```
5. Optionally regenerate the EPlus comparison:
   ```bash
   python test/resstock_golden/compare_ochre_and_eplus.py
   ```
6. Commit the updated `ochre_annual_result.csv` and `comparison/` files.

## Recreating the Dataset

These steps are only needed when the building set itself needs to change
(e.g., updating to a new ResStock version).

### 1. Select a minimal buildstock

```bash
uv run test/resstock_golden/generate_minimal_buildstock.py
```

`generate_minimal_buildstock.py` uses
[buildstock-query](https://github.com/NatLabRockies/buildstock-query) to downsample a
large buildstock into the smallest set of buildings that covers most housing characteristics
with cardinality less than 20. The script expects a sibling `resstock/` directory and uses
the following files from ResStock:

```
resstock/
  resources/res_ochre_550K.csv                          # large buildstock CSV
  project_national/sdr_upgrades_tmy3.yml                # project YAML config
  project_national/resources/options_saturations.csv    # upgrade options
```

[uv](https://docs.astral.sh/uv/) installs all Python dependencies automatically from the
inline script metadata.

If you don't already have the 550K buildstock (`res_ochre_550K.csv`), create it first
from the ResStock directory:

```bash
resstock>  openstudio resources/run_sampling.rb -n 550000 -o res_ochre_550K.csv -p project_national
```

The script writes `ochre_minimal_buildstock.csv` into the `project_national` directory.

### 2. Run ResStock simulation with the minimal buildstock

Update `project_national/national_baseline.yml` to use the minimal buildstock:

```yml
output_directory: ochre_minimal_run

sampler:
  type: precomputed
  args:
    sample_file: ochre_minimal_buildstock.csv
```

Then run the ResStock simulation. Use the `-k` flag to preserve all run folders:

```bash
resstock>  openstudio ./workflow/run_analysis.rb -y ./project_national/national_baseline.yml -k
```

### 3. Extract results into this directory

```bash
uv run test/resstock_golden/copy_eplus_result.py /path/to/ochre_minimal_run
```

This copies building files into `eplus_result/`, weather files into `weather/`, and builds
`eplus_result/eplus_annual_result.csv` from the ResStock outputs. Absolute paths are
automatically sanitized during the copy.

### 4. Generate OCHRE results and comparison

```bash
python test/resstock_golden/generate_ochre_result.py
python test/resstock_golden/compare_ochre_and_eplus.py
```

This runs OCHRE simulations, writes `ochre_annual_result_new.csv`, and generates the
EPlus comparison CSVs. Copy the new results to the reference and commit everything:

```bash
cp test/resstock_golden/ochre_result/ochre_annual_result_new.csv \
   test/resstock_golden/ochre_result/ochre_annual_result.csv
```
