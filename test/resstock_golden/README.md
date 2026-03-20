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
| `ochre_result/` | Contains `ochre_annual_result.csv` (committed) with OCHRE's reference annual energy results. Per-building simulation outputs (`results_annual.csv`, `results_timeseries.csv`) are written here by `generate_ochre_result.py` but are **gitignored**. |
| `weather/` | Shared EPW weather files referenced by the buildings' `home.xml` files. |
| `comparison/` | One CSV per energy metric comparing OCHRE vs EnergyPlus. Auto-generated and committed so that changes are visible in PR diffs. |

### Scripts

| Script | Purpose |
|--------|---------|
| `generate_ochre_result.py` | Run OCHRE simulations on all buildings and update `ochre_annual_result.csv`. |
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
simulations complete, the script automatically updates `ochre_result/ochre_annual_result.csv`
with the new annual energy values.

**When to run:** whenever OCHRE changes are expected to alter simulation outputs.

### Step 2: Run Golden Tests

```bash
pytest test/test_dwelling/test_resstock_golden.py -v --tb=short       # all buildings
pytest test/test_dwelling/test_resstock_golden.py -k bldg0108019 -v   # one building
```

Each building's `results_annual.csv` is compared against the reference values in
`ochre_annual_result.csv` across 9 energy metrics (see below). The tolerance is
**0.01 MBtu** per metric. Buildings without simulation output are skipped.

### Step 3: Compare with EnergyPlus (optional)

```bash
python test/resstock_golden/compare_ochre_and_eplus.py
```

Produces one CSV per energy metric in `comparison/`, showing each building's OCHRE value,
EnergyPlus value, and percent difference. Buildings that failed OCHRE simulation appear
with `NA` values. These CSVs are committed to the repo and CI automatically updates them
on pull requests. This helps us track alignment between OCHRE and Energy Plus simulation.

## How to Recreate the Dataset

These steps are only needed when the building set itself needs to change
(e.g., updating to a new ResStock version).

### 1. Select a minimal buildstock

Run from the OCHRE root:

```bash
uv run test/resstock_golden/generate_minimal_buildstock.py
```

`generate_minimal_buildstock.py` uses
[buildstock-query](https://github.com/NatLabRockies/buildstock-query) to downsample a
large buildstock into the smallest set of buildings that covers most housing characteristic
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

If you don't already have the 550K buildstock (`res_ochre_550K.csv`) - you should create it first.
Running the following command from ResStock directory can be used to generate it.

```bash
resstock>  openstudio resources/run_sampling.rb -n 550000 -o res_ochre_550K.csv  -p project_national 
```

The `generate_minimal_buildstock.py` will write `ochre_minimal_buildstock.csv` into the project_national
directory.

### 2. Run ResStock Simulation with the minimal buildstock.

Update the ResStock project_national/national_baseline.yml to use the `ochre_minimal_buildstock.csv`

````yml
output_directory: ocher_minimal_run

sampler:
  type: precomputed
  args:
    sample_file: ochre_minimal_buildstock.csv
````

Then run ResStock simulation. Make sure to use the `-k` flag to preserve all run folders.

```bash
resstock>  openstudio ./workflow/run_analysis.rb -y ./project_national/national_baseline.yml -k 
```

### 3. Extract results into this directory

Pass the path to the ResStock output directory as an argument:

```bash
uv run test/resstock_golden/copy_eplus_result.py /path/to/ochre_minimal_run
```

This copies building files into `eplus_result/`, weather files into `weather/`, and builds
`eplus_result/eplus_annual_result.csv` from the ResStock outputs.

### 4. Generate OCHRE results

```bash
python test/resstock_golden/generate_ochre_result.py
```

This runs OCHRE simulation on the copied buildings and populates `ochre_result/` with
 OCHRE simulation outputs and creates the initial `ochre_annual_result.csv`.

## Compare OCHRE and Energy Plus

```bash
uv run test/resstock_golden/compare_ochre_and_eplus.py
```
Will compare between the OCHRE and Eplus result and generate `comparison/` files.

The golden tests and EnergyPlus comparison dynamically discover all
`report_simulation_output` columns with non-empty numeric values in both the
OCHRE and EPlus results. No hardcoded metric list is maintained.

The golden test compares `ochre_annual_result_new.csv` (freshly generated)
against the committed `ochre_annual_result.csv` reference. If results change
due to legitimate OCHRE changes, update the reference:

```bash
cp test/resstock_golden/ochre_result/ochre_annual_result_new.csv \
   test/resstock_golden/ochre_result/ochre_annual_result.csv
```
