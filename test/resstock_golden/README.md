# ResStock Golden Test Dataset

## Overview

This directory contains a curated set of 69 residential buildings used as golden test
inputs for OCHRE's ResStock simulation pipeline. The test verifies that OCHRE can
successfully simulate a diverse range of building configurations with ResStock output mode.

## How the Dataset Was Created

### 1. Minimal Buildstock Selection (super69_buildstock.csv)

The `super69_buildstock.csv` was created by downsampling from a 5M-sample buildstock.csv
using the **buildstock-query** upgrades analyzer's minimal buildstock creation tool. The
algorithm selects the smallest set of buildings such that every housing characteristic with
a cardinality of 25 or lower appears in at least one building. This ensures coverage of all:

- HVAC system types (heat pumps, gas furnaces, electric resistance, etc.)
- Insulation levels
- Climate zones
- Water heater types
- Foundation types
- Building geometries
- And other key housing characteristics with cardinality <= 25

This process initially produced 68 buildings. One building was then manually duplicated and
modified to be fully electric with all available electric end-uses (building ID starting
with `9` — bldg9521019). This brings the total to 69 buildings.

### 2. ResStock Simulation (BuildStockBatch)

The `super69_buildstock.csv` was run through BuildStockBatch with OCHRE output mode. From
the simulation output, only the input files needed by OCHRE were extracted:

- `home.xml` — the HPXML building definition file
- `in.schedules.csv` — normalized occupancy and equipment schedule profiles

The OCHRE output mode setting does not affect these input files; they are produced by
ResStock's standard HPXML generation pipeline.

### 3. Reference Results (results_up00.csv)

`results_up00.csv` contains the annual simulation results from the BuildStockBatch run.
For buildings that OCHRE successfully simulates, their annual energy results should match
the values in this file.

## Directory Structure

```
resstock_golden/
  README.md                  # This file
  super69_buildstock.csv     # The 69-building buildstock input (186 columns)
  results_up00.csv           # Reference annual results from BuildStockBatch
  buildings/                 # 69 building directories
    bldg0120236/
      home.xml               # HPXML building definition
      in.schedules.csv       # Normalized schedule profiles
    bldg0163682/
      ...
    ...
  weather/                   # 62 unique EPW weather files
    G0100890.epw
    G0200200.epw
    ...
```

## Test Status

Currently **56 out of 69** buildings simulate successfully. The 13 known failures are
tracked in `test/test_dwelling/test_resstock_golden.py` and fall into these categories:

| Error | Count | Description |
|-------|-------|-------------|
| `KeyError: 'HeatingSystemType'` | 6 | HPXML parsing issue for certain HVAC configs |
| `Unable to parse multiple attic floor areas` | 2 | Multi-attic geometry not supported |
| `Cannot find material properties for Garage Roof` | 2 | Missing material lookup |
| `'HeatPumpWaterHeater' has no attribute 'hp_cop'` | 2 | HPWH model issue |
| `HVAC Heating system and heat pump cannot both be specified` | 1 | Dual-system config |

As these bugs are fixed, the known failures list should be updated and the pass count
should increase toward 69.
