![OCHRE](docs\source\images\OCHRE-Logo-Horiz-2Color.png)

# OCHRE: The Object-oriented Controllable High-resolution Residential Energy Model

OCHRE&trade; is a Python-based building energy modeling (BEM) tool designed to model flexible loads in residential buildings. OCHRE includes detailed models and controls for flexible devices including HVAC equipment, water heaters, electric vehicles, solar PV, and batteries. It is designed to run in co-simulation with custom controllers, aggregators, and grid models.

The full documentation for OCHRE can be found at https://ochre-nrel.readthedocs.io/

Contact: jeff.maguire@nrel.gov, michael.blonsky@nrel.gov, killian.mckenna@nrel.gov

## Installation

Note that OCHRE requires Python version >=3.9 and <3.12

### Stand-alone Installation

For a stand-alone installation, OCHRE can be installed using `pip` from the command line:

```
pip install ochre-nrel
```

Alternatively, you can install a specific branch, for example:

```
pip install git+https://github.com/NREL/OCHRE@dev
```

### In Co-simulation
To embed this in a co-simulation and a separate conda environment, create an `environment.yml` file in the co-simulation
project and include the following lines:
```
dependencies:
  - pip:
    - ochre-nrel
```


## Usage

OCHRE can be used to simulate a residential dwelling or an individual piece of equipment. In either case, a python
object is instantiated and then simulated. A set of input parameters and input files must be defined. 

Below is a simple example of simulating a dwelling:
```
import datetime as dt
from ochre import Dwelling
house = Dwelling(simulation_name, 
                 start_time=dt.datetime(2018, 1, 1, 0, 0),
                 time_res=dt.timedelta(minutes=10),       
                 duration=dt.timedelta(days=3),
                 properties_file='sample_resstock_house.xml',
                 schedule_file='sample_resstock_schedule.csv',
                 weather_file='USA_CO_Denver.Intl.AP.725650_TMY3.epw',
                 verbosity=3,
                 )
df, metrics, hourly = dwelling.simulate()
```

This will output 3 variables:
 * `df`: a Pandas DataFrame with 10 minute resolution
 * `metrics`: a dictionary of energy metrics
 * `hourly`: a Pandas DataFrame with 1 hour resolution (verbosity >= 3 only)

For more examples, see the following python scripts in the `bin` folder:
* Run a single dwelling: `bin/run_dwelling.py`
* Run a single piece of equipment: `bin/run_equipment.py`
* Run a dwelling with an external controller: `bin/run_external_control.py`
* Run multiple dwellings: `bin/run_multiple.py`
* Run a fleet of equipment: `bin/run_fleet.py`

Required and optional input parameters and files are described below for a dwelling.

### Required Dwelling Parameters

* `name`: Name of the simulation
* `start_time`: Simulation start time as a datetime.datetime
* `time_res`: Simulation time resolution as a datetime.timedelta
* `duration`: Simulation duration as a datetime.timedelta
* `properties_file`: Path to building properties file (HPXML, yaml, or BEopt properties file)
* `schedule_file`: Path to building schedule file (csv)
* `weather_file` or `weather_path`: Path to weather file (epw or NSRDB file). `weather_path` can be used if the 
Weather Station name is specified in the properties file.

### Optional Dwelling Parameters

* `input_path`: Path with additional input files (defaults to a built-in directory)
* `output_path`: Path to output files
* `save_results`: if True, saves results to output files (default is True if `output_path` is specified)
* `initialization_time`: Duration to initialize the building temperatures as a datetime.timedelta (default is no 
initialization)
* `water_draw_file`: File name for water draw schedule file. For BEopt inputs only (default is no water draw)
* `verbosity`: Verbosity of the output files as integer from 1 to 9 (default is 1)
* `metrics_verbosity`: Verbosity of the metrics output file as integer from 1 to 9 (default is 6)

### Equipment-specific Parameters

Equipment arguments can be included to override information from the properties file. See `bin/run_dwelling.py` or
`bin/run_equipment.py` for examples. Below is a list of all of OCHRE's equipment names:
* HVAC Heating:
  * Electric Furnace
  * Electric Baseboard
  * Electric Boiler
  * Gas Furnace
  * Gas Boiler
  * Heat Pump Heater
  * Air Source Heat Pump (ASHP Heater)
  * Minisplit Heat Pump (MSHP Heater)
  * Ideal Heater
* HVAC Cooling:
  * Air Conditioner
  * Room AC
  * Air Source Heat Pump (ASHP Cooler)
  * Minisplit Heat Pump (MSHP Cooler)
  * Ideal Cooler
* Water Heating:
  * Electric Resistance Water Heater
  * Heat Pump Water Heater
  * Gas Water Heater
  * Modulating Water Heater
  * Tankless Water Heater
  * Gas Tankless Water Heater
* DERs and Controllable Loads:
  * PV
  * Battery
  * Electric Vehicle (EV)
* Scheduled Loads:
  * Lighting
  * Exterior Lighting
  * Range
  * Dishwasher
  * Refrigerator
  * Clothes Washer
  * Clothes Dryer
  * MELs
  * Scheduled EV 


## Overview

OCHRE is an object-oriented residential building model that simulates a variety of behind-the-meter equipment.
It simulates dwelling energy consumption (electricity and gas) at a high resolution (up to 1-minute) and is designed 
to integrate in co-simulation with controllers, distribution systems, and other agents.
Most equipment types are controllable though an external controller to simulate the impact of device
controllers, HEMS, demand response, or other control strategies.
The initialization integrates with ResStock and BEopt output files to simplify the building modeling.

The key features of the code are:

* High-fidelity, high-resolution residential building simulation
* Controllable equipment via external controllers
* Simple integration with co-simulation using object-oriented principles
* Voltage-dependent electric power and reactive power using an equipment-level ZIP model
* Large variety of equipment types including HVAC, water heating, PV, batteries, and EVs
* Envelope, HVAC, and water heating validation with EnergyPlus (in progress)

OCHRE integrates with the following models and tools:
* ResStock (for generating input files)
* BEopt (for generating input files)
* HELICS (for co-simulation)
* Foresee (for HEMS control)
* SAM (for PV modeling)
* EVIpro (for EV modeling)
* Distribution models, e.g. OpenDSS, through co-simulation

