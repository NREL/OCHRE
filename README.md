![OCHRE](https://github.com/NREL/OCHRE/blob/main/docs/source/images/OCHRE-Logo-Horiz-2Color.png)

# OCHRE: The Object-oriented Controllable High-resolution Residential Energy Model

OCHRE&trade; is a Python-based energy modeling tool designed to model end-use
loads and distributed energy resources in residential buildings. It can model
flexible devices---including HVAC equipment, water heaters, electric vehicles,
solar PV, and batteries---and the thermal and electrical interactions between
them. OCHRE has been used to generate diverse and high-resolution load
profiles, examine the impacts of advanced control strategies on energy costs
and occupant comfort, and assess grid reliability and resilience through
building-to-grid co-simulation.

More information about OCHRE can be found in [our
documentation](https://ochre-nrel.readthedocs.io/), on [NREL's
website](https://www.nrel.gov/grid/ochre.html), and from the [Powered By
OCHRE](https://www.youtube.com/watch?v=B5elLVtYDbI) webinar recording.

If you use OCHRE for your research or other projects, please fill out our [user survey](https://forms.office.com/g/U4xYhaWEvs).

## Installation

OCHRE can be installed using `pip` from the command line:

```
pip install ochre-nrel
```

Alternatively, you can install a specific branch, for example:

```
pip install git+https://github.com/NREL/OCHRE@dev
```

Note that OCHRE requires Python version >=3.9 and <3.13.

## Usage

OCHRE can be used to simulate a residential dwelling or an individual piece of
equipment. In either case, a python object is instantiated and then simulated.
A set of input parameters and/or input files must be defined. 

Below is a simple example of simulating a dwelling:
```
import os
import datetime as dt
from ochre import Dwelling
from ochre.utils import default_input_path # for using sample files
house = Dwelling(
    simulation_name, 
    start_time=dt.datetime(2018, 1, 1, 0, 0),
    time_res=dt.timedelta(minutes=10),       
    duration=dt.timedelta(days=3),
    hpxml_file=os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml"),
    hpxml_schedule_file=os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv"),
    weather_file=os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
)

df, metrics, hourly = dwelling.simulate()
```

This will return 3 variables:
 * `df`: a Pandas DataFrame with 10 minute resolution
 * `metrics`: a dictionary of energy metrics
 * `hourly`: a Pandas DataFrame with 1 hour resolution (verbosity >= 3 only)

For more examples, see:
* The [OCHRE User
  Tutorial](https://colab.research.google.com/github/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb)
  Jupyter notebook 
* Python example scripts to:
  * Run a [single dwelling](https://github.com/NREL/OCHRE/blob/main/bin/run_dwelling.py)
  * Run a [single piece of equipment](https://github.com/NREL/OCHRE/blob/main/bin/run_equipment.py)
  * Run a [fleet of equipment](https://github.com/NREL/OCHRE/blob/main/bin/run_fleet.py)
  * Run [multiple dwellings](https://github.com/NREL/OCHRE/blob/main/bin/run_multiple.py)
  * Run a [OCHRE with an external controller](https://github.com/NREL/OCHRE/blob/main/bin/run_external_control.py)
  * Run a [OCHRE in co-simulation using HELICS](https://github.com/NREL/OCHRE/blob/main/bin/run_cosimulation.py)
