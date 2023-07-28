Main index
==========

Getting Started
===============

OCHRE Overview
--------------

OCHRE is a Python-based building energy modeling tool designed
specifically for modeling flexible loads in residential buildings. OCHRE
includes detailed models and controls for flexible devices including
HVAC equipment, water heaters, electric vehicles, solar PV, and
batteries. It is designed to run in co-simulation with custom
controllers, aggregators, and grid models.

More information about OCHRE is available on `NREL’s
website <https://www.nrel.gov/grid/ochre.html>`__ and on
`Github <https://github.com/NREL/OCHRE>`__.

Scope
-----

OCHRE is closely integrated with other OS-HPXML based tools. This means
that OCHRE inherits any limitations of these tools. OCHRE is capable of
modeling residential single family, multifamily, and manufactured
housing. For multifamily buildings, each unit is modeled individually
rather than modeling the entire building all at once. See the \`OS-HPXML
Documentation <https://openstudio-hpxml.readthedocs.io/en/latest/>`\_.
for more information.

Installation
------------

For a stand-alone installation, OCHRE can be installed using \`pip\`
from the command line:

\``\`

pip install git+https://github.nrel.gov/Customer-Modeling/ochre.git

\``\`

Alternatively, you can download the repo and run the \`setup.py\` file:

\``\`

python setup.py install

\``\`

To embed OCHRE in a co-simulation using a conda environment, create an
\`environment.yml\` file in the co-simulation project and include the
following lines:

\``\`

dependencies:

- pip:

- git+https://github.nrel.gov/Customer-Modeling/ochre

\``\`

Usage
-----

OCHRE can be used to simulate a residential dwelling or an individual
piece of equipment. In either case, a python object is instantiated and
then simulated. A set of input parameters must be defined.

Below is a simple example to simulate a dwelling:

\``\`

import os

import datetime as dt

from ochre import Dwelling

from ochre.utils import default_input_path # for using sample files

house = Dwelling(

start_time=dt.datetime(2018, 5, 1, 0, 0),

time_res=dt.timedelta(minutes=10),

duration=dt.timedelta(days=3),

hpxml_file =os.path.join(default_input_path, 'Input Files',
'sample_resstock_properties.xml'),

schedule_input_file=os.path.join(default_input_path, 'Input Files',
'sample_resstock_schedule.csv'),

weather_file=os.path.join(default_input_path, 'Weather',
'USA_CO_Denver.Intl.AP.725650_TMY3.epw'),

verbosity=3,

)

df, metrics, hourly = dwelling.simulate()

\``\`

This will return 3 variables:

\* \`df\`: a Pandas DataFrame with 10 minute resolution

\* \`metrics\`: a dictionary of energy metrics

\* \`hourly\`: a Pandas DataFrame with 1 hour resolution (verbosity >= 3
only)

OCHRE can also be used to model a specific piece of equipment so long as
the boundary conditions are appropriately defined. For example, a water
heater could be simulated alone so long as draw profile, ambient air
temperature, and mains temperature are defined.

For more examples, see the following python scripts in the \`bin\`
folder:

\* Run a single dwelling: \`bin/run_dwelling.py\`

\* Run a single piece of equipment: \`bin/run_equipment.py\`

\* Run a dwelling with an external controller:
\`bin/run_external_control.py\`

\* Run multiple dwellings: \`bin/run_multiple.py\`

\* Run a fleet of equipment: \`bin/run_fleet.py\`

License
-------

This project is available under a BSD-3-like license, which is a free,
open-source, and permissive license. For more information, check out the
\`license file
<https://github.nrel.gov/Customer-Modeling/ochre/blob/main/LICENSE>`\_.

Citation and Publications
-------------------------

When using OCHRE in your publications, please cite:

1. Blonsky, M., Maguire, J., McKenna, K., Cutler, D., Balamurugan, S.
   P., & Jin, X. (2021). **OCHRE: The Object-oriented, Controllable,
   High-resolution Residential Energy Model for Dynamic Integration
   Studies.** *Applied Energy*, *290*, 116732.
   https://doi.org/10.1016/j.apenergy.2021.116732

Below is a list of publications that have used OCHRE:

2.  Munankarmi, P., Maguire, J., Balamurugan, S. P., Blonsky, M.,
    Roberts, D., & Jin, X. (2021). Community-scale interaction of energy
    efficiency and demand flexibility in residential buildings. *Applied
    Energy*, *298*, 117149.
    https://doi.org/10.1016/j.apenergy.2021.117149

3.  Pattawi, K., Munankarmi, P., Blonsky, M., Maguire, J., Balamurugan,
    S. P., Jin, X., & Lee, H. (2021). Sensitivity Analysis of Occupant
    Preferences on Energy Usage in Residential Buildings. *Proceedings
    of the ASME 2021 15th International Conference on Energy
    Sustainability, ES 2021*. https://doi.org/10.1115/ES2021-64053

4.  Blonsky, M., Munankarmi, P., & Balamurugan, S. P. (2021).
    Incorporating residential smart electric vehicle charging in home
    energy management systems. *IEEE Green Technologies Conference*,
    *2021-April*, 187–194.
    https://doi.org/10.1109/GREENTECH48523.2021.00039

5.  Cutler, D., Kwasnik, T., Balamurugan, S., Elgindy, T., Swaminathan,
    S., Maguire, J., & Christensen, D. (2021). Co-simulation of
    transactive energy markets: A framework for market testing and
    evaluation. *International Journal of Electrical Power & Energy
    Systems*, *128*, 106664.
    https://doi.org/10.1016/J.IJEPES.2020.106664

6.  Utkarsh, K., Ding, F., Jin, X., Blonsky, M., Padullaparti, H., &
    Balamurugan, S. P. (2021). A Network-Aware Distributed Energy
    Resource Aggregation Framework for Flexible, Cost-Optimal, and
    Resilient Operation. *IEEE Transactions on Smart Grid*.
    https://doi.org/10.1109/TSG.2021.3124198

7.  Blonsky, M., McKenna, K., Maguire, J., & Vincent, T. (2022). Home
    energy management under realistic and uncertain conditions: A
    comparison of heuristic, deterministic, and stochastic control
    methods. *Applied Energy*, *325*, 119770.
    https://doi.org/10.1016/J.APENERGY.2022.119770

8.  Munankarmi, P., Maguire, J., & Jin, X. (2022). *Occupancy-Based
    Controls for an All-Electric Residential Community in a Cold
    Climate*. 1–5. https://doi.org/10.1109/PESGM48719.2022.9917067

9.  Wang, J., Munankarmi, P., Maguire, J., Shi, C., Zuo, W., Roberts,
    D., & Jin, X. (2022). Carbon emission responsive building control: A
    case study with an all-electric residential community in a cold
    climate. *Applied Energy*, *314*, 118910.
    https://doi.org/10.1016/J.APENERGY.2022.118910

10. O’Shaughnessy, E., Cutler, D., Farthing, A., Elgqvist, E., Maguire,
    J., Blonsky, M., Li, X., Ericson, S., Jena, S., & Cook, J. J.
    (2022). *Savings in Action: Lessons from Observed and Modeled
    Residential Solar Plus Storage Systems*.
    https://doi.org/10.2172/1884300

11. Earle, L., Maguire, J., Munankarmi, P., & Roberts, D. (2023). The
    impact of energy-efficiency upgrades and other distributed energy
    resources on a residential neighborhood-scale electrification
    retrofit. *Applied Energy*, *329*, 120256.
    https://doi.org/10.1016/J.APENERGY.2022.120256

Contact
-------

For any questions, concerns, or suggestions for new features in OCHRE,
contact the developers directly at Jeff.Maguire@nrel.gov and
Michael.Blonsky@nrel.gov
