Getting Started
===============

.. image:: images/OCHRE-Logo-Horiz-2Color.png
  :width: 500
  :alt: OCHRE logo

.. note::
  If you use OCHRE for your research or other projects, please fill out our
  `user survey <https://forms.office.com/g/U4xYhaWEvs>`__.

OCHRE Overview
--------------

OCHRE\ |tm| is a Python-based energy modeling tool designed to model flexible
end-use loads and distributed energy resources in residential buildings. OCHRE
includes detailed models for flexible devices including HVAC equipment, water
heaters, electric vehicles, solar PV, and batteries. It can examine the
impacts of novel control strategies on energy consumption and occupant comfort
metrics. OCHRE integrates with many of NREL's established modeling tools,
including EnergyPlus\ |tm|, BEopt\ |tm|, ResStock\ |tm|, SAM, and EVI-Pro.

.. |tm| unicode:: U+2122

More information about OCHRE can be found on `NREL's website
<https://www.nrel.gov/grid/ochre.html>`__ and from the `Powered By OCHRE
<https://www.youtube.com/watch?v=B5elLVtYDbI>`__ webinar recording. 

Installation
------------

OCHRE can be installed using ``pip`` from the command line:

.. code-block:: python

    pip install ochre-nrel

Alternatively, you can install a specific branch, for example:

.. code-block:: python

    pip install git+https://github.com/NREL/OCHRE@dev

Note that OCHRE requires Python version >=3.9 and <3.12.

Usage
-----

OCHRE can be used to simulate a residential dwelling or individual pieces of
equipment. In either case, a python object is instantiated and then simulated.

Below is a simple example to simulate a dwelling:

.. code-block:: python

    import os
    import datetime as dt
    from ochre import Dwelling
    from ochre.utils import default_input_path  # for using sample files

    house = Dwelling(
        start_time=dt.datetime(2018, 5, 1, 0, 0),
        time_res=dt.timedelta(minutes=10),
        duration=dt.timedelta(days=3),
        hpxml_file=os.path.join(default_input_path, "Input Files", "bldg0112631-up11.xml"),
        schedule_input_file=os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv"),
        weather_file=os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
        verbosity=3,
    )

    df, metrics, hourly = house.simulate()

This will return 3 variables:

- ``df``: a Pandas DataFrame with 10 minute resolution

- ``metrics``: a dictionary of energy metrics

- ``hourly``: a Pandas DataFrame with 1 hour resolution (``verbosity >= 3`` only)

OCHRE can also be used to model a single piece of equipment, a fleet of
equipment, or multiple dwellings. It can also be run in co-simulation with
custom controllers, home energy management systems, aggregators, and grid
models. 

For more examples, see:

- The `OCHRE User Tutorial
  <https://github.com/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb>`__
  Jupyter notebook 

- Python example scripts to:

  - Run a `single dwelling
    <https://github.com/NREL/OCHRE/blob/main/bin/run_dwelling.py>`__

  - Run a `single piece of equipment
    <https://github.com/NREL/OCHRE/blob/main/bin/run_equipment.py>`__

  - Run a `fleet of equipment
    <https://github.com/NREL/OCHRE/blob/main/bin/run_fleet.py>`__

  - Run `multiple dwellings
    <https://github.com/NREL/OCHRE/blob/main/bin/run_multiple.py>`__

  - Run OCHRE with `an external controller
    <https://github.com/NREL/OCHRE/blob/main/bin/run_external_control.py>`__

  - Run OCHRE in `co-simulation
    <https://github.com/NREL/OCHRE/blob/main/bin/run_cosimulation.py>`__ using
    HELICS

License
-------

This project is available under a BSD-3-like license, which is a free,
open-source, and permissive license. For more information, check out the
`license file <https://github.com/NREL/OCHRE/blob/main/LICENSE>`__.


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

For any usage questions or suggestions for new features in OCHRE, please
create an issue on Github. For any other questions or concerns, contact the
developers directly at Jeff.Maguire@nrel.gov and Michael.Blonsky@nrel.gov.
