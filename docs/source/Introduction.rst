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

OCHRE\ |tm| is a Python-based energy modeling tool designed to model end-use
loads and distributed energy resources in residential buildings. OCHRE
includes detailed models for building thermal envelopes and for flexible
devices including HVAC equipment, water heaters, electric vehicles, solar PV,
and batteries. OCHRE can:

- Generate diverse and representative end-use load profiles at a high temporal
  resolution

- Simulate advanced control strategies for single devices, fleets, individual
  homes, and neighborhoods 

- Examine the impacts of energy efficiency and flexibility on customers
  through energy costs and occupant comfort

- Assess grid reliability and resilience through building-to-grid
  co-simulation

- Integrate with many of NREL's established modeling tools, including
  `ResStock <https://resstock.nrel.gov/>__`\ |tm|, `BEopt
  <https://www.nrel.gov/buildings/beopt.html>__`\ |tm|, `EVI-Pro
  <https://www.nrel.gov/transportation/evi-pro.html>`__, `SAM
  <https://sam.nrel.gov/>`__, and `HELICS <https://helics.org/>`__.


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

OCHRE can be run in Python or through a command line interface. The `OCHRE
User Tutorial
<https://colab.research.google.com/github/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb>`__
is available on Google Colab, an interactive online platform for running
Python code.

Python Interface
~~~~~~~~~~~~~~~~

OCHRE can be used to simulate a residential dwelling or individual pieces of
equipment. In either case, a Python object is instantiated and then simulated.

The following code will simulate a dwelling model using `sample files
<https://github.com/NREL/OCHRE/tree/main/ochre/defaults/Input%20Files>`__ that
contain building and equipment properties, occupancy schedules, and weather
data. In addition to `input files <#generating-input-files>`_, OCHRE requires
`input arguments <#dwelling-arguments>`_ to specify the simulation start time,
time resolution, and duration. `Time series results
<#dwelling-time-series-results>`_ and simulation `metrics <#all-metrics>`_ can
be saved to memory and/or in output files.

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
        hpxml_schedule_file=os.path.join(default_input_path, "Input Files", "bldg0112631_schedule.csv"),
        weather_file=os.path.join(default_input_path, "Weather", "USA_CO_Denver.Intl.AP.725650_TMY3.epw"),
    )

    house.simulate()

OCHRE can also be used to model a single piece of equipment, a fleet of
equipment, or multiple dwellings. It can be run in co-simulation with custom
controllers, home energy management systems, aggregators, and grid models. 

For more examples, see:

- The `OCHRE User Tutorial
  <https://github.com/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb>`__
  Jupyter notebook (also available on `Google Colab
  <https://colab.research.google.com/github/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb>`__)

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

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

OCHRE can be run from the command line using the following commands:

- ``ochre single``: Run a single dwelling simulation

- ``ochre local``: Run multiple dwellings in parallel or in series

- ``ochre hpc``: Run multiple dwellings using Slurm

- ``ochre-gui-basic``: Run a single dwelling by specifying a run directory.
  Uses default options only.

- ``ochre-gui-detailed``: Run a single dwelling using a graphical user
  interface

A small set of simulation options is available for most of these commands,
including time resolution and duration, file paths, and verbosity level. Run
``ochre single --help`` for more information on the available options. To run
simulations for single pieces of equipment or with more advanced controls, use
the Python interface.

License
-------

This project is available under a BSD-3-like license, which is a free,
open-source, and permissive license. For more information, check out the
`license file <https://github.com/NREL/OCHRE/blob/main/LICENSE>`__.


Citation and Publications
-------------------------

When using OCHRE in your publications, please cite:

1. Blonsky, M., Maguire, J., McKenna, K., Cutler, D., Balamurugan, S. P., &
   Jin, X. (2021). **OCHRE: The Object-oriented, Controllable, High-resolution
   Residential Energy Model for Dynamic Integration Studies.** *Applied
   Energy*, *290*, 116732. https://doi.org/10.1016/j.apenergy.2021.116732

Below is a list of select publications that have used OCHRE:

2.  Jeff Maguire, Michael Blonsky, Sean Ericson, Amanda Farthing, Indu
    Manogaran, and Sugi Ramaraj. 2024. *Nova Analysis: Holistically Valuing
    the Contributions of Residential Efficiency, Solar and Storage*. Golden,
    CO: National Renewable Energy Laboratory. NREL/TP-5500-84658.
    https://www.nrel.gov/docs/fy24osti/84658.pdf.

3.  Earle, L., Maguire, J., Munankarmi, P., & Roberts, D. (2023). The impact
    of energy-efficiency upgrades and other distributed energy resources on a
    residential neighborhood-scale electrification retrofit. *Applied Energy*,
    *329*, 120256. https://doi.org/10.1016/J.APENERGY.2022.120256

4.  Blonsky, M., McKenna, K., Maguire, J., & Vincent, T. (2022). Home energy
    management under realistic and uncertain conditions: A comparison of
    heuristic, deterministic, and stochastic control methods. *Applied
    Energy*, *325*, 119770. https://doi.org/10.1016/J.APENERGY.2022.119770

5.  Wang, J., Munankarmi, P., Maguire, J., Shi, C., Zuo, W., Roberts, D., &
    Jin, X. (2022). Carbon emission responsive building control: A case study
    with an all-electric residential community in a cold climate. *Applied
    Energy*, *314*, 118910. https://doi.org/10.1016/J.APENERGY.2022.118910

6.	Munankarmi P., Maguire J., Jin X. (2023). Control of Behind-the-Meter
  	Resources for Enhancing the Resilience of Residential Buildings. *IEEE
  	Power and Energy Society General Meeting*, 2023-July.
  	https://doi.org/10.1109/PESGM52003.2023.10253443

7.	Graf, P. and Emami, P. (2024). Three Pathways to Neurosymbolic
  	Reinforcement Learning with Interpretable Model and Policy Networks.
  	*arXiv*. https://arxiv.org/abs/2402.05307 (see also: `Github: ochre-gym
  	<https://nrel.github.io/ochre_gym/>`__)

8.  Utkarsh, K., Ding, F., Jin, X., Blonsky, M., Padullaparti, H., &
    Balamurugan, S. P. (2021). A Network-Aware Distributed Energy Resource
    Aggregation Framework for Flexible, Cost-Optimal, and Resilient Operation.
    *IEEE Transactions on Smart Grid*.
    https://doi.org/10.1109/TSG.2021.3124198


Contact
-------

For any usage questions or suggestions for new features in OCHRE, please
create an issue on Github. For any other questions or concerns, contact the
developers directly at Jeff.Maguire@nrel.gov and Michael.Blonsky@nrel.gov.
