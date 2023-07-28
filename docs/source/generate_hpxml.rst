Input Files and Arguments
=========================

HPXML File
----------

OCHRE uses the Home Performance eXtensible Markup Language, or
`HPXML <https://www.hpxmlonline.com/>`__, file format for defining
building properties. HPXML provides a standardized format for all the
relevant inputs for a building energy model of a residential building.
The full HPXML schema and a validator tool is available
`here <https://hpxml.nrel.gov/>`__. HPXML is continuously updated to
account for additional relevant properties, and in some cases extension
elements can be used to store additional information not currently
included in the schema.

-  Standardized data format designed for interoperability between
   stakeholders

-  Generated during audits with REM/Rate, but also by other NREL tools
   like `ResStock <https://resstock.nrel.gov/>`__ (or any other
   `OS-HPXML <https://github.com/NREL/OpenStudio-HPXML>`__ based
   workflow)

-  HPXML integration allows us to quickly generate corresponding models
   suitable for co-simulation based on other workflows

Schedule Input File
-------------------

A schedule input file is optional but highly recommended. OS-HPXML has
two different types of occupancy models it supports: “asset” and
“operational” (see
`here <https://openstudio-hpxml.readthedocs.io/en/latest/workflow_inputs.html?highlight=occupant#buildingoccupancy>`__
for more information). The “asset” occupant model uses the schedules
defined in `ANSI/RESNET
301 <http://www.resnet.us/wp-content/uploads/archive/resblog/2019/01/ANSIRESNETICC301-2019_vf1.23.19.pdf>`__
for all occupant driven loads, but note that these schedules represent a
smooth average usage for all of the occupant driven loads (hot water
usage, dishwasher, clothes washer, clothes dryer, and cooking range) as
well as occupancy itself. The “operational” calculation uses a
stochastic event generator (described
`here <https://www.sciencedirect.com/science/article/pii/S0306261922011540>`__)
to model more realistic events associated with the occupant driven
loads. The operational (or stochastic) model is most often used in OCHRE
as it more realistically models the on/off usage of these devices and
therefore gets a better estimate of the power spikes associated with
their usage.

The schedule file is generated when using ResStock (named
“schedules.csv”) or when using BEopt and selecting stochastic schedules
for each end use (also named “schedules.csv”). The file contains all
necessary times series schedule information for load profiles as well as
hourly temperature setpoints for both thermostats and water heaters (if
. See the `OS-HPXML
documentation <https://openstudio-hpxml.readthedocs.io/en/latest/workflow_inputs.html#detailed-schedule-inputs>`__
for more information.

Tip: The load profile values in the schedule input file are normalized.
OCHRE can save a schedule file after initialization that contains load
profiles for each scheduled equipment in units of kW.

Weather File
------------

A weather input file is required for simulating a dwelling. OCHRE
accepts
`EnergyPlus <https://bigladdersoftware.com/epx/docs/8-3/auxiliary-programs/energyplus-weather-file-epw-data-dictionary.html>`__
and `National Solar Radiation Database <https://nsrdb.nrel.gov/>`__
(NSRDB) weather file formats.

Generating Input Files
----------------------

A large advantage to using HPXML is the interoperability it provides,
particularly with other NREL building energy modeling tools. HPXML files
can be generated using the
`OS-HPXML <https://github.com/NREL/OpenStudio-HPXML>`__ workflow, which
is documented
`here <https://openstudio-hpxml.readthedocs.io/en/latest/intro.html>`__.
This workflow is used in both
`BEopt <https://www.nrel.gov/buildings/beopt.html>`__ (version 3.0 or
later) and `ResStock <https://github.com/NREL/resstock>`__ (version 3.0
or later). As a result, a user familiar with these other tools generates
OCHRE input files as part of their normal workflow. This allows these
other tools to be used as a front end and enables quick comparisons
between OCHRE and EnergyPlus. OCHRE has been tested with HPXML files
from both workflows, but note it does not currently support all of the
features of these tools.

HPXML and occupancy schedule input files can be generated from:

-  `BEopt <https://www.nrel.gov/buildings/beopt.html>`__ 3.0 or later:
   best for designing a single building model. Includes a user interface
   to select building features. Note that the occupancy schedule file is
   optional.

-  `End-Use Load
   Profiles <https://www.nrel.gov/buildings/end-use-load-profiles.html>`__
   Database: best for using pre-existing building models

-  `ResStock <https://resstock.nrel.gov/>`__: best for existing ResStock
   users and for users in need of a large sample of building models

Weather input files can be generated from:

-  `BEopt <https://www.nrel.gov/buildings/beopt.html>`__ or
   `EnergyPlus <https://energyplus.net/weather>`__: for TMY weather
   files in EPW format

-  `NSRDB <https://nsrdb.nrel.gov/data-viewer>`__: for TMY and AMY
   weather files in NSRDB format

