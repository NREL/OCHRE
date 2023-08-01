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

The schedule file (usually named “schedules.csv”) is generated when using ResStock 
or when using BEopt and selecting stochastic schedules
for each end use. The file contains all
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
   optional; users must specify stochastic occupancy in BEopt. **Note
   that BEopt is also under active development. OCHRE may not currently
   work with BEopt due to using different versions of OS-HPXML. OCHRE
   will be re-synchronized with BEopt after the next release in mid
   August (using OS-HXPML v1.7).**

-  `End-Use Load
   Profiles <https://www.nrel.gov/buildings/end-use-load-profiles.html>`__
   Database: best for using pre-existing building models

-  `ResStock <https://resstock.nrel.gov/>`__: best for existing ResStock
   users and for users in need of a large sample of building models.

Weather input files can be generated from:

-  `BEopt <https://www.nrel.gov/buildings/beopt.html>`__ or
   `EnergyPlus <https://energyplus.net/weather>`__: for TMY weather
   files in EPW format

-  `NSRDB <https://nsrdb.nrel.gov/data-viewer>`__: for TMY and AMY
   weather files in NSRDB format

Dwelling Arguments
------------------

A Dwelling model can be initialized using:

.. code-block:: python

   from OCHRE import Dwelling
   house = Dwelling(\**dwelling_args)

where ``dwelling_args`` is a Python dictionary of Dwelling arguments.

The table below lists the required arguments for creating a Dwelling
model.

=======================  =========================  ========================================================================= 
**Argument Name**        **Argument Type**          **Description**     
=======================  =========================  ========================================================================= 
``start_time``           ``datetime.datetime``      Simulation start time
``time_res``             ``datetime.timedelta``     Simulation timestep
``duration``             ``datetime.timedelta``     Simulation duration
``hpxml_file``           string                     Path to hpxml file
``weather_file``         string                     Path to weather file
``weather_path``         string                     Path to directory of weather files [#]_
=======================  =========================  =========================================================================

.. [#] If ``weather_path`` is used, ``weather_file`` will be read from the HPXML file. Useful if 
       running a batch of files with different weather files (ie from ResStock)

The table below lists the optional arguments for creating a ``Dwelling`` model.

==========================  =========================  ==============================  ====================================================================================================================================================================
**Argument Name**           **Argument Type**          **Default Value**               **Description**                                                                                                                                                     
==========================  =========================  ==============================  ====================================================================================================================================================================
``name``                    string                     None                            Name of the simulation                                                                                                                                           
``schedule_input_file``     string                     None                            Path to schedule input file                                                                                                                                      
``initialization_time``     ``datetime.timedelta``     None                            Length of "warm up" simulation for initial conditions [#]_                                                                                                       
``time_zone``               string                     None [#]_                       Use ``DST`` for local U.S. time zone with daylight savings, ``noDST`` for local U.S. time zone without [#]_                                                      
``verbosity``               int                        1                               Verbosity of the outputs, from 0-9. See `Outputs and Analysis <https://github.com/NREL/OCHRE/blob/documentation/docs/source/Outputs.rst>`__ for details.                
``metrics_verbosity``       int                        1                               Verbosity of metrics, from 0-9. See `Dwelling Metrics <https://github.com/NREL/OCHRE/blob/documentation/docs/source/Outputs.rst#dwelling-metrics>`__ for details.
``output_path``             string                     [#]_                            Path to saved output files                                                                                                                                       
``output_to_parquet``       boolean                    False                           Save time series data as parquet (instead of .csv)                                                                                                               
``export_res``              ``datetime.timedelta``     None [#]_                       Time resolution to save results                                                                                                                                  
``save_results``            boolean                    ``TRUE`` if ``verbosity > 0``   Save results, including time series, metrics, status, and schedule outputs                                                                                       
``save_args_to_json``       boolean                    ``FALSE``                       Save all input arguments to .json file, including user defined arguments. [#]_                                                                                    
``save_status``             boolean                    ``TRUE`` [#]_                   Save status file for is simulation completed or failed                                                                                                            
``save_schedule_columns``   list                       Empty list                      List of time series inputs to save to schedule outputs file                                                                                                       
``schedule``                pandas.DataFrame           None                            Schedule with equipment and weather data that overrides the ``schedule_input_file`` and the ``equipment_schedule_file``. Not required for ``Dwelling``                          
``ext_time_res``            datetime.timedelta         None                            Time resolution for external controller. Required for Duty Cycle control.                                                                                            
``seed``                    int or string              HPXML or schedule file          Random seed for initial temperatures and EV event data                                                                                                               
``modify_hpxml_dict``       dict                       empty dict                      Dictionary that directly modifies values from HPXML file                                                                                                          
``Equipment``               dict                       empty dict                      Includes equipment specific arguments                                                                                                                             
``Envelope``                dict                       empty dict                      Includes envelope specific arguments                                                                                                                              
==========================  =========================  ==============================  ====================================================================================================================================================================

.. [#] While not required, a warm up period **is recommended**. The warm up gets more accurate initial conditions
       for the simulation by running a few prior days. Warm up is particularly helpful for simulation with a 
       shorter ``duration``
.. [#] ``None`` means no time zone is modeled or considered.
.. [#] Can also accept any time zone in ``pyzt.all_timezones``
.. [#] Default location is same as HPXML file
.. [#] Default is time step for time series data
.. [#] If ``False`` and ``verbosity > 3``, .json will only include HPXML properties
.. [#] If ``verbosity > 0``, else ``FALSE``

Equipment-specific Arguments
----------------------------

An Equipment model can be initialized in a very similar way to a
Dwelling. For example, to initialize a battery:



.. code-block:: python
   from OCHRE import Battery
   equipment = Battery(name, \**equipment_args)


where equipment_args is a Python dictionary of Equipment arguments.
A full set of the equipment classes available are listed in this
section, by end use.

The table below lists the required arguments for creating any standalone
Equipment model. Some equipment have additional required arguments as
described in the sections below.

+----------------------------+--------------+-------------------------+
| **Argument Name**          | **Argument   | **Description**         |
|                            | Type**       |                         |
+============================+==============+=========================+
| start_time                 | datet        | Simulation start time   |
|                            | ime.datetime |                         |
+----------------------------+--------------+-------------------------+
| time_res                   | dateti       | Simulation time         |
|                            | me.timedelta | resolution              |
+----------------------------+--------------+-------------------------+
| duration                   | dateti       | Simulation duration     |
|                            | me.timedelta |                         |
+----------------------------+--------------+-------------------------+
|                            |              |                         |
+----------------------------+--------------+-------------------------+
|                            |              |                         |
+----------------------------+--------------+-------------------------+
|                            |              |                         |
+----------------------------+--------------+-------------------------+

The table below lists the optional arguments for creating any standalone
Equipment model. Some equipment have additional optional arguments as
described in the sections below.

+-------+-----+---------+---------------------------------------------+
| **Arg | *   | **      | **Description**                             |
| ument | *Ar | Default |                                             |
| N     | gum | Value** |                                             |
| ame** | ent |         |                                             |
|       | Typ |         |                                             |
|       | e** |         |                                             |
+=======+=====+=========+=============================================+
| name  | str | OCHRE   | Name of the simulation                      |
|       | ing |         |                                             |
+-------+-----+---------+---------------------------------------------+
| init  | dat | None    | Runs a "warm up" simulation to improve      |
| ializ | eti | (no     | initial temperature values                  |
| ation | me. | i       |                                             |
| _time | tim | nitiali |                                             |
|       | ede | zation) |                                             |
|       | lta |         |                                             |
+-------+-----+---------+---------------------------------------------+
| zone  | str | None    | Name of Envelope zone if envelope model     |
| _name | ing |         | exists                                      |
+-------+-----+---------+---------------------------------------------+
| enve  | oc  | None    | Envelope model for measuring temperature    |
| lope_ | hre |         | impacts (required for HVAC equipment)       |
| model | .En |         |                                             |
|       | vel |         |                                             |
|       | ope |         |                                             |
+-------+-----+---------+---------------------------------------------+
| verb  | int | 1       | Verbosity of the outputs, from 0-9. See     |
| osity |     |         | Outputs and Analysis for details            |
+-------+-----+---------+---------------------------------------------+
| o     | str | HPXML   | Path to saved output files                  |
| utput | ing | or      |                                             |
| _path |     | eq      |                                             |
|       |     | uipment |                                             |
|       |     | s       |                                             |
|       |     | chedule |                                             |
|       |     | file    |                                             |
|       |     | di      |                                             |
|       |     | rectory |                                             |
+-------+-----+---------+---------------------------------------------+
| ou    | b   | FALSE   | Save time series files as parquet files     |
| tput_ | ool |         | (False saves as csv files)                  |
| to_pa | ean |         |                                             |
| rquet |     |         |                                             |
+-------+-----+---------+---------------------------------------------+
| expor | dat | None    | Time resolution to save results to files    |
| t_res | eti | (no     |                                             |
|       | me. | inter   |                                             |
|       | tim | mediate |                                             |
|       | ede | data    |                                             |
|       | lta | export) |                                             |
+-------+-----+---------+---------------------------------------------+
| sa    | b   | True if | Save results files, including time series   |
| ve_re | ool | ve      | files, metrics file, schedule output file,  |
| sults | ean | rbosity | and status file                             |
|       |     | > 0     |                                             |
+-------+-----+---------+---------------------------------------------+
| sa    | b   | FALSE   | Save all input arguments to json file,      |
| ve_ar | ool |         | including user defined arguments. If False  |
| gs_to | ean |         | and verbosity >= 3, the json file will only |
| _json |     |         | include HPXML properties.                   |
+-------+-----+---------+---------------------------------------------+
| s     | b   | True if | Save status file to indicate whether the    |
| ave_s | ool | save_   | simulation is complete or failed            |
| tatus | ean | results |                                             |
|       |     | is True |                                             |
+-------+-----+---------+---------------------------------------------+
| s     | b   | FALSE   | Include equivalent battery model data in    |
| ave_e | ool |         | results                                     |
| bm_re | ean |         |                                             |
| sults |     |         |                                             |
+-------+-----+---------+---------------------------------------------+
| s     | l   | Empty   | List of time series inputs to save to       |
| ave_s | ist | list    | schedule output file                        |
| chedu |     |         |                                             |
| le_co |     |         |                                             |
| lumns |     |         |                                             |
+-------+-----+---------+---------------------------------------------+
| equ   | str | None    | File with equipment time series data.       |
| ipmen | ing |         | Optional for most equipment                 |
| t_sch |     |         |                                             |
| edule |     |         |                                             |
| _file |     |         |                                             |
+-------+-----+---------+---------------------------------------------+
| sch   | d   | None    | Dictionary of {file_column_name:            |
| edule | ict |         | ochre_schedule_name} to rename columns in   |
| _rena |     |         | equipment_schedule_file. Sometimes used for |
| me_co |     |         | PV                                          |
| lumns |     |         |                                             |
+-------+-----+---------+---------------------------------------------+
| s     | num | 1       | Scaling factor to normalize data in         |
| chedu | ber |         | equipment_schedule_file. Sometimes used for |
| le_sc |     |         | PV to convert units                         |
| ale_f |     |         |                                             |
| actor |     |         |                                             |
+-------+-----+---------+---------------------------------------------+
| sch   | p   | None    | Schedule with equipment or weather data     |
| edule | and |         | that overrides the schedule_input_file and  |
|       | as. |         | the equipment_schedule_file. Not required   |
|       | Dat |         | for Dwelling and some equipment             |
|       | aFr |         |                                             |
|       | ame |         |                                             |
+-------+-----+---------+---------------------------------------------+
| ex    | dat | None    | Time resolution for external controller.    |
| t_tim | eti |         | Required if using Duty Cycle control        |
| e_res | me. |         |                                             |
|       | tim |         |                                             |
|       | ede |         |                                             |
|       | lta |         |                                             |
+-------+-----+---------+---------------------------------------------+
| seed  | int | HPXML   | Random seed for setting initial             |
|       | or  | or      | temperatures and EV event data              |
|       | str | eq      |                                             |
|       | ing | uipment |                                             |
|       |     | s       |                                             |
|       |     | chedule |                                             |
|       |     | file    |                                             |
+-------+-----+---------+---------------------------------------------+

The following sections list the names and arguments for all OCHRE
equipment by end use. Many equipment types have all of their required
arguments included in the HPXML properties. These properties can be
overwritten by specifying the arguments in the \`equipment_args\`
dictionary.

HVAC Heating and Cooling
~~~~~~~~~~~~~~~~~~~~~~~~

OCHRE includes models for the following HVAC equipment:

+---------+-------------------+--------------------+-----------------------------------------------------------+
| End Use | Equipment Class   | Equipment Name     | Description                                               |
+=========+===================+====================+===========================================================+
| Heating | ElectricFurnace   | Electric Furnace   |                                                           |
| Heating | ElectricBaseboard | Electric Baseboard |                                                           |
| Heating | ElectricBoiler    | Electric Boiler    |                                                           |
| Heating | GasFurnace        | Gas Furnace        |                                                           |
| Heating | GasBoiler         | Gas Boiler         |                                                           |
| Heating | HeatPumpHeater    | Heat Pump Heater   | Air Source Heat Pump  with no electric resistance backup  |
| Heating | ASHPHeater        | ASHP Heater        | Air Source Heat Pump, heating only                        |
| Heating | MSHPHeater        | MSHP Heater        | Minisplit Heat Pump, heating only                         |
| Cooling | AirConditioner    | Air Conditioner    | Central air conditioner                                   |
| Cooling | RoomAC            | Room AC            | Room air conditioner                                      |
| Cooling | ASHPCooler        | ASHP Cooler        | Air Source Heat Pump, cooling only                        |
| Cooling | MSHPCooler        | MSHP Cooler        | Minisplit Heat Pump, cooling only                         |
+---------+-------------------+--------------------+-----------------------------------------------------------+


The table below shows the required and optional equipment-specific
arguments for HVAC equipment.

+---------------+--------+---------+--------------+------------------+
| Argument Name | Ar     | Re      | Default      | Description      |
|               | gument | quired? | Value        |                  |
|               | Type   |         |              |                  |
+===============+========+=========+==============+==================+
| Capacity (W)  | number | Yes     | N/A          | Number: Rated    |
|               | or     |         |              | capacity         |
|               | list   |         |              |                  |
|               |        |         |              | List: Rated      |
|               |        |         |              | capacity by      |
|               |        |         |              | speed            |
+---------------+--------+---------+--------------+------------------+
| use_i         | b      | No      | True only if | Method to        |
| deal_capacity | oolean |         | time_res >=  | determine HVAC   |
|               |        |         | 5 minutes or | capacity.        |
|               |        |         | for          |                  |
|               |        |         | va           | If True, use     |
|               |        |         | riable-speed | ideal setpoint   |
|               |        |         | equipment    | method.          |
|               |        |         |              |                  |
|               |        |         |              | If False, use    |
|               |        |         |              | equipment        |
|               |        |         |              | cycling method   |
|               |        |         |              | with thermostat  |
|               |        |         |              | deadband         |
+---------------+--------+---------+--------------+------------------+
| …             |        |         |              |                  |
+---------------+--------+---------+--------------+------------------+

Water Heating
~~~~~~~~~~~~~

OCHRE includes models for the following Water Heating equipment:

+-------------------+----------------------+--------------------------+
| End Use           | Equipment Class      | Equipment Name           |
+===================+======================+==========================+
| Water Heating     | ElectricR            | Electric Tank Water      |
|                   | esistanceWaterHeater | Heater                   |
+-------------------+----------------------+--------------------------+
| Water Heating     | GasWaterHeater       | Gas Tank Water Heater    |
+-------------------+----------------------+--------------------------+
| Water Heating     | HeatPumpWaterHeater  | Heat Pump Water Heater   |
+-------------------+----------------------+--------------------------+
| Water Heating     | TanklessWaterHeater  | Tankless Water Heater    |
+-------------------+----------------------+--------------------------+
| Water Heating     | Ga                   | Gas Tankless Water       |
|                   | sTanklessWaterHeater | Heater                   |
+-------------------+----------------------+--------------------------+

The table below shows the required and optional equipment-specific
arguments for Water Heating equipment.

+---+----------+---+-------+----------------+--------------------------+
| e | **       | * | **R   | **Default      | **Description**          |
| n | Argument | * | equir | Value**        |                          |
| d | Name**   | A | ed?** |                |                          |
| u |          | r |       |                |                          |
| s |          | g |       |                |                          |
| e |          | u |       |                |                          |
|   |          | m |       |                |                          |
|   |          | e |       |                |                          |
|   |          | n |       |                |                          |
|   |          | t |       |                |                          |
|   |          | T |       |                |                          |
|   |          | y |       |                |                          |
|   |          | p |       |                |                          |
|   |          | e |       |                |                          |
|   |          | * |       |                |                          |
|   |          | * |       |                |                          |
+===+==========+===+=======+================+==========================+
| W | us       | b | No    | True if        | If True, OCHRE sets      |
| a | e_ideal_ | o |       | time_res >= 5  | water heater capacity to |
| t | capacity | o |       | minutes        | meet the setpoint. If    |
| e |          | l |       |                | False, OCHRE uses        |
| r |          | e |       |                | thermostat deadband      |
| H |          | a |       |                | control                  |
| e |          | n |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | wat      | i | No    | 12 if Heat     | Number of nodes in water |
| a | er_nodes | n |       | Pump Water     | tank model               |
| t |          | t |       | Heater, 1 if   |                          |
| e |          |   |       | Tankless Water |                          |
| r |          |   |       | Heater,        |                          |
| H |          |   |       | otherwise 2    |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Capacity | n | No    | 4500           | Water heater capacity    |
| a | (W)      | u |       |                |                          |
| t |          | m |       |                |                          |
| e |          | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Ef       | n | No    | 1              | Water heater efficiency  |
| a | ficiency | u |       |                | (or supplemental heater  |
| t | (-)      | m |       |                | efficiency for HPWH)     |
| e |          | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Setpoint | n | Yes   | Taken from     | Water heater setpoint    |
| a | Tem      | u |       | HPXML file, or | temperature. Can also be |
| t | perature | m |       | 51.67          | set in schedule          |
| e | (C)      | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Deadband | n | No    | 8.17 for Heat  | Water heater deadband    |
| a | Tem      | u |       | Pump Water     | size. Can also be set in |
| t | perature | m |       | Heater,        | schedule                 |
| e | (C)      | b |       | otherwise 5.56 |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Max Tank | n | No    | 60             | Maximum water tank       |
| a | Tem      | u |       |                | temperature              |
| t | perature | m |       |                |                          |
| e | (C)      | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Mixed    | n | No    | 40.56          | Hot water temperature    |
| a | Delivery | u |       |                | for tempered water draws |
| t | Tem      | m |       |                | (sinks, showers, and     |
| e | perature | b |       |                | baths)                   |
| r | (C)      | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Initial  | n | No    | Setpoint       | Initial temperature of   |
| a | Tem      | u |       | temperature -  | the entire tank (before  |
| t | perature | m |       | 10% of         | initialization routine)  |
| e | (C)      | b |       | deadband       |                          |
| r |          | e |       | temperature    |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Max      | n | No    | None           | Maximum rate of change   |
| a | Setpoint | u |       |                | for setpoint temperature |
| t | Ramp     | m |       |                |                          |
| e | Rate     | b |       |                |                          |
| r | (C/min)  | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Tank     | n | Yes   | Taken from     | Size of water tank, in L |
| a | Volume   | u |       | HPXML file     |                          |
| t | (L)      | m |       |                |                          |
| e |          | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Tank     | n | Yes   | Taken from     | Height of water tank,    |
| a | Height   | u |       | HPXML file     | used to determine        |
| t | (m)      | m |       |                | surface area             |
| e |          | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Heat     | n | Yes   | Taken from     | Heat transfer            |
| a | Transfer | u |       | HPXML file     | coefficient of water     |
| t | Coe      | m |       |                | tank                     |
| e | fficient | b |       |                |                          |
| r | (        | e |       |                |                          |
| H | W/m^2/K) | r |       |                |                          |
| e | or UA    |   |       |                |                          |
| a | (W/K)    |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | hp_o     | b | No    | FALSE          | Disable supplemental     |
| a | nly_mode | o |       |                | heater for HPWH          |
| t |          | o |       |                |                          |
| e |          | l |       |                |                          |
| r |          | e |       |                |                          |
| H |          | a |       |                |                          |
| e |          | n |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH COP | n | Only  |                | Coefficient of           |
| a | (-)      | u | for   |                | Performance for HPWH     |
| t |          | m | Heat  |                |                          |
| e |          | b | Pump  |                |                          |
| r |          | e | Water |                |                          |
| H |          | r | H     |                |                          |
| e |          |   | eater |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH     | n | No    | 500 (for HPWH  | Capacity or rated power  |
| a | Capacity | u |       | Power)         | for HPWH                 |
| t | (W) or   | m |       |                |                          |
| e | HPWH     | b |       |                |                          |
| r | Power    | e |       |                |                          |
| H | (W)      | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH     | n | No    | 1              | Parasitic power for HPWH |
| a | Pa       | u |       |                |                          |
| t | rasitics | m |       |                |                          |
| e | (W)      | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH Fan | n | No    | 35             | Fan power for HPWH       |
| a | Power    | u |       |                |                          |
| t | (W)      | m |       |                |                          |
| e |          | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH SHR | n | No    | 0.88           | Sensible heat ratio for  |
| a | (-)      | u |       |                | HPWH                     |
| t |          | m |       |                |                          |
| e |          | b |       |                |                          |
| r |          | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH     | n | No    | 0.75 if in     | Fraction of HPWH         |
| a | Int      | u |       | Indoor Zone    | sensible gains to        |
| t | eraction | m |       | else 1         | envelope                 |
| e | Factor   | b |       |                |                          |
| r | (-)      | e |       |                |                          |
| H |          | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | HPWH     | n | No    | 0.5            | Fraction of HPWH         |
| a | Wall     | u |       |                | sensible gains to wall   |
| t | Int      | m |       |                | boundary, remainder goes |
| e | eraction | b |       |                | to zone                  |
| r | Factor   | e |       |                |                          |
| H | (-)      | r |       |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | Energy   | n | Only  | Taken from     | Water heater energy      |
| a | Factor   | u | for   | HPXML file     | factor (EF) for getting  |
| t | (-)      | m | Gas   |                | skin loss fraction       |
| e |          | b | Water |                |                          |
| r |          | e | H     |                |                          |
| H |          | r | eater |                |                          |
| e |          |   |       |                |                          |
| a |          |   |       |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+
| W | P        | n | Only  | Taken from     | Parasitic power for Gas  |
| a | arasitic | u | for   | HPXML file     | Tankless Water Heater    |
| t | Power    | m | Gas   |                |                          |
| e | (W)      | b | Tan   |                |                          |
| r |          | e | kless |                |                          |
| H |          | r | Water |                |                          |
| e |          |   | H     |                |                          |
| a |          |   | eater |                |                          |
| t |          |   |       |                |                          |
| i |          |   |       |                |                          |
| n |          |   |       |                |                          |
| g |          |   |       |                |                          |
+---+----------+---+-------+----------------+--------------------------+

Electric Vehicle
~~~~~~~~~~~~~~~~

OCHRE includes an electric vehicle (EV) model. The equipment name can be
“EV” or “Electric Vehicle”. The table below shows the required and
optional equipment-specific arguments for EVs.

+---+------------+-----+----------+------------------+--------------+
| e | **Argument | *   | **Req    | **Default        | **D          |
| n | Name**     | *Ar | uired?** | Value**          | escription** |
| d |            | gum |          |                  |              |
| u |            | ent |          |                  |              |
| s |            | Typ |          |                  |              |
| e |            | e** |          |                  |              |
+===+============+=====+==========+==================+==============+
| E | ve         | str | Yes      | BEV, if taken    | EV vehicle   |
| V | hicle_type | ing |          | from HPXML file  | type,        |
|   |            |     |          |                  | options are  |
|   |            |     |          |                  | "PHEV" or    |
|   |            |     |          |                  | "BEV"        |
+---+------------+-----+----------+------------------+--------------+
| E | char       | str | Yes      | Level 2, if      | EV charging  |
| V | ging_level | ing |          | taken from HPXML | type,        |
|   |            |     |          | file             | options are  |
|   |            |     |          |                  | "Level 1" or |
|   |            |     |          |                  | "Level 2"    |
+---+------------+-----+----------+------------------+--------------+
| E | capacity   | num | Yes      | 100 miles if     | EV battery   |
| V | or mileage | ber |          | HPXML Annual EV  | capacity in  |
|   |            |     |          | Energy < 1500    | kWh or       |
|   |            |     |          | kWh, otherwise   | mileage in   |
|   |            |     |          | 250 miles        | miles        |
+---+------------+-----+----------+------------------+--------------+
| E | enable     | b   | No       | True if          | Allows EV to |
| V | _part_load | ool |          | charging_level = | charge at    |
|   |            | ean |          | Level 2          | partial load |
+---+------------+-----+----------+------------------+--------------+
| E | ambie      | num | No       | Taken from       | Ambient      |
| V | nt_ev_temp | ber |          | schedule, or 20  | temperature  |
|   |            |     |          | C                | used to      |
|   |            |     |          |                  | estimate EV  |
|   |            |     |          |                  | usage per    |
|   |            |     |          |                  | day          |
+---+------------+-----+----------+------------------+--------------+

Battery
~~~~~~~

OCHRE includes a battery model. The table below shows the required and
optional equipment-specific arguments for batteries.

+---+----------+---+------+--------------+----------------------------+
| e | **       | * | *    | **Default    | **Description**            |
| n | Argument | * | *Req | Value**      |                            |
| d | Name**   | A | uire |              |                            |
| u |          | r | d?** |              |                            |
| s |          | g |      |              |                            |
| e |          | u |      |              |                            |
|   |          | m |      |              |                            |
|   |          | e |      |              |                            |
|   |          | n |      |              |                            |
|   |          | t |      |              |                            |
|   |          | T |      |              |                            |
|   |          | y |      |              |                            |
|   |          | p |      |              |                            |
|   |          | e |      |              |                            |
|   |          | * |      |              |                            |
|   |          | * |      |              |                            |
+===+==========+===+======+==============+============================+
| B | capa     | n | No   | 10           | Nominal energy capacity of |
| a | city_kwh | u |      |              | battery, in kWh            |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | capacity | n | No   | 5            | Max power of battery, in   |
| a |          | u |      |              | kW                         |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | ef       | n | No   | 0.98         | Battery Discharging        |
| a | ficiency | u |      |              | Efficiency, unitless       |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | e        | n | No   | 0.98         | Battery Charging           |
| a | fficienc | u |      |              | Efficiency, unitless       |
| t | y_charge | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | eff      | n | No   | 0.97         | Inverter Efficiency,       |
| a | iciency_ | u |      |              | unitless                   |
| t | inverter | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | efficie  | s | No   | "advanced"   | Efficiency calculation     |
| a | ncy_type | t |      |              | option. Options are        |
| t |          | r |      |              | "advanced", "constant",    |
| t |          | i |      |              | "curve", and "quadratic"   |
| e |          | n |      |              |                            |
| r |          | g |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | soc_init | n | No   | 0.5          | Initial State of Charge,   |
| a |          | u |      |              | unitless                   |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | soc_max  | n | No   | 0.95         | Maximum SOC, unitless      |
| a |          | u |      |              |                            |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | soc_min  | n | No   | 0.15         | Minimum SOC, unitless      |
| a |          | u |      |              |                            |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | en       | b | No   | TRUE         | If True, runs an energy    |
| a | able_deg | o |      |              | capacity degradation model |
| t | radation | o |      |              | daily                      |
| t |          | l |      |              |                            |
| e |          | e |      |              |                            |
| r |          | a |      |              |                            |
| y |          | n |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | initial  | n | No   | 50.4         | Initial open circuit       |
| a | _voltage | u |      |              | voltage, in V. Used for    |
| t |          | m |      |              | advanced efficiency and    |
| t |          | b |      |              | degradation models.        |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | v_cell   | n | No   | 3.6          | Cell voltage, in V. Used   |
| a |          | u |      |              | for advanced efficiency    |
| t |          | m |      |              | and degradation models.    |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | ah_cell  | n | No   | 70           | Cell capacity, in Ah. Used |
| a |          | u |      |              | for advanced efficiency    |
| t |          | m |      |              | and degradation models.    |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | r_cell   | n | No   | 0            | Cell resistance, in ohm.   |
| a |          | u |      |              | Used for advanced          |
| t |          | m |      |              | efficiency and degradation |
| t |          | b |      |              | models.                    |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | c        | n | No   | 9            | Schedule: Charge Start     |
| a | harge_st | u |      |              | Time, in hour of day       |
| t | art_hour | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | disc     | n | No   | 16           | Schedule: Discharge Start  |
| a | harge_st | u |      |              | Time, in hour of day       |
| t | art_hour | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | char     | n | No   | 1            | Schedule: Charge Power, in |
| a | ge_power | u |      |              | kW                         |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | dischar  | n | No   | 1            | Schedule: Discharge Power, |
| a | ge_power | u |      |              | in kW                      |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | c        | n | No   | 0            | Self-Consumption: Force    |
| a | harge_fr | u |      |              | Charge from Solar, in      |
| t | om_solar | m |      |              | boolean                    |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | impo     | n | No   | 0            | Self-Consumption: Grid     |
| a | rt_limit | u |      |              | Import Limit, in kW        |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | expo     | n | No   | 0            | Self-Consumption: Grid     |
| a | rt_limit | u |      |              | Export Limit, in kW        |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | enab     | b | No   | True only if | If True, creates 1R-1C     |
| a | le_therm | o |      | zone_name or | thermal model for battery  |
| t | al_model | o |      | envelope is  | temperature. Temperature   |
| t |          | l |      | specified    | is used in degradation     |
| e |          | e |      |              | model                      |
| r |          | a |      |              |                            |
| y |          | n |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | t        | n | No   | 0.5          | Thermal Resistance, in K/W |
| a | hermal_r | u |      |              |                            |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | t        | n | No   | 90000        | Thermal Mass, in J/K       |
| a | hermal_c | u |      |              |                            |
| t |          | m |      |              |                            |
| t |          | b |      |              |                            |
| e |          | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+
| B | Initial  | n | No   | zone         |                            |
| a | Battery  | u |      | temperature  |                            |
| t | Tem      | m |      |              |                            |
| t | perature | b |      |              |                            |
| e | (C)      | e |      |              |                            |
| r |          | r |      |              |                            |
| y |          |   |      |              |                            |
+---+----------+---+------+--------------+----------------------------+

Solar PV
~~~~~~~~

OCHRE includes a solar PV model. The table below shows the required and
optional equipment-specific arguments for PV.

+---+--------+---+--------------+-------------+----------------------+
| e | **Ar   | * | *            | **Default   | **Description**      |
| n | gument | * | *Required?** | Value**     |                      |
| d | Name** | A |              |             |                      |
| u |        | r |              |             |                      |
| s |        | g |              |             |                      |
| e |        | u |              |             |                      |
|   |        | m |              |             |                      |
|   |        | e |              |             |                      |
|   |        | n |              |             |                      |
|   |        | t |              |             |                      |
|   |        | T |              |             |                      |
|   |        | y |              |             |                      |
|   |        | p |              |             |                      |
|   |        | e |              |             |                      |
|   |        | * |              |             |                      |
|   |        | * |              |             |                      |
+===+========+===+==============+=============+======================+
| P | ca     | n | Only if      |             | PV panel capacity,   |
| V | pacity | u | use_sam is   |             | in kW                |
|   |        | m | True         |             |                      |
|   |        | b |              |             |                      |
|   |        | e |              |             |                      |
|   |        | r |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | u      | b | No           | True if     | If True, runs PySAM  |
| V | se_sam | o |              | e           | to generate PV power |
|   |        | o |              | quipment_sc | profile              |
|   |        | l |              | hedule_file |                      |
|   |        | e |              | not         |                      |
|   |        | a |              | specified   |                      |
|   |        | n |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | tilt   | n | No           | Taken from  | Tilt angle from      |
| V |        | u |              | HPXML roof  | horizontal, in       |
|   |        | m |              | pitch       | degrees. Used for    |
|   |        | b |              |             | SAM                  |
|   |        | e |              |             |                      |
|   |        | r |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | orien  | n | No           | Taken from  | Orientation angle    |
| V | tation | u |              | HPXML       | from south, in       |
|   |        | m |              | building    | degrees. Used for    |
|   |        | b |              | orientation | SAM                  |
|   |        | e |              |             |                      |
|   |        | r |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | incl   | b | No           | TRUE        | If True, outputs AC  |
| V | ude_in | o |              |             | power and            |
|   | verter | o |              |             | incorporates         |
|   |        | l |              |             | inverter efficiency  |
|   |        | e |              |             | and power            |
|   |        | a |              |             | constraints          |
|   |        | n |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | i      | n | No           | 1           | Efficiency of the    |
| V | nverte | u |              |             | inverter, unitless   |
|   | r_effi | m |              |             |                      |
|   | ciency | b |              |             |                      |
|   |        | e |              |             |                      |
|   |        | r |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | inver  | s | No           | "Var"       | PV inverter          |
| V | ter_pr | t |              |             | priority. Options    |
|   | iority | r |              |             | are "Var", "Watt",   |
|   |        | i |              |             | or "CPF" (constant   |
|   |        | n |              |             | power factor)        |
|   |        | g |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | inver  | n | No           | PV.capacity | Inverter apparent    |
| V | ter_ca | u |              |             | power capacity, in   |
|   | pacity | m |              |             | kVA (i.e., kW)       |
|   |        | b |              |             |                      |
|   |        | e |              |             |                      |
|   |        | r |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | inv    | n | No           | 0.8         | Inverter minimum     |
| V | erter_ | u |              |             | power factor,        |
|   | min_pf | m |              |             | unitless             |
|   |        | b |              |             |                      |
|   |        | e |              |             |                      |
|   |        | r |              |             |                      |
+---+--------+---+--------------+-------------+----------------------+
| P | sam_   | s | Only if      |             | Weather file in SAM  |
| V | weathe | t | use_sam is   |             | format               |
|   | r_file | r | True and     |             |                      |
|   |        | i | running      |             |                      |
|   |        | n | without a    |             |                      |
|   |        | g | Dwelling     |             |                      |
+---+--------+---+--------------+-------------+----------------------+

Gas Generator
~~~~~~~~~~~~~

OCHRE includes models for the following gas generator equipment:

+-------------------+----------------------+--------------------------+
| End Use           | Equipment Class      | Equipment Name           |
+===================+======================+==========================+
| Gas Generator     | GasGenerator         | Gas Generator            |
+-------------------+----------------------+--------------------------+
| Gas Generator     | GasFuelCell          | Gas Fuel Cell            |
+-------------------+----------------------+--------------------------+

The table below shows the required and optional equipment-specific
arguments for gas generators.

+----+-----------------+--------+---------------+---------------------+
| e  | **Argument      | **Ar   | **Required?** | **Default Value**   |
| nd | Name**          | gument |               |                     |
| u  |                 | Type** |               |                     |
| se |                 |        |               |                     |
+====+=================+========+===============+=====================+
| G  | capacity        | number | No            | 6                   |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | efficiency      | number | No            | 0.95                |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | efficiency_type | string | No            | "curve" if          |
| en |                 |        |               | GasFuelCell,        |
| er |                 |        |               | otherwise           |
| at |                 |        |               | "constant"          |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | control_type    | string | No            | "Off"               |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | ramp_rate       | number | No            | 0.1                 |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | ch              | number | No            | 9                   |
| en | arge_start_hour |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | disch           | number | No            | 16                  |
| en | arge_start_hour |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | charge_power    | number | No            | 1                   |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | discharge_power | number | No            | 1                   |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | import_limit    | number | No            | 0                   |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+
| G  | export_limit    | number | No            | 0                   |
| en |                 |        |               |                     |
| er |                 |        |               |                     |
| at |                 |        |               |                     |
| or |                 |        |               |                     |
+----+-----------------+--------+---------------+---------------------+

Other Equipment
~~~~~~~~~~~~~~~

OCHRE includes basic models for other loads, including appliances,
lighting, and miscellaneous electric and gas loads:

+-------------------+----------------------+--------------------------+
| End Use           | Equipment Class      | Equipment Name           |
+===================+======================+==========================+
| Lighting          | LightingLoad         | Lighting                 |
+-------------------+----------------------+--------------------------+
| Lighting          | LightingLoad         | Exterior Lighting        |
+-------------------+----------------------+--------------------------+
| Lighting          | LightingLoad         | Basement Lighting        |
+-------------------+----------------------+--------------------------+
| Lighting          | LightingLoad         | Garage Lighting          |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Clothes Washer           |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Clothes Dryer            |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Dishwasher               |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Refrigerator             |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Cooking Range            |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | MELs                     |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | TV                       |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Well Pump                |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Gas Grill                |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Gas Fireplace            |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Gas Lighting             |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Pool Pump                |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Pool Heater              |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Hot Tub Pump             |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Hot Tub Heater           |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Ceiling Fan              |
+-------------------+----------------------+--------------------------+
| Other             | ScheduledLoad        | Ventilation Fan          |
+-------------------+----------------------+--------------------------+
| EV                | ScheduledEV          | Scheduled EV             |
+-------------------+----------------------+--------------------------+

The table below shows the required and optional equipment-specific
arguments for other equipment.

+---+------------+-----+---------+--------------+--------------------+
| e | **Argument | *   | **Requ  | **Default    | **Description**    |
| n | Name**     | *Ar | ired?** | Value**      |                    |
| d |            | gum |         |              |                    |
| u |            | ent |         |              |                    |
| s |            | Typ |         |              |                    |
| e |            | e** |         |              |                    |
+===+============+=====+=========+==============+====================+
| O | Convective | num | No      | Taken from   | Fraction of power  |
| t | Gain       | ber |         | HPXML file,  | consumption that   |
| h | Fraction   |     |         | or 0         | is dissipated      |
| e | (-)        |     |         |              | through convection |
| r |            |     |         |              | into zone          |
| / |            |     |         |              |                    |
| L |            |     |         |              |                    |
| i |            |     |         |              |                    |
| g |            |     |         |              |                    |
| h |            |     |         |              |                    |
| t |            |     |         |              |                    |
| i |            |     |         |              |                    |
| n |            |     |         |              |                    |
| g |            |     |         |              |                    |
+---+------------+-----+---------+--------------+--------------------+
| O | Radiative  | num | No      | Taken from   | Fraction of power  |
| t | Gain       | ber |         | HPXML file,  | consumption that   |
| h | Fraction   |     |         | or 0         | is dissipated      |
| e | (-)        |     |         |              | through radiation  |
| r |            |     |         |              | into zone          |
| / |            |     |         |              |                    |
| L |            |     |         |              |                    |
| i |            |     |         |              |                    |
| g |            |     |         |              |                    |
| h |            |     |         |              |                    |
| t |            |     |         |              |                    |
| i |            |     |         |              |                    |
| n |            |     |         |              |                    |
| g |            |     |         |              |                    |
+---+------------+-----+---------+--------------+--------------------+
| O | Latent     | num | No      | Taken from   | Fraction of power  |
| t | Gain       | ber |         | HPXML file,  | consumption that   |
| h | Fraction   |     |         | or 0         | is dissipated as   |
| e | (-)        |     |         |              | latent heat into   |
| r |            |     |         |              | zone               |
| / |            |     |         |              |                    |
| L |            |     |         |              |                    |
| i |            |     |         |              |                    |
| g |            |     |         |              |                    |
| h |            |     |         |              |                    |
| t |            |     |         |              |                    |
| i |            |     |         |              |                    |
| n |            |     |         |              |                    |
| g |            |     |         |              |                    |
+---+------------+-----+---------+--------------+--------------------+
