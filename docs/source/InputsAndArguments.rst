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
hourly temperature setpoints for both thermostats and water heaters (See the `OS-HPXML
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
   optional; users must specify stochastic occupancy in BEopt. To generate
   input files from BEopt, run your model as usual. The input files you need
   for OCHRE (in.hpxml and schedules.csv) will be automatically generated
   and are located in 'C:/Users/*your_username*/Documents/BEopt_3.0.x/TEMP1/1/run'.
   BEopt generates several xml files as part of the workflow, but the one
   OCHRE is looking for is always within the run directory.

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

-  The `ResStock dataset <https://data.nrel.gov/submissions/156>`__: 
   for weather files that align with ResStock-generated HPXML files.
  
Dwelling Arguments
------------------

A Dwelling model can be initialized using:

.. code-block:: python

   from OCHRE import Dwelling
   house = Dwelling(**dwelling_args)

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
       running a batch of files with different weather files (i.e., from ResStock)

The table below lists the optional arguments for creating a ``Dwelling`` model.

==========================  =========================  ==============================  ====================================================================================================================================================================
**Argument Name**           **Argument Type**          **Default Value**               **Description**                                                                                                                                                     
==========================  =========================  ==============================  ====================================================================================================================================================================
``name``                    string                     None                            Name of the simulation                                                                                                                                           
``schedule_input_file``     string                     None                            Path to schedule input file                                                                                                                                      
``initialization_time``     ``datetime.timedelta``     None                            Length of "warm up" simulation for initial conditions [#]_                                                                                                       
``time_zone``               string                     None [#]_                       Use ``DST`` for local U.S. time zone with daylight savings, ``noDST`` for local U.S. time zone without [#]_                                                      
``verbosity``               int                        1                               Verbosity of the outputs, from 0-9. See `Outputs and Analysis <https://ochre-docs-final.readthedocs.io/en/latest/Outputs.html>`__ for details.                
``metrics_verbosity``       int                        1                               Verbosity of metrics, from 0-9. See `Dwelling Metrics <https://ochre-docs-final.readthedocs.io/en/latest/Outputs.html#dwelling-metrics>`__ for details.
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
``Envelope``                dict                       empty dict                      Includes envelope specific arguments                                                                                                                              
``Equipment``               dict                       empty dict                      Includes equipment specific arguments                                                                                                                             
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

``Envelope`` arguments can be included to modify the default envelope model
that is based on the HPXML file. The table below lists optional arguments for
the ``Envelope`` dictionary.

+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| Argument Name                 | Argument Type          | Default Value                           | Description                                                                                               |
+===============================+========================+=========================================+===========================================================================================================+
| ``initial_temp_setpoint``     | number                 | Random temperature within HVAC deadband | Initial temperature for Indoor zone. It is set before the initialization time                             |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``linearize_infiltration``    | boolean                | FALSE                                   | Linearizes infiltration heat pathways and incorporates in state space matrices                            |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``external_radiation_method`` | string                 | "full"                                  | Option to use detailed radiation method ("full"), linearized radiation ("linear"), or no radiation (None) |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``internal_radiation_method`` | string                 | "full"                                  | Option to use detailed radiation method ("full"), linearized radiation ("linear"), or no radiation (None) |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``reduced_states``            | integer                | None                                    | Number of states for envelope model reduction                                                             |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``reduced_min_accuracy``      | number                 | None                                    | Minimum accuracy to determine number of states for envelope model reduction                               |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``save_matrices``             | boolean                | FALSE                                   | Saves envelope state space matrices to files                                                              |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``save_matrices_time_res``    | ``datetime.timedelta`` | None                                    | Time resolution for discretizing saved matrices. If None, saves continuous time matrices                  |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+
| ``zones``                     | dict of dicts          | Empty dict                              | Includes arguments for individual zones                                                                   |
+-------------------------------+------------------------+-----------------------------------------+-----------------------------------------------------------------------------------------------------------+

The ``zones`` dictionary keys can be from the list: ``['Indoor', 'Attic',
'Garage', 'Foundation']``. The table below lists optional arguments for
each zone dictionary.

+-----------------------------+---------------+----------------------------------+--------------------------------------------------------+
| Argument Name               | Argument Type | Default Value                    | Description                                            |
+=============================+===============+==================================+========================================================+
| ``enable_humidity``         | boolean       | True for Indoor zone, else False | If True, OCHRE models humidity in the given zone       |
+-----------------------------+---------------+----------------------------------+--------------------------------------------------------+
| ``Thermal Mass Multiplier`` | number        | 7                                | Multiplier for zone's thermal mass (i.e., capacitance) |
+-----------------------------+---------------+----------------------------------+--------------------------------------------------------+
| ``Volume (m^3)``            | number        | Taken from HPXML file            | Volume of the given zone                               |
+-----------------------------+---------------+----------------------------------+--------------------------------------------------------+

We note that it is possible, though not recommended, to create an ``Envelope``
object without initializing a ``Dwelling``. This can be done for very simple
Envelope models. As an example, see the ``run_hvac`` function in
`run_equipment.py
<https://github.com/NREL/OCHRE/blob/main/bin/run_equipment.py>`__.

Equipment-specific Arguments
----------------------------

An Equipment model can be initialized in a very similar way to a
Dwelling. For example, to initialize a battery:

.. code-block:: python

   from OCHRE import Battery
   equipment = Battery(**equipment_args)

where equipment_args is a Python dictionary of Equipment arguments.
A full set of the equipment classes available are listed in this
section, by end use.

The table below lists the required arguments for creating any standalone
Equipment model. Some equipment have additional required arguments as
described in the sections below.

+----------------+------------------------+----------------------------+
| Argument Name  | Argument Type          | Description                |
+================+========================+============================+
| ``start_time`` | ``datetime.datetime``  | Simulation start time      |
+----------------+------------------------+----------------------------+
| ``time_res``   | ``datetime.timedelta`` | Simulation time resolution |
+----------------+------------------------+----------------------------+
| ``duration``   | ``datetime.timedelta`` | Simulation duration        |
+----------------+------------------------+----------------------------+

The table below lists the optional arguments for creating any standalone
Equipment model. Some equipment have additional optional arguments as
described in the sections below.

+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument Name               | Argument Type          | Default Value                              | Description                                                                                                                                                  |
+=============================+========================+============================================+==============================================================================================================================================================+
| ``name``                    | string                 | OCHRE                                      | Name of the simulation                                                                                                                                       |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``initialization_time``     | ``datetime.timedelta`` | None (no initialization)                   | Runs a "warm up" simulation to improve initial temperature values                                                                                            |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``zone_name``               | string                 | None                                       | Name of Envelope zone if envelope model exists                                                                                                               |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``envelope_model``          | ``ochre.Envelope``     | None                                       | Envelope model for measuring temperature impacts (required for HVAC equipment)                                                                               |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``verbosity``               | int                    | 1                                          | Verbosity of the outputs, from 0-9. See Outputs and Analysis for details                                                                                     |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_path``             | string                 | HPXML or equipment schedule file directory | Path to saved output files                                                                                                                                   |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_to_parquet``       | boolean                | FALSE                                      | Save time series files as parquet files (False saves as csv files)                                                                                           |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``export_res``              | ``datetime.timedelta`` | None (no intermediate data export)         | Time resolution to save results to files                                                                                                                     |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_results``            | boolean                | True if verbosity > 0                      | Save results files, including time series files, metrics file, schedule output file, and status file                                                         |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_args_to_json``       | boolean                | FALSE                                      | Save all input arguments to json file, including user defined arguments. If False and verbosity >= 3, the json file will only include HPXML properties.      |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_status``             | boolean                | True if save_results is True               | Save status file to indicate whether the simulation is complete or failed                                                                                    |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_ebm_results``        | boolean                | FALSE                                      | Include equivalent battery model data in results                                                                                                             |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_schedule_columns``   | list                   | Empty list                                 | List of time series inputs to save to schedule output file                                                                                                   |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``equipment_schedule_file`` | string                 | None                                       | File with equipment time series data. Optional for most equipment                                                                                            |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``schedule_rename_columns`` | dict                   | None                                       | Dictionary of {file_column_name: ochre_schedule_name} to rename columns in equipment_schedule_file. Sometimes used for PV                                    |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``schedule_scale_factor``   | number                 | 1                                          | Scaling factor to normalize data in equipment_schedule_file. Sometimes used for PV to convert units                                                          |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``schedule``                | ``pandas.DataFrame``   | None                                       | Schedule with equipment or weather data that overrides the schedule_input_file and the equipment_schedule_file. Not required for Dwelling and some equipment |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``ext_time_res``            | ``datetime.timedelta`` | None                                       | Time resolution for external controller. Required if using Duty Cycle control                                                                                |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``seed``                    | int or string          | HPXML or equipment schedule file           | Random seed for setting initial temperatures and EV event data                                                                                               |
+-----------------------------+------------------------+--------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+

The following sections list the names and arguments for all OCHRE
equipment by end use. Many equipment types have all of their required
arguments included in the HPXML properties. These properties can be
overwritten by specifying the arguments in the ``equipment_args``
dictionary.

HVAC Heating and Cooling
~~~~~~~~~~~~~~~~~~~~~~~~

OCHRE includes models for the following HVAC equipment:

+--------------+-----------------------+--------------------+----------------------------------------------------------+
| End Use      | Equipment Class       | Equipment Name     | Description                                              |
+==============+=======================+====================+==========================================================+
| HVAC Heating | ``ElectricFurnace``   | Electric Furnace   |                                                          |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``ElectricBaseboard`` | Electric Baseboard |                                                          |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``ElectricBoiler``    | Electric Boiler    |                                                          |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``GasFurnace``        | Gas Furnace        |                                                          |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``GasBoiler``         | Gas Boiler         |                                                          |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``HeatPumpHeater``    | Heat Pump Heater   | Air Source Heat Pump  with no electric resistance backup |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``ASHPHeater``        | ASHP Heater        | Air Source Heat Pump, heating only                       |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Heating | ``MSHPHeater``        | MSHP Heater        | Minisplit Heat Pump, heating only                        |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Cooling | ``AirConditioner``    | Air Conditioner    | Central air conditioner                                  |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Cooling | ``RoomAC``            | Room AC            | Room air conditioner                                     |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Cooling | ``ASHPCooler``        | ASHP Cooler        | Air Source Heat Pump, cooling only                       |
+--------------+-----------------------+--------------------+----------------------------------------------------------+
| HVAC Cooling | ``MSHPCooler``        | MSHP Cooler        | Minisplit Heat Pump, cooling only                        |
+--------------+-----------------------+--------------------+----------------------------------------------------------+

The table below shows the required and optional equipment-specific
arguments for HVAC equipment.

+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| Argument Name                                  | Argument Type             | Required?                    | Default Value                                                      | Description                                                                                                        |
+================================================+===========================+==============================+====================================================================+====================================================================================================================+
| ``envelope_model``                             | ``ochre.Envelope``        | Yes                          |                                                                    | Envelope model for measuring temperature impacts                                                                   |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``use_ideal_capacity``                         | boolean                   | No                           | True if time_res >= 5 minutes or for variable-speed equipment      | If True, OCHRE sets HVAC capacity to meet the setpoint. If False, OCHRE uses thermostat deadband control           |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Capacity (W)``                               | number or list of numbers | Yes                          | Taken from HPXML                                                   | Rated capacity of equipment. If a list, it is the rated capacity by speed                                          |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Minimum Capacity (W)``                       | number                    | No                           | 0                                                                  | Minimum equipment capacity for ideal capacity equipment models                                                     |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Conditioned Space Fraction (-)``             | number                    | No                           | Taken from HPXML file, or 1                                        | Conditioned space fraction, e.g., for Room Air Conditioners                                                        |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``EIR (-)``                                    | number or list of numbers | Yes                          | Taken from HPXML file, or from Rated Efficiency                    | Energy input ratio (i.e., the inverse of the COP). If a list, it is the EIR by speed                               |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``SHR (-)``                                    | number or list of numbers | No                           | Taken from HPXML file, or from Rated Efficiency, or 1              | Sensible heat ratio. If a list, it is the SHR by speed. Only for HVAC Cooling equipment                            |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Rated Auxiliary Power (W)``                  | number                    | Yes                          | Taken from HPXML file                                              | Rated auxiliary power, including fan or pump power                                                                 |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``initial_schedule``                           | dict                      | Yes                          | Taken from first row of schedule                                   | Dictionary of initial values in schedule                                                                           |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Ducts``                                      | dict                      | No                           | Taken from HPXML file, or sets distribution system efficiency to 1 | Dictionary of inputs to determine HVAC distribution system efficiency                                              |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Basement Airflow Ratio (-)``                 | number                    | No                           | 0.2 for heaters if there is a conditioned basement, otherwise 0    | Ratio of airflow and HVAC capacity to send to conditioned basement. For heaters only                               |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Setpoint Temperature (C)``                   | number                    | No                           | Taken from HPXML file or schedule file                             | Constant setpoint temperature                                                                                      |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Weekday Setpoints (C)``                      | list of 24 numbers        | No                           | Taken from HPXML file or schedule file                             | Hourly weekday setpoint temperatures by hour                                                                       |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Weekend Setpoints (C)``                      | list of 24 numbers        | No                           | Taken from HPXML file or schedule file                             | Hourly weekend setpoint temperatures by hour. Defaults to weekday temperatures if they are included.               |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Deadband Temperature (C)``                   | number                    | No                           | Taken from HPXML file, or 1                                        | Size of temperature deadband in degC. Can also be specified in the schedule                                        |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``setpoint_ramp_rate``                         | number                    | No                           | 0.2 for ASHP Heater, otherwise None                                | Maximum ramp rate of thermostat setpoint, in degC/min                                                              |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``show_eir_shr``                               | boolean                   | No                           | FALSE                                                              | If True, show EIR and SHR in results for all time steps. If False, they will be set to 0 when the equipment is off |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Number of Speeds (-)``                       | int                       | No                           | Taken from HPXML file, or 1                                        | Number of speeds. Options are 1 (single speed), 2 (double speed), 4 (variable speed), or 10 (mini-split HP only)   |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Rated Efficiency``                           | string                    | Only if Number of Speeds > 1 | Taken from HPXML file, or None                                     | Rated SEER or HSPF. Used to determine the capacity, EIR, and SHR ratios of each speed                              |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Supplemental Heater Capacity (W)``           | number                    | Only for ASHP Heater         |                                                                    | ASHP Heater supplemental heater capacity                                                                           |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Supplemental Heater EIR (-)``                | number                    | No                           | 1                                                                  | ASHP Heater supplemental heater energy input ratio                                                                 |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``Supplemental Heater Cut-in Temperature (C)`` | number                    | No                           | None                                                               | Temperature to shut off heat pump for ASHP Heater                                                                  |
+------------------------------------------------+---------------------------+------------------------------+--------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------+


Water Heating
~~~~~~~~~~~~~

OCHRE includes models for the following Water Heating equipment:

+---------------+-----------------------------------+----------------------------+
| End Use       | Equipment Class                   | Equipment Name             |
+===============+===================================+============================+
| Water Heating | ``ElectricResistanceWaterHeater`` | Electric Tank Water Heater |
+---------------+-----------------------------------+----------------------------+
| Water Heating | ``GasWaterHeater``                | Gas Tank Water Heater      |
+---------------+-----------------------------------+----------------------------+
| Water Heating | ``HeatPumpWaterHeater``           | Heat Pump Water Heater     |
+---------------+-----------------------------------+----------------------------+
| Water Heating | ``TanklessWaterHeater``           | Tankless Water Heater      |
+---------------+-----------------------------------+----------------------------+
| Water Heating | ``GasTanklessWaterHeater``        | Gas Tankless Water Heater  |
+---------------+-----------------------------------+----------------------------+


The table below shows the required and optional equipment-specific
arguments for Water Heating equipment.

+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| Argument Name                                       | Argument Type | Required?                          | Default Value                                                         | Description                                                                                                      |
+=====================================================+===============+====================================+=======================================================================+==================================================================================================================+
| ``use_ideal_capacity``                              | boolean       | No                                 | True if time_res >= 5 minutes                                         | If True, OCHRE sets water heater capacity to meet the setpoint. If False, OCHRE uses thermostat deadband control |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``water_nodes``                                     | int           | No                                 | 12 if Heat Pump Water Heater, 1 if Tankless Water Heater, otherwise 2 | Number of nodes in water tank model                                                                              |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Capacity (W)``                                    | number        | No                                 | 4500                                                                  | Water heater capacity                                                                                            |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Efficiency (-)``                                  | number        | No                                 | 1                                                                     | Water heater efficiency (or supplemental heater efficiency for HPWH)                                             |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Setpoint Temperature (C)``                        | number        | Yes                                | Taken from HPXML file, or 51.67                                       | Water heater setpoint temperature. Can also be set in schedule                                                   |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Deadband Temperature (C)``                        | number        | No                                 | 8.17 for Heat Pump Water Heater, otherwise 5.56                       | Water heater deadband size. Can also be set in schedule                                                          |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Max Tank Temperature (C)``                        | number        | No                                 | 60                                                                    | Maximum water tank temperature                                                                                   |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Mixed Delivery Temperature (C)``                  | number        | No                                 | 40.56                                                                 | Hot water temperature for tempered water draws (sinks, showers, and baths)                                       |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Initial Temperature (C)``                         | number        | No                                 | Setpoint temperature - 10% of deadband temperature                    | Initial temperature of the entire tank (before initialization routine)                                           |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Max Setpoint Ramp Rate (C/min)``                  | number        | No                                 | None                                                                  | Maximum rate of change for setpoint temperature                                                                  |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Tank Volume (L)``                                 | number        | Yes                                | Taken from HPXML file                                                 | Size of water tank, in L                                                                                         |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Tank Height (m)``                                 | number        | Yes                                | Taken from HPXML file                                                 | Height of water tank, used to determine surface area                                                             |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Heat Transfer Coefficient (W/m^2/K) or UA (W/K)`` | number        | Yes                                | Taken from HPXML file                                                 | Heat transfer coefficient of water tank                                                                          |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``hp_only_mode``                                    | boolean       | No                                 | FALSE                                                                 | Disable supplemental heater for HPWH                                                                             |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH COP (-)``                                    | number        | Only for Heat Pump Water Heater    |                                                                       | Coefficient of Performance for HPWH                                                                              |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH Capacity (W) or HPWH Power (W)``             | number        | No                                 | 500 (for HPWH Power)                                                  | Capacity or rated power for HPWH                                                                                 |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH Parasitics (W)``                             | number        | No                                 | 1                                                                     | Parasitic power for HPWH                                                                                         |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH Fan Power (W)``                              | number        | No                                 | 35                                                                    | Fan power for HPWH                                                                                               |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH SHR (-)``                                    | number        | No                                 | 0.88                                                                  | Sensible heat ratio for HPWH                                                                                     |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH Interaction Factor (-)``                     | number        | No                                 | 0.75 if in Indoor Zone else 1                                         | Fraction of HPWH sensible gains to envelope                                                                      |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``HPWH Wall Interaction Factor (-)``                | number        | No                                 | 0.5                                                                   | Fraction of HPWH sensible gains to wall boundary, remainder goes to zone                                         |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Energy Factor (-)``                               | number        | Only for Gas Water Heater          | Taken from HPXML file                                                 | Water heater energy factor (EF) for getting skin loss fraction                                                   |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+
| ``Parasitic Power (W)``                             | number        | Only for Gas Tankless Water Heater | Taken from HPXML file                                                 | Parasitic power for Gas Tankless Water Heater                                                                    |
+-----------------------------------------------------+---------------+------------------------------------+-----------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------+

Electric Vehicle
~~~~~~~~~~~~~~~~

OCHRE includes an electric vehicle (EV) model. The equipment name can be
“EV” or “Electric Vehicle”. The table below shows the required and
optional equipment-specific arguments for EVs.

+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+
| Argument Name            | Argument Type | Required? | Default Value                                                       | Description                                           |
+==========================+===============+===========+=====================================================================+=======================================================+
| ``vehicle_type``         | string        | Yes       | BEV, if taken from HPXML file                                       | EV vehicle type, options are "PHEV" or "BEV"          |
+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+
| ``charging_level``       | string        | Yes       | Level 2, if taken from HPXML file                                   | EV charging type, options are "Level 1" or "Level 2"  |
+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+
| ``capacity or mileage``  | number        | Yes       | 100 miles if HPXML Annual EV Energy < 1500 kWh, otherwise 250 miles | EV battery capacity in kWh or mileage in miles        |
+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+
| ``enable_part_load``     | boolean       | No        | True if charging_level = Level 2                                    | Allows EV to charge at partial load                   |
+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+
| ``ambient_ev_temp``      | number        | No        | Taken from schedule, or 20 C                                        | Ambient temperature used to estimate EV usage per day |
+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+
| ``equipment_event_file`` | string        | No        | Default file based on ``vehicle_type`` and mileage                  | File for EV event scenarios                           |
+--------------------------+---------------+-----------+---------------------------------------------------------------------+-------------------------------------------------------+

Battery
~~~~~~~

OCHRE includes a battery model. The table below shows the required and
optional equipment-specific arguments for batteries.

+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| Argument Name                       | Argument Type | Required? | Default Value                                   | Description                                                                                            |
+=====================================+===============+===========+=================================================+========================================================================================================+
| ``capacity_kwh``                    | number        | No        | 10                                              | Nominal energy capacity of battery, in kWh                                                             |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``capacity``                        | number        | No        | 5                                               | Max power of battery, in kW                                                                            |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``efficiency``                      | number        | No        | 0.98                                            | Battery Discharging Efficiency, unitless                                                               |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``efficiency_charge``               | number        | No        | 0.98                                            | Battery Charging Efficiency, unitless                                                                  |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``efficiency_inverter``             | number        | No        | 0.97                                            | Inverter Efficiency, unitless                                                                          |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``efficiency_type``                 | string        | No        | "advanced"                                      | Efficiency calculation option. Options are "advanced", "constant", "curve", and "quadratic"            |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``soc_init``                        | number        | No        | 0.5                                             | Initial State of Charge, unitless                                                                      |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``soc_max``                         | number        | No        | 0.95                                            | Maximum SOC, unitless                                                                                  |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``soc_min``                         | number        | No        | 0.15                                            | Minimum SOC, unitless                                                                                  |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``enable_degradation``              | boolean       | No        | TRUE                                            | If True, runs an energy capacity degradation model daily                                               |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``initial_voltage``                 | number        | No        | 50.4                                            | Initial open circuit voltage, in V. Used for advanced efficiency and degradation models.               |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``v_cell``                          | number        | No        | 3.6                                             | Cell voltage, in V. Used for advanced efficiency and degradation models.                               |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``ah_cell``                         | number        | No        | 70                                              | Cell capacity, in Ah. Used for advanced efficiency and degradation models.                             |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``r_cell``                          | number        | No        | 0                                               | Cell resistance, in ohm. Used for advanced efficiency and degradation models.                          |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``charge_start_hour``               | number        | No        | 9                                               | Schedule: Charge Start Time, in hour of day                                                            |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``discharge_start_hour``            | number        | No        | 16                                              | Schedule: Discharge Start Time, in hour of day                                                         |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``charge_power``                    | number        | No        | 1                                               | Schedule: Charge Power, in kW                                                                          |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``discharge_power``                 | number        | No        | 1                                               | Schedule: Discharge Power, in kW                                                                       |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``charge_from_solar``               | number        | No        | 0                                               | Self-Consumption: Force Charge from Solar, in boolean                                                  |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``import_limit``                    | number        | No        | 0                                               | Self-Consumption: Grid Import Limit, in kW                                                             |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``export_limit``                    | number        | No        | 0                                               | Self-Consumption: Grid Export Limit, in kW                                                             |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``enable_thermal_model``            | boolean       | No        | True only if zone_name or envelope is specified | If True, creates 1R-1C thermal model for battery temperature. Temperature is used in degradation model |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``thermal_r``                       | number        | No        | 0.5                                             | Thermal Resistance, in K/W                                                                             |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``thermal_c``                       | number        | No        | 90000                                           | Thermal Mass, in J/K                                                                                   |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+
| ``Initial Battery Temperature (C)`` | number        | No        | zone temperature                                |                                                                                                        |
+-------------------------------------+---------------+-----------+-------------------------------------------------+--------------------------------------------------------------------------------------------------------+

Solar PV
~~~~~~~~

OCHRE includes a solar PV model. The table below shows the required and
optional equipment-specific arguments for PV.

+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| Argument Name           | Argument Type | Required?             | Default Value                            | Description                                                                        |
+=========================+===============+=======================+==========================================+====================================================================================+
| ``capacity``            | number        | Only when running SAM |                                          | PV panel capacity, in kW                                                           |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| ``tilt``                | number        | No                    | Taken from HPXML roof pitch              | Tilt angle from horizontal, in degrees. Used for SAM                               |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| ``azimuth``             | number        | No                    | Taken from HPXML, south-most facing roof | Azimuth angle from south, in degrees. Used for SAM                                 |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| ``inverter_capacity``   | number        | No                    | PV.capacity                              | Inverter apparent power capacity, in kVA. Used for SAM                             |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| ``inverter_efficiency`` | number        | No                    | Use default from SAM                     | Efficiency of the inverter, unitless. Used for SAM                                 |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| ``inverter_priority``   | string        | No                    | "Var"                                    | PV inverter priority. Options are "Var", "Watt", or "CPF" (constant power factor)  |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+
| ``inverter_min_pf``     | number        | No                    | 0.8                                      | Inverter minimum power factor, unitless                                            |
+-------------------------+---------------+-----------------------+------------------------------------------+------------------------------------------------------------------------------------+

Gas Generator
~~~~~~~~~~~~~

OCHRE includes models for the following gas generator equipment:

+---------------+------------------+----------------+
| End Use       | Equipment Class  | Equipment Name |
+===============+==================+================+
| Gas Generator | ``GasGenerator`` | Gas Generator  |
+---------------+------------------+----------------+
| Gas Generator | ``GasFuelCell``  | Gas Fuel Cell  |
+---------------+------------------+----------------+

The table below shows the required and optional equipment-specific
arguments for gas generators.

+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| Argument Name            | Argument Type | Required? | Default Value                                | Description                                                                     |
+==========================+===============+===========+==============================================+=================================================================================+
| ``capacity``             | number        | No        | 6                                            | Maximum power, in kW                                                            |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``efficiency``           | number        | No        | 0.95                                         | Discharging Efficiency, unitless                                                |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``efficiency_type``      | string        | No        | "curve" if GasFuelCell, otherwise "constant" | Efficiency calculation option. Options are "constant", "curve", and "quadratic" |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``control_type``         | string        | No        | "Off"                                        | Control option. Options are "Off", "Schedule", and "Self-Consumption"           |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``ramp_rate``            | number        | No        | 0.1                                          | Max ramp rate, in kW/min                                                        |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``charge_start_hour``    | number        | No        | 9                                            | Schedule: Charge Start Hour                                                     |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``discharge_start_hour`` | number        | No        | 16                                           | \Schedule: Discharge Start Hour                                                 |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``charge_power``         | number        | No        | 1                                            | Schedule: Charge Power, in kW                                                   |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``discharge_power``      | number        | No        | 1                                            | Schedule: Discharge Power, in kW                                                |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``import_limit``         | number        | No        | 0                                            | Self-Consumption: Grid Import Limit, in kW                                      |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+
| ``export_limit``         | number        | No        | 0                                            | Self-Consumption: Grid Export Limit, in kW                                      |
+--------------------------+---------------+-----------+----------------------------------------------+---------------------------------------------------------------------------------+

Other Equipment
~~~~~~~~~~~~~~~

OCHRE includes basic models for other loads, including appliances,
lighting, and miscellaneous electric and gas loads:

+----------+-------------------+-------------------+
| End Use  | Equipment Class   | Equipment Name    |
+==========+===================+===================+
| Lighting | ``LightingLoad``  | Lighting          |
+----------+-------------------+-------------------+
| Lighting | ``LightingLoad``  | Exterior Lighting |
+----------+-------------------+-------------------+
| Lighting | ``LightingLoad``  | Basement Lighting |
+----------+-------------------+-------------------+
| Lighting | ``LightingLoad``  | Garage Lighting   |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Clothes Washer    |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Clothes Dryer     |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Dishwasher        |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Refrigerator      |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Cooking Range     |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | MELs              |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | TV                |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Well Pump         |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Gas Grill         |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Gas Fireplace     |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Gas Lighting      |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Pool Pump         |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Pool Heater       |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Hot Tub Pump      |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Hot Tub Heater    |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Ceiling Fan       |
+----------+-------------------+-------------------+
| Other    | ``ScheduledLoad`` | Ventilation Fan   |
+----------+-------------------+-------------------+
| EV       | ``ScheduledEV``   | Scheduled EV      |
+----------+-------------------+-------------------+

The table below shows the required and optional equipment-specific
arguments for other equipment.

+----------------------------------+---------------+-----------+-----------------------------+-------------------------------------------------------------------------------+
| Argument Name                    | Argument Type | Required? | Default Value               | Description                                                                   |
+==================================+===============+===========+=============================+===============================================================================+
| ``Convective Gain Fraction (-)`` | number        | No        | Taken from HPXML file, or 0 | Fraction of power consumption that is dissipated through convection into zone |
+----------------------------------+---------------+-----------+-----------------------------+-------------------------------------------------------------------------------+
| ``Radiative Gain Fraction (-)``  | number        | No        | Taken from HPXML file, or 0 | Fraction of power consumption that is dissipated through radiation into zone  |
+----------------------------------+---------------+-----------+-----------------------------+-------------------------------------------------------------------------------+
| ``Latent Gain Fraction (-)``     | number        | No        | Taken from HPXML file, or 0 | Fraction of power consumption that is dissipated as latent heat into zone     |
+----------------------------------+---------------+-----------+-----------------------------+-------------------------------------------------------------------------------+
