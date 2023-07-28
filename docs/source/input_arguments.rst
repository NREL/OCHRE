Dwelling Arguments
------------------

A Dwelling model can be initialized using:

\``\`

from OCHRE import Dwelling

house = Dwelling(\**dwelling_args)

\``\`

where \`dwelling_args\` is a Python dictionary of Dwelling arguments.

The table below lists the required arguments for creating a Dwelling
model.

+--------------+---------+---------------------------------------------+
| **Argument   | **A     | **Description**                             |
| Name**       | rgument |                                             |
|              | Type**  |                                             |
+==============+=========+=============================================+
| start_time   | dat     | Simulation start time                       |
|              | etime.d |                                             |
|              | atetime |                                             |
+--------------+---------+---------------------------------------------+
| time_res     | date    | Simulation time resolution                  |
|              | time.ti |                                             |
|              | medelta |                                             |
+--------------+---------+---------------------------------------------+
| duration     | date    | Simulation duration                         |
|              | time.ti |                                             |
|              | medelta |                                             |
+--------------+---------+---------------------------------------------+
| hpxml_file   | string  | Path to HPXML file                          |
+--------------+---------+---------------------------------------------+
| weather_file | string  | weather_file: Path to weather file          |
| or           |         |                                             |
| weather_path |         |                                             |
+--------------+---------+---------------------------------------------+
|              |         | weather_path: Path to directory of weather  |
|              |         | files. The file name can be read from       |
|              |         | “Weather Station” in the HPXML file.        |
+--------------+---------+---------------------------------------------+

The table below lists the optional arguments for creating a Dwelling
model.

+------+-----+---------+----------------------------------------------+
| **   | *   | **      | **Description**                              |
| Argu | *Ar | Default |                                              |
| ment | gum | Value** |                                              |
| Na   | ent |         |                                              |
| me** | Typ |         |                                              |
|      | e** |         |                                              |
+======+=====+=========+==============================================+
| name | str | OCHRE   | Name of the simulation                       |
|      | ing |         |                                              |
+------+-----+---------+----------------------------------------------+
| sch  | str | None    | Path to schedule input file                  |
| edul | ing |         |                                              |
| e_in |     |         |                                              |
| put_ |     |         |                                              |
| file |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| ini  | dat | None    | Runs a "warm up" simulation to improve       |
| tial | eti | (no     | initial temperature values                   |
| izat | me. | i       |                                              |
| ion_ | tim | nitiali |                                              |
| time | ede | zation) |                                              |
|      | lta |         |                                              |
+------+-----+---------+----------------------------------------------+
| t    | str | None    | Use "DST" for local U.S. time zone with      |
| ime_ | ing | (no     | daylight savings, "noDST" for local U.S.     |
| zone |     | time    | time zone without daylight savings, or any   |
|      |     | zone    | time zone in pytz.all_timezones              |
|      |     | m       |                                              |
|      |     | odeled) |                                              |
+------+-----+---------+----------------------------------------------+
| v    | int | 1       | Verbosity of the outputs, from 0-9. See      |
| erbo |     |         | Outputs and Analysis for details             |
| sity |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| m    | int | 6       | Verbosity of the output metrics, from 0-9.   |
| etri |     |         | See Dwelling and Equipment Metrics for       |
| cs_v |     |         | details                                      |
| erbo |     |         |                                              |
| sity |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| out  | str | HPXML   | Path to saved output files                   |
| put_ | ing | or      |                                              |
| path |     | eq      |                                              |
|      |     | uipment |                                              |
|      |     | s       |                                              |
|      |     | chedule |                                              |
|      |     | file    |                                              |
|      |     | di      |                                              |
|      |     | rectory |                                              |
+------+-----+---------+----------------------------------------------+
| o    | b   | FALSE   | Save time series files as parquet files      |
| utpu | ool |         | (False saves as csv files)                   |
| t_to | ean |         |                                              |
| _par |     |         |                                              |
| quet |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| ex   | dat | None    | Time resolution to save results to files     |
| port | eti | (no     |                                              |
| _res | me. | inter   |                                              |
|      | tim | mediate |                                              |
|      | ede | data    |                                              |
|      | lta | export) |                                              |
+------+-----+---------+----------------------------------------------+
| save | b   | True if | Save results files, including time series    |
| _res | ool | ve      | files, metrics file, schedule output file,   |
| ults | ean | rbosity | and status file                              |
|      |     | > 0     |                                              |
+------+-----+---------+----------------------------------------------+
| s    | b   | FALSE   | Save all input arguments to json file,       |
| ave_ | ool |         | including user defined arguments. If False   |
| args | ean |         | and verbosity >= 3, the json file will only  |
| _to_ |     |         | include HPXML properties.                    |
| json |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| sav  | b   | True if | Save status file to indicate whether the     |
| e_st | ool | save_   | simulation is complete or failed             |
| atus | ean | results |                                              |
|      |     | is True |                                              |
+------+-----+---------+----------------------------------------------+
| s    | l   | Empty   | List of time series inputs to save to        |
| ave_ | ist | list    | schedule output file                         |
| sche |     |         |                                              |
| dule |     |         |                                              |
| _col |     |         |                                              |
| umns |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| sche | p   | None    | Schedule with equipment or weather data that |
| dule | and |         | overrides the schedule_input_file and the    |
|      | as. |         | equipment_schedule_file. Not required for    |
|      | Dat |         | Dwelling and some equipment                  |
|      | aFr |         |                                              |
|      | ame |         |                                              |
+------+-----+---------+----------------------------------------------+
| ext_ | dat | None    | Time resolution for external controller.     |
| time | eti |         | Required if using Duty Cycle control         |
| _res | me. |         |                                              |
|      | tim |         |                                              |
|      | ede |         |                                              |
|      | lta |         |                                              |
+------+-----+---------+----------------------------------------------+
| seed | int | HPXML   | Random seed for setting initial temperatures |
|      | or  | or      | and EV event data                            |
|      | str | eq      |                                              |
|      | ing | uipment |                                              |
|      |     | s       |                                              |
|      |     | chedule |                                              |
|      |     | file    |                                              |
+------+-----+---------+----------------------------------------------+
| m    | d   | Empty   | Dictionary that directly modifies values     |
| odif | ict | dict    | from HPXML file                              |
| y_hp |     |         |                                              |
| xml_ |     |         |                                              |
| dict |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| E    | d   | Empty   | Includes Equipment-specific arguments        |
| quip | ict | dict    |                                              |
| ment |     |         |                                              |
+------+-----+---------+----------------------------------------------+
| Enve | d   | Empty   | Includes arguments for the building Envelope |
| lope | ict | dict    |                                              |
+------+-----+---------+----------------------------------------------+

Equipment-specific Arguments
----------------------------

An Equipment model can be initialized in a very similar way to a
Dwelling. For example, to initialize a battery:

\``\`

from OCHRE import Battery

equipment = Battery(name, \**equipment_args)

\``\`

where \` equipment_args\` is a Python dictionary of Equipment arguments.
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

+------------+---------------------+----------------+----------------+
| End Use    | Equipment Class     | Equipment Name | Description    |
+============+=====================+================+================+
| HVAC       | ElectricFurnace     | Electric       |                |
| Heating    |                     | Furnace        |                |
+------------+---------------------+----------------+----------------+
| HVAC       | ElectricBaseboard   | Electric       |                |
| Heating    |                     | Baseboard      |                |
+------------+---------------------+----------------+----------------+
| HVAC       | ElectricBoiler      | Electric       |                |
| Heating    |                     | Boiler         |                |
+------------+---------------------+----------------+----------------+
| HVAC       | GasFurnace          | Gas Furnace    |                |
| Heating    |                     |                |                |
+------------+---------------------+----------------+----------------+
| HVAC       | GasBoiler           | Gas Boiler     |                |
| Heating    |                     |                |                |
+------------+---------------------+----------------+----------------+
| HVAC       | HeatPumpHeater      | Heat Pump      | Air Source     |
| Heating    |                     | Heater         | Heat Pump with |
|            |                     |                | no electric    |
|            |                     |                | resistance     |
|            |                     |                | backup         |
+------------+---------------------+----------------+----------------+
| HVAC       | ASHPHeater          | ASHP Heater    | Air Source     |
| Heating    |                     |                | Heat Pump,     |
|            |                     |                | heating only   |
+------------+---------------------+----------------+----------------+
| HVAC       | MSHPHeater          | MSHP Heater    | Minisplit Heat |
| Heating    |                     |                | Pump, heating  |
|            |                     |                | only           |
+------------+---------------------+----------------+----------------+
| HVAC       | AirConditioner      | Air            | Central air    |
| Cooling    |                     | Conditioner    | conditioner    |
+------------+---------------------+----------------+----------------+
| HVAC       | RoomAC              | Room AC        | Room air       |
| Cooling    |                     |                | conditioner    |
+------------+---------------------+----------------+----------------+
| HVAC       | ASHPCooler          | ASHP Cooler    | Air Source     |
| Cooling    |                     |                | Heat Pump,     |
|            |                     |                | cooling only   |
+------------+---------------------+----------------+----------------+
| HVAC       | MSHPCooler          | MSHP Cooler    | Minisplit Heat |
| Cooling    |                     |                | Pump, cooling  |
|            |                     |                | only           |
+------------+---------------------+----------------+----------------+

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

.. _water-heating-1:

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

.. _solar-pv-1:

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

Outputs and Analysis

At the end of any OCHRE simulation, time series outputs are saved. These
time series outputs are used to calculate metrics that describe the
simulation results. The set of time series outputs depends on the
\`verbosity\` of the simulation, and the set of metrics depends on the
\`metrics_verbosity\`. The tables below describe the Dwelling and
Equipment-specific outputs and metrics that are reported.

Dwelling Time Series Outputs
----------------------------

+-----------------------+--------------------+------------------------+
| Time Series Output    | Available if       | Description            |
| Name                  | \`verbosity>=\_\`  |                        |
+=======================+====================+========================+
| Total Electric Power  | 1                  | Total dwelling real    |
| (kW)                  |                    | electric power, in kW  |
+-----------------------+--------------------+------------------------+
| …                     |                    |                        |
+-----------------------+--------------------+------------------------+

Dwelling Metrics
----------------

+-----------------------+--------------------+------------------------+
| Time Series Output    | Available if       | Description            |
| Name                  | \`metri            |                        |
|                       | cs_verbosity>=\_\` |                        |
+=======================+====================+========================+
| Total Electric Energy | 1                  | Total dwelling real    |
| (kWh)                 |                    | electric energy        |
|                       |                    | consumption, in kWh    |
+-----------------------+--------------------+------------------------+
| …                     |                    |                        |
+-----------------------+--------------------+------------------------+

Equipment-Specific Time Series Outputs
--------------------------------------

+--------------+-----------------+----------------+------------------+
| End Use      | Time Series     | Available if   | Description      |
|              | Output Name     | \`v            |                  |
|              |                 | erbosity>=\_\` |                  |
+==============+=================+================+==================+
| HVAC Heating | HVAC Heating    | 2?             | Real electric    |
|              | Electric Power  |                | power from HVAC  |
|              | (kW)            |                | heating          |
|              |                 |                | equipment, in kW |
+--------------+-----------------+----------------+------------------+
|              | …               |                |                  |
+--------------+-----------------+----------------+------------------+

Equipment-Specific Metrics
--------------------------

+-------------+-----------------+-------------------+------------------+
| End Use     | Time Series     | Available if      | Description      |
|             | Output Name     | \`metric          |                  |
|             |                 | s_verbosity>=\_\` |                  |
+=============+=================+===================+==================+
| HVAC        | HVAC Heating    | 2?                | Total electric   |
| Heating     | Electric Energy |                   | energy           |
|             | (kWh)           |                   | consumption from |
|             |                 |                   | HVAC heating     |
|             |                 |                   | equipment, in    |
|             |                 |                   | kWh              |
+-------------+-----------------+-------------------+------------------+
|             | …               |                   |                  |
+-------------+-----------------+-------------------+------------------+

Data Analysis
-------------

The \`Analysis\` module has useful data analysis functions for OCHRE
output data:

\``\`

from ochre import Analysis

# load existing ochre simulation data

df, metrics, df_hourly = Analysis.load_ochre(folder)

# calculate metrics from a pandas DataFrame

metrics = Analysis.calculate_metrics(df)

\``\`

Some analysis functions are useful for analyzing or combining results
from multiple OCHRE simulations:

\``\`

# Combine OCHRE metrics files from multiple simulations (in subfolders
of path)

df_metrics = Analysis.combine_metrics_files(path=path)

# Combine 1 output column from multiple OCHRE simulations into a single
DataFrame

results_files = Analysis.find_files_from_ending(path, ‘ochre.csv’)

df_powers = Analysis.combine_time_series_column(results_files, 'Total
Electric Power (kW)')

\``\`

Data Visualization
------------------

The \`CreateFigures\` module has useful visualization functions for
OCHRE output data:

\``\`

from ochre import Analysis, CreateFigures

df, metrics, df_hourly = Analysis.load_ochre(folder)

# Create standard HVAC output plots

CreateFigures.plot_hvac(df)

# Create stacked plot of power by end use

CreateFigures.plot_power_stack(df)

\``\`

Many functions work on any generic pandas DataFrame with a
DateTimeIndex.

Controller Integration
======================

External Control Signals
------------------------

While OCHRE can simulate a stand-alone dwelling or piece of equipment,
it is designed to integrate with external controllers and other modeling
tools. External controllers can adjust the power consumption of any
OCHRE equipment using multiple control methods.

Below is a simple example that will create a battery model and discharge
it at 5 kW.

\``\`

battery = Battery(capacity_kwh=10, # energy capacity = 10 kWh

capacity=5, # power capacity = 5 kW

soc_init=1, # Initial SOC=100%

start_time=dt.datetime(2018, 1, 1, 0, 0),

time_res=dt.timedelta(minutes=15),

duration=dt.timedelta(days=1),

)

control_signal = {'P Setpoint': -5} # Discharge at 5 kW

status = battery.update(control_signal) # Run for 1 time step with
control signal

\``\`

The following table lists the control signals available to OCHRE
equipment, by end use.

+----------------+----------------+-----------------+-----------------+
| End Uses       | Control Signal | Control Signal  | Description     |
|                | Name           | Type and Units  |                 |
+================+================+=================+=================+
| HVAC Heating,  | Setpoint       | Number, in      | Setpoint        |
| HVAC Cooling,  |                | degrees C       | temperature for |
| or Water       |                |                 | thermostat      |
| Heating        |                |                 | control         |
+----------------+----------------+-----------------+-----------------+
| …              |                |                 |                 |
+----------------+----------------+-----------------+-----------------+

External Model Signals
----------------------

OCHRE can also integrate with external models that modify default
schedule values and other settings.

The most common use case is to integrate with a grid simulator that
modifies the dwelling voltage. OCHRE includes a ZIP model for all
equipment that modifies the real and reactive electric power based on
the grid voltage.

The following code sends a voltage of 0.97 p.u. to a Dwelling model:

\``\`

status = dwelling.update(ext_model_args={‘Voltage (-)’: 0.97})

\``\`

External model signals can also modify any time series schedule values
including weather and occupancy variables. The names and units of these
variables can be found in the header of the schedule output file.
Alternatively, these variables can be reset at the beginning of the
simulation; see notebooks/… for more details.

Status Variables
----------------

The \`update\` function for equipment and dwellings returns a Python
dictionary with status variables that can be sent to the external
controller. These status variables are equivalent to the Time Series
Outputs described in Outputs and Analysis. Note that the \`verbosity\`
applies to the status variables in the same way as the outputs.

Example Use Case – Dwelling
---------------------------

The following code creates a Dwelling model and runs a simulation that
controls the HVAC heating setpoint. For more details and examples, see
bin/run_external_control.py and notebooks/…

Example Use Case – Equipment
----------------------------

The following code creates a water heater model and runs a simulation
that controls the water heater setpoint. For more details and examples,
see bin/run_external_control.py and notebooks/…

.. _co-simulation-1:

Co-simulation
-------------

Multiple OCHRE instances have been run in co-simulation using the HELICS
platform. OCHRE models can communicate with other agents via its
external control signals, external model signals, and status variables.

See the publications list for examples of co-simulation architectures
that use OCHRE. We do not currently have public code for using OCHRE in
co-simulation.

API Reference (automatically generated, low priority)
=====================================================

.. |A picture containing diagram Description automatically generated| image:: media/image1.png
   :width: 4.57271in
   :height: 1.80509in
.. |Diagram, schematic Description automatically generated| image:: media/image2.png
   :width: 2.87476in
   :height: 4.27811in
.. |image1| image:: media/image3.emf
   :width: 4.59037in
   :height: 3.80417in
