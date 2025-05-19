.. _outputs:

Outputs and Analysis
====================

OCHRE saves many time series results throughout the simulation. These time
series results are used to calculate metrics that summarize the simulation
results. The set of time series results depends on the ``verbosity`` of the
simulation, and the set of metrics depends on the ``metrics_verbosity``. OCHRE
also includes modules with useful code for analysis and visualization. 

.. _dwelling-results:

Dwelling Time Series Results
----------------------------

The table below shows the Dwelling-level time series results, including units
and the minimum ``verbosity`` required to output the data.

+-------------------------------+-------------+-----------+-------------------------------------------------------+
| OCHRE Name                    | OCHRE Units | Verbosity | Description                                           |
+===============================+=============+===========+=======================================================+
| Total Electric Power (kW)     | kW          | 1         | Total dwelling real electric power                    |
+-------------------------------+-------------+-----------+-------------------------------------------------------+
| Total Electric Energy (kWh)   | kWh         | 6         | Total dwelling real electric energy for 1 time step   |
+-------------------------------+-------------+-----------+-------------------------------------------------------+
| Total Gas Power (therms/hour) | therms/hour | 1         | Total dwelling gas power                              |
+-------------------------------+-------------+-----------+-------------------------------------------------------+
| Total Gas Energy (therms)     | therms      | 6         | Total dwelling gas energy consumption for 1 time step |
+-------------------------------+-------------+-----------+-------------------------------------------------------+
| Total Reactive Power (kVAR)   | kVAR        | 1         | Total dwelling reactive power                         |
+-------------------------------+-------------+-----------+-------------------------------------------------------+
| Total Reactive Energy (kVARh) | kVARh       | 6         | Total dwelling reactive energy for 1 time step        |
+-------------------------------+-------------+-----------+-------------------------------------------------------+
| Grid Voltage (-)              | p.u.        | 8         | Per-unit grid voltage                                 |
+-------------------------------+-------------+-----------+-------------------------------------------------------+


Equipment Time Series Results
-----------------------------

The tables below show equipment-level time series results, including units and
the minimum ``verbosity`` required to output the data. Note that some rows
represent multiple results; for example, ``<end use> Electric Power (kW)`` is
output for each end use.


All Equipment
~~~~~~~~~~~~~

The table below shows generic results for all equipment types and end uses.

+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| OCHRE Name                          | OCHRE Units | Verbosity | Description                                                              |
+=====================================+=============+===========+==========================================================================+
| <end use> Electric Power (kW)       | kW          | 2         | Real electric power of all equipment within the end use                  |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| <end use> Gas Power (therms/hour)   | therms/hour | 2         | Gas power of all equipment within the end use                            |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| <end use> Reactive Power (kVAR)     | kVAR        | 8         | Reactive electric power of all equipment within the end use              |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| <equipment> Mode                    | N/A         | 7         | Current mode of equipment operation                                      |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| <equipment> Electric Power (kW)     | kW          | 6         | Real electric power of the equipment (Lighting and Other equipment only) |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| <equipment> Gas Power (therms/hour) | therms/hour | 6         | Gas power of the equipment                                               |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+
| <equipment> Reactive Power (kVAR)   | kVAR        | 8         | Reactive electric power of the equipment                                 |
+-------------------------------------+-------------+-----------+--------------------------------------------------------------------------+


HVAC Heating and Cooling
~~~~~~~~~~~~~~~~~~~~~~~~

The following results are specific to HVAC equipment.

+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| OCHRE Name                                 | OCHRE Units | Verbosity | Description                                                               |
+============================================+=============+===========+===========================================================================+
| HVAC <Heating or Cooling> Delivered (W)    | W           | 4         | HVAC sensible heat gain delivered to indoor zone                          |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Setpoint (C)     | degC        | 4         | HVAC temperature setpoint                                                 |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> COP (-)          | unitless    | 4         | HVAC coefficient of performance of main unit                              |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Duct Losses (W)  | W           | 5         | HVAC heat loss due to ducts                                               |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Main Power (kW)  | kW          | 7         | HVAC electric or gas power excluding fan, peripherals, and backup element |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Fan Power (kW)   | kW          | 7         | HVAC fan and peripherals power                                            |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Latent Gains (W) | W           | 7         | HVAC latent heat gain delivered to indoor zone                            |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> SHR (-)          | unitless    | 7         | HVAC sensible heat ratio                                                  |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Capacity (W)     | W           | 7         | HVAC heat capacity of main unit                                           |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> Max Capacity (W) | W           | 7         | HVAC maximum heat capacity of main unit                                   |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+
| HVAC <Heating or Cooling> ER Power (kW)    | kW          | 7         | HVAC backup element power (ASHPHeater only)                               |
+--------------------------------------------+-------------+-----------+---------------------------------------------------------------------------+

The following results are specific to the Envelope model.

+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| OCHRE Name                                        | OCHRE Units | Verbosity                      | Description                                                                                                                               |
+===================================================+=============+================================+===========================================================================================================================================+
| Temperature - <zone> (C)                          | degC        | 3 for Indoor zone, otherwise 5 | Temperature of envelope zone                                                                                                              |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Unmet HVAC Load (C)                               | degC        | 3                              | Absolute difference between Indoor temperature and thermal comfort limit (positive if hot, negative if cold)                              |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Occupancy (Persons)                               | Persons     | 8                              | Number of current occupants                                                                                                               |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Net Sensible Heat Gain - <zone> (W)               | W           | 5 for Indoor zone, otherwise 8 | Net sensible heat injected into zone. Includes heat gains from infiltration, ventilation, radiation, HVAC, other equipment, and occupants |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Window Transmitted Solar Gain (W)                 | W           | 5                              | Heat gains from solar transmitted through windows to Indoor zone                                                                          |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Infiltration Flow Rate - <zone> (m^3/s)           | m^3/s       | 8                              | Infiltration flow rate between zone and outdoors                                                                                          |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Forced Ventilation Flow Rate - Indoor (m^3/s)     | m^3/s       | 8                              | Mecahnical ventilation flow rate                                                                                                          |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Natural Ventilation Flow Rate - Indoor (m^3/s)    | m^3/s       | 8                              | Natural ventilation flow rate (open windows)                                                                                              |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Infiltration Heat Gain - <zone> (W)               | W           | 5 for Indoor zone, otherwise 8 | Infiltration heat gain into zone                                                                                                          |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Forced Ventilation Heat Gain - Indoor (W)         | W           | 5                              | Heat gain from mechanical ventilation                                                                                                     |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Natural Ventilation Heat Gain - Indoor (W)        | W           | 5                              | Heat gain from natural ventilation                                                                                                        |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Occupancy Heat Gain - Indoor (W)                  | W           | 8                              | Heat gain from occupancy                                                                                                                  |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Internal Heat Gain - Indoor (W)                   | W           | 5                              | Heat gain from non-HVAC equipment                                                                                                         |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Radiation Heat Gain - Indoor (W)                  | W           | 8                              | Heat gain from radiation. Includes transmitted solar and internal radiation to zone                                                       |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Net Latent Heat Gain - Indoor (W)                 | W           | 8                              | Net latent heat injected into zone. Includes heat gains from infiltration, ventilation, HVAC, other equipment, and occupants              |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Relative Humidity - Indoor (-)                    | unitless    | 8                              | Relative humidity of zone                                                                                                                 |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Humidity Ratio - Indoor (-)                       | unitless    | 8                              | Humidity ratio of zone                                                                                                                    |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Wet Bulb - Indoor (C)                             | W           | 8                              | Wet bulb temperature in zone                                                                                                              |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| Air Density - Indoor (kg/m^3)                     | unitless    | 8                              | Air density of zone                                                                                                                       |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> Ext. Solar Gain (W)               | W           | 9                              | Solar heat gain on external boundary surface                                                                                              |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> Ext. LWR Gain (W)                 | W           | 9                              | Long wave radiation heat gain on external boundary surface                                                                                |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> Ext. Surface Temperature (C)      | degC        | 9                              | External boundary surface temperature                                                                                                     |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> Ext. Film Coefficient (m^2-K/W)   | m^2-K/W     | 9                              | Film coefficient of external boundary surface                                                                                             |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> <zone> LWR Gain (W)               | W           | 9                              | Long wave radiation heat gain on internal boundary surface                                                                                |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> <zone> Surface Temperature (C)    | C           | 9                              | Internal boundary surface temperature                                                                                                     |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+
| <boundary name> <zone> Film Coefficient (m^2-K/W) | m^2-K/W     | 9                              | Film coefficient of internal boundary surface                                                                                             |
+---------------------------------------------------+-------------+--------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+

.. [#] Includes heat gains from infiltration, ventilation, radiation, HVAC,
    other equipment, and occupants. Does not include heat gains intrinsic to
    the linear model (usually only convection or conduction).
.. [#] Includes heat gains from infiltration, ventilation, HVAC, other
    equipment, and occupants.

Water Heating
~~~~~~~~~~~~~

The following results are specific to Water Heater equipment.

+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| OCHRE Name                                 | OCHRE Units | Verbosity | Description                                               |
+============================================+=============+===========+===========================================================+
| Water Heating Delivered (W)                | W           | 4         | Heat delivered by water heater to tank                    |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating COP (-)                      | unitless    | 4         | Water heater coefficient of performance                   |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating Total Sensible Heat Gain (W) | W           | 7         | Sensible heat gain from water tank to envelope zone       |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating Deadband Upper Limit (C)     | C           | 7         | Upper temperature limit for water heater deadband control |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating Deadband Lower Limit (C)     | C           | 7         | Lower temperature limit for water heater deadband control |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating Heat Pump Max Capacity (W)   | W           | 7         | Maximum capacity of HPWH heat pump element                |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating Heat Pump On Fraction (-)    | unitless    | 7         | Fraction of time HPWH heat pump element is on             |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+
| Water Heating Heat Pump COP (-)            | unitless    | 7         | HPWH heat pump coefficient of performance                 |
+--------------------------------------------+-------------+-----------+-----------------------------------------------------------+

The following results are specific to the Water Tank model.

+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| OCHRE Name                        | OCHRE Units | Verbosity | Description                                                        |
+===================================+=============+===========+====================================================================+
| Hot Water Unmet Demand (kW)       | kW          | 3         | Unmet hot water demand, based on flow rate and desired temperature |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Outlet Temperature (C)  | degC        | 3         | Hot water outlet temperature                                       |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Delivered (L/min)       | L/min       | 4         | Hot water draw volumetric flow rate                                |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Delivered (W)           | W           | 4         | Hot water draw heat flow rate                                      |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Heat Injected (W)       | W           | 7         | Water tank heat gains from water heater                            |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Heat Loss (W)           | W           | 7         | Water tank heat losses to envelope zone                            |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Average Temperature (C) | degC        | 7         | Water tank average temperature                                     |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Maximum Temperature (C) | degC        | 7         | Water tank maximum temperature                                     |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Minimum Temperature (C) | degC        | 7         | Water tank minimum temperature                                     |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+
| Hot Water Mains Temperature (C)   | degC        | 7         | Water mains temperature                                            |
+-----------------------------------+-------------+-----------+--------------------------------------------------------------------+

Electric Vehicle
~~~~~~~~~~~~~~~~

The following results are specific to Electric Vehicle equipment.

+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+
| OCHRE Name                     | OCHRE Units | Verbosity | Description                                                             |
+================================+=============+===========+=========================================================================+
| EV SOC (-)                     | unitless    | 3         | EV state of charge                                                      |
+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+
| EV Unmet Load (kW)             | kW          | 3         | Unmet EV demand, determined at parking End Time. Negative value         |
+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+
| EV Parked                      | N/A         | 4         | True if EV is parked at home                                            |
+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+
| EV Start Time                  | N/A         | 7         | If parked, time that EV arrived. If away, next time that EV will arrive |
+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+
| EV End Time                    | N/A         | 7         | Next time that EV will depart                                           |
+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+
| EV Remaining Charge Time (min) | minutes     | 7         | Estimated time to fully charge, based on SOC and max charge rate        |
+--------------------------------+-------------+-----------+-------------------------------------------------------------------------+

Solar PV
~~~~~~~~

The following results are specific to Solar PV equipment.

+--------------------+-------------+-----------+----------------------------+
| OCHRE Name         | OCHRE Units | Verbosity | Description                |
+====================+=============+===========+============================+
| PV P Setpoint (kW) | kW          | 6         | PV real power setpoint     |
+--------------------+-------------+-----------+----------------------------+
| PV Q Setpoint (kW) | kVAR        | 6         | PV reactive power setpoint |
+--------------------+-------------+-----------+----------------------------+

Battery
~~~~~~~

The following results are specific to Battery equipment.

+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+
| OCHRE Name                        | OCHRE Units | Verbosity | Description                                                                   |
+===================================+=============+===========+===============================================================================+
| Battery SOC (-)                   | unitless    | 3         | Battery state of charge                                                       |
+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+
| Battery Setpoint (kW)             | kW          | 6         | Battery real power setpoint                                                   |
+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+
| Battery Efficiency (-)            | unitless    | 6         | Battery efficiency                                                            |
+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+
| Battery Energy to Discharge (kWh) | kWh         | 7         | Estimated energy available for discharge, based on SOC and max discharge rate |
+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+
| Battery Nominal Capacity (kWh)    | kWh         | 7         | Nominal battery capacity, including degradation model                         |
+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+
| Battery Actual Capacity (kWh)     | kWh         | 7         | Actual battery capacity, including degradation and temperature models         |
+-----------------------------------+-------------+-----------+-------------------------------------------------------------------------------+

Equivalent Battery Model
~~~~~~~~~~~~~~~~~~~~~~~~

The following results are not reported at any verbosity, but they can be
output using the ``Equipment.make_equivalent_battery_model`` function.
Currently, this functions works for the following end uses:

- HVAC Heating
- HVAC Cooling
- Water Heating
- EV
- Battery

+----------------------------------------+-------------+-----------+---------------------------------------------------+
| OCHRE Name                             | OCHRE Units | Verbosity | Description                                       |
+========================================+=============+===========+===================================================+
| <end use> EBM Energy (kWh)             | kWh         | N/A       | Energy state of equivalent battery model (EBM)    |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Min Energy (kWh)         | kWh         | N/A       | Minimum energy constraint                         |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Max Energy (kWh)         | kWh         | N/A       | Maximum energy constraint                         |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Max Power (kW)           | kW          | N/A       | Maximum power constraint                          |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Efficiency (-)           | unitless    | N/A       | Input/output power efficiency                     |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Baseline Power (kW)      | kW          | N/A       | Power to maintain constant energy state           |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Max Discharge Power (kW) | kW          | N/A       | Minimum power constraint (negative for discharge) |
+----------------------------------------+-------------+-----------+---------------------------------------------------+
| <end use> EBM Discharge Efficiency (-) | unitless    | N/A       | Input/output power efficiency while discharging   |
+----------------------------------------+-------------+-----------+---------------------------------------------------+

.. _output-files:

Additional Output Files and Print Statements
--------------------------------------------

The ``verbosity`` parameter determines whether additional output files will be
saved. Regardless of ``verbosity``, no files will be saved if ``save_results``
is False. Additional output files include:

- ``<simulation_name>_complete`` or ``<simulation_name>_failed``: Empty file
  indicating if the simulation completed successfully or failed. Saved if
  ``verbosity > 0``.

- ``<simulation_name>.json``: JSON file with HPXML properties. Can also
  include dwelling parameters if ``save_args_to_json`` is set to True. Saved
  if ``verbosity >= 3`` or if ``save_args_to_json`` is set to True.

- ``<simulation_name>_hourly.csv``: Time series output file resampled to
  hourly resolution. Can be a parquet file if ``output_to_parquet`` is set to
  True. Saved if ``verbosity >= 3``.

- ``<simulation_name>_schedule.csv``: OCHRE schedule file including all
  scheduled time series data. Unlike the ``hpxml_schedule_file``, the values
  are absolute, not normalized, and the units are specified. Can be a parquet
  file if ``output_to_parquet`` is set to True. Saved if ``verbosity >= 7`` or
  if ``save_schedule_columns`` is specified. Only for ``Dwelling`` simulations.

- ``<equipment_name>_events.csv``: Event-based schedule file for event-based
  equipment. Includes event start and stop times and other relevant
  information. Saved if ``verbosity >= 7``.

The ``verbosity`` will also impact the print statements provided during the
simulation. Setting ``verbosity >= 3`` will allow most print statements to be
written.

.. _all-metrics:

All Metrics
-----------

Metrics are calculated at the end of a simulation and summarize the results
over the simulation period. The tables below show all potential metrics,
including the minimum ``metrics_verbosity`` required to output the data. Note
that some rows represent multiple results; for example, ``<end use> Electric
Energy (kWh)`` is output for each end use.

Dwelling Metrics
~~~~~~~~~~~~~~~~

The table below shows dwelling-level metrics.

+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Metric                                           | Verbosity | Description                                                             |
+==================================================+===========+=========================================================================+
| Total Electric Energy (kWh)                      | 1         | Total dwelling real electric energy consumption                         |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Total Gas Energy (therms)                        | 1         | Total dwelling gas energy consumption                                   |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Total Reactive Energy (kVARh)                    | 8         | Total dwelling reactive electric energy consumption                     |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Average Electric Power (kW)                      | 1         | Average dwelling real electric power                                    |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Peak Electric Power (kW)                         | 1         | Peak dwelling real electric power, using simulation time resolution     |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Peak Electric Power - <time resolution> avg (kW) | 7         | Peak dwelling real electric power, using specified time resolution [#]_ |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Number of Outages                                | 1         | Total number of outages during simulation                               |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Average Outage Duration (hours)                  | 1         | Average duration of outages during simulation                           |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+
| Longest Outage Duration (hours)                  | 1         | Duration of longest outage during simulation                            |
+--------------------------------------------------+-----------+-------------------------------------------------------------------------+

.. [#] OCHRE calculates peak power using 15-, 30-, and 60-minute resolution

Generic Equipment Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

The table below shows generic equipment and end-use metrics.

+------------------------------------------+-----------+--------------------------------------------------------------------------+
| Metric                                   | Verbosity | Description                                                              |
+==========================================+===========+==========================================================================+
| <end use> Electric Energy (kWh)          | 2         | Real electric energy consumption of all equipment within the end use     |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <end use> Gas Energy (therms)            | 2         | Gas energy consumption of all equipment within the end use               |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <end use> Reactive Energy (kVARh)        | 8         | Reactive electric energy consumption of all equipment within the end use |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <equipment name> Electric Energy (kWh)   | 6         | Real electric energy consumption of the equipment                        |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <equipment name> Gas Energy (therms)     | 6         | Gas energy consumption of the equipment                                  |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <equipment name> Reactive Energy (kVARh) | 8         | Reactive electric energy consumption of the equipment                    |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <equipment name> Cycles                  | 7         | Number of cycles of the equipment (On/Off cycles only)                   |
+------------------------------------------+-----------+--------------------------------------------------------------------------+
| <equipment name> <mode name> Cycles      | 7         | Number of cycles of the equipment with multiple modes                    |
+------------------------------------------+-----------+--------------------------------------------------------------------------+

Specific Equipment Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below shows equipment-level metrics by end use.

+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| End Use                  | Metric                                                | Verbosity | Description                                                                                     |
+==========================+=======================================================+===========+=================================================================================================+
| HVAC Heating and Cooling | Unmet <Heating or Cooling> Load (C-hours)             | 3         | Unmet HVAC load [#]_                                                                            |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Heating and Cooling | Total HVAC <Heating or Cooling> Delivered (kWh)       | 4         | Total heat delivered to the Indoor zone                                                         |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Heating and Cooling | Average HVAC <Heating or Cooling> COP (-)             | 4         | Average coefficient of performance of HVAC equipment (excludes fan and other peripherals)       |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Heating and Cooling | HVAC <Heating or Cooling> Main Energy (kWh)           | 7         | Total electric or gas energy consumed by main HVAC element (excludes fan and other peripherals) |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Heating and Cooling | HVAC <Heating or Cooling> Fan Energy (kWh)            | 7         | Total energy consumed by HVAC fan and other peripherals                                         |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Heating and Cooling | Average HVAC <Heating or Cooling> Capacity (kW)       | 7         | Average heating capacity of HVAC equipment                                                      |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Heating and Cooling | Average HVAC <Heating or Cooling> Duct Efficiency (-) | 7         | Average duct efficiency of HVAC equipment                                                       |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| HVAC Cooling             | Average HVAC Cooling SHR (-)                          | 7         | Average sensible heat ratio of HVAC cooling equipment                                           |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Water Heating            | Total Hot Water Unmet Demand (kWh)                    | 3         | Unmet water heating load [#]_                                                                   |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Water Heating            | Total Hot Water Delivered (gal/day)                   | 4         | Total volume of hot water delivered to water draws                                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Water Heating            | Total Hot Water Delivered (kWh)                       | 4         | Total energy of hot water delivered to water draws                                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Water Heating            | Total Water Heating Delivered (kWh)                   | 4         | Total energy of hot water delivered by the water heater                                         |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Water Heating            | Average Water Heating COP (-)                         | 4         | Average coefficient of performance of water heater                                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| EV                       | Average EV SOC (-)                                    | 4         | Average SOC of the EV                                                                           |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| EV                       | Total EV Unmet Load (kWh)                             | 4         | Unmet EV load [#]_                                                                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Battery                  | Average Islanding Time (hours)                        | 4         | The average duration that the battery could prevent an outage [#]_                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Battery                  | Battery Charging Energy (kWh)                         | 4         | Total real electric energy consumed by the battery during charging                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Battery                  | Battery Discharging Energy (kWh)                      | 4         | Total real electric energy produced by the battery during discharging                           |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Battery                  | Battery Round-trip Efficiency (-)                     | 4         | Average round-trip efficiency [#]_                                                              |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+
| Gas Generator            | Gas Generator Efficiency (-)                          | 4         | Average efficiency of electricity outputs to gas inputs                                         |
+--------------------------+-------------------------------------------------------+-----------+-------------------------------------------------------------------------------------------------+

.. [#] Calculated as the difference between the actual temperature and the
    minimum (maximum) deadband temperature for HVAC Heating (Cooling), summed
    across all time steps
.. [#] Calculated as the difference between the actual temperature and the
    minimum deadband temperature, summed across all time steps
.. [#] Unmet load is incurred when the EV SOC lost from driving is greater
    than the EV SOC at the end of the previous charging session. EVs can shift
    energy between charging sessions without incurring unmet load as long as
    the SOC remains positive.
.. [#] Calculated based on battery SOC and future dwelling net load
.. [#] Ignores differences between initial and final SOC, which may be
    significant for short simulations

Envelope Metrics
~~~~~~~~~~~~~~~~

The table below shows envelope metrics.

+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Metric                                     | Verbosity                      | Description                                                            |
+============================================+================================+========================================================================+
| Average Temperature - <zone name> (C)      | 3 for Indoor zone, otherwise 5 | Average temperature of the zone                                        |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Component Load - Internal Gains (kWh)      | 5                              | Total HVAC load (heating minus cooling) induced by internal gains      |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Component Load - Infiltration (kWh)        | 5                              | Total HVAC load (heating minus cooling) induced by infiltration        |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Component Load - Forced Ventilation (kWh)  | 5                              | Total HVAC load (heating minus cooling) induced by forced ventilation  |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Component Load - Natural Ventilation (kWh) | 5                              | Total HVAC load (heating minus cooling) induced by natural ventilation |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Component Load - Ducts, Heating (kWh)      | 5                              | Total HVAC heating load induced by duct losses                         |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Component Load - Ducts, Cooling (kWh)      | 5                              | Total HVAC cooling load induced by duct losses                         |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Std. Dev. Temperature - <zone name> (C)    | 8                              | Standard deviation of zone temperature                                 |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Average Relative Humidity - Indoor (-)     | 9                              | Average Relative Humidity of Indoor zone                               |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+
| Average Humidity Ratio - Indoor (-)        | 9                              | Average Humidity Ratio of Indoor zone                                  |
+--------------------------------------------+--------------------------------+------------------------------------------------------------------------+


Data Analysis
-------------

The ``Analysis`` module has useful functions for analyzing OCHRE output data,
manipulating output files, and other tasks.

This code will load an existing OCHRE simulation and recalculate the metrics:

.. code-block:: python

    from ochre import Analysis
    
    # load existing ochre simulation data
    df, metrics, df_hourly = Analysis.load_ochre(folder)
    
    # calculate metrics from a pandas DataFrame
    metrics = Analysis.calculate_metrics(df)

Some analysis functions are useful for analyzing or combining results from
multiple OCHRE simulations:

.. code-block:: python

    # combine input json files
    json_files = {folder: os.path.join(folder, "ochre.json") for folder in ochre_folders}
    df = Analysis.combine_json_files(json_files)

    # combine a single time series column for each simulation (e.g., total electricity consumption)
    results_files = {folder: os.path.join(folder, "ochre.csv") for folder in ochre_folders}
    df = Analysis.combine_time_series_column("Total Electric Power (kW)", results_files)

    # aggregate time series data across all simulations
    df = Analysis.combine_time_series_files(results_files, aggregate=True)

For a more complete example to compile data across multiple OCHRE simulations,
see the ``compile_results`` function in `run_multiple.py
<https://github.com/NREL/OCHRE/blob/main/bin/run_multiple.py#L16>`__.

Other functions can:

- Download ResStock model files

- Compare OCHRE and EnergyPlus results

- Find all OCHRE simulation folders within a root directory


Data Visualization
------------------

The ``CreateFigures`` module has useful visualization functions for OCHRE
output data. Many functions work on any generic pandas DataFrame with a
DateTimeIndex.

This code will load an existing OCHRE simulation and create a stacked plot of
power by end use and various HVAC output plots:

.. code-block:: python

    from ochre import Analysis, CreateFigures
    
    df, metrics, df_hourly = Analysis.load_ochre(folder)

    # Create stacked plot of power by end use
    CreateFigures.plot_power_stack(df)

    # Create standard HVAC output plots
    CreateFigures.plot_hvac(df)

Other functions can:

- Plot one or more time series columns across one or more result dataframes

- Plot daily or monthly load profiles

- Plot powers for all end uses or all equipment in a dwelling simulation

- Plot standard results for HVAC equipment, water heaters, and the dwelling
  envelope
