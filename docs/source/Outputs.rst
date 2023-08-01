Outputs and Analysis
====================

At the end of any OCHRE simulation, time series outputs are saved. These
time series outputs are used to calculate metrics that describe the
simulation results. The set of time series outputs depends on the
``verbosity``` of the simulation, and the set of metrics depends on the
``metrics_verbosity``. The tables below describe the Dwelling and
Equipment-specific outputs and metrics that are reported.

Time Series Outputs
----------------------------
Time series outputs are sorted by category.

Dwelling Level Outputs
----------------------

+----------------+-----------------------------------------+-----------------+
| **Category**   | **OCHRE Name**                          | **OCHRE Units** |
+================+=========================================+=================+
| Dwelling       | Total Electric Power                    | kW              |
+----------------+-----------------------------------------+-----------------+
| Dwelling       | Total Gas Power                         | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Electric Power             | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Main Power                 | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating ER Power                   | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Fan Power                  | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating  |  HVAC Heating Gas Power                  | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Delivered                  | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Duct Losses                | W               |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Capacity                   | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating EIR Ratio                  | unitless        |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Capacity Ratio             | unitless        |
+----------------+-----------------------------------------+-----------------+
| HVAC Heating   | HVAC Heating Setpoint                   | degC            |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Electric Power             | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Main Power                 | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Fan Power                  | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Delivered                  | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Latent Gains               | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Duct Losses                | W               |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Capacity                   | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Sensible Capacity          | kW              |
+----------------+-----------------------------------------+-----------------+
| HVAC Cooling   | HVAC Cooling Setpoint                   | degC            |
+----------------+-----------------------------------------+-----------------+
| Water Heating  | Water Heating Electric Power            | kW              |
+----------------+-----------------------------------------+-----------------+
| Water Heating  |  Water Heating Gas Power                | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| Water Heating  | Hot Water Delivered                     | kW              |
+----------------+-----------------------------------------+-----------------+
| Water Heating  | Hot Water Outlet Temperature            | degC            |
+----------------+-----------------------------------------+-----------------+
| Water Heating  | Hot Water Delivered                     | L/min           |
+----------------+-----------------------------------------+-----------------+
| EV             | EV Electric Power                       | kW              |
+----------------+-----------------------------------------+-----------------+
| PV             | PV Electric Power                       | kW              |
+----------------+-----------------------------------------+-----------------+
| Battery        | Battery Electric Power                  | kW              |
+----------------+-----------------------------------------+-----------------+
| Gas Generator  | Gas Generator Electric Power            | kW              |
+----------------+-----------------------------------------+-----------------+
| Gas Generator  | Generator Gas Power                     | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| Lighting       | Lighting Electric Power                 | kW              |
+----------------+-----------------------------------------+-----------------+
| Lighting       | Indoor Lighting Electric Power          | kW              |
+----------------+-----------------------------------------+-----------------+
| Lighting       | Basement Lighting Electric Power        | kW              |
+----------------+-----------------------------------------+-----------------+
| Lighting       | Garage Lighting Electric Power          | kW              |
+----------------+-----------------------------------------+-----------------+
| Lighting       | Exterior Lighting Electric Power        | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Other Electric Power                    | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Clothes Washer Electric Power           | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Clothes Dryer Electric Power            | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Dishwasher Electric Power               | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Refrigerator Electric Power             | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Cooking Range Electric Power            | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | MELs Electric Power                     | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | TV Electric Power                       | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Well Pump Electric Power                | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Pool Pump Electric Power                | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Pool Heater Electric Power              | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Hot Tub Pump Electric Power             | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Hot Tub Heater Electric Power           | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Ceiling Fan Electric Power              | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Ventilation Fan Electric Power          | kW              |
+----------------+-----------------------------------------+-----------------+
| Other          | Other Gas Power                         | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| Other          | Clothes Dryer Gas Power                 | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| Other          | Cooking Range Gas Power                 | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| Other          | MGLs Gas Power                          | therms/hour     |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Temperature - Indoor                    | degC            |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Temperature - Garage                    | degC            |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Temperature - Foundation                | degC            |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Temperature - Attic                     | degC            |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Net Latent Heat Gain - Indoor           | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Relative Humidity - Indoor              | unitless        |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Humidity Ratio - Indoor                 | unitless        |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Unmet HVAC Load                         | degC            |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Forced Ventilation Flow Rate - Indoor   | m^3/s           |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Natural Ventilation Flow Rate - Indoor  | m^3/s           |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Flow Rate - Indoor         | m^3/s           |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Flow Rate - Foundation     | m^3/s           |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Flow Rate - Garage         | m^3/s           |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Flow Rate - Attic          | m^3/s           |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Forced Ventilation Heat Gain - Indoor   | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Natural Ventilation Heat Gain - Indoor  | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Heat Gain - Indoor         | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Heat Gain - Foundation     | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Heat Gain - Garage         | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Infiltration Heat Gain - Attic          | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Occupancy Heat Gain - Indoor            | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Internal Heat Gain - Indoor             | W               |
+----------------+-----------------------------------------+-----------------+
| Envelope       | Window Transmitted Solar Gain           | W               |
+----------------+-----------------------------------------+-----------------+
| Schedule       | Temperature - Outdoor                   | degC            |
+----------------+-----------------------------------------+-----------------+
| Schedule       | Ambient Humidity Ratio                  | unitless        |
+----------------+-----------------------------------------+-----------------+
| Schedule       | Ambient Relative Humidity               | %               |
+----------------+-----------------------------------------+-----------------+
| Schedule       | Wind Speed                              | m/s             |
+----------------+-----------------------------------------+-----------------+
| Schedule       | DNI                                     | W/m^2           |
+----------------+-----------------------------------------+-----------------+
| Schedule       | DHI                                     | W/m^2           |
+----------------+-----------------------------------------+-----------------+
| Schedule       | GHI                                     | W/m^2           |
+----------------+-----------------------------------------+-----------------+

Dwelling Metrics
----------------

Metrics are calculated at the end of a simulation and summarize the
results over the simulation period (generally a year in most use cases).

+-----------------+--------+------------------------------------------------------------------+
| Metric          | Minimul Metrics Verbosity      | Description                              |
+=================+================================+==========================================+
| Total Electric  | 1                              | Total dwelling real electric energy      |
| Energy (kWh)    |                                | consumption                              |
+-----------------+--------------------------------+------------------------------------------+
| <end use>       | 2                              | Real electric energy consumption of all  |
| Electric Energy |                                | equipment within the end use             |
| (kWh)           |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| <equipment      | 5                              | Real electric energy consumption of the  |
| name> Electric  |                                | equipment                                |
| Energy (kWh)    |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Total Reactive  | 7                              | Total dwelling reactive electric energy  |
| Energy (kVARh)  |                                | consumption                              |
+-----------------+--------------------------------+------------------------------------------+
| <end use>       | 7                              | Reactive electric energy consumption of  |
| Reactive Energy |                                | all equipment within the end use         |
| (kVARh)         |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| <equipment      | 7                              | Reactive electric energy consumption of  |
| name> Reactive  |                                | the equipment                            |
| Energy (kVARh)  |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Total Gas       | 1                              | Total dwelling gas energy consumption    |
| Energy (therms) |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| <end use> Gas   | 2                              | Gas energy consumption of all equipment  |
| Energy (therms) |                                | within the end use                       |
+-----------------+--------------------------------+------------------------------------------+
| <equipment      | 5                              | Gas energy consumption of the equipment  |
| name> Gas       |                                |                                          |
| Energy (therms) |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average         | 3                              | Average temperature of the zone          |
| Temperature -   |                                |                                          |
| <zone name> (C) |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Unmet <Heating  | 4                              | Unmet HVAC load. Based on the difference |
| or Cooling>     |                                | between actual and desired temperature   |
| Load (C-hours)  |                                | and the duration of the unmet load       |
+-----------------+--------------------------------+------------------------------------------+
| Component Load  | 6                              | Total HVAC load (heating minus cooling)  |
| - Internal      |                                | induced by internal gains                |
| Gains (kWh)     |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Component Load  | 6                              | Total HVAC load (heating minus cooling)  |
| - Infiltration  |                                | induced by infiltration                  |
| (kWh)           |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Component Load  | 6                              | Total HVAC load (heating minus cooling)  |
| - Forced        |                                | induced by forced ventilation            |
| Ventilation     |                                |                                          |
| (kWh)           |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Component Load  | 6                              | Total HVAC load (heating minus cooling)  |
| - Natural       |                                | induced by natural ventilation           |
| Ventilation     |                                |                                          |
| (kWh)           |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Component Load  | 6                              | Total HVAC heating load induced by duct  |
| - Ducts,        |                                | losses                                   |
| Heating (kWh)   |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Component Load  | 6                              | Total HVAC cooling load induced by duct  |
| - Ducts,        |                                | losses                                   |
| Cooling (kWh)   |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average         | 6                              | Average dwelling real electric power     |
| Electric Power  |                                |                                          |
| (kW)            |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Peak Electric   | 6                              | Peak dwelling real electric power, using |
| Power (kW)      |                                | simulation time resolution               |
+-----------------+--------------------------------+------------------------------------------+
| Peak Electric   | 6                              | Peak dwelling real electric power, using |
| Power - <time   |                                | specified time resolution                |
| resolution> avg |                                |                                          |
| (kW)            |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average HVAC    | 8                              | Average heating capacity of HVAC         |
| <Heating or     |                                | equipment                                |
| Cooling>        |                                |                                          |
| Capacity (kW)   |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| HVAC <Heating   | **5**                          | Total electric or gas energy consumed by |
| or Cooling>     |                                | main HVAC element (excludes fan and      |
| Main Energy     |                                | other peripherals)                       |
| (kWh)           |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| HVAC <Heating   | 4                              | Total energy consumed by HVAC fan and    |
| or Cooling> Fan |                                | other peripherals                        |
| Energy (kWh)    |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average HVAC    | 4                              | Average coefficient of performance of    |
| <Heating or     |                                | HVAC equipment (excludes fan and other   |
| Cooling> COP    |                                | peripherals)                             |
| (-)             |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average HVAC    | 4                              | Average duct efficiency of HVAC          |
| <Heating or     |                                | equipment                                |
| Cooling> Duct   |                                |                                          |
| Efficiency (-)  |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average HVAC    | 4                              | Average sensible heat ratio of HVAC      |
| Cooling SHR (-) |                                | cooling equipment                        |
+-----------------+--------------------------------+------------------------------------------+
| Std. Dev.       | 8                              | Standard deviation of zone temperature   |
| Temperature -   |                                |                                          |
| <zone name> (C) |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average         |                                | Average Relative Humidity of Indoor zone |
| Relative        |                                |                                          |
| Humidity -      |                                |                                          |
| Indoor (-)      |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average         |                                | Average Humidity Ratio of Indoor zone    |
| Humidity Ratio  |                                |                                          |
| - Indoor (-)    |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Total Hot Water | 4                              | Unmet water heating load. Based on the   |
| Unmet Demand    |                                | difference between actual and desired    |
| (kWh)           |                                | temperature and the duration of the      |
|                 |                                | unmet load                               |
+-----------------+--------------------------------+------------------------------------------+
| Total Hot Water | 4                              | Total volume of hot water delivered to   |
| Delivered       |                                | water draws                              |
| (gal/day)       |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Total Hot Water | 4                              | Total energy of hot water delivered to   |
| Delivered (kWh) |                                | water draws                              |
+-----------------+--------------------------------+------------------------------------------+
| Total Water     | 4                              | Total energy of hot water delivered by   |
| Heating         |                                | the water heater                         |
| Delivered (kWh) |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Average Water   | 4                              | Average coefficient of performance of    |
| Heating COP (-) |                                | water heater                             |
+-----------------+--------------------------------+------------------------------------------+
| Average         | 4                              | The average duration that the home could |
| Islanding Time  |                                | island using battery power given no      |
| (hours)         |                                | changes in other equipment power         |
+-----------------+--------------------------------+------------------------------------------+
| Battery         | 4                              | Total real electric energy consumed by   |
| Charging Energy |                                | the battery during charging              |
| (kWh)           |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Battery         | 4                              | Total real electric energy produced by   |
| Discharging     |                                | the battery during discharging           |
| Energy (kWh)    |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Battery         | 4                              | Average round-trip efficiency. Ignores   |
| Round-trip      |                                | differences between initial and final    |
| Efficiency (-)  |                                | SOC                                      |
+-----------------+--------------------------------+------------------------------------------+
| Gas Generator   | 4                              | Average efficiency of electricity        |
| Efficiency (-)  |                                | outputs to gas inputs                    |
+-----------------+--------------------------------+------------------------------------------+
| Number of       | 4                              | Total number of outages during           |
| Outages         |                                | simulation                               |
+-----------------+--------------------------------+------------------------------------------+
| Average Outage  | 4                              | Average duration of outages during       |
| Duration        |                                | simulation                               |
| (hours)         |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| Longest Outage  | 4                              | Duration of longest outage during        |
| Duration        |                                | simulation                               |
| (hours)         |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+
| <equipment      | 5                              | Number of cycles of the equipment        |
| name> Cycles    |                                | (On/Off cycles only)                     |
+-----------------+--------------------------------+------------------------------------------+
| <equipment      | 5                              | Number of cycles of the equipment with   |
| name> <mode     |                                | multiple modes                           |
| name> Cycles    |                                |                                          |
+-----------------+--------------------------------+------------------------------------------+

Data Analysis
-------------

The ``Analysis`` module has useful data analysis functions for OCHRE
output data:

.. code-block:: python
    from ochre import Analysis
    
    # load existing ochre simulation data
    df, metrics, df_hourly = Analysis.load_ochre(folder)
    # calculate metrics from a pandas DataFrame
    metrics = Analysis.calculate_metrics(df)



Some analysis functions are useful for analyzing or combining results
from multiple OCHRE simulations:

.. code-block:: python
    # Combine OCHRE metrics files from multiple simulations (in subfolders of path)
    df_metrics = Analysis.combine_metrics_files(path=path)
    
    # Combine 1 output column from multiple OCHRE simulations into a single DataFrame
    results_files = Analysis.find_files_from_ending(path, ‘ochre.csv’)
    df_powers = Analysis.combine_time_series_column(results_files, 'Total Electric Power (kW)')

Data Visualization
------------------

The ``CreateFigures`` module has useful visualization functions for
OCHRE output data:

.. code-block:: python
    from ochre import Analysis, CreateFigures
    df, metrics, df_hourly = Analysis.load_ochre(folder)
    # Create standard HVAC output plots
    CreateFigures.plot_hvac(df)
    # Create stacked plot of power by end use
    CreateFigures.plot_power_stack(df)

Many functions work on any generic pandas DataFrame with a
DateTimeIndex.
