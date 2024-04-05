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

.. code-block:: python

    battery = Battery(
        capacity_kwh=10,  # energy capacity = 10 kWh
        capacity=5,       # power capacity = 5 kW
        soc_init=0.5,     # Initial SOC = 50%
        start_time=dt.datetime(2018, 1, 1, 0, 0),
        time_res=dt.timedelta(minutes=15),
        duration=dt.timedelta(days=1),
    )
    
    control_signal = {'P Setpoint': -5}      # Discharge at 5 kW
    status = battery.update(control_signal)  # Run for 1 time step

The following table lists the control signals available to OCHRE
equipment, by end use.

HVAC Heating or HVAC Cooling
----------------------------

+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| **End Use or Equipment Name** | **Control Command**      | **Units** | **Resets?**         | **Description**                                                           |
+===============================+==========================+===========+=====================+===========================================================================+
| HVAC Heating or HVAC Cooling  | Load Fraction            | unitless  | TRUE                | 1 (no effect) or 0 (forces equipment off)                                 |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating or HVAC Cooling  | Setpoint                 | C         | TRUE                | Sets temperature setpoint (then reverts back to schedule)                 |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating or HVAC Cooling  | Deadband                 | C         | Only if in schedule | Sets temperature deadband                                                 |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating or HVAC Cooling  | Capacity                 | W         | TRUE                | Sets HVAC capacity directly, ideal capacity mode only                     |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating or HVAC Cooling  | Max Capacity Fraction    | unitless  | Only if in schedule | Limits HVAC max capacity, ideal capacity only                             |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating or HVAC Cooling  | Duty Cycle               | unitless  | TRUE                | Sets the equipment duty cycle for ext_time_res, non-ideal capacity only   |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating or HVAC Cooling  | Disable Speed X          | N/A       | FALSE               | Flag to disable low (X=1) or high (X=2) speed, only for 2 speed equipment |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating (ASHP only)      | ER Capacity              | W         | TRUE                | Sets ER element capacity directly, ideal capacity only                    |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating (ASHP only)      | Max ER Capacity Fraction | unitless  | Only if in schedule | Limits ER element max capacity, ideal capacity only                       |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+
| HVAC Heating (ASHP only)      | ER Duty Cycle            | unitless  | TRUE                | Sets the ER element duty cycle for ext_time_res, non-ideal capacity only  |
+-------------------------------+--------------------------+-----------+---------------------+---------------------------------------------------------------------------+

Water Heating
-----------------------------

+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| **Control Command** | **Units** | **Resets?**         | **Description**                                                    |
+=====================+===========+=====================+====================================================================+
| Load Fraction       | unitless  | TRUE                | 1 (no effect) or 0 (forces equipment off)                          |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| Setpoint            | C         | Only if in schedule | Sets temperature setpoint [#]_                                     |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| Deadband            | C         | Only if in schedule | Sets temperature deadband [#]_                                     |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| Max Power           | kW        | Only if in schedule | Sets the maximum power. Does not work for HPWH in HP mode          |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| Duty Cycle          | unitless  | TRUE                | Sets the equipment duty cycle for ext_time_res                     |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| HP Duty Cycle       | unitless  | TRUE                | Sets the HPWH heat pump duty cycle for ext_time_res                |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+
| ER Duty Cycle       | unitless  | TRUE                | Sets the HPWH electric resistance duty cycle for ext_time_res [#]_ |
+---------------------+-----------+---------------------+--------------------------------------------------------------------+

.. [#] Sending {'Setpoint': None} will reset the setpoint to the default schedule. Note that a 10 F (5.56 C)
       decrease in setpoint corresponds to a CTA-2045 'Load Shed' command. A 10 F increase corresponds to an
       'Advanced Load Add' command (only available in B version of standard).
.. [#] Decreasing the deadband to about 2 C corresponds to a CTA 'Load Add' command. A typical deadband for
       gas and electric water heaters is 10 F (5.56 C).
.. [#] Most, but not all HPWHs have backup electric resistance. 120 V HPWHs (coming soon in OCHRE) do not
         have backup ER heaters.

Electric Vehicle (EV)
-----------------------------

+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+
| **Control Command** | **Units** | **Resets?**         | **Description**                                                                                                                 |
+=====================+===========+=====================+=================================================================================================================================+
| Delay               | N/A       | TRUE                | Delays EV charge for a given time. Value can be a datetime.timedelta or an integer to specify the number of time steps to delay |
+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+
| P Setpoint          | kW        | TRUE                | Sets AC power setpoint                                                                                                          |
+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+
| SOC                 | unitless  | TRUE                | Sets AC power to achieve desired SOC setpoint                                                                                   |
+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+
| SOC Rate            | 1/hour    | TRUE                | Sets AC power setpoint based on SOC rate, EV capacity, and efficiency of charging                                               |
+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Max Power           | kW        | Only if in schedule | Maximum power limit                                                                                                             |
+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Max SOC             | unitless  | Only if in schedule | Maximum SOC limit                                                                                                               |
+---------------------+-----------+---------------------+---------------------------------------------------------------------------------------------------------------------------------+

Photovoltaics (PV)
-----------------------------

+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+
| **Control Command** | **Units** | **Resets?** | **Description**                                                                        |
+=====================+===========+=============+========================================================================================+
| P Setpoint          | kW        | TRUE        | Sets real AC power setpoint                                                            |
+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+
| P Curtailment (kW)  | kW        | TRUE        | Sets real power setpoint by specifying absolute curtailment                            |
+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+
| P Curtailment (%)   | %         | TRUE        | Sets real power setpoint by specifying curtailment relative to maximum power point     |
+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+
| Q Setpoint          | kVAR      | TRUE        | Sets reactive power setpoint                                                           |
+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+
| Power Factor        | unitless  | TRUE        | Sets reactive power setpoint based on power factor                                     |
+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+
| Priority            | N/A       | FALSE       | Changes internal controller priority setting. Options are 'Watt', 'Var', or 'CPF' [#]_ |
+---------------------+-----------+-------------+----------------------------------------------------------------------------------------+

.. [#] CPF: Constant Power Factor

Battery
-----------------------------

+-----------------------+-----------+---------------------+--------------------------------------------------------+
| **Control Command**   | **Units** | **Resets?**         | **Description**                                        |
+=======================+===========+=====================+========================================================+
| P Setpoint            | kW        | TRUE                | Sets AC power setpoint                                 |
+-----------------------+-----------+---------------------+--------------------------------------------------------+
| SOC                   | unitless  | TRUE                | Sets AC power to achieve desired SOC setpoint          |
+-----------------------+-----------+---------------------+--------------------------------------------------------+
| Self Consumption Mode | N/A       | FALSE               | Flag to turn on Self-Consumption Mode [#]_             |
+-----------------------+-----------+---------------------+--------------------------------------------------------+
| Min SOC               | unitless  | Only if in schedule | Minimum SOC limit for self-consumption control         |
+-----------------------+-----------+---------------------+--------------------------------------------------------+
| Max SOC               | unitless  | Only if in schedule | Maximum SOC limit for self-consumption control         |
+-----------------------+-----------+---------------------+--------------------------------------------------------+
| Max Import Limit      | kW        | Only if in schedule | Max dwelling import power for self-consumption control |
+-----------------------+-----------+---------------------+--------------------------------------------------------+
| Max Export Limit      | kW        | Only if in schedule | Max dwelling export power for self-consumption control |
+-----------------------+-----------+---------------------+--------------------------------------------------------+

.. [#] Self-Consumption Mode aims to minimize grid imports and exports. This
    strategy will charge the battery when net energy consumption is larger
    than the Max Import Limit and discharge when net energy generation is
    larger than the Max Export Limit.

Lighting and Other
-----------------------------

+---------------------+-------------+-------------+----------------------------------------------------------------------------+
| **Control Command** | **Units**   | **Resets?** | **Description**                                                            |
+=====================+=============+=============+============================================================================+
| Load Fraction       | unitless    | TRUE        | Adjusts the scheduled power consumption. Applied to electric and gas power |
+---------------------+-------------+-------------+----------------------------------------------------------------------------+
| P Setpoint          | kW          | TRUE        | Sets electric power setpoint                                               |
+---------------------+-------------+-------------+----------------------------------------------------------------------------+
| Gas Setpoint        | therms/hour | TRUE        | Sets gas power setpoint                                                    |
+---------------------+-------------+-------------+----------------------------------------------------------------------------+

External Model Signals
------------------------------

OCHRE can also integrate with external models that modify default schedule
values and other settings.

The most common use case is to integrate with a grid simulator that modifies
the dwelling voltage. OCHRE includes a ZIP model for all equipment that
modifies the real and reactive electric power based on the grid voltage.

The following code sends a voltage of 0.97 p.u. to a Dwelling model:

.. code-block:: python

    status = dwelling.update(schedule_inputs={'Voltage (-)': 0.97})

External model signals can also modify any time series schedule values
including weather and occupancy variables. The names and units of these
variables can be found in the header of the schedule output file.
Alternatively, these variables can be reset at the beginning of the
simulation; see `this example code
<https://github.com/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb>`__ for
more details.

Status Variables
----------------

The ``update`` function for equipment and dwellings returns a Python
dictionary with status variables that can be sent to the external controller.
These status variables are equivalent to the Time Series Outputs described in
Outputs and Analysis. Note that the ``verbosity`` applies to the status
variables in the same way as the outputs.

Example Use Cases
-----------------

See `bin/run_external_control.py
<https://github.com/NREL/OCHRE/blob/main/bin/run_external_control.py>`__ and
`notebooks/user_tutorial.ipynb
<https://github.com/NREL/OCHRE/blob/main/notebook/user_tutorial.ipynb>`__ for
more details.

Co-simulation
-------------

Multiple OCHRE instances have been run in co-simulation using the `HELICS
<https://helics.org/>`__ platform. OCHRE models can communicate with other
agents via its external control signals, external model signals, and status
variables.

See the publications list for examples of co-simulation architectures that use
OCHRE. We do not currently have public code for using OCHRE in co-simulation.
