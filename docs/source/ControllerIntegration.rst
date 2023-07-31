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

The following table lists the control signals available to OCHRE
equipment, by end use.

+----------+-----------+-----+---------------------------------------+
| End Use  | Control   | Un  | Description                           |
| or       | Command   | its |                                       |
| E        |           |     |                                       |
| quipment |           |     |                                       |
| Name     |           |     |                                       |
+==========+===========+=====+=======================================+
| HVAC     | Load      | un  | 1 (no effect) or 0 (forces equipment  |
| Heating  | Fraction  | itl | off)                                  |
| or HVAC  |           | ess |                                       |
| Cooling  |           |     |                                       |
+----------+-----------+-----+---------------------------------------+
| HVAC     | Setpoint  | C   | Sets temperature setpoint for one     |
| Heating  |           |     | timestep (then reverts back to        |
| or HVAC  |           |     | schedule)                             |
| Cooling  |           |     |                                       |
+----------+-----------+-----+---------------------------------------+
| HVAC     | Deadband  | C   | Sets temperature deadband (does not   |
| Heating  |           |     | revert back unless deadband is in the |
| or HVAC  |           |     | schedule)                             |
| Cooling  |           |     |                                       |
+----------+-----------+-----+---------------------------------------+
| HVAC     | Duty      | un  | Sets the equipment duty cycle for     |
| Heating  | Cycle     | itl | ext_time_res                          |
| or HVAC  |           | ess |                                       |
| Cooling  |           |     |                                       |
+----------+-----------+-----+---------------------------------------+
| HVAC     | Disable   | N/A | For 2 speed equipment, disables low   |
| Heating  | Speed X   |     | (X=1) or high (X=2) speed if value is |
| or HVAC  |           |     | True                                  |
| Cooling  |           |     |                                       |
+----------+-----------+-----+---------------------------------------+
| Water    | Load      | un  | 1 (no effect) or 0 (forces equipment  |
| Heating  | Fraction  | itl | off)                                  |
|          |           | ess |                                       |
+----------+-----------+-----+---------------------------------------+
| Water    | Setpoint  | C   | Sets temperature setpoint. Sending    |
| Heating  |           |     | {'Setpoint': None} will reset the     |
|          |           |     | setpoint to the default schedule      |
+----------+-----------+-----+---------------------------------------+
| Water    | Deadband  | C   | Sets temperature deadband (does not   |
| Heating  |           |     | reset)                                |
+----------+-----------+-----+---------------------------------------+
| Water    | Duty      | un  | Sets the equipment duty cycle for     |
| Heating  | Cycle     | itl | ext_time_res                          |
|          |           | ess |                                       |
+----------+-----------+-----+---------------------------------------+
| Water    | HP Duty   | un  | Sets the HPWH heat pump duty cycle    |
| Heating  | Cycle     | itl | for ext_time_res                      |
|          |           | ess |                                       |
+----------+-----------+-----+---------------------------------------+
| Water    | ER Duty   | un  | Sets the HPWH electric resistance     |
| Heating  | Cycle     | itl | duty cycle for ext_time_res           |
|          |           | ess |                                       |
+----------+-----------+-----+---------------------------------------+
| EV       | Delay     | N/A | Delays EV charge for a given time.    |
|          |           |     | Value can be a datetime.timedelta or  |
|          |           |     | an integer to specify the number of   |
|          |           |     | time steps to delay                   |
+----------+-----------+-----+---------------------------------------+
| EV       | P         | kW  | Sets AC power setpoint                |
|          | Setpoint  |     |                                       |
+----------+-----------+-----+---------------------------------------+
| EV       | SOC Rate  | 1/h | Sets AC power setpoint based on SOC   |
|          |           | our | rate, EV capacity, and efficiency of  |
|          |           |     | charging                              |
+----------+-----------+-----+---------------------------------------+
| PV       | P         | kW  | Sets real AC power setpoint           |
|          | Setpoint  |     |                                       |
+----------+-----------+-----+---------------------------------------+
| PV       | P         | kW  | Sets real power setpoint by           |
|          | Cu        |     | specifying absolute curtailment       |
|          | rtailment |     |                                       |
|          | (kW)      |     |                                       |
+----------+-----------+-----+---------------------------------------+
| PV       | P         | %   | Sets real power setpoint by           |
|          | Cu        |     | specifying curtailment relative to    |
|          | rtailment |     | maximum power point                   |
|          | (%)       |     |                                       |
+----------+-----------+-----+---------------------------------------+
| PV       | Q         | k   | Sets reactive power setpoint          |
|          | Setpoint  | VAR |                                       |
+----------+-----------+-----+---------------------------------------+
| PV       | Power     | un  | Sets reactive power setpoint based on |
|          | Factor    | itl | power factor                          |
|          |           | ess |                                       |
+----------+-----------+-----+---------------------------------------+
| PV       | Priority  | N/A | Changes internal controller priority  |
|          |           |     | setting. Options are 'Watt', 'Var',   |
|          |           |     | or 'CPF'                              |
+----------+-----------+-----+---------------------------------------+
| Battery  | P         | kW  | Sets AC power setpoint                |
|          | Setpoint  |     |                                       |
+----------+-----------+-----+---------------------------------------+
| Battery  | SOC Rate  | 1/h | Sets AC power setpoint based on SOC   |
|          |           | our | rate, battery capacity, and           |
|          |           |     | efficiency                            |
+----------+-----------+-----+---------------------------------------+
| Battery  | Control   | N/A | Changes default control type. Options |
|          | Type      |     | are 'Schedule', 'Self-Consumption',   |
|          |           |     | and 'Off'                             |
+----------+-----------+-----+---------------------------------------+
| Battery  | P         | N/A | Dictionary of updated control         |
|          | arameters |     | parameters. See battery input args    |
|          |           |     | for details                           |
+----------+-----------+-----+---------------------------------------+
| Lighting | Load      | un  | Adjusts the scheduled power           |
| or Other | Fraction  | itl | consumption. Applied to electric and  |
|          |           | ess | gas power                             |
+----------+-----------+-----+---------------------------------------+
| Lighting | P         | kW  | Sets electric power setpoint          |
| or Other | Setpoint  |     |                                       |
+----------+-----------+-----+---------------------------------------+
| Lighting | Gas       | th  | Sets gas power setpoint               |
| or Other | Setpoint  | erm |                                       |
|          |           | s/h |                                       |
|          |           | our |                                       |
+----------+-----------+-----+---------------------------------------+

External Model Signals

OCHRE can also integrate with external models that modify default
schedule values and other settings.

The most common use case is to integrate with a grid simulator that
modifies the dwelling voltage. OCHRE includes a ZIP model for all
equipment that modifies the real and reactive electric power based on
the grid voltage.

The following code sends a voltage of 0.97 p.u. to a Dwelling model:

.. code-block:: python
    status = dwelling.update(ext_model_args={‘Voltage (-)’: 0.97})

External model signals can also modify any time series schedule values
including weather and occupancy variables. The names and units of these
variables can be found in the header of the schedule output file.
Alternatively, these variables can be reset at the beginning of the
simulation; see notebooks/… for more details.

Status Variables
----------------

The ``update`` function for equipment and dwellings returns a Python
dictionary with status variables that can be sent to the external
controller. These status variables are equivalent to the Time Series
Outputs described in Outputs and Analysis. Note that the ``verbosity``
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

Co-simulation
-------------

Multiple OCHRE instances have been run in co-simulation using the HELICS
platform. OCHRE models can communicate with other agents via its
external control signals, external model signals, and status variables.

See the publications list for examples of co-simulation architectures
that use OCHRE. We do not currently have public code for using OCHRE in
co-simulation.
