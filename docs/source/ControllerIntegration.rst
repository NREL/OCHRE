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

HVAC Heating or HVAC Cooling
----------------------------
================================  ==========  ========================================================================= 
Control Command                   Units       Description     
================================  ==========  ========================================================================= 
Load Fraction                     unitless    1 (no effect) or 0 (force equipment off)
Setpoint                          C           Sets temperature setpoint for one timestep (then reverts to schedule)
Deadband                          C           Sets thermostat deadband (does not revert unless deadband is scheduled)
Duty Cycle                        unitless    Sets the equipment duty cycle for ``ext_time_res``
Disable Speed X                   unitless    Disables low (X=1) or high (X=2) speed if value is ``True`` [#]_
================================  ==========  =========================================================================

.. [#] Only available for 2 speed equipment, either ASHP or AC. Variable speed equipment modulates between all speeds to
         perfectly maintain setpoint ( deadband = 0 C)

Water Heating
-----------------------------
================================  ==========  ========================================================================= 
Control Command                   Units       Description     
================================  ==========  ========================================================================= 
Load Fraction                     unitless    1 (no effect) or 0 (force equipment off)
Setpoint                          C           Sets temperature setpoint for one timestep. [#]_
Deadband                          C           Sets temperature deadband (does not reset) [#]_
Duty Cycle                        unitless    Sets the equipment duty cycle for ``ext_time_res``
HP Duty Cycle                     unitless    Sets the heat pump duty cycle for a heat pump water heater
ER Duty Cycle                     unitless    Sets the electric resistance duty cycle for a heat pump water heater [#]_
================================  ==========  =========================================================================

.. [#] Sending {'Setpoint': None} will reset the setpoint to the default schedule. Note that a 10 F (5.56 C)
       decrease in setpoint corresponds to a CTA-2045 'Load Shed' command. A 10 F increase corresponds to an
       'Advanced Load Add' command (only available in B version of standard).
.. [#] Decreasing the deadband to about 2 C corresponds to a CTA 'Load Add' command.
.. [#] Most, but not all HPWHs have backup electric resistance. 120 V HPWHs (coming soon in OCHRE) do not
         have backup ER heaters.

Electric Vehicle (EV)
-----------------------------

================================  ==========  ========================================================================================================= 
Control Command                   Units       Description     
================================  ==========  =========================================================================================================
Delay                             unitless    Delay EV chage for a given time. Value can either be ``datetime.timedelta`` or integer for # of timesteps
P Setpoint                        kW          Set real AC power setpoint
SOC Rate                          1/hour      Set AC power setpoint based on SOC rate, EV capacity, and efficiency of charging
================================  ==========  =========================================================================================================

Photovoltaics (PV)
-----------------------------

================================  ==========  ========================================================================================================= 
Control Command                   Units       Description     
================================  ==========  =========================================================================================================
P Setpoint                        kW          Sets real AC power setpoint
P Curtailment (kW)                kW          Set real power setpoint by specifying absolute curtailment
P Curtailment (%)                 %           Set real power setpoint by specifying curtailment relative to maximum power point
Q Setpoint                        kVar        Set reactive power setpoint
Power Factor                      unitless    Set reactive power setpoint based on power factor
Priority                          N/A         Changes internal controller priority setting. Options are ``Watt``, ``Var``, or ``CPF`` [#]_
================================  ==========  =========================================================================================================

.. [#] CPF: Constant Power Factor

Battery
-----------------------------

================================  ==========  ========================================================================================================= 
Control Command                   Units       Description     
================================  ==========  =========================================================================================================
P Setpoint                        kW          Sets AC power setpoint
SOC Rate                          1/hour      Set AC power setpoint based on SOC rate, battery capacity, and efficiency
Control Type                      N/A         Change default control type. Supported options are ``Schedule``, ``Self-Consumption`` [#]_, and ``Off``
Parameters                        N/A         Dictionary of updated control parameters. See battery input arguments for details.
================================  ==========  =========================================================================================================

.. [#] 'Self-Consumption' mode, sometimes referred to as minimizing grid import, only applies for homes with PV and a battery.
         This strategy will charge the battery when PV production is larger than electricty consumption and vice versa.

Lighting and Other
-----------------------------
================================  ============  ============================================================================ 
Control Command                   Units         Description                                                                 
================================  ============  ============================================================================
Load Fraction                     unitless      Adjust the scheduled power consumption. Can apply to both electric and gas  
P Setpoint                        kW            Set electric power setpoint                                                 
Gas Setpoint                      therms/hour   Set gas power setpoint [#]_                                                   
================================  ============  ============================================================================
.. [#] Most useful for modeling backup gas generators

External Model Signals
------------------------------

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
bin/run_external_control.py and notebooks/user_tutorial.ipynb

Example Use Case – Equipment
----------------------------

The following code creates a water heater model and runs a simulation
that controls the water heater setpoint. For more details and examples,
see bin/run_external_control.py and notebooks/user_tutorial.ipynb

Co-simulation
-------------

Multiple OCHRE instances have been run in co-simulation using the HELICS
platform. OCHRE models can communicate with other agents via its
external control signals, external model signals, and status variables.

See the publications list for examples of co-simulation architectures
that use OCHRE. We do not currently have public code for using OCHRE in
co-simulation.
