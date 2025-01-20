Modeling Approach
=================

This section describes the modeling approach for each piece of equipment and
for the building envelope. OCHRE models the energy usage of each piece of
equipment in a dwelling and the electrical and thermal dynamics of various
systems. The equipment models are mostly independent of each other, but there
are some interactions as described below.  

Envelope
--------

The envelope model is a simplified resistor-capacitor (RC) model that tracks
temperatures throughout the dwelling. The model is flexible and can handle
multiple zones and boundaries, including:

-  Temperature Zones

   -  Living space

   -  Garage

   -  Attic

   -  Foundation (conditioned basement, unconditioned basement, or
      crawlspace)

-  Boundaries

   -  Exterior walls

   -  Interior walls and furniture

   -  Windows and doors to living space

   -  Roof (flat or tilted)

   -  Floor (slab or raised floor)

   -  Ceiling and gable walls (if attic exists)

   -  Garage walls, door, ceiling, roof, and floor (if garage exists)

   -  Foundation walls, slab, ceiling, and rim joists (if foundation
      exists)

   -  Walls between adjacent units in multifamily buildings

Each boundary is modeled using a resistance/capacitance (RC) network. OCHRE
treats each individual material within the boundary separately, with a
capacitor representing the thermal mass of the material and resistors with
half of the overall material resistance on each side of the capacitor.
Convection, solar radiation, and thermal (long-wave) radiation are accounted
for at both surfaces of each boundary. Convection is incorporated using
constant film coefficients that are based on the orientation of the surface
and its location (interior or exterior). Radiation is treated as heat added to
the surface and is calculated using the surface temperature, the temperature
of other surfaces in the connected zone, and view factors for each surface.
External surface radiation incorporates the ambient temperature and the sky
temperature. The figures below show a a high-level overview of the heat
transfer pathways and an example of how a boundary is converted into an RC
network.

.. image:: images/Heat_Transfer_Pathways.png
  :width: 500
  :alt: Schematic of a home with all optional zones and heat transfer pathways

.. image:: images/Wall_RC_Network.png
  :width: 800
  :alt: Schematic of how RC networks are generated for each surface

Thermal resistance and capacitance coefficients are determined from the HPXML
file and are based on values from EnergyPlus input/output (.eio) files. Some
coefficients (in particular for slabs and foundation walls) have been modified
based on validation efforts to more closely match EnergyPlus results.

The full RC network for the building is generated dynamically depending on
what features are included in the building. For example, a building with a
slab on grade and without an attic or garage will be modeled as a single zone.
OCHRE will generate more complicated RC networks if multiple zones are
included in the building. Additional zones are used to model attics, basements
or crawlspaces, and garages. The figure below shows the most complicated RC
network in OCHRE, where an attic, crawlspace/basement, and garage are all
included in the building.

.. image:: images/RC_network.png
  :width: 500
  :alt: The full RC network for a building. Each rectangle represents the RC network shown in Figure 1

OCHRE includes the capability to model multifamily buildings using a unit by
unit-based approach. Each unit is modeled as a separate dwelling unit with
adiabatic surfaces separating different units. OCHRE does not currently
support modeling a whole multifamily building with multiple units
simultaneously or the modeling of central space and water heating systems.

Thermal mass due to furniture and interior partition walls is also accounted
for in the living space. Partition walls and furniture are modeled explicitly
with surface areas and material properties like any other surface and exchange
heat through both convection and radiation. The heat capacity of the air is
also modeled to determine the living zone temperature. However, a multiplier
is generally applied to this capacitance. `Numerous studies
<https://docs.google.com/spreadsheets/d/1ebSmvDFdXEXVRdvkzqMF1C9MwHrHCQKFF75QMkPgd7A/edit?pli=1#gid=0>`__
have shown that applying a multiplier to the air capacitance provides a much
better match to experimental data when trying to model explicit cycling of the
HVAC equipment conditioning the living space. This multiplier helps account
for the volume of ducts and the time required for warm and cold air to diffuse
through the living space. Values for this multiplier in the literature range
from 3-15 depending on the study. OCHRE uses a default multiplier of 7.

The envelope includes a humidity model for the living space zone. The model
determines the indoor humidity and wet bulb temperature based on a mass
balance. Moisture can be added or removed from the space based on airflow from
outside through infiltration and ventilation, internal latent gains from
appliances and occupants, and latent cooling provided by HVAC equipment. OCHRE
does not currently include a dehumidifier or other models to control indoor
humidity.

Sensible and latent heat gains within the dwelling are taken from multiple
sources:

-  Conduction between zones and material layers

-  Convection and long-wave radiation from zone surfaces

-  Infiltration, mechanical ventilation, and natural ventilation

-  Solar irradiance, including absorbed and transmitted irradiance
   through windows

-  Occupancy and equipment heat gains

-  HVAC delivered heat, including duct losses and heat delivered to the
   basement zone

HVAC
----

OCHRE models several different types of heating, ventilation, and air
conditioning (HVAC) technologies commonly found in residential buildings in
the United States. This includes furnaces, boilers, electric resistance
baseboards, central air conditioners (ACs), room air conditioners, air source
heat pumps (ASHPs), and minisplit heat pumps (MSHPs). OCHRE also includes
“ideal” heating and cooling equipment models that perfectly maintain the
indoor setpoint temperature with a constant efficiency.

HVAC equipment use one of two algorithms to determine equipment max capacity
and efficiency:

-  Static: System max capacity and efficiency is set at initialization and
   does not change (e.g., Gas Furnace, Electric Baseboard).

-  Dynamic: System max capacity and efficiency varies based on indoor and
   outdoor temperatures and air flow rate using biquadratic formulas. These
   curves are based on `this paper
   <https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/r781wg40j>`__.

In addition, HVAC equipment use one of two modes to determine real-time
capacity and power consumption:

-  Thermostatic mode: A thermostat control with a deadband is used to turn the
   equipment on and off. Capacity and power are zero or at their maximum
   values.

-  Ideal mode: Capacity is calculated at each time step to perfectly maintain
   the indoor setpoint temperature. Power is determined by the fraction of
   time that the equipment is on in various modes.

By default, most HVAC equipment operate in thermostatic mode for simulations
with a time resolution of less than 5 minutes. Otherwise, the ideal mode is
used. The only exceptions are variable speed equipment, which always operate
in ideal capacity mode.

ASHPs, central ACs, and room ACs include single-speed, two-speed, and variable
speed options. MSHPs are always modeled as variable speed equipment.

The ASHP and MSHP models include heating and cooling functionality. The heat
pump heating model includes a few unique features:

-  An electric resistance backup element with additional controls, including
   an offset thermostat deadband.
-  A heat pump shut off control when the outdoor air temperature is below a
   threshold.
-  A reverse cycle defrost algorithm that reduces heat pump efficiency and
   capacity at low temperatures.

All HVAC equipment can be externally controlled by updating the thermostat
setpoints and deadband or by direct load control (i.e., shut-off). Specific
speeds can be disabled in multi-speed equipment. Equipment capacity can also
be set directly or controlled using a maximum capacity fraction in ideal mode.

Ducts
~~~~~

Ducts are modeled using a Distribution System Efficiency (DSE) based approach.
DSE values are calculated according to `ASHRAE 152
<https://webstore.ansi.org/standards/ashrae/ansiashrae1522004>`__ and
represent the seasonal DSE in both heating and cooling. The DSE is affected by
the location, duct length, duct insulation, and airflow rate through ducts.
Sensible heat gains and losses associated with the ducts do end up in the
space the ducts are primarily located in and affect the temperature of that
zone. Changes in humidity in these zones due to duct losses are not included.

For homes with a finished basement, this zone has a separate temperature from
the living zone and does not have it's own thermostat. Instead, a fixed
fraction of the space heating to be delivered to the zone is diverted into the
basement. This approximates having dampers with a fixed position in a home
with a single thermostat. OCHRE currently assumes that 20% of space heating
energy goes to a finished basement.

Water Heating
-------------

OCHRE models electric resistance and gas tank water heaters, electric and gas
tankless water heaters, and heat pump water heaters (HPWHs).

In tank water heaters, stratification occurs as cold water is brought into the
bottom of the tank and buoyancy drives the hottest water to the top of the
tank. OCHRE's stratified water tank model captures this buoyancy using a
multi-node RC network that tracks temperatures vertically throughout the tank
and an algorithm to simulate temperature inversion mixing (i.e.,
stratification). The tank model also accounts for internal and external
conduction, heat flows from water draws, and the location of upper and lower
heating elements when determining tank temperatures. RC coefficients are
derived from the physical properties of the tank.

The tank model can handle multiple nodes, although 12-node, 2-node, and 1-node
models are currently implemented. The 1-node model ignores the effects of
stratification and maintains a uniform temperature in the tank. This model is
best suited for large timesteps.

Similar to HVAC equipment, electric resistance and gas heating elements are
modeled with static capacity and efficiency. The electric resistance model
includes upper and lower heating elements and two temperature sensors for the
thermostatic control.

In HPWHs, the heat pump capacity and efficiency are functions of the ambient
air wet bulb temperature (calculated using the humidity module in OCHRE) and
the temperature of water adjacent to the condenser (typically the bottom half
of the tank in most products on the market today). The model also includes an
electric resistance backup element at the top of the tank.

Tankless water heaters operate similarly to Ideal HVAC equipment, although an
8% derate is applied to the nominal efficiency of the unit to account for
cycling losses in accordance with ANSI/RESNET 301.

The model accounts for regular and tempered water draws. Sink, shower, and
bath water draws are modeled as tempered (i.e., the volume of hot water
depends on the outlet temperature), and appliance draws are modeled as regular
(i.e., the volume is fixed).

Similar to HVAC equipment, water heater equipment has a thermostat control,
and can be externally controlled by updating the thermostat setpoints and
deadband or by direct shut-off.

Electric Vehicles
-----------------

Electric vehicles are modeled using event-based data. EV parking events are
randomly generated using event-based datasets for each day of the simulation.
Zero, one, or more events may occur per day. Each event has a prescribed start
time, end time, and starting state-of-charge (SOC). When the event starts, the
EV will charge using a linear model similar to the battery model described
below.

OCHRE's default event-based datasets are taken from `EVI-Pro
<https://www.nrel.gov/transportation/evi-pro.html>`__. Additional datasets
used for the `2030 National Charging Network
<https://www.nrel.gov/docs/fy23osti/85654.pdf>`__ study may be available upon
request.

Electric vehicles can be externally controlled through a delay signal, a
direct power signal, or charging constraints. A delay signal will delay the
start time of the charging event. A direct power signal (in kW, or SOC rate)
will set the charging power directly at each time step, and it is only
suggested for Level 2 charging. Max power and max SOC contraints can also
limit the charging rate and can optionally be set as a schedule.

Batteries
---------

The battery model incorporates standard battery parameters including battery
energy capacity, power capacity, and efficiency. The model tracks battery SOC
and maintains upper and lower SOC limits. It tracks AC and DC power, and it
can report losses as sensible heat to the building envelope. It can also model
self-discharge.

The battery model can optionally track internal battery temperature and
battery degradation. Battery temperature is modeling using a 1R-1C thermal
model and can use any envelope zone as the ambient temperature. The battery
degradation model tracks energy capacity degradation using temperature and SOC
data and a rainflow algorithm.

The battery model can be controlled through a direct power signal or using a
self-consumption controller. Direct power signals (or desired SOC setpoints)
can be included in the schedule or sent at each time step. The
self-consumption controller sets the battery power setpoint to the opposite of
the house net load (including PV) to achieve desired grid import and export
limits (defaults are zero, i.e., maximize self-consumption). The battery will
follow these controls while maintaining SOC and power limits. There is also an
option to only allow battery charging from PV. There is currently no reactive
power control for the battery model.

Solar PV
--------

Solar photovoltaics (PV) is modeled using `PySAM
<https://pysam.readthedocs.io/en/latest/api.html>`__, a python wrapper for the
System Advisory Model (`SAM <https://sam.nrel.gov/>`__), using the PVWatts
module. SAM's default values are used for the PV model, although the user must
select the PV system capacity and can specify the tilt angle, azimuth, and
inverter properties.

PV can be externally controlled through a direct setpoint for real and
reactive power. The user can define an inverter size and a minimum power
factor threshold to curtail real or reactive power. Watt- and Var-priority
modes are available.

Generators
----------

Gas-based generators and fuel cells can be modeled for resilience analysis.
These models include power capacity and efficiency parameters similar to the
battery model. Control options are also similar to the battery model.

Other Loads
-----------

OCHRE includes many other common end-use loads that are modeled using a
load profile schedule. Load profiles, as well as sensible and latent
heat gain coefficients, are included in the input files. These loads can
be electric or natural gas loads. Schedule-based loads include:

-  Appliances (clothes washer, clothes dryer, dishwasher, refrigerator,
   cooking range)

-  Lighting (indoor, exterior, garage, basement)

-  Ceiling fan and ventilation fan

-  Pool and spa equipment (spump and heaters)

-  Miscellaneous electric loads (television, small kitchen appliances, other)

-  Miscellaneous gas loads (grill, fireplace, lighting)

These loads are not typically controlled, but they can be externally
controlled using a load fraction. For example, a user can set the load
fraction to zero to simulate an outage or a resilience use case.

Equipment Interactions
----------------------

Equipment models in OCHRE are mostly independent of each other, but there are
some electrical and thermal interactions between them. These interactions are
modeled when simulating a full dwelling; they will not be modeled when
simulating a single piece of equipment unless explicitly specified by the
user. Interactions include:

- Many equipment models have sensible and latent heat gains that impact the
  envelope temperatures and therefore impact the energy consumption of the
  HVAC system. For this reason, we do not recommend running HVAC equipment
  without a full dwelling model.

- HVAC equipment, water heaters, and batteries can have some variables that
  depend on envelope temperatures. Envelope tempreatures may need to be
  specified when simulating water heaters, and batteries by themselves.

- Battery and generator energy use will depend on other equipment energy use
  when running in self-consumption mode. They can impact the energy use of
  other equipment when running in islanded mode by forcing the power to shut
  off if they can't provide enough backup power. Batteries can also be
  constrained to only charge from net or gross solar generation. 

Co-simulation
-------------

OCHRE is designed to be run in co-simulation with controllers, aggregators,
grid models, and other agents. The inputs and outputs of key functions are
designed to connect with these agents for streamlined integration. See
`Controller Integration
<https://ochre-nrel.readthedocs.io/en/latest/ControllerIntegration.html>`__
and `Outputs and Analysis
<https://ochre-nrel.readthedocs.io/en/latest/Outputs.html>`__ for details on
the inputs and outputs, respectively.

See `here <https://github.com/NREL/OCHRE/blob/main/bin/run_cosimulation.py>`__
for a simple example that implements OCHRE in co-simulation using HELICS.
There are also co-simulation examples in our `publications list
<https://ochre-nrel.readthedocs.io/en/latest/Introduction.html#citation-and-publications>`__

Unsupported Features
--------------------

While OCHRE is intended to work with OS-HPXML and files created through either
BEopt or ResStock, not every feature in those tools is currently supported in
OCHRE. Features not currently supported are generally lower priority features
that are considered future work. Depending on the impact of the feature, OCHRE
should either return a warning or error when an HPXML file including these
options is supplied. Warnings are used if the option is likely to have a
minimal impact on energy results (such as eaves) and errors are used for a
feature with a substantial impact (such as a ground source heat pump). **Note
that correctly throwing warnings and errors is currently under development.**
The current list of technologies not supported in OCHRE is:

-  Eaves

-  Overhangs

-  Cathedral ceilings

-  Structural Insulated Panel (SIP) walls

-  Ground source heat pumps

-  Fuels other than electricity, natural gas, propane, or oil

   -  Propane and oil equipment are converted to natural gas

-  Dehumidifiers

-  Solar water heaters

-  Desuperheaters
