# OCHRE: NREL's Object-oriented Controllable High-resolution Residential Energy Model

A high-fidelity, high-resolution residential building model with behind-the-meter DERs and flexible load models 
that integrates with controllers and distribution models in building-to-grid co-simulation platforms.

Contact: jeff.maguire@nrel.gov, michael.blonsky@nrel.gov, killian.mckenna@nrel.gov, dylan.cutler@nrel.gov

## Installation

### Stand-alone Installation

For a stand-alone installation, OCHRE can be installed using `pip` from the command line:

```
pip install --editable=git+https://github.com/NREL/OCHRE
```
 
Alternatively, you can download the repo and run the `setup.py` file:

```
python setup.py install
```

### In Co-simulation
To embed this in a co-simulation and a separate conda environment, create an `environment.yml` file in the co-simulation
project and include the following lines:
```
dependencies:
  - pip:
    - --editable=git+https://github.com/NREL/OCHRE
```


## Usage

To simulate a single dwelling, a set of input parameters and input files must be defined. For example, see the python
script in `bin/run_dwelling.py`. This section describes the required and optional input parameters and files.

### Initialization Arguments

Required arguments include:
* `name`: Name of the simulation
* `equipment_dict`: Dictionary of equipment to include in the dwelling (see below)
* `start_time`: Simulation start time as a datetime.datetime
* `time_res`: Simulation time resolution as a datetime.timedelta
* `duration`: Simulation duration as a datetime.timedelta
* `properties_file`: File name of BEopt properties file
* `schedule_file`: File name of BEopt schedule file
* `weather_file`: File name of weather file, typically a .epw file

Optional arguments include:
* `input_path`: Directory for input files (defaults to a built-in directory)
* `output_path`: Path to output files (defaults to a built-in directory)
* `initialization_time`: Duration to initialize the building temperatures as a datetime.timedelta (default is no 
initialization)
* `water_draw_file`: File name for water draw schedule file (default is no water draw)
* `save_results`: if True, saves results to output files (default is True)
* `verbosity`: Verbosity of the output files as integer from 1 to 9 (default is 1)
* `assume_equipment`: If True, some equipment is assumed from the properties file (default is False) 
* `uncontrolled_equipment`: List of equipment names to be considered as "Uncontrolled" for external controller 
communication

### Equipment-specific Arguments

The `equipment_dict` dictionary includes all equipment-specific initialization arguments that vary by type of equipment.
Dictionary keys are the equipment names, and values are dictionaries with equipment arguments. See `bin/run_dwelling.py`
for examples.

Below is a list of all equipment types and the equipment names:

* HVAC Heating:
  * Electric Furnace
  * Electric Baseboard
  * Electric Boiler
  * Gas Furnace
  * Gas Boiler
  * Heat Pump Heater
  * Air Source Heat Pump (ASHP Heater)
  * Minisplit Heat Pump (MSHP Heater)
  * Ideal Heater
* HVAC Cooling:
  * Air Conditioner
  * Room AC
  * Air Source Heat Pump (ASHP Cooler)
  * Minisplit Heat Pump (MSHP Cooler)
  * Ideal Cooler
* Water Heating:
  * Electric Resistance Water Heater
  * Heat Pump Water Heater
  * Gas Water Heater
  * Modulating Water Heater
  * Tankless Water Heater
  * Gas Tankless Water Heater
* DERs and Controllable Loads:
  * PV
  * Battery
  * Electric Vehicle (EV)
* Scheduled Loads:
  * Lighting
  * Exterior Lighting
  * Range
  * Dishwasher
  * Refrigerator
  * Clothes Washer
  * Clothes Dryer
  * MELs
  * Scheduled EV 


## Overview

OCHRE is an object-oriented residential building model that simulates a variety of behind-the-meter equipment.
It simulates dwelling energy consumption (electricity and gas) at a high resolution (up to 1-minute) and is designed 
to integrate in co-simulation with controllers, distribution systems, and other agents.
Most equipment types are controllable though an external controller to simulate the impact of device
controllers, HEMS, demand response, or other control strategies. The initialization integrates with BEopt output files 
to simplify the building modeling.

The key features of the code are:

* High-fidelity, high-resolution residential building simulation
* Simple integration with co-simulation using object-oriented principles
* Controllable equipment via external controllers
* Voltage-dependent electric power and reactive power using an equipment-level ZIP model
* Large variety of equipment types including HVAC, water heating, PV, batteries, and EVs
* Envelope, HVAC, and water heating validation with EnergyPlus (in progress)

OCHRE integrates with the following models and tools:
* BEopt (for generating input files)
* ResStock (for generating input files, in progress)
* HELICS (for co-simulation)
* Foresee (for HEMS control)
* SAM (for PV modeling)
* EVIpro (for EV modeling)
* Distribution models, e.g. OpenDSS, through co-simulation

The following sections describe the main modules in the code, including the envelope model, the water tank model, and
key equipment models.

### Envelope

The envelope model is a simplified RC model that tracks temperatures throughout the dwelling. The model 
is flexible and can handle multiple nodes and boundaries, including:

* Temperature Nodes
  * Living space (modeled as a single zone)
  * Garage
  * Attic
  * Foundation (basement or crawlspace)
* Boundaries
  * Exterior walls
  * Interior walls
  * Windows
  * Roof
  * Floor
  * Ceiling and gable walls (if attic exists)
  * Garage walls, roof, and floor (if garage exists)
  * Above- and below-ground foundation walls and ceiling (if foundation exists)

RC coefficients are determined from the BEopt properties file. Sensible and latent heat gains within the dwelling are 
taken from multiple sources:

* HVAC delivered heat
* Solar irradiance
* Infiltration and ventilation
* Internal and external long-wave radiation
* Occupancy and equipment heat gains

The envelope also includes a humidity model to estimate indoor humidity and wet bulb temperature.

### HVAC

HVAC equipment use three types of algorithms for determining equipment capacity and efficiency:

* Static capacity: System capacity and efficiency is set at initialization and does not change
(e.g Gas Furnace, Electric Baseboard)
* Dynamic capacity: System capacity and efficiency varies based on indoor and outdoor temperatures and air flow rate 
(e.g. Air Conditioner, Air Source Heat Pump)
* Ideal capacity: System capacity is calculated at each time step to maintain constant indoor temperature.
(e.g. Ideal Heater, Ideal Cooler)

Some HVAC models include multi-speed options, including single-speed, two-speed, and variable speed options.
One- and two-speed options typically use the dynamic capacity algorithm, while variable speed option typically uses the
ideal capacity algorithm.

Note that the Air Source Heat Pump includes heating and cooling functionality, and includes an electric resistance 
element that is enabled when outdoor air temperatures are below a threshold.

By default, all HVAC equipment are controlled using a thermostat control. Heating and cooling setpoints are defined in
the BEopt schedule file and can vary over time.

All HVAC equipment can be externally controlled by updating the thermostat setpoints and deadband or by direct load 
control (i.e. shut-off). Static and dynamic HVAC equipment can also be controlled using duty cycle control. The 
equipment will follow the external control exactly while minimizing temperature deviation from setpoint and minimizing
cycling.

### Water Heating

The water tank model is an RC model that tracks temperature throughout the tank. It is a flexible model that can handle
multiple nodes in the water tank. Currently, a 12-node, 2-node, and 1-node model are implemented. RC coefficients are
derived from the BEopt properties file.

The tank model accounts for internal and external conduction and heat flows from water draws, and includes an algorithm
to simulate temperature inversion mixing. The model can handle regular and tempered water draws. A separate water draw
file is currently required to set the water draw profile. 

Water heaters with a tank follow a similar structure to HVAC equipment. For example, the Electric Resistance Water
Heater has a static capacity, while the Heat Pump Water Heater has a dynamic capacity (and a backup electric resistance
element similar to the Air Source Heat Pump). Tankless water heaters operate similarly to Ideal HVAC equipment.

Similar to HVAC equipment, water heater equipment has a thermostat control, and can be externally controlled by 
updating the thermostat setpoints and deadband, specifying a duty cycle, or direct shut-off. Tankless equipment can only
be controlled through thermostat control and direct-shut-off.

### PV

PV is modeled using PySAM, a python wrapper for the System Advisory Model (SAM). Standard values are used for the PV
model, although the user can select the PV system capacity, the tilt angle, and the orientation.

PV can be externally controlled through a direct setpoint for real and reactive power. The user can define an inverter
size and a minimum power factor threshold to curtail real or reactive power. Watt- and Var-priority modes are 
available.

### Batteries

The battery is modeled as a linear system with typical assumptions for system capacity, power capacity, and efficiency.
The model tracks battery state-of-charge and maintains upper and lower SOC limits. It tracks AC and DC power,
and reports losses as sensible heat to the building envelope. It can also model self-discharge.
  
The model includes a schedule-based battery controller and a self-consumption controller. The schedule-based controller
runs a daily charge and discharge schedule, where the user can define the charging and discharging start times and 
power setpoints. The self-consumption controller sets the battery power setpoint to the opposite of the house net load
(including PV) to achieve zero grid import and export. There is also an option to only allow battery charging from PV.
The battery will follow that control until the SOC limits are reached.    

The battery can be externally controlled through a direct setpoint for real power. 

### Electric Vehicles

Electric vehicles are modeled using an event-based model and a charging event dataset from EVI-Pro. EV parking events
are randomly generated using the EVI-Pro dataset for each day of the simulation. One or more events may occur per day.
Each event has a prescribed start time, end time, and starting SOC. When the event starts, the EV will charge using
a linear model similar to the battery model described above.

Electric vehicles can be externally controlled through a delay signal or a direct power signal. A delay signal will 
delay the start time of the charging event. The direct power signal will set the charging power directly at each time
step, and is suggested for Level 2 charging only.  

### Scheduled Equipment

A wide variety of scheduled equipment are available. Equipment schedules are defined in the BEopt schedule file or in a 
separate input file. Schedules from the BEopt file include power output as well as sensible and latent heat gains. 

Scheduled equipment are typically not controlled, but can be externally controlled using a load fraction. For example,
a co-simulation can set the load fraction to zero to simulate an outage or a resiliency use case. 

### FUTURE: Wet appliances

Wet appliances, including clothes washers, clothes dryers, and dishwashers are not currently developed. We plan to
develop an event-based model for these appliances in the future, similar to the EV event-based model.


## File Structure

The basic file structure of code is described below.

* `bin/`: contains all executable python files (i.e. scripts)
* `ochre/`: contains all underlying python files to run the model
  * `Equipment/`: contains all equipment classes
  * `Model/`: contains all model classes, including the building envelope and water tank
  * `Dwelling.py`: main python file for defining a dwelling object
* `defaults/`: contains default input files
* `test/`: contains all files for running unit tests

### Input Files

A number of input files are required to run the simulation. An example file structure is included in `defaults/`. It is
recommended to copy files from this folder to a separate location to make changes in inputs. This folder includes:

* `BEopt_Files/`: contains BEopt properties and schedule files
* `Weather/`: contains .epw weather files
* `<equipment_name>/`: contains all equipment-specific files
* `ZIP_loads.csv`: includes voltage-dependency ZIP data for all equipment


## Glossary of Building-Related Terms

### Equipment:
A catch all term that could refer to HVAC, water heaters, appliances, and other devices that consume and/or generate
energy.

### Appliance:
Large pieces of equipment that serve a load OTHER than water heating or HVAC. Examples include clothes washers, clothes
dryers, dishwashers, refrigerators, and stoves/ranges

### HVAC (Heating, Ventilation, and Air Conditioning):
Any piece of equipment that serves a building's heating and cooling load. Common examples for residential buildings 
include:
* Heating: Furnaces, boilers, electric baseboards, heat pumps
* Cooling: Air conditioner (central or room, AC), heat pumps
* Dehumidification: Dehumidifiers can either be stand alone units or in line with other pieces of HVAC equipment (i.e.
ducted). Most cooling equipment also dehumidifies and may be controlled to meet dehumidification requirements.

### Heat Pump (HP):
A piece of HVAC equipment that serves both heating and cooling loads. Heat pumps work similar to an air conditioner
that's reversible. Examples of residential heat pumps include:
* Air Source Heat Pump (ASHP): A large, central heat pump that uses the outdoor air as a heat source/sink. Depending on
the efficiency of the heat pump (denoted by SEER for cooling and HSPF for heating), ASHPs can either be single speed,
two speed, or variable speed. Higher efficiency units tend to use more speeds.
* Mini Split Heat Pump (MSHP): A smaller heat pump similar in appearance to a room air conditioner. MSHPs are a common
retrofit for efficiency, particularly in heating dominated climates. MSHPs are sometimes also referred to as "ductless
heat pumps", although they can be ducted in certain situations.
* Ground Source Heat Pump (GSHP): A large, central heat pump that uses the ground temperature as a heat source/sink.
GSHPs tend to be efficient year round due to the relatively stable ground temperature, but the cost associated with
installing a ground loop makes this HVAC equipment much less common.

### Ideal HVAC:
HVAC equipment with a constant efficiency of 1.0 for heating or cooling. This equipment exactly meets the load to
maintain the setpoint. These aren't real pieces of equipment (although electric baseboards can do this for heating), but
are primarily used for debugging purposes.

### Water Heater (WH):
A piece of equipment primarily used to heat water for domestic hot water (DHW) use. Water heaters can either be fueled
by electricity or natural gas. Electric water heaters can either use electric resistance elements, a heat pump (in
HPWHs), or some combination to meet the water heating load.

### Miscellaneous Electric Loads (MELs)
The small end uses in the home. Examples of MELs include TVs, cable boxes, microwaves and other small appliances, and 
chargers.

### Internal Gain:
A heat gain in the space. Internal gains are caused by:
* Appliances
* Occupants
* Lighting
* MELS

Internal gains are not always equal to the power consumption of everything that could become an internal gain in the
home. The power consumed can become either:
* Sensible gain: Change in the sensible heat content in the space. This causes a change in temperature.
* Latent gain: Change in the humidity of the space. This causes a change in humidity.
* Lost: Power consumption that doesn't impact the indoor conditions. Energy can be lost due to some heat being vented
outside the building or otherwise ending up outside of the building envelope.

### Minimal Building:
A building used for debugging purposes that is designed to have a minimal amount of energy consumption. The minimal
building has:
* No internal gains
* No infiltration/ventilation
* No windows
* "Superinslated (R~1000)" constructions on the floor, walls, and ceiling.
* Ideal HVAC
* No solar absorptivity
* No water heater

The minimal building is primarily used for debugging. Adding in one realistic feature at a time helps isolate that
particular feature for testing purposes.

### Film resistance:
A thermal resistance due to convection heat transfer coefficients between a surface and the ambient air. Film
resistances may vary based on temperature and wind speed, but are currently modeled as constant.

### Infiltration:
Airflow through the building envelope due to leaks and cracks in the building envelope. Infiltration rates depend on the
indoor temperature, outdoor temperature, and local wind speed.

### Ventilation:
Airflow through the building due to an active attempt to increase airflow. This can either be mechanical (driven by
fans) or natural (driven by opening windows). Most buildings, unless they have a high infiltration rate, require some
mechanical ventilation.
