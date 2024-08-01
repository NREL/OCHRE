## OCHRE Changelog

### New from PRs

- Added multi-speed HVAC parameters for ResStock 2024 dataset

### OCHRE v0.8.5-beta

- Updated PV model to integrate with PVWatts using PySAM v5.0 (not backwards compatible)
- Removed and renamed PV input arguments related to PySAM
- Added HVAC capacity and max capacity controls, ideal mode only
- Require HVAC duty cycle control for thermostatic mode only
- Added water heater max power control
- Added EV max power and max SOC controls
- Added `equipment_event_file` input for EVs
- Added OCHREException class to handle errors
- Added warnings for HVAC and WH heat pumps with low COP
- Moved default input file path for package installation
- Replaced setup.py with pyproject.toml
- Fixed bug with schedule file import using Pandas v2.2
- Fixed bug with accounting for HVAC delivered heat for standalone HVAC runs 
- Fixed bug with ASHP backup heater units
- Fixed bug with named HVAC/Water Heating equipment arguments
- Fixed bug in ASHP duty cycle control
- Fixed bug with accounting for HVAC delivered heat for standalone HVAC runs 
- Fixed bug with ASHP backup heater units
- Fixed bug with battery/generator self-consumption controls
- Fixed bug with WH and battery islanding time metrics
- Fixed bug with garage area outside of typical building rectangle
- Fixed bug with state space model reduction algorithm
- Fixed syntax warning for Python 3.12

### OCHRE v0.8.4-beta

- Fixed bug with air infiltration inputs (works with ResStock 3.0 and 3.1, and OS-HPXML 1.6.0)
- Fixed bug with 2-speed HVAC control
- Fixed bug with nested dictionary arguments for Envelope zones and boundaries
- Fixed bug when setting ScheduledLoad power to 0
- Fixed bug with Lighting end use power and total power figures
- Fixed floor area check for garage geometry. 
- Removed requirement for HVAC setpoint schedule in HPXML file
- Added garage door boundary. Uses the same material as a regular door
- Added check to explicitly not handle garage windows, attic windows, and attic doors
- Added errors for unknown zones (mainly for foundation windows and doors)

### OCHRE v0.8.3-beta (initial beta release)

- Compatible with OS-HPXML 1.5.0 and 1.6.1 (includes ResStock 3.0, ResStock 3.1, and BEopt 3.0)
- Incorporated adjusted number of bedrooms calculation
- Changed end use for lighting loads
- Added garage roof material properties
- Made "OCHRE" the default Simulator name
- Fixed bug in 2-speed HVAC controls
- Fixed bug in calculating cooling DSE
- Fixed bug in calculating indoor zone infiltration
- Fixed bug with HPWH ER-only mode control and sensible gains
- Fixed issue with garage geometry and added warnings
- Fixed issue importing garage lighting
- Fixed issue importing adiabatic doors
- Fixed issue importing HPXML AirInfiltrationMeasurement
- Updated tankless water heater derate factor
- Fixed potential memory issue for saving results
- Replaced private file paths and Github Enterprise paths
- Removed old scripts and notebooks
- Removed validation files
- Added (some) component loads for validation
- Added comparison metrics for validation
- Added OS-HPXML sample files for testing (testing not started)
- Added documentation files and pushed to readthedocs
- Added PR template
- Added changelog
