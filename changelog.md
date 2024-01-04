## OCHRE Changelog

### Changes from PRs

- Fixed bug with accounting for HVAC delivered heat for standalone HVAC runs 
- Fixed bug with ASHP backup heater units
- Added OCHREException class to handle errors

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
