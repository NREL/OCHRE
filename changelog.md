## OCHRE V0.8.3-beta

- Initial Beta Release

### List of changes since v0.8.1 (private version)

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