"""
Output registry defining which outputs are generated at each verbosity level.

Template patterns use {placeholder} syntax:
- {end_use}: Equipment end use (e.g., "HVAC Heating", "Water Heating", "EV")
- {results_name}: Equipment results name (usually same as end_use)
- {zone_name}: Zone name (e.g., "Indoor", "Foundation", "Garage")
- {boundary_name}: Boundary name (e.g., "Wall", "Window", "Roof")

This registry is a superset - not all outputs will be present in every simulation.
The enabled_outputs set is used to check if an output should be generated.
"""

# All possible placeholder values for pattern expansion
END_USES = [
    # HVAC
    "HVAC Heating",
    "HVAC Cooling",
    # Water Heating
    "Water Heating",
    # EV and Battery
    "EV",
    "Battery",
    # Solar and Generation
    "PV",
    "Generator",
    "Gas Generator",
    # Lighting
    "Lighting",
    "Indoor Lighting",
    "Exterior Lighting",
    "Garage Lighting",
    # Appliances
    "Refrigerator",
    "Freezer",
    "Dishwasher",
    "Clothes Washer",
    "Clothes Dryer",
    "Cooking Range",
    "Ceiling Fan",
    "TV",
    # Pool/Spa
    "Pool Heater",
    "Pool Pump",
    "Spa Heater",
    "Spa Pump",
    # Other
    "Well Pump",
    "Ventilation Fan",
    "MELs",
    "Other",
    # Individual equipment names that may differ from end_use
    "Electric Furnace",
    "Gas Furnace",
    "Electric Boiler",
    "Gas Boiler",
    "Air Source Heat Pump",
    "ASHP Heater",
    "ASHP Cooler",
    "Mini-split Heat Pump",
    "MSHP Heater",
    "MSHP Cooler",
    "Room AC",
    "Central AC",
    "Electric Resistance Water Heater",
    "Heat Pump Water Heater",
    "Gas Tankless Water Heater",
    "Gas Water Heater",
    # Test equipment (used in unit tests)
    "Test Equipment",
]

ZONE_NAMES = [
    "Indoor",
    "Foundation",
    "Garage",
    "Attic",
    "Outdoor",
]

BOUNDARY_NAMES = [
    "Wall",
    "Window",
    "Roof",
    "Floor",
    "Door",
    "Ceiling",
    "Slab",
    "Exterior Wall",
    "Interior Wall",
    "Partition Wall",
    "Ground",
    "Attic Floor",
    "Garage Wall",
    "Foundation Wall",
    "Rim Joist",
]

OUTPUT_REGISTRY = {
    "ochre": {
        0: [
            # Total power - always included
            "Total Electric Power (kW)",
            "Total Reactive Power (kVAR)",
            "Total Gas Power (therms/hour)",
            # EBM outputs - always available (controlled by save_ebm_results flag)
            "{end_use} EBM Energy (kWh)",
            "{end_use} EBM Min Energy (kWh)",
            "{end_use} EBM Max Energy (kWh)",
            "{end_use} EBM Max Power (kW)",
            "{end_use} EBM Efficiency (-)",
            "{end_use} EBM Baseline Power (kW)",
            "{end_use} EBM Max Discharge Power (kW)",
            "{end_use} EBM Discharge Efficiency (-)",
        ],
        1: [
            # Reserved for StateSpaceModel outputs (handled separately)
        ],
        2: [
            # End-use level power aggregation
            "{end_use} Electric Power (kW)",
            "{end_use} Gas Power (therms/hour)",
        ],
        3: [
            # Standard outputs - Indoor conditions, SOC, unmet loads
            "Temperature - Indoor (C)",
            "Unmet HVAC Load (C)",
            "Hot Water Unmet Demand (kW)",
            "Hot Water Outlet Temperature (C)",
            "{end_use} SOC (-)",
            "{end_use} Unmet Load (kWh)",
        ],
        4: [
            # Detailed equipment outputs - delivered heat, setpoints, efficiency
            "{end_use} Delivered (W)",
            "{end_use} Setpoint (C)",
            "{end_use} COP (-)",
            "Hot Water Delivered (L/min)",
            "Hot Water Delivered (W)",
            "{end_use} Parked",
        ],
        5: [
            # Component loads, zone temperatures
            "Temperature - {zone_name} (C)",
            "Net Sensible Heat Gain - {zone_name} (W)",
            "Infiltration Heat Gain - Indoor (W)",
            "Forced Ventilation Heat Gain - Indoor (W)",
            "Natural Ventilation Heat Gain - Indoor (W)",
            "Internal Heat Gain - Indoor (W)",
            "Window Transmitted Solar Gain (W)",
            "{boundary_name} Heat Gain - Indoor (W)",
            "{end_use} Duct Losses (W)",
        ],
        6: [
            # Energy totals, individual equipment power, setpoints
            "Total Electric Energy (kWh)",
            "Total Reactive Energy (kVARh)",
            "Total Gas Energy (therms)",
            "{results_name} Electric Power (kW)",
            "{results_name} Gas Power (therms/hour)",
            "{end_use} P Setpoint (kW)",
            "{end_use} Q Setpoint (kW)",
            "{end_use} Setpoint (kW)",
            "{end_use} Efficiency (-)",
        ],
        7: [
            # Debug outputs - modes, detailed equipment internals
            "{results_name} Mode",
            "{end_use} Main Power (kW)",
            "{end_use} Fan Power (kW)",
            "{end_use} ER Power (kW)",
            "{end_use} Latent Gains (W)",
            "{end_use} SHR (-)",
            "{end_use} Speed (-)",
            "{end_use} Capacity (W)",
            "{end_use} Max Capacity (W)",
            "{end_use} Total Sensible Heat Gain (W)",
            "{end_use} Deadband Upper Limit (C)",
            "{end_use} Deadband Lower Limit (C)",
            # Water heater details
            "Hot Water Heat Injected (W)",
            "Hot Water Heat Loss (W)",
            "Hot Water Average Temperature (C)",
            "Hot Water Maximum Temperature (C)",
            "Hot Water Minimum Temperature (C)",
            "Hot Water Mains Temperature (C)",
            # Heat pump water heater
            "{end_use} Heat Pump Max Capacity (W)",
            "{end_use} Heat Pump On Fraction (-)",
            "{end_use} Heat Pump COP (-)",
            # EV details
            "{end_use} Start Time",
            "{end_use} End Time",
            "{end_use} Remaining Charge Time (min)",
            # Battery details
            "{end_use} Energy to Discharge (kWh)",
            "{end_use} Nominal Capacity (kWh)",
            "{end_use} Actual Capacity (kWh)",
            "{end_use} Degradation State Q1",
            "{end_use} Degradation State Q2",
            "{end_use} Degradation State Q3",
            # Battery thermal model
            "Battery Temperature (C)",
        ],
        8: [
            # Reactive power, humidity, detailed zone data
            "{results_name} Reactive Power (kVAR)",
            "{end_use} Reactive Power (kVAR)",
            "Grid Voltage (-)",
            "Occupancy (Persons)",
            "Infiltration Flow Rate - {zone_name} (m^3/s)",
            "Infiltration Heat Gain - {zone_name} (W)",
            "Forced Ventilation Flow Rate - Indoor (m^3/s)",
            "Natural Ventilation Flow Rate - Indoor (m^3/s)",
            "Air Changes per Hour - Indoor (1/hour)",
            "Occupancy Heat Gain - Indoor (W)",
            "Internal Heat Gain - {zone_name} (W)",
            "Radiation Heat Gain - {zone_name} (W)",
            "Relative Humidity - {zone_name} (-)",
            "Wet Bulb - {zone_name} (C)",
            "Humidity Ratio - {zone_name} (-)",
            "Net Latent Heat Gain - {zone_name} (W)",
            "Air Density - {zone_name} (kg/m^3)",
        ],
        9: [
            # Full diagnostic - surface temps, radiation details
            "{boundary_name} Ext. Solar Gain (W)",
            "{boundary_name} Ext. LWR Gain (W)",
            "{boundary_name} Ext. Surface Temperature (C)",
            "{boundary_name} Ext. Film Coefficient (m^2-K/W)",
            "{boundary_name} {zone_name} LWR Gain (W)",
            "{boundary_name} {zone_name} Surface Temperature (C)",
            "{boundary_name} {zone_name} Film Coefficient (m^2-K/W)",
        ],
    },
    "resstock": {
        # ResStock format - all outputs needed for ResStock compatibility
        # Using level 0 to include everything needed for ResStock output
        0: [
            # Total power
            "Total Electric Power (kW)",
            "Total Gas Power (therms/hour)",
            # End-use power
            "{end_use} Electric Power (kW)",
            "{end_use} Gas Power (therms/hour)",
            # Temperatures
            "Temperature - Indoor (C)",
            "Temperature - Outdoor (C)",
            # Delivered loads
            "{end_use} Delivered (W)",
            "{end_use} Setpoint (C)",
            # Fan power (for HVAC fans/pumps)
            "{end_use} Fan Power (kW)",
            "{end_use} ER Power (kW)",
            # Humidity
            "Humidity Ratio - Indoor (-)",
            "Relative Humidity - Indoor (-)",
            # Airflow
            "Infiltration Flow Rate - Indoor (m^3/s)",
            "Forced Ventilation Flow Rate - Indoor (m^3/s)",
            "Natural Ventilation Flow Rate - Indoor (m^3/s)",
            # Water heating
            "Hot Water Heat Loss (W)",
            # EBM outputs
            "{end_use} EBM Energy (kWh)",
            "{end_use} EBM Min Energy (kWh)",
            "{end_use} EBM Max Energy (kWh)",
            "{end_use} EBM Max Power (kW)",
            "{end_use} EBM Efficiency (-)",
            "{end_use} EBM Baseline Power (kW)",
            "{end_use} EBM Max Discharge Power (kW)",
            "{end_use} EBM Discharge Efficiency (-)",
        ],
    },
}
