from .Equipment import Equipment, EquipmentException
from .ScheduledLoad import ScheduledLoad
from .EventBasedLoad import EventBasedLoad, DailyLoad
from .HVAC import *
from .WaterHeater import *
from .Generator import Generator, GasGenerator, GasFuelCell
from .PV import PV
from .Battery import Battery
from .EV import ElectricVehicle, ScheduledEV

# from .WetAppliance import WetAppliance
ALL_EQUIPMENT = [
    # HVAC Heating
    Heater, ElectricFurnace, ElectricBaseboard, ElectricBoiler, GasFurnace, GasBoiler, HeatPumpHeater, ASHPHeater,
    MinisplitAHSPHeater,

    # HVAC Cooling
    Cooler, AirConditioner, ASHPCooler, RoomAC, MinisplitAHSPCooler,

    # Water Heating
    ElectricResistanceWaterHeater, HeatPumpWaterHeater, GasWaterHeater,
    TanklessWaterHeater, GasTanklessWaterHeater,

    # EV
    ElectricVehicle,
    ScheduledEV,

    # DERs
    PV, Battery, Generator, GasGenerator, GasFuelCell
]

scheduled_load_names = ['Lighting', 'Exterior Lighting', 'Garage Lighting', 'Basement Lighting',
                        'Range', 'Dishwasher', 'Refrigerator', 'Clothes Washer', 'Clothes Dryer',
                        'MELs', 'Basement MELs', 'Ventilation Fan']

EQUIPMENT_BY_NAME = {
    'Electric Vehicle': ElectricVehicle,
    **{equipment.name: equipment for equipment in ALL_EQUIPMENT},
    **{name: ScheduledLoad for name in scheduled_load_names}
    # Wet appliances
    # 'Clothes Washer Ctrl': WetAppliance,
    # 'Clothes Dryer Ctrl': WetAppliance,
    # 'Dishwasher Ctrl': WetAppliance,
}

ALL_END_USES = ['HVAC Heating', 'HVAC Cooling', 'Water Heating', 'EV', 'PV', 'Gas Generator', 'Battery', 'Other']
assert all([e.end_use in ALL_END_USES for e in ALL_EQUIPMENT])

EQUIPMENT_BY_END_USE = {
    end_use: [name for name, e in EQUIPMENT_BY_NAME.items() if e.end_use == end_use] for end_use in ALL_END_USES
}
