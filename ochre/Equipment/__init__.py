from .Equipment import Equipment
from .ScheduledLoad import ScheduledLoad, LightingLoad
from .EventBasedLoad import EventBasedLoad, DailyLoad, EventDataLoad
from .HVAC import (
    HVAC,
    Heater,
    ElectricFurnace,
    ElectricBaseboard,
    ElectricBoiler,
    GasFurnace,
    GasBoiler,
    HeatPumpHeater,
    ASHPHeater,
    MinisplitAHSPHeater,
    Cooler,
    AirConditioner,
    ASHPCooler,
    RoomAC,
    MinisplitAHSPCooler,
)
from .WaterHeater import (
    WaterHeater,
    ElectricResistanceWaterHeater,
    HeatPumpWaterHeater,
    GasWaterHeater,
    TanklessWaterHeater,
    GasTanklessWaterHeater,
)
from .Generator import Generator, GasGenerator, GasFuelCell
from .PV import PV
from .Battery import Battery
from .EV import ElectricVehicle, ScheduledEV
# from .WetAppliance import WetAppliance


EQUIPMENT_BY_NAME = {
    # 'HVAC Heating'
    **{
        equipment.name: equipment
        for equipment in [
            Heater,
            ElectricFurnace,
            ElectricBaseboard,
            ElectricBoiler,
            GasFurnace,
            GasBoiler,
            HeatPumpHeater,
            ASHPHeater,
            MinisplitAHSPHeater,
        ]
    },
    # 'HVAC Cooling'
    **{
        equipment.name: equipment
        for equipment in [Cooler, AirConditioner, ASHPCooler, RoomAC, MinisplitAHSPCooler]
    },
    # 'Water Heating'
    **{
        equipment.name: equipment
        for equipment in [
            ElectricResistanceWaterHeater,
            HeatPumpWaterHeater,
            GasWaterHeater,
            TanklessWaterHeater,
            GasTanklessWaterHeater,
        ]
    },
    # 'EV'
    ElectricVehicle.name: ElectricVehicle,
    "Electric Vehicle": ElectricVehicle,
    ScheduledEV.name: ScheduledEV,
    # 'PV'
    PV.name: PV,
    # 'Battery'
    Battery.name: Battery,
    # 'Gas Generator'
    GasGenerator.name: GasGenerator,
    GasFuelCell.name: GasFuelCell,
    # 'Lighting'
    "Indoor Lighting": LightingLoad,
    "Exterior Lighting": LightingLoad,
    "Basement Lighting": LightingLoad,
    "Garage Lighting": LightingLoad,
    # 'Other'
    "Clothes Washer": EventBasedLoad,
    "Clothes Dryer": EventBasedLoad,
    "Dishwasher": EventBasedLoad,
    "Refrigerator": ScheduledLoad,
    "Cooking Range": EventBasedLoad,
    "MELs": ScheduledLoad,
    # 'Basement MELs',  # not modeled
    "TV": ScheduledLoad,
    "Well Pump": ScheduledLoad,
    "Pool Pump": ScheduledLoad,
    "Pool Heater": ScheduledLoad,
    "Spa Pump": ScheduledLoad,
    "Spa Heater": ScheduledLoad,
    "Gas Grill": ScheduledLoad,
    "Gas Fireplace": ScheduledLoad,
    "Gas Lighting": ScheduledLoad,
    "Ceiling Fan": ScheduledLoad,
    "Ventilation Fan": ScheduledLoad,
}

ALL_END_USES = {cls.end_use for cls in EQUIPMENT_BY_NAME.values()}
