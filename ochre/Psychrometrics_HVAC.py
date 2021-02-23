# $Date: 2015-01-29 08:56:26 -0700 (Thu, 29 Jan 2015) $
# $Rev: 5050 $
# $Author: shorowit $
# $HeadURL: https://cbr.nrel.gov/BEopt2/svn/trunk/Modeling/util.py $

"""
Add classes or functions here that can be used across a variety of our
python classes and modules.
"""
import math
# import numpy as np
import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)

# from ochre import Units
#
#
# class Constants(object):
#     # Values
#     AtticIsVentedMinSLA = 0.001  # The minimum SLA at which an attic is assumed to be vented
#     # DOE2 default for INSIDE-SOL-ABS for ceilings
#     DefaultSolarAbsCeiling = 0.3
#     # DOE2 default for SOLAR-FRACTION for floors, which is used because
#     # we have Minimal/FullExterior solar distribution in E+; If we ever
#     # switch to FullInterior solar distribution, we should use the DOE2
#     # default of 0.8 for INSIDE-SOL-ABS for walls
#     DefaultSolarAbsFloor = 0.6
#     # DOE2 default for INSIDE-SOL-ABS for walls
#     DefaultSolarAbsWall = 0.5
#     g = 32.174  # gravity (ft/s2)
#     GSHP_CFM_Btuh = Units.Btu_h2Ton(400)
#     GSHP_GPM_Btuh = Units.Btu_h2Ton(3)
#     HXPipeCond = 0.23  # Pipe thermal conductivity, default to high density polyethylene
#     large = 1e9
#     NoDataFlag = -9999999
#     MonthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#     MonthNumDays = 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
#     DaysOfWeek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#     StartDayYears = {'Monday': 1990, 'Tuesday': 1991, 'Wednesday': 1997, 'Thursday': 1998, 'Friday': 1993,
#                      'Saturday': 1994,
#                      'Sunday': 1995}  # For DOE-2: Non-leap years with Jan 1 corresponding to the day of the week
#     Patm = 14.696  # standard atmospheric pressure (psia)
#     Pi = math.pi
#     R = 1.9858  # gas constant (Btu/lbmol-R)
#     small = 1e-9
#     MinCoolingCapacity = 1  # Btu/h
#
#     # Mini-split heat pump constants
#     Num_Speeds_MSHP = 10
#     MSHP_Cd_Cooling = 0.25
#     MSHP_Cd_Heating = 0.25
#
#     # Room AC constants:
#     Num_Speeds_RoomAC = 1
#
#     # Strings
#     Auto = 'auto'
#     BAZoneCold = 'Cold'
#     BAZoneHotDry = 'Hot-Dry'
#     BAZoneSubarctic = 'Subarctic'
#     BAZoneHotHumid = 'Hot-Humid'
#     BAZoneMixedHumid = 'Mixed-Humid'
#     BAZoneMarine = 'Marine'
#     BAZoneVeryCold = 'Very Cold'
#     BoilerTypeCondensing = 'hot water, condensing'
#     BoilerTypeNaturalDraft = 'hot water, natural draft'
#     BoilerTypeForcedDraft = 'hot water, forced draft'
#     BoilerTypeSteam = 'steam'
#     BoreConfigSingle = 'single'
#     BoreConfigLine = 'line'
#     BoreConfigLconfig = 'l-config'
#     BoreConfigRectangle = 'rectangle'
#     BoreConfigUconfig = 'u-config'
#     BoreConfigL2config = 'l2-config'
#     BoreConfigOpenRectangle = 'open-rectangle'
#     BoreTypeVertical = 'vertical bore'
#     CollectorTypeClosedLoop = 'closed loop'
#     CollectorTypeICS = 'ics'
#     ColorWhite = 'white'
#     ColorMedium = 'medium'
#     ColorDark = 'dark'
#     ColorLight = 'light'
#     CoordRelative = 'relative'
#     CoordAbsolute = 'absolute'
#     CondenserTypeWater = 'watercooled'
#     CondenserTypeAir = 'aircooled'
#     DehumidDucted = 'ducted'
#     DehumidStandalone = 'standalone'
#     DayTypeWeekend = 'weekend'
#     DayTypeWeekday = 'weekday'
#     DayTypeVacation = 'vacation'
#     DRControlAuto = 'automatic'
#     DRControlManual = 'manual'
#     FacadeFront = 'front'
#     FacadeBack = 'back'
#     FacadeLeft = 'left'
#     FacadeRight = 'right'
#     FanControlSmart = 'smart'
#     FluidWater = 'water'
#     FluidPropyleneGlycol = 'propylene-glycol'
#     FluidEthyleneGlycol = 'ethylene-glycol'
#     FoundationCalcSimple = 'simple'
#     FoundationCalcPreProcess = 'preprocess'
#     FuelTypeElectric = 'electric'
#     FuelTypeGas = 'gas'
#     FuelTypePropane = 'propane'
#     FuelTypeOil = 'oil'
#     FurnTypeLight = 'LIGHT'
#     FurnTypeHeavy = 'HEAVY'
#     HeatTransferMethodCTF = 'ctf'
#     HeatTransferMethodCondFD = 'confd'
#     HERSReference = "ReferenceHome"
#     HERSRated = "RatedHome"
#     InfMethodSG = 'S-G'
#     InfMethodASHRAE = 'ASHRAE-ENHANCED'
#     InfMethodRes = 'RESIDENTIAL'
#     InsulationCellulose = 'cellulose'
#     InsulationFiberglass = 'fiberglass'
#     InsulationFiberglassBatt = 'fiberglass batt'
#     InsulationPolyiso = 'polyiso'
#     InsulationSIP = 'sip'
#     InsulationClosedCellSprayFoam = 'closed cell spray foam'
#     InsulationOpenCellSprayFoam = 'open cell spray foam'
#     InsulationXPS = 'xps'
#     LocationInterior = 'interior'
#     LocationExterior = 'exterior'
#     MaterialAdiabatic = 'Adiabatic'
#     MaterialSoil12in = 'Soil-12in'
#     MaterialPlywood1_2in = 'Plywood-1_2in'
#     MaterialPlywood3_2in = 'Plywood-3_2in'
#     MaterialStudandAirWall = 'StudandAirWall'
#     MaterialPartitionWallMass = 'PartitionWallMass'
#     MaterialPartitionWallMassLayer2 = 'PartitionWallMassLayer2'
#     MaterialIntWallIns = 'IntWallIns'
#     MaterialConcrete4in = 'Concrete-4in'
#     MaterialUFBaseFloorFicR = 'UFBaseFloor-FicR'
#     Material2x6 = '2x6'
#     Material2x4 = '2x4'
#     Material2x = '2x'  # for rim joist
#     MaterialGrgRoofStudandAir = 'GrgRoofStudandAir'
#     MaterialRoofingMaterial = 'RoofingMaterial'
#     MaterialPlywood3_4in = 'Plywood-3_4in'
#     MaterialRadiantBarrier = 'RadiantBarrier'
#     MaterialConcrete8in = 'Concrete-8in'
#     MaterialCWallIns = 'CWallIns'
#     MaterialCWallFicR = 'CWall-FicR'
#     MaterialCFloorFicR = 'CFloor-FicR'
#     MaterialConcPCMCeilWall = 'ConcPCMCeilWall'
#     MaterialConcPCMExtWall = 'ConcPCMExtWall'
#     MaterialConcPCMPartWall = 'ConcPCMPartWall'
#     MaterialOSB = 'osb'
#     MaterialGypsum = 'gyp'
#     MaterialGypcrete = 'crete'
#     MaterialCopper = 'copper'
#     MaterialPEX = 'pex'
#     MaterialTile = 'tile'
#     MaterialStudandCavity_Att = 'StudandCavity_Att'
#     MaterialExteriorFinish = 'ExteriorFinish'
#     MaterialIntWallRigidIns = 'IntWallRigidIns'
#     MaterialStudandAirFloor = 'StudandAirFloor'
#     MaterialFloorMass = 'FloorMass'
#     MaterialIntFloorIns = 'IntFloorIns'
#     MaterialUAAdditionalCeilingIns = 'UAAdditionalCeilingIns'
#     MaterialUATrussandIns = 'UATrussandIns'
#     MaterialAddforCTFCalc = 'AddforCTFCalc'
#     MaterialUARigidRoofIns = 'UARigidRoofIns'
#     MaterialUARoofIns = 'UARoofIns'
#     MaterialRigidRoofIns = 'RigidRoofIns'
#     MaterialRoofIns = 'RoofIns'
#     MaterialGypsumBoardCeiling = 'GypsumBoard-Ceiling'
#     MaterialCeilingMass = 'CeilingMass'
#     MaterialCeilingMassLayer2 = 'CeilingMassLayer2'
#     MaterialStudandAirRoof = 'StudandAirRoof'
#     MaterialDoorMaterial = 'DoorMaterial'
#     MaterialGarageDoorMaterial = 'GarageDoorMaterial'
#     MaterialWoodFlooring = 'WoodFlooring'
#     MaterialGypsumBoard1_2in = 'GypsumBoard-1_2in'
#     MaterialFBaseWallFicR = 'FBaseWall-FicR'
#     MaterialFBaseWallIns = 'FBaseWallIns'
#     MaterialFBaseFloorFicR = 'FBaseFloor-FicR'
#     MaterialUFBsmtCeilingIns = 'UFBsmtCeilingIns'
#     MaterialUFBaseWallIns = 'UFBaseWallIns'
#     MaterialUFBaseWallFicR = 'UFBaseWall-FicR'
#     MaterialCrawlCeilingIns = 'CrawlCeilingIns'
#     MaterialStudandCavity = 'StudandCavity'
#     MaterialFBsmtJoistandCavity = 'FBsmtJoistandCavity'
#     MaterialUFBsmtJoistandCavity = 'UFBsmtJoistandCavity'
#     MaterialCSJoistandCavity = 'CSJoistandCavity'
#     MaterialFurring = 'Furring'
#     MaterialCMU = 'CMU'
#     MaterialIntSheathing = 'IntSheathing'
#     MaterialSplineLayer = 'SplineLayer'
#     MaterialWallIns = 'WallIns'
#     MaterialICFInsForm = 'ICFInsForm'
#     MaterialICFConcrete = 'ICFConcrete'
#     MaterialGypsumBoardExtWall = 'GypsumBoard-ExtWall'
#     MaterialExtWallMass = 'ExtWallMass'
#     MaterialExtWallMassLayer2 = 'ExtWallMass2'
#     MaterialCarpetLayer = 'CarpetLayer'
#     MaterialCarpetBareLayer = 'CarpetBareLayer'
#     MaterialWallRigidIns = 'WallRigidIns'
#     MaterialCavity = 'Cavity'
#     MaterialAluminum = 'aluminum'
#     MaterialBrick = 'brick'
#     MaterialFiberCement = 'fiber-cement'
#     MaterialStucco = 'stucco'
#     MaterialVinyl = 'vinyl'
#     MaterialWood = 'wood'
#     MaterialTypeProperties = 'PROPERTIES'
#     MaterialTypeResistance = 'RESISTANCE'
#     PipeTypeTrunkBranch = 'trunkbranch'
#     PipeTypeHomeRun = 'homerun'
#     PCMtypeDistributed = 'distributed'
#     PCMtypeConcentrated = 'concentrated'
#     RecircTypeTimer = 'timer'
#     RecircTypeDemand = 'demand'
#     RoofMaterialAsphalt = 'asphaltshingles'
#     RoofMaterialWoodShakes = 'woodshakes'
#     RoofMaterialTarGravel = 'targravel'
#     RoofMaterialMetal = 'metal'
#     RoofMaterialMembrane = 'membrane'
#     RoofStructureRafter = 'rafter'
#     RoofTypeHip = "hip"
#     ScheduleTypeTemperature = 'TEMPERATURE'
#     ScheduleTypeFraction = 'FRACTION'
#     ScheduleTypeMultiplier = 'MULTIPLIER'
#     ScheduleTypeFlag = 'FLAG'
#     ScheduleTypeOnOff = 'ON/OFF'
#     ScheduleTypeNumber = 'NUMBER'
#     ScheduleTypeMonth = 'MONTH'
#     SeasonHeating = "Heating"
#     SeasonCooling = "Cooling"
#     SeasonOverlap = "Overlap"
#     SeasonNone = "None"
#     SimEngineDOE2 = 0
#     SimEngineEnergyPlus = 1
#     SimEngineCSE = 2
#     SimEngineSEEM = 3
#     SizingFixed = 'fixed'
#     SizingAuto = 'autosize'
#     SpaceLiving = 'living'
#     SpaceGarage = 'garage'
#     SpaceGround = 'ground'
#     SpaceAttic = 'attic'
#     SpaceUnfinAttic = 'unfinishedattic'
#     SpaceFinAttic = 'finishedattic'
#     SpaceCrawl = 'crawlspace'
#     SpacePierbeam = 'pierbeam'
#     SpaceBasement = 'basement'
#     SpaceUnfinBasement = 'unfinishedbasement'
#     SpaceFinBasement = 'finishedbasement'
#     SpaceOutside = 'outside'
#     SpaceDummy = 'dummy'
#     # Wall surface types
#     SurfaceTypeExtInsFinWall = 'ExtInsFinWall'
#     SurfaceTypeExtInsUnfinWall = 'ExtInsUnfinWall'
#     SurfaceTypeExtUninsUnfinWall = 'ExtUninsUnfinWall'
#     SurfaceTypeFinUninsFinWall = 'FinUninsFinWall'
#     SurfaceTypeGrndAdiabaticFinWall = 'GrndAdiabaticFinWall'
#     SurfaceTypeGrndAdiabaticUnfinWall = 'GrndAdiabaticUnfinWall'
#     SurfaceTypeGrndInsFinWall = 'GrndInsFinWall'
#     SurfaceTypeGrndInsUnfinBWall = 'GrndInsUnfinBWall'
#     SurfaceTypeGrndInsUnfinCSWall = 'GrndInsUnfinCSWall'
#     SurfaceTypeNghbrAdiabaticFinWall = 'NghbrAdiabaticFinWall'
#     SurfaceTypeNghbrAdiabaticUnfinWall = 'NghbrAdiabaticUnfinWall'
#     SurfaceTypeUnfinInsFinWall = 'UnfinInsFinWall'
#     SurfaceTypeUnfinInsUnfinWall = 'UnfinInsUnfinWall'
#     SurfaceTypeUnfinUninsFinWall = 'UnfinUninsFinWall'
#     SurfaceTypeUnfinUninsUnfinWall = 'UnfinUninsUnfinWall'
#     SurfaceTypeCSRimJoist = 'CSRimJoist'
#     SurfaceTypeUFBsmtRimJoist = 'UFBsmtRimJoist'
#     SurfaceTypeFBsmtRimJoist = 'FBsmtRimJoist'
#     # Roof surface types
#     SurfaceTypeFinInsExtRoof = 'FinInsExtRoof'
#     SurfaceTypeShadingRoof = 'ShadingRoof'
#     SurfaceTypeUnfinInsExtRoof = 'UnfinInsExtRoof'
#     SurfaceTypeUnfinUninsExtGrgRoof = 'UnfinUninsExtGrgRoof'
#     SurfaceTypeUnfinUninsExtRoof = 'UnfinUninsExtRoof'
#     # Floor surface types
#     SurfaceTypeFinInsUnfinUAFloor = 'FinInsUnfinUAFloor'
#     SurfaceTypeFinUninsFinFloor = 'FinUninsFinFloor'
#     SurfaceTypeGrndAdiabaticFinLivFloor = 'GrndAdiabaticFinLivFloor'
#     SurfaceTypeGrndInsFinLivFloor = 'GrndInsFinLivFloor'
#     SurfaceTypeGrndUninsFinBFloor = 'GrndUninsFinBFloor'
#     SurfaceTypeGrndUninsUnfinBFloor = 'GrndUninsUnfinBFloor'
#     SurfaceTypeGrndUninsUnfinCSFloor = 'GrndUninsUnfinCSFloor'
#     SurfaceTypeGrndUninsUnfinGrgFloor = 'GrndUninsUnfinGrgFloor'
#     SurfaceTypeShadingFloor = 'ShadingFloor'
#     SurfaceTypeUnfinBInsFinFloor = 'UnfinBInsFinFloor'
#     SurfaceTypeUnfinCSInsFinFloor = 'UnfinCSInsFinFloor'
#     SurfaceTypeUnfinInsFinFloor = 'UnfinInsFinFloor'
#     SurfaceTypeUnfinInsUnfinFloor = 'UnfinInsUnfinFloor'
#     SurfaceTypeUnfinUninsUnfinFloor = 'UnfinUninsUnfinFloor'
#     # Other surface types
#     SurfaceTypeShadingSurface = 'ShadingSurface'
#     SurfaceTypeDoor = 'Door'
#     SurfaceTypeGarageDoor = 'GarageDoor'
#     TankTypeStratified = 'Stratified'
#     TankTypeMixed = 'Mixed'
#     TerrainOcean = 'ocean'
#     TerrainPlains = 'plains'
#     TerrainRural = 'rural'
#     TerrainSuburban = 'suburban'
#     TerrainCity = 'city'
#     TestBldgMinimal = 'minimal'
#     TestBldgTypical = 'typical'
#     TestBldgExisting = 'existing'
#     TestTypeStandard = 'standard'
#     TestTypeCSEValidation = 'cse'
#     TestTypeSEEMValidation = 'seem'
#     TiltPitch = 'pitch'
#     TiltLatitude = 'latitude'
#     TubeSpacingB = 'b'
#     TubeSpacingC = 'c'
#     TubeSpacingAS = 'as'
#     VentTypeExhaust = 'exhaust'
#     VentTypeSupply = 'supply'
#     VentTypeBalanced = 'balanced'
#     WallTypeWoodStud = 'woodstud'
#     WallTypeDoubleStud = 'doublestud'
#     WallTypeCMU = 'cmus'
#     WallTypeSIP = 'sips'
#     WallTypeICF = 'icfs'
#     WallTypeMisc = 'misc'
#     WaterHeaterTypeTankless = 'tankless'
#     WaterHeaterTypeTank = 'tank'
#     WaterHeaterTypeHeatPump = 'heatpump'
#     WaterHeaterTypeHeatPumpStratified = 'heatpump_strat'
#     WindowClear = 'clear'
#     WindowHighSHGCLowe = 'high-gain low-e'
#     WindowLowSHGCLowe = 'low-gain low-e'
#     WindowMedSHGCLowe = 'medium-gain low-e'
#     WindowFrameInsulated = 'insulated'
#     WindowFrameMTB = 'metal with thermal breaks'
#     WindowFrameNonMetal = 'non-metal'
#     WindowFrameMetal = 'metal'
#     WindowTypeSingleCasement = "single casement"
#     WindowTypeDoubleCasement = "double casement"
#     WindowTypeHorizontalSlider = "horizontal slider"
#     WindowTypeVerticalSlider = "vertical slider"
#     WindowTypeFixedPicture = "fixed"
#     WindowTypeDoor = "door"
#
#
# class mat_solid(object):
#
#     def __init__(self, rho, Cp, k):
#         self.rho = rho  # Density (lb/ft3)
#         self.Cp = Cp  # Specific Heat (Btu/lbm-R)
#         self.k = k  # Thermal Conductivity (Btu/h-ft-R)
#
#
# class mat_liq(object):
#
#     def __init__(self, rho, Cp, k, mu, H_fg, T_frz, T_boil, T_crit):
#         self.rho = rho  # Density (lb/ft3)
#         self.Cp = Cp  # Specific Heat (Btu/lbm-R)
#         self.k = k  # Thermal Conductivity (Btu/h-ft-R)
#         self.mu = mu  # Dynamic Viscosity (lbm/ft-h)
#         self.H_fg = H_fg  # Latent Heat of Vaporization (Btu/lbm)
#         self.T_frz = T_frz  # Freezing Temperature (degF)
#         self.T_boil = T_boil  # Boiling Temperature (degF)
#         self.T_crit = T_crit  # Critical Temperature (degF)
#
#
# class mat_gas(object):
#
#     def __init__(self, rho, Cp, k, mu, M):
#         self.rho = rho  # Density (lb/ft3)
#         self.Cp = Cp  # Specific Heat (Btu/lbm-R)
#         self.k = k  # Thermal Conductivity (Btu/h-ft-R)
#         self.mu = mu  # Dynamic Viscosity (lbm/ft-h)
#         self.M = M  # Molecular Weight (lbm/lbmol)
#         if M:
#             self.R = Constants.R / M  # Gas Constant (Btu/lbm-R)
#         else:
#             self.R = None
#
#
# class Properties(object):
#     # From EES at STP
#     Air = mat_gas(0.07518, 0.2399, 0.01452, 0.04415, 28.97)
#     H2O_l = mat_liq(62.32, 0.9991, 0.3386, 2.424, 1055, 32.0, 212.0, None)
#     H2O_v = mat_gas(None, 0.4495, None, None, 18.02)
#
#     # Converted from EnthDR22 f77 in ResAC (Brandemuehl)
#     R22_l = mat_liq(None, 0.2732, None, None, 100.5, None, -41.35, 204.9)
#     R22_v = mat_gas(None, 0.1697, None, None, None)
#
#     # From wolframalpha.com
#     Wood = mat_solid(630, 2500, 0.14)
#
#     PsychMassRat = H2O_v.M / Air.M
#
#
def Iterate(x0, f0, x1, f1, x2, f2, icount, cvg, TolRel=1e-5, small=1e-9):
    '''
    Description:
    ------------
        Determine if a guess is within tolerance for convergence
        if not, output a new guess using the Newton-Raphson method

    Source:
    -------
        Based on XITERATE f77 code in ResAC (Brandemuehl)

    Inputs:
    -------
        x0      float    current guess value
        f0      float    value of function f(x) at current guess value

        x1,x2   floats   previous two guess values, used to create quadratic
                         (or linear fit)
        f1,f2   floats   previous two values of f(x)

        icount  int      iteration count
        cvg     bool     Has the iteration reached convergence?

    Outputs:
    --------
        x_new   float    new guess value
        cvg     bool     Has the iteration reached convergence?

        x1,x2   floats   updated previous two guess values, used to create quadratic
                         (or linear fit)
        f1,f2   floats   updated previous two values of f(x)

    Example:
    --------

        # Find a value of x that makes f(x) equal to some specific value f:

        # initial guess (all values of x)
        x = 1.0
        x1 = x
        x2 = x

        # initial error
        error = f - f(x)
        error1 = error
        error2 = error

        itmax = 50  # maximum iterations
        cvg = False # initialize convergence to 'False'

        for i in range(1,itmax+1):
            error = f - f(x)
            x,cvg,x1,error1,x2,error2 = \
                                     Iterate(x,error,x1,error1,x2,error2,i,cvg)

            if cvg == True:
                break
        if cvg == True:
            print 'x converged after', i, 'iterations'
        else:
            print 'x did NOT converge after', i, 'iterations'

        print 'x, when f(x) is', f,'is', x
    '''

    dx = 0.1

    # Test for convergence
    if (abs(x0 - x1) < TolRel * max(abs(x0), small) and icount != 1) or f0 == 0:
        x_new = x0
        cvg = True
    else:
        x_new = None
        cvg = False

        if icount == 1:  # Perturbation
            mode = 1
        elif icount == 2:  # Linear fit
            mode = 2
        else:  # Quadratic fit
            mode = 3

        if mode == 3:
            # Quadratic fit
            if x0 == x1:  # If two xi are equal, use a linear fit
                x1 = x2
                f1 = f2
                mode = 2
            elif x0 == x2:  # If two xi are equal, use a linear fit
                mode = 2
            else:
                # Set up quadratic coefficients
                c = ((f2 - f0) / (x2 - x0) - (f1 - f0) / (x1 - x0)) / (x2 - x1)
                b = (f1 - f0) / (x1 - x0) - (x1 + x0) * c
                a = f0 - (b + c * x0) * x0

                if abs(c) < small:  # If points are co-linear, use linear fit
                    mode = 2
                elif abs((a + (b + c * x1) * x1 - f1) / f1) > small:
                    # If coefficients do not accurately predict data points due to
                    # round-off, use linear fit
                    mode = 2
                else:
                    D = b ** 2 - 4.0 * a * c  # calculate discriminant to check for real roots
                    if D < 0.0:  # if no real roots, use linear fit
                        mode = 2
                    else:
                        if D > 0.0:  # if real unequal roots, use nearest root to recent guess
                            x_new = (-b + math.sqrt(D)) / (2 * c)
                            x_other = -x_new - b / c
                            if abs(x_new - x0) > abs(x_other - x0):
                                x_new = x_other
                        else:  # If real equal roots, use that root
                            x_new = -b / (2 * c)

                        if f1 * f0 > 0 and f2 * f0 > 0:  # If the previous two f(x) were the same sign as the new
                            if abs(f2) > abs(f1):
                                x2 = x1
                                f2 = f1
                        else:
                            if f2 * f0 > 0:
                                x2 = x1
                                f2 = f1
                        x1 = x0
                        f1 = f0

        if mode == 2:
            # Linear Fit
            m = (f1 - f0) / (x1 - x0)
            if m == 0:  # If slope is zero, use perturbation
                mode = 1
            else:
                x_new = x0 - f0 / m
                x2 = x1
                f2 = f1
                x1 = x0
                f1 = f0

        if mode == 1:
            # Perturbation
            if abs(x0) > small:
                x_new = x0 * (1 + dx)
            else:
                x_new = dx
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0

    return x_new, cvg, x1, f1, x2, f2


class Psychrometrics:

    # @staticmethod
    # def H_fg_fT(T):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the latent heat of vaporization at a given drybulb
    #         temperature.
    #
    #         Valid for temperatures between 0 and 200 degC (32 - 392 degF)
    #
    #     Source:
    #     -------
    #         Based on correlation from steam tables - "Introduction to
    #         Thermodynamics, Classical and Statistical" by Sonntag and Wylen
    #
    #         H_fg = 2518600 - 2757.1*T (J/kg with T in degC)
    #              = 2581600 - 2757.1*(T - 32)*5/9 (J/kg with T in degF)
    #              = 2581600 - 1531.72*T + 49015.1 (J/kg with T in degF)
    #              = 1083 - 0.6585*T + 21.07 (Btu/lbm with T in degF)
    #
    #     Inputs:
    #     -------
    #         T       float      temperature         (degF)
    #
    #     Outputs:
    #     --------
    #         H_fg    float      latent heat of vaporization (Btu/lbm)
    #     '''
    #     H_fg = 1083 - 0.6585 * T + 21.07
    #
    #     return H_fg
    #
    # @staticmethod
    # def Psat_fT(Tdb):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the saturation pressure of water vapor at a given temperature
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature      (degF)
    #
    #     Outputs:
    #     --------
    #         Psat    float      saturated vapor pressure (psia)
    #     '''
    #     C1 = -1.0214165e4
    #     C2 = -4.8932428
    #     C3 = -5.3765794e-3
    #     C4 = 1.9202377e-7
    #     C5 = 3.5575832e-10
    #     C6 = -9.0344688e-14
    #     C7 = 4.1635019
    #     C8 = -1.0440397e4
    #     C9 = -1.1294650e1
    #     C10 = -2.7022355e-2
    #     C11 = 1.2890360e-5
    #     C12 = -2.4780681e-9
    #     C13 = 6.5459673
    #
    #     T_abs = Units.F2R(Tdb)
    #     T_frz_abs = Units.F2R(Properties.H2O_l.T_frz)
    #
    #     # If below freezing, calculate saturation pressure over ice
    #     if T_abs < T_frz_abs:
    #         Psat = math.exp(C1 / T_abs + C2 + T_abs * (C3 + T_abs * (C4 + T_abs * (C5 + C6 * T_abs))) +
    #                         C7 * math.log(T_abs))
    #
    #     # If above freezing, calculate saturation pressure over liquid water
    #     else:
    #         Psat = math.exp(C8 / T_abs + C9 + T_abs * (C10 + T_abs * (C11 + C12 * T_abs)) + C13 * math.log(T_abs))
    #
    #     return Psat
    #
    # @staticmethod
    # def Psat_fT_SI(Tdb):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the saturation pressure of water vapor at a given temperature
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature      (degC)
    #
    #     Outputs:
    #     --------
    #         Psat    float      saturated vapor pressure (kPa)
    #     '''
    #     C1 = -1.0214165e4
    #     C2 = -4.8932428
    #     C3 = -5.3765794e-3
    #     C4 = 1.9202377e-7
    #     C5 = 3.5575832e-10
    #     C6 = -9.0344688e-14
    #     C7 = 4.1635019
    #     C8 = -1.0440397e4
    #     C9 = -1.1294650e1
    #     C10 = -2.7022355e-2
    #     C11 = 1.2890360e-5
    #     C12 = -2.4780681e-9
    #     C13 = 6.5459673
    #
    #     Tdb = Units.C2F(Tdb)
    #     T_abs = Units.F2R(Tdb)
    #     T_frz_abs = Units.F2R(Properties.H2O_l.T_frz)
    #
    #     # If below freezing, calculate saturation pressure over ice
    #     if T_abs < T_frz_abs:
    #         Psat = math.exp(C1 / T_abs + C2 + T_abs * (C3 + T_abs * (C4 + T_abs * (C5 + C6 * T_abs))) +
    #                         C7 * math.log(T_abs))
    #
    #     # If above freezing, calculate saturation pressure over liquid water
    #     else:
    #         Psat = math.exp(C8 / T_abs + C9 + T_abs * (C10 + T_abs * (C11 + C12 * T_abs)) + C13 * math.log(T_abs))
    #
    #     Psat = Units.psi2kPa(Psat)
    #     return Psat
    #
    # @staticmethod
    # def Tsat_fP(P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the saturation temperature of water vapor at a given pressure
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         P       float      pressure                    (psia)
    #
    #     Outputs:
    #     --------
    #         Tsat    float      saturated vapor temperature (degF)
    #     '''
    #     # Initialize
    #     Tsat = 212.0  # (degF)
    #     Tsat1 = Tsat  # (degF)
    #     Tsat2 = Tsat  # (degF)
    #
    #     error = P - Psychrometrics.Psat_fT(Tsat)  # (psia)
    #     error1 = error  # (psia)
    #     error2 = error  # (psia)
    #
    #     itmax = 50  # maximum iterations
    #     cvg = False
    #
    #     for i in range(1, itmax + 1):
    #
    #         error = P - Psychrometrics.Psat_fT(Tsat)  # (psia)
    #
    #         Tsat, cvg, Tsat1, error1, Tsat2, error2 = \
    #             Iterate(Tsat, error, Tsat1, error1, Tsat2, error2, i, cvg, TolRel=0.01)
    #
    #         if cvg == True:
    #             break
    #
    #     if cvg == False:
    #         print('Warning: Tsat_fP failed to converge')
    #
    #     return Tsat
    #
    # @staticmethod
    # def Tsat_fh_P(h, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the drybulb temperature at saturation a given enthalpy and
    #         pressure.
    #
    #     Source:
    #     -------
    #         Based on TAIRSAT f77 code in ResAC (Brandemuehl)
    #
    #     Inputs:
    #     -------
    #         h       float      enathalpy           (Btu/lbm)
    #         P       float      pressure            (psia)
    #
    #     Outputs:
    #     --------
    #         Tdb     float      drybulb temperature (degF)
    #     '''
    #     # Initialize
    #     Tdb = 50
    #     Tdb1 = Tdb  # (degF)
    #     Tdb2 = Tdb  # (degF)
    #
    #     error = h - Psychrometrics.hsat_fT_P(Tdb, P)  # (Btu/lbm)
    #     error1 = error
    #     error2 = error
    #
    #     itmax = 50  # maximum iterations
    #     cvg = False
    #
    #     for i in range(1, itmax + 1):
    #
    #         error = h - Psychrometrics.hsat_fT_P(Tdb, P)  # (Btu/lbm)
    #
    #         Tdb, cvg, Tdb1, error1, Tdb2, error2 = \
    #             Iterate(Tdb, error, Tdb1, error1, Tdb2, error2, i, cvg)
    #
    #         if cvg == True:
    #             break
    #
    #     if cvg == False:
    #         print('Warning: Tsat_fh_P failed to converge')
    #
    #     return Tdb
    #
    # @staticmethod
    # def Tsat_fh_P_SI(h, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the drybulb temperature at saturation a given enthalpy and
    #         pressure.
    #
    #     Source:
    #     -------
    #         Based on TAIRSAT f77 code in ResAC (Brandemuehl)
    #
    #     Inputs:
    #     -------
    #         h       float      enathalpy           (J/kg)
    #         P       float      pressure            (kPa)
    #
    #     Outputs:
    #     --------
    #         Tdb     float      drybulb temperature (degC)
    #     '''
    #     return Units.F2C(Psychrometrics.Tsat_fh_P(Units.J_kg2Btu_lb(h), Units.kPa2psi(P)))
    #
    # @staticmethod
    # def w_fT_R_P(Tdb, R, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given drybulb temperature,
    #         relative humidity and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degF)
    #         R       float      relative humidity     (1/1)
    #         P       float      pressure              (psia)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (lbm/lbm)
    #     '''
    #     Pws = Psychrometrics.Psat_fT(Tdb)
    #     Pw = R * Pws
    #     w = 0.62198 * Pw / (P - Pw)  # Rd / Rv = 0.622 Gas constant for dry air, Rv is the gas constant for water vapor
    #     # w - mixing ratio, or ratio of the mass of water vapor to the mass of dry air ( m_v / m_d )
    #
    #     return w
    #
    # @staticmethod
    # def w_fT_R_P_SI(Tdb, R, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given drybulb temperature,
    #         relative humidity and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degC)
    #         R       float      relative humidity     (1/1)
    #         P       float      pressure              (kPa)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (g/g)
    #     '''
    #     Pws = Units.psi2kPa(Psychrometrics.Psat_fT(Units.C2F(Tdb)))
    #     Pw = R * Pws
    #     w = 0.62198 * Pw / (P - Pw)
    #
    #     return w
    #
    # @staticmethod
    # def Twb_fT_R_SI(Tdb, RH):
    #     '''
    #     Description:
    #     ------------
    #         Calculates wetbulb temperature at a given drybulb
    #         and RH, using an empirical correlation for speed
    #
    #     Source:
    #     -------
    #     Wet-Bulb Temperature from Relative Humidity and Air Temperature
    #         ROLAND STULL University of British Columbia, Vancouver, British Columbia, Canada
    #
    #     Twb = T*atan(0.151977*(RH+8.313659)^0.5)+atan(T+RH)-atan(RH-1.676331)+0.00391838*(RH)^1.5*atan(0.023101*RH)-4.686035
    #     atan in radians
    #     T in degrees C
    #     RH as a number (eg 75%RH=75)
    #     '''
    #
    #     Twb = Tdb * np.arctan(0.151977 * (RH + 8.313659) ** 0.5) + np.arctan(Tdb + RH) - np.arctan(RH - 1.676331) + (
    #             0.00391838 * RH ** 1.5) * np.arctan(0.023101 * RH) - 4.686035
    #
    #     return Twb
    #
    # @staticmethod
    # def Twb_fT_R_P_SI(Tdb, R, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the wetbulb temperature at a given drybulb temperature,
    #         relative humidity, and pressure.
    #
    #     Source:
    #     -------
    #         None (calls other pyschrometric functions)
    #
    #     Inputs:
    #     -------
    #         Tdb    float    drybulb temperature    (degC)
    #         R      float    relative humidity      (1/1)
    #         P      float    pressure               (kPa)
    #
    #     Output:
    #     ------
    #         Twb    float    wetbulb temperautre    (degC)
    #     '''
    #     return Units.F2C(Psychrometrics.Twb_fT_R_P(Units.C2F(Tdb), R, Units.kPa2psi(P)))
    #
    # @staticmethod
    # def Twb_fT_R_P(Tdb, R, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the wetbulb temperature at a given drybulb temperature,
    #         relative humidity, and pressure.
    #
    #     Source:
    #     -------
    #         None (calls other pyschrometric functions)
    #
    #     Inputs:
    #     -------
    #         Tdb    float    drybulb temperature    (degF)
    #         R      float    relative humidity      (1/1)
    #         P      float    pressure               (psia)
    #
    #     Output:
    #     ------
    #         Twb    float    wetbulb temperautre    (degF)
    #     '''
    #
    #     w = Psychrometrics.w_fT_R_P(Tdb, R, P)
    #     Twb = Psychrometrics.Twb_fT_w_P(Tdb, w, P)
    #
    #     return Twb
    #
    # @staticmethod
    # def Pstd_fZ(Z):
    #     '''
    #     Description:
    #     ------------
    #         Calculate standard pressure of air at a given altitude
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Z        float        altitude     (feet)
    #
    #     Outputs:
    #     --------
    #         Pstd    float        barometric pressure (psia)
    #     '''
    #
    #     Pstd = 14.696 * pow(1 - 6.8754e-6 * Z, 5.2559)
    #
    #     return Pstd
    #
    # @staticmethod
    # def Twb_fT_w_P(Tdb, w, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the wetbulb temperature at a given drybulb temperature,
    #         humidity ratio, and pressure.
    #
    #     Source:
    #     -------
    #         Based on WETBULB f77 code in ResAC (Brandemuehl)
    #
    #         Converted into IP units
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature (degF)
    #         w       float      humidity ratio      (lbm/lbm)
    #         P       float      pressure            (psia)
    #
    #     Outputs:
    #     --------
    #         Twb     float      wetbulb temperature (degF)
    #     '''
    #     # Initialize
    #     Tboil = Psychrometrics.Tsat_fP(P)  # (degF)
    #     Twb = max(min(Tdb, Tboil - 0.1), 0.0)  # (degF)
    #
    #     Twb1 = Twb  # (degF)
    #     Twb2 = Twb  # (degF)
    #
    #     Psat_star = Psychrometrics.Psat_fT(Twb)  # (psia)
    #     w_star = Psychrometrics.w_fP(P, Psat_star)  # (lbm/lbm)
    #     w_new = ((Properties.H2O_l.H_fg - (
    #             Properties.H2O_l.Cp - Properties.H2O_v.Cp) * Twb) * w_star - Properties.Air.Cp * (Tdb - Twb)) / (
    #                     Properties.H2O_l.H_fg + Properties.H2O_v.Cp * Tdb - Properties.H2O_l.Cp * Twb)  # (lbm/lbm)
    #
    #     error = w - w_new
    #     error1 = error
    #     error2 = error
    #
    #     itmax = 50  # maximum iterations
    #     cvg = False
    #
    #     for i in range(1, itmax + 1):
    #
    #         Psat_star = Psychrometrics.Psat_fT(Twb)  # (psia)
    #         w_star = Psychrometrics.w_fP(P, Psat_star)  # (lbm/lbm)
    #         w_new = ((Properties.H2O_l.H_fg - (
    #                 Properties.H2O_l.Cp - Properties.H2O_v.Cp) * Twb) * w_star - Properties.Air.Cp * (
    #                          Tdb - Twb)) / (
    #                         Properties.H2O_l.H_fg + Properties.H2O_v.Cp * Tdb - Properties.H2O_l.Cp * Twb)  # (lbm/lbm)
    #
    #         error = w - w_new
    #
    #         Twb, cvg, Twb1, error1, Twb2, error2 = \
    #             Iterate(Twb, error, Twb1, error1, Twb2, error2, i, cvg, TolRel=0.01)
    #
    #         if cvg == True:
    #             break
    #
    #     if cvg == False:
    #         print('Warning: Twb_fT_w_P failed to converge. Inputs are: Tdb={}, w={}, P={}'.format(Tdb, w, P))
    #
    #     if Twb > Tdb:
    #         Twb = Tdb  # (degF)
    #
    #     return Twb
    #
    # @staticmethod
    # def Pw_fP_w(P, w):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the partial vapor pressure at a given pressure and
    #         humidity ratio.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         P       float      pressure              (psia)
    #         w       float      humidity ratio        (lbm/lbm)
    #
    #     Outputs:
    #     --------
    #         Pw      float      partial pressure      (psia)
    #     '''
    #
    #     Pw = P * w / (Properties.PsychMassRat + w)
    #
    #     return Pw
    #
    # @staticmethod
    # def Tdp_fP_w(P, w):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the dewpoint temperature at a given pressure
    #         and humidity ratio.
    #
    #         There are two available methods:
    #
    #         CalcMethod == 1: Uses the correlation method from ASHRAE Handbook
    #         CalcMethod != 1: Uses the saturation temperature at the partial
    #                          pressure
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         P       float      pressure              (psia)
    #         w       float      humidity ratio        (lbm/lbm)
    #
    #     Outputs:
    #     --------
    #         Tdp     float      dewpoint temperature  (degF)
    #     '''
    #
    #     CalcMethod = 1
    #
    #     if CalcMethod == 1:
    #
    #         C14 = 100.45
    #         C15 = 33.193
    #         C16 = 2.319
    #         C17 = 0.17074
    #         C18 = 1.2063
    #
    #         Pw = Psychrometrics.Pw_fP_w(P, w)  # (psia)
    #         alpha = math.log(Pw)
    #         Tdp1 = C14 + C15 * alpha + C16 * alpha ** 2 + C17 * alpha ** 3 + C18 * Pw ** 0.1984
    #         Tdp2 = 90.12 + 26.142 * alpha + 0.8927 * alpha ** 2
    #         if Tdp1 >= Properties.H2O_l.T_frz:
    #             Tdp = Tdp1
    #         else:
    #             Tdp = Tdp2
    #
    #     else:
    #
    #         # based on DEWPOINT f77 code in ResAC (Brandemuehl)
    #         if w < Constants.small:
    #             Tdp = -999.0
    #         else:
    #             Pw = Psychrometrics.Pw_fP_w(P, w)
    #             Tdp = Psychrometrics.Tsat_fP(Pw)
    #
    #     return Tdp
    #
    # @staticmethod
    # def Tdp_fP_w_SI(P, w):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the dewpoint temperature at a given pressure
    #         and humidity ratio.
    #
    #         There are two available methods:
    #
    #         CalcMethod == 1: Uses the correlation method from ASHRAE Handbook
    #         CalcMethod != 1: Uses the saturation temperature at the partial
    #                          pressure
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         P       float      pressure              (kPa)
    #         w       float      humidity ratio        (g/g)
    #
    #     Outputs:
    #     --------
    #         Tdp     float      dewpoint temperature  (degC)
    #     '''
    #     return Units.F2C(Psychrometrics.Tdp_fP_w(Units.kPa2psi(P), w))
    #
    # @staticmethod
    # def Tdb_fh_w(h, w):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the drybulb temperature at a given enthalpy
    #         and humidity ratio.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         h       float      enthalpy              (Btu/lbm)
    #         w       float      humidity ratio        (lbm/lbm)
    #
    #     Outputs:
    #     --------
    #         Tdb     float      drypbulb temperature  (degF)
    #     '''
    #     Tdb = (h - Properties.H2O_l.H_fg * w) / (Properties.Air.Cp + Properties.H2O_v.Cp * w)
    #
    #     return Tdb
    #
    # @staticmethod
    # def h_fT_w(Tdb, w):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the enthalpy at a given drybulb temperature
    #         and humidity ratio.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature  (degF)
    #         w       float      humidity ratio        (lbm/lbm)
    #
    #     Outputs:
    #     --------
    #         h       float      enthalpy              (Btu/lbm)
    #     '''
    #     h_dryair = Properties.Air.Cp * Tdb
    #     h_satv = Properties.H2O_l.H_fg + Properties.H2O_v.Cp * Tdb
    #     h = h_dryair + w * h_satv
    #
    #     return h
    #
    # @staticmethod
    # def h_fT_w_SI(Tdb, w):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the enthalpy at a given drybulb temperature
    #         and humidity ratio.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degC)
    #         w       float      humidity ratio        (kg/kg)
    #
    #     Outputs:
    #     --------
    #         h       float      enthalpy              (J/kg)
    #     '''
    #
    #     h = 1000 * (1.006 * Tdb + w * (2501 + 1.86 * Tdb))
    #     return h
    #
    # @staticmethod
    # def w_fT_h_SI(Tdb, h):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given drybulb temperature
    #         and enthalpy.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature  (degC)
    #         h       float      enthalpy              (J/kg)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (kg/kg)
    #     '''
    #
    #     w = (h / 1000 - 1.006 * Tdb) / (2501 + 1.86 * Tdb)
    #     return w
    #
    # @staticmethod
    # def T_fw_h_SI(w, h):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the drybulb temperature at a given humidity ratio
    #         and enthalpy.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         w       float      humidity ratio        (kg/kg)
    #         h       float      enthalpy              (J/kg)
    #
    #     Outputs:
    #     --------
    #         T       float      drybulb temperature  (degC)
    #     '''
    #
    #     T = (h / 1000 - w * 2501) / (1.006 + w * 1.86)
    #     return T
    #
    # @staticmethod
    # def hsat_fT_P(Tdb, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the enthalpy at saturation given drybulb temperature
    #         and Pressure.
    #
    #     Source:
    #     -------
    #         Based on ENTHSAT f77 code in ResAC (Brandemuehl)
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drypbulb temperature  (degF)
    #         w       float      humidity ratio        (lbm/lbm)
    #
    #     Outputs:
    #     --------
    #         h       float      enthalpy              (Btu/lbm)
    #     '''
    #
    #     Psat = Psychrometrics.Psat_fT(Tdb)
    #     w = Psychrometrics.w_fP(P, Psat)
    #     hsat = Psychrometrics.h_fT_w(Tdb, w)
    #
    #     return hsat
    #
    # @staticmethod
    # def w_fP(P, Pw):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given pressure and partial pressure.
    #
    #     Source:
    #     -------
    #         Based on HUMRATIO f77 code in ResAC (Brandemuehl)
    #
    #     Inputs:
    #     -------
    #         P       float      pressure              (psia)
    #         Pw      float      partial pressure      (psia)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (lbm/lbm)
    #     '''
    #     w = Properties.PsychMassRat * Pw / (P - Pw)
    #
    #     return w
    #
    # @staticmethod
    # def w_fT_h(Tdb, h):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given drybulb temperature and
    #         enthalpy.
    #
    #     Source:
    #     -------
    #         Based on HUMTH f77 code in ResAC (Brandemuehl)
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degF)
    #         h       float      enthalpy              (Btu/lbm)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (lbm/lbm)
    #     '''
    #     w = (h - Properties.Air.Cp * Tdb) / (Properties.H2O_l.H_fg + Properties.H2O_v.Cp * Tdb)
    #
    #     return w
    #
    # @staticmethod
    # def w_fT_Twb_P(Tdb, Twb, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given drybulb temperature,
    #         wetbulb temperature and pressure.
    #
    #     Source:
    #     -------
    #         ASHRAE Handbook 2009
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degF)
    #         Twb     float      wetbulb temperature   (degF)
    #         P       float      pressure              (psia)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (lbm/lbm)
    #     '''
    #     w_star = Psychrometrics.w_fP(P, Psychrometrics.Psat_fT(Twb))
    #
    #     w = ((Properties.H2O_l.H_fg - (
    #             Properties.H2O_l.Cp - Properties.H2O_v.Cp) * Twb) * w_star - Properties.Air.Cp * (Tdb - Twb)) / (
    #                 Properties.H2O_l.H_fg + Properties.H2O_v.Cp * Tdb - Properties.H2O_l.Cp * Twb)  # (lbm/lbm)
    #
    #     return w
    #
    # @staticmethod
    # def w_fT_Twb_P_SI(Tdb, Twb, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the humidity ratio at a given drybulb temperature,
    #         wetbulb temperature and pressure.
    #
    #     Source:
    #     -------
    #         ASHRAE Handbook 2009
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degC)
    #         Twb     float      wetbulb temperature   (degC)
    #         P       float      pressure              (kPa)
    #
    #     Outputs:
    #     --------
    #         w       float      humidity ratio        (g/g)
    #     '''
    #
    #     return Psychrometrics.w_fT_Twb_P(Units.C2F(Tdb), Units.C2F(Twb), Units.kPa2psi(P))
    #
    # @staticmethod
    # def R_fT_w_P(Tdb, w, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the relative humidity at a given drybulb temperature,
    #         humidity ratio and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degF)
    #         w       float      humidity ratio        (lbm/lbm)
    #         P       float      pressure              (psia)
    #
    #     Outputs:
    #     --------
    #         R       float      relative humidity     (1/1)
    #     '''
    #     Pw = Psychrometrics.Pw_fP_w(P, w)
    #     R = Pw / Psychrometrics.Psat_fT(Tdb)
    #
    #     return R
    #
    # @staticmethod
    # def R_fT_w_P_SI(Tdb, w, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the relative humidity at a given drybulb temperature,
    #         humidity ratio and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degC)
    #         w       float      humidity ratio        (g/g)
    #         P       float      pressure              (kPa)
    #
    #     Outputs:
    #     --------
    #         R       float      relative humidity     (1/1)
    #     '''
    #     return Psychrometrics.R_fT_w_P(Units.C2F(Tdb), w, Units.kPa2psi(P))
    #
    # @staticmethod
    # def rhoD_fT_w_P(Tdb, w, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the density of dry air at a given drybulb temperature,
    #         humidity ratio and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degF)
    #         w       float      humidity ratio        (lbm/lbm)
    #         P       float      pressure              (psia)
    #
    #     Outputs:
    #     --------
    #         rhoD    float      density of dry air    (lbm/ft3)
    #     '''
    #     Pair = Properties.PsychMassRat * P / (Properties.PsychMassRat + w)  # (psia)
    #     rhoD = Units.psi2Btu_ft3(Pair) / (Constants.R / Properties.Air.M) / (Units.F2R(Tdb))  # (lbm/ft3)
    #
    #     return rhoD
    #
    # @staticmethod
    # def rhoD_fT_w_P_SI(Tdb, w, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the density of dry air at a given drybulb temperature,
    #         humidity ratio and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degC)
    #         w       float      humidity ratio        (g/g)
    #         P       float      pressure              (kPa)
    #
    #     Outputs:
    #     --------
    #         rhoD    float      density of dry air    (kg/m3)
    #     '''
    #
    #     return Units.lbm_ft32kg_m3(Psychrometrics.rhoD_fT_w_P(Units.C2F(Tdb), w, Units.kPa2psi(P)))
    #
    # @staticmethod
    # def rhoM_fT_w_P(Tdb, w, P):
    #     '''
    #     Description:
    #     ------------
    #         Calculate the density of moist air at a given drybulb temperature,
    #         humidity ratio and pressure.
    #
    #     Source:
    #     -------
    #         2009 ASHRAE Handbook
    #
    #     Inputs:
    #     -------
    #         Tdb     float      drybulb temperature   (degF)
    #         w       float      humidity ratio        (lbm/lbm)
    #         P       float      pressure              (psia)
    #
    #     Outputs:
    #     --------
    #         rhoM    float      density of moist air  (lbm/ft3)
    #     '''
    #     rhoM = Psychrometrics.rhoD_fT_w_P(Tdb, w, P) * (1.0 + w)  # (lbm/ft3)
    #
    #     return rhoM
    #
    # @staticmethod
    # def T_adp(Tdb, Twb, cfm, sens, total):
    #     '''
    #    Description:
    #     ------------
    #         Find the dry bulb temp at the Apparatus Dew Point given incoming air state (entering drybulb and wetbulb) and
    #         CFM, sensible and total capacities
    #
    #
    #     Source:
    #     -------
    #         RECONCILING DIFFERENCES BETWEEN RESIDENTIAL DX COOLING MODELS IN DOE-2 AND ENERGYPLUS
    #         Nathanael Kruis (Interpreted by Dylan Cutler 2/16/11)
    #
    #     Inputs:
    #     -------
    #         Tdb    float    Entering Dry Bulb (degF)
    #         Twb    float    Entering Wet Bulb (degF)
    #         sens   float    Sensible capacity of unit (MBtu/h)
    #         total  float    Total capacity of unit (MBtu/h)
    #         cfm    float    Volumetric flow rate of unit (CFM)
    #     Outputs:
    #     --------
    #         db_ADP  float    Drybulb temperature of appartus dew point (degF)
    #     '''
    #
    #     # define necessary variables for Bypass Factor
    #     w_i = Psychrometrics.w_fT_Twb_P(Tdb, Twb, Constants.Patm)
    #     m_dot = Units.hr2min(float(cfm) * Psychrometrics.rhoM_fT_w_P(Tdb, w_i, Constants.Patm))  # lbm/hr
    #     h_i = Psychrometrics.h_fT_w(Tdb, w_i)
    #     ldb = Tdb - Units.kBtu_h2Btu_h(float(sens)) / (m_dot * Properties.Air.Cp)
    #     h_o = h_i - Units.kBtu_h2Btu_h(float(total)) / m_dot
    #     w_o = Psychrometrics.w_fT_h(ldb, h_o)
    #     # calculate BF by iteration
    #
    #     # initial guess (all values of x)
    #     db_ADP = 55
    #     db_ADP1 = db_ADP
    #     db_ADP2 = db_ADP
    #
    #     h_ADP = Psychrometrics.hsat_fT_P(db_ADP, Constants.Patm)
    #     w_ADP1 = Psychrometrics.w_fT_h(db_ADP, h_ADP)
    #     w_ADP2 = w_i - (Tdb - db_ADP) * ((w_i - w_o) / (Tdb - ldb))
    #
    #     # initial error
    #     error = w_ADP1 - w_ADP2
    #     error1 = error
    #     error2 = error
    #
    #     itmax = 50  # maximum iterations
    #     cvg = False  # initialize convergence to 'False'
    #
    #     for x in range(1, itmax + 1):
    #         h_ADP = Psychrometrics.hsat_fT_P(db_ADP, Constants.Patm)
    #         w_ADP1 = Psychrometrics.w_fT_h(db_ADP, h_ADP)
    #         w_ADP2 = w_i - (Tdb - db_ADP) * ((w_i - w_o) / (Tdb - ldb))
    #
    #         error = w_ADP1 - w_ADP2
    #
    #         db_ADP, cvg, db_ADP1, error1, db_ADP2, error2 = \
    #             Iterate(db_ADP, error, db_ADP1, error1, db_ADP2, error2, x, cvg)
    #
    #         if cvg == True:
    #             break
    #
    #     if cvg == False:
    #         print('Warning: T_adp failed to converge ' + 'Error=' + str(error))
    #
    #     return db_ADP  # degF
    #
    # @staticmethod
    # def CalculateMassflowRate(DBin, WBin, P, cfm):
    #     '''
    #    Description:
    #     ------------
    #         Calculate the mass flow rate at the given incoming air state (entering drybubl and wetbulb) and CFM
    #
    #     Source:
    #     -------
    #
    #
    #     Inputs:
    #     -------
    #         Tdb    float    Entering Dry Bulb (degF)
    #         Twb    float    Entering Wet Bulb (degF)
    #         P      float    Barometric pressure (psi)
    #         cfm    float    Volumetric flow rate of unit (CFM)
    #     Outputs:
    #     --------
    #         mfr    float    mass flow rate (lbm/min)
    #     '''
    #     Win = Psychrometrics.w_fT_Twb_P(DBin, WBin, P)
    #     rho_in = Psychrometrics.rhoD_fT_w_P(DBin, Win, P)
    #     mfr = cfm * rho_in
    #     return mfr

    @staticmethod
    def CalculateMassflowRate_SI(DBin, WBin, P, flow):
        '''
       Description:
        ------------
            Calculate the mass flow rate at the given incoming air state (entering drybubl and wetbulb) and CFM

        Source:
        -------


        Inputs:
        -------
            Tdb    float    Entering Dry Bulb (degC)
            Twb    float    Entering Wet Bulb (degC)
            P      float    Barometric pressure (kPa)
            flow   float    Volumetric flow rate of unit (m^3/s)
        Outputs:
        --------
            mfr    float    mass flow rate (kg/s)
        '''
        Win = psychrolib.GetHumRatioFromTWetBulb(DBin, WBin, P * 1000)
        rho_in = psychrolib.GetMoistAirDensity(DBin, Win, P * 1000)
        mfr = flow * rho_in
        return mfr

    @staticmethod
    def CalculateSHR_SI(DBin, WBin, P, Q, flow, Ao):
        '''
               Description:
                ------------
                    Calculate the coil SHR at the given incoming air state, CFM, total capacity, and coil
                    Ao factor

                Source:
                -------
                    EnergyPlus source code

                Inputs:
                -------
                    Tdb    float    Entering Dry Bulb (degC)
                    Twb    float    Entering Wet Bulb (degC)
                    P      float    Barometric pressure (kPa)
                    Q      float    Total capacity of unit (kW)
                    flow   float    Volumetric flow rate of unit (m^3/s)
                    Ao     float    Coil Ao factor (=UA/Cp - IN SI UNITS)
                Outputs:
                --------
                    SHR    float    Sensible Heat Ratio
                '''
        mfr = Psychrometrics.CalculateMassflowRate_SI(DBin, WBin, P, flow)
        bf = math.exp(-1.0 * Ao / mfr)

        Win = psychrolib.GetHumRatioFromTWetBulb(DBin, WBin, P * 1000)
        # P = Units.psi2kPa(P)
        # DBin = Units.F2C(DBin)
        # Hin = Psychrometrics.h_fT_w_SI(Tin, Win)
        Hin = psychrolib.GetMoistAirEnthalpy(DBin, Win)  # in J/kg
        dH = Q * 1000 / mfr
        H_ADP = Hin - dH / (1 - bf)

        # T_ADP = Psychrometrics.Tsat_fh_P_SI(H_ADP, P)
        # W_ADP = Psychrometrics.w_fT_h_SI(T_ADP, H_ADP)

        # Initialize
        T_ADP = psychrolib.GetTDewPointFromHumRatio(DBin, Win, P * 1000)
        T_ADP_1 = T_ADP  # (degC)
        T_ADP_2 = T_ADP  # (degC)
        W_ADP = psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, P * 1000)
        # error = H_ADP - Psychrometrics.h_fT_w_SI(T_ADP, W_ADP)
        error = H_ADP - psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)
        error1 = error
        error2 = error

        itmax = 50  # maximum iterations
        cvg = False

        for i in range(1, itmax + 1):

            W_ADP = psychrolib.GetHumRatioFromRelHum(T_ADP, 1.0, P * 1000)
            # error = H_ADP - Psychrometrics.h_fT_w_SI(T_ADP, W_ADP)
            error = H_ADP - psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)

            T_ADP, cvg, T_ADP_1, error1, T_ADP_2, error2 = \
                Iterate(T_ADP, error, T_ADP_1, error1, T_ADP_2, error2, i, cvg)

            if cvg:
                break

        if not cvg:
            print('Warning: Tsat_fh_P failed to converge')

        # h_Tin_Wadp = Psychrometrics.h_fT_w_SI(Tin, W_ADP)
        h_Tin_Wadp = psychrolib.GetMoistAirEnthalpy(DBin, W_ADP)

        if Hin - H_ADP != 0:
            shr = min((h_Tin_Wadp - H_ADP) / (Hin - H_ADP), 1.0)
        else:
            shr = 1

        return shr

    #
    # @staticmethod
    # def CalculateSHR(DBin, WBin, P, Q, cfm, Ao):
    #     '''
    #    Description:
    #     ------------
    #         Calculate the coil SHR at the given incoming air state, CFM, total capacity, and coil
    #         Ao factor
    #
    #     Source:
    #     -------
    #         EnergyPlus source code
    #
    #     Inputs:
    #     -------
    #         Tdb    float    Entering Dry Bulb (degF)
    #         Twb    float    Entering Wet Bulb (degF)
    #         P      float    Barometric pressure (psi)
    #         Q      float    Total capacity of unit (kBtu/h)
    #         cfm    float    Volumetric flow rate of unit (CFM)
    #         Ao     float    Coil Ao factor (=UA/Cp - IN SI UNITS)
    #     Outputs:
    #     --------
    #         SHR    float    Sensible Heat Ratio
    #     '''
    #
    #     mfr = Units.lbm_min2kg_s(Psychrometrics.CalculateMassflowRate(DBin, WBin, P, cfm))
    #     bf = math.exp(-1.0 * Ao / mfr)
    #
    #     Win = Psychrometrics.w_fT_Twb_P(DBin, WBin, P)
    #     P = Units.psi2kPa(P)
    #     Tin = Units.F2C(DBin)
    #     Hin = Psychrometrics.h_fT_w_SI(Tin, Win)
    #     dH = Units.kBtu_h2W(Q) / mfr
    #     H_ADP = Hin - dH / (1 - bf)
    #
    #     # T_ADP = Psychrometrics.Tsat_fh_P_SI(H_ADP, P)
    #     # W_ADP = Psychrometrics.w_fT_h_SI(T_ADP, H_ADP)
    #
    #     # Initialize
    #     T_ADP = Psychrometrics.Tdp_fP_w_SI(P, Win)
    #     T_ADP_1 = T_ADP  # (degC)
    #     T_ADP_2 = T_ADP  # (degC)
    #     W_ADP = Psychrometrics.w_fT_R_P_SI(T_ADP, 1.0, P)
    #     error = H_ADP - Psychrometrics.h_fT_w_SI(T_ADP, W_ADP)
    #     error1 = error
    #     error2 = error
    #
    #     itmax = 50  # maximum iterations
    #     cvg = False
    #
    #     for i in range(1, itmax + 1):
    #
    #         W_ADP = Psychrometrics.w_fT_R_P_SI(T_ADP, 1.0, P)
    #         error = H_ADP - Psychrometrics.h_fT_w_SI(T_ADP, W_ADP)
    #
    #         T_ADP, cvg, T_ADP_1, error1, T_ADP_2, error2 = \
    #             Iterate(T_ADP, error, T_ADP_1, error1, T_ADP_2, error2, i, cvg)
    #
    #         if cvg == True:
    #             break
    #
    #     if cvg == False:
    #         print('Warning: Tsat_fh_P failed to converge')
    #
    #     h_Tin_Wadp = Psychrometrics.h_fT_w_SI(Tin, W_ADP)
    #
    #     if (Hin - H_ADP != 0):
    #         shr = min((h_Tin_Wadp - H_ADP) / (Hin - H_ADP), 1.0)
    #     else:
    #         shr = 1
    #
    #     return shr

    @staticmethod
    def CoilAoFactor_SI(DBin, WBin, P, Qdot, flow, shr):
        '''
       Description:
        ------------
            Find the coil Ao factor at the given incoming air state (entering drybubl and wetbulb) and CFM,
            total capacity and SHR


        Source:
        -------
            EnergyPlus source code

        Inputs:
        -------
            Tdb    float    Entering Dry Bulb (degC)
            Twb    float    Entering Wet Bulb (degC)
            P      float    Barometric pressure (kPa)
            Qdot   float    Total capacity of unit (kW)
            cfm    float    Volumetric flow rate of unit (m^3/s)
            shr    float    Sensible heat ratio
        Outputs:
        --------
            Ao    float    Coil Ao Factor
        '''
        bf = Psychrometrics.CoilBypassFactor_SI(DBin, WBin, P, Qdot, flow, shr)
        mfr = Psychrometrics.CalculateMassflowRate_SI(DBin, WBin, P, flow)

        ntu = -1.0 * math.log(bf)
        Ao = ntu * mfr
        return Ao

    # @staticmethod
    # def CoilAoFactor(DBin, WBin, P, Qdot, cfm, shr):
    #     '''
    #    Description:
    #     ------------
    #         Find the coil Ao factor at the given incoming air state (entering drybubl and wetbulb) and CFM,
    #         total capacity and SHR
    #
    #
    #     Source:
    #     -------
    #         EnergyPlus source code
    #
    #     Inputs:
    #     -------
    #         Tdb    float    Entering Dry Bulb (degF)
    #         Twb    float    Entering Wet Bulb (degF)
    #         P      float    Barometric pressure (psi)
    #         Qdot   float    Total capacity of unit (kBtu/h)
    #         cfm    float    Volumetric flow rate of unit (CFM)
    #         shr    float    Sensible heat ratio
    #     Outputs:
    #     --------
    #         Ao    float    Coil Ao Factor
    #     '''
    #
    #     bf = Psychrometrics.CoilBypassFactor(DBin, WBin, P, Qdot, cfm, shr)
    #     mfr = Units.lbm_min2kg_s(Psychrometrics.CalculateMassflowRate(DBin, WBin, P, cfm))
    #
    #     ntu = -1.0 * math.log(bf)
    #     Ao = ntu * mfr
    #     return Ao
    #
    # @staticmethod
    # def CoilBypassFactor(DBin, WBin, P, Qdot, cfm, shr):
    #     '''
    #    Description:
    #     ------------
    #         Find the coil bypass factor at the given incoming air state (entering drybubl and wetbulb) and CFM,
    #         total capacity and SHR
    #
    #
    #     Source:
    #     -------
    #         EnergyPlus source code
    #
    #     Inputs:
    #     -------
    #         Tdb    float    Entering Dry Bulb (degF)
    #         Twb    float    Entering Wet Bulb (degF)
    #         P      float    Barometric pressure (psi)
    #         Qdot   float    Total capacity of unit (kBtu/h)
    #         cfm    float    Volumetric flow rate of unit (CFM)
    #         shr    float    Sensible heat ratio
    #     Outputs:
    #     --------
    #         CBF    float    Coil Bypass Factor
    #     '''
    #
    #     mfr = Units.lbm_min2kg_s(Psychrometrics.CalculateMassflowRate(DBin, WBin, P, cfm))
    #
    #     Tin = Units.F2C(DBin)
    #     Win = Psychrometrics.w_fT_Twb_P(DBin, WBin, P)
    #     P = Units.psi2kPa(P)
    #
    #     dH = Units.kBtu_h2W(Qdot) / mfr
    #     Hin = Psychrometrics.h_fT_w_SI(Tin, Win)
    #     Hin = psychrolib.GetMoistAirEnthalpy(Tin, Win)
    #     h_Tin_Wout = Hin - (1 - shr) * dH
    #     Wout = Psychrometrics.w_fT_h_SI(Tin, h_Tin_Wout)
    #     dW = Win - Wout
    #     Hout = Hin - dH
    #     Tout = Psychrometrics.T_fw_h_SI(Wout, Hout)
    #     RH_out = Psychrometrics.R_fT_w_P_SI(Tout, Wout, P)
    #
    #     T_ADP = Psychrometrics.Tdp_fP_w_SI(P, Wout)  # Initial guess for iteration
    #
    #     if shr == 1:
    #         W_ADP = Psychrometrics.w_fT_Twb_P_SI(T_ADP, T_ADP, P)
    #         H_ADP = Psychrometrics.h_fT_w_SI(T_ADP, W_ADP)
    #         H_ADP = psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)
    #         BF = (Hout - H_ADP) / (Hin - H_ADP)
    #         return max(BF, 0.01)
    #
    #     if RH_out > 1:
    #         print('Error: Conditions passed to CoilBypassFactor result in outlet RH > 100%')
    #
    #     dT = Tin - Tout
    #     M_c = dW / dT
    #
    #     cnt = 0
    #     tol = 1.0
    #     errorLast = 100
    #     d_T_ADP = 5.0
    #
    #     W_ADP = None
    #     while cnt < 100 and tol > 0.001:
    #         # for i in range(1,itmax+1):
    #
    #         if cnt > 0:
    #             T_ADP = T_ADP + d_T_ADP
    #
    #         W_ADP = Psychrometrics.w_fT_Twb_P_SI(T_ADP, T_ADP, P)
    #
    #         M = (Win - W_ADP) / (Tin - T_ADP)
    #         error = (M - M_c) / M_c
    #
    #         if error > 0 and errorLast < 0:
    #             d_T_ADP = -1.0 * d_T_ADP / 2.0
    #
    #         if error < 0 and errorLast > 0:
    #             d_T_ADP = -1.0 * d_T_ADP / 2.0
    #
    #         errorLast = error
    #         tol = math.fabs(error)
    #         cnt = cnt + 1
    #
    #     H_ADP = Psychrometrics.h_fT_w_SI(T_ADP, W_ADP)
    #     H_ADP = psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)
    #
    #     BF = (Hout - H_ADP) / (Hin - H_ADP)
    #     return max(BF, 0.01)

    @staticmethod
    def CoilBypassFactor_SI(DBin, WBin, P, Qdot, flow, shr):
        '''
       Description:
        ------------
            Find the coil bypass factor at the given incoming air state (entering drybubl and wetbulb) and CFM,
            total capacity and SHR


        Source:
        -------
            EnergyPlus source code

        Inputs:
        -------
            Tdb    float    Entering Dry Bulb (degC)
            Twb    float    Entering Wet Bulb (degC)
            P      float    Barometric pressure (kPa)
            Qdot   float    Total capacity of unit (kW)
            flow   float    Volumetric flow rate of unit (m^3/s)
            shr    float    Sensible heat ratio
        Outputs:
        --------
            CBF    float    Coil Bypass Factor
        '''

        mfr = Psychrometrics.CalculateMassflowRate_SI(DBin, WBin, P, flow)

        Win = psychrolib.GetHumRatioFromTWetBulb(DBin, WBin, P * 1000)

        dH = Qdot * 1000 / mfr  # W / kg/s == J/kg
        Hin = psychrolib.GetMoistAirEnthalpy(DBin, Win)
        h_Tin_Wout = Hin - (1 - shr) * dH
        Wout = psychrolib.GetHumRatioFromEnthalpyAndTDryBulb(h_Tin_Wout, DBin)
        dW = Win - Wout
        Hout = Hin - dH
        Tout = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(Hout, Wout)
        RH_out = psychrolib.GetRelHumFromHumRatio(Tout, Wout, P * 1000)

        T_ADP = psychrolib.GetTDewPointFromHumRatio(Tout, Wout, P * 1000)  # Initial guess for iteration

        if shr == 1:
            W_ADP = psychrolib.GetHumRatioFromTWetBulb(T_ADP, T_ADP, P * 1000)
            H_ADP = psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)
            BF = (Hout - H_ADP) / (Hin - H_ADP)
            return max(BF, 0.01)

        if RH_out > 1:
            print('Error: Conditions passed to CoilBypassFactor result in outlet RH > 100%')

        dT = DBin - Tout
        M_c = dW / dT

        cnt = 0
        tol = 1.0
        errorLast = 100
        d_T_ADP = 5.0

        W_ADP = None
        while cnt < 100 and tol > 0.001:
            # for i in range(1,itmax+1):

            if cnt > 0:
                T_ADP = T_ADP + d_T_ADP

            W_ADP = psychrolib.GetHumRatioFromTWetBulb(T_ADP, T_ADP, P * 1000)

            M = (Win - W_ADP) / (DBin - T_ADP)
            error = (M - M_c) / M_c

            if error > 0 and errorLast < 0:
                d_T_ADP = -1.0 * d_T_ADP / 2.0

            if error < 0 and errorLast > 0:
                d_T_ADP = -1.0 * d_T_ADP / 2.0

            errorLast = error
            tol = math.fabs(error)
            cnt = cnt + 1

        H_ADP = psychrolib.GetMoistAirEnthalpy(T_ADP, W_ADP)

        BF = (Hout - H_ADP) / (Hin - H_ADP)
        return max(BF, 0.01)

    # @staticmethod
    # def T_ldb(Tdb, Twb, cfm, sens, total):
    #     '''
    #    Description:
    #     ------------
    #         Find the leaving dry bulb temp of the unit given incoming air state (entering drybulb and wetbulb) and
    #         CFM, sensible and total capacities
    #
    #
    #     Source:
    #     -------
    #         Dylan Cutler 2/16/11
    #
    #     Inputs:
    #     -------
    #         Tdb    float    Entering Dry Bulb (degF)
    #         Twb    float    Entering Wet Bulb (degF)
    #         sens   float    Sensible capacity of unit (MBtu/h)
    #         total  float    Total capacity of unit (MBtu/h)
    #         cfm    float    Volumetric flow rate of unit (CFM)
    #     Outputs:
    #     --------
    #         ldb  float    Drybulb temperature of appartus dew point (degF)
    #     '''
    #
    #     # define necessary variables for leaving wet bulb
    #     w_i = Psychrometrics.w_fT_Twb_P(Tdb, Twb, Constants.Patm)
    #     m_dot = Units.hr2min(float(cfm) * Psychrometrics.rhoM_fT_w_P(Tdb, w_i, Constants.Patm))  # lbm/hr
    #
    #     ldb = Tdb - Units.kBtu_h2Btu_h(float(sens)) / (m_dot * Properties.Air.Cp)
    #
    #     return ldb
