import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)


class HumidityModel:
    h_vap = 2454  # kJ/kg
    humidity_cap_mult = 15.0

    def __init__(self, time_res, initial_schedule, volume, t_zone=20):
        """
        Dwelling humidity model
        """
        self.time_res = time_res
        self.latent_gains = 0  # in W
        self.latent_gains_init = self.latent_gains  # for saving values from update_inputs step
        self.volume = volume
        self.max_latent_flow = self.volume * self.humidity_cap_mult / self.time_res.total_seconds()  # in m^3/s

        # Initial conditions - use outdoor humidity ratio (w) for initial indoor w
        # t_outdoor = kwargs['initial_schedule']['Ambient Dry Bulb (C)']
        p_outdoor = initial_schedule.get('Ambient Pressure (kPa)', 101.325)
        w_outdoor = initial_schedule['Ambient Humidity Ratio (-)']

        self.pressure = p_outdoor * 1000  # in Pa
        self.w = w_outdoor  # assume same starting humidity ratio as outdoor, in kgH20/kgAir
        self.rh = psychrolib.GetRelHumFromHumRatio(t_zone, self.w, self.pressure)
        self.density = psychrolib.GetMoistAirDensity(t_zone, self.w, self.pressure)  # moist air, in kg/m^3
        self.wet_bulb = psychrolib.GetTWetBulbFromHumRatio(t_zone, self.w, self.pressure)  # in deg C

    def update_humidity(self, t_indoor):
        """
        Latent gains incorporate:
         - HVAC latent cooling
         - Air changes (infiltration and ventilation)
         - Appliances
         - Occupancy

        Inputs are in units of degC, kPa, fraction (for RH), W (for latent gains)
        """
        # FUTURE: Dehumidifier?

        # t_outdoor = schedule['Ambient Dry Bulb (C)']
        # w_outdoor = schedule['Ambient Humidity Ratio (-)']
        # self.pressure = schedule['Ambient Pressure (kPa)'] * 1000  # already updated in Envelope.update_inputs

        # calculate unitless latent gains
        latent_gains_w = self.latent_gains * self.time_res.total_seconds() / 1000 / (
                self.density * self.volume * self.h_vap)
        # w_outdoor = psychrolib.GetHumRatioFromRelHum(t_outdoor, rh_outdoor, p_outdoor)

        # Update moisture balance calculations
        self.w += latent_gains_w / self.humidity_cap_mult
        if self.w < 0:
            self.w = 0
            # TODO: add warnings back after running test suite (intgain test has high latent gains)
            # print("WARNING: Indoor Relative Humidity less than 0%, double check inputs.")

        # Calculate relative humidity, density, and wet bulb temp
        self.rh = psychrolib.GetRelHumFromHumRatio(t_indoor, self.w, self.pressure)
        if self.rh > 1:
            # print("WARNING: Indoor Relative Humidity greater than 100%, condensation is occurring.")
            self.rh = 1
            self.w = psychrolib.GetHumRatioFromRelHum(t_indoor, self.rh, self.pressure)

        self.density = psychrolib.GetMoistAirDensity(t_indoor, self.w, self.pressure)  # kg/m^3
        self.wet_bulb = psychrolib.GetTWetBulbFromHumRatio(t_indoor, self.w, self.pressure)

    @staticmethod
    def get_dry_air_density(t, w, p):
        # calculate dry air density using moist air density and humidity ratio
        # assumes SI units: deg C, kgH20/kgAir, kPa -> kg/m^3
        density = psychrolib.GetMoistAirDensity(t, w, p * 1000)  # moist air, in kg/m^3
        dry_density = density * 1 / (1 + w)
        return dry_density
