import psychrolib

psychrolib.SetUnitSystem(psychrolib.SI)

h_vap = 2454  # kJ/kg
humidity_cap_mult = 15.0


class HumidityModel:
    def __init__(self, t_indoor, time_res, **kwargs):
        """
        Dwelling humidity model
        """
        self.time_res = time_res
        self.latent_gains = 0  # in W

        # Ventilated space parameters
        self.indoor_volume = kwargs['building volume (m^3)']

        # Initial conditions - use outdoor humidity ratio (w) for initial indoor w
        # t_outdoor = kwargs['initial_schedule']['ambient_dry_bulb']
        p_outdoor = kwargs['initial_schedule']['ambient_pressure']
        w_outdoor = kwargs['initial_schedule']['ambient_humidity']

        p_outdoor *= 1000  # kPa to Pa
        self.indoor_w = w_outdoor  # assume same starting humidity ratio as outdoor, in kgH20/kgAir
        self.indoor_rh = psychrolib.GetRelHumFromHumRatio(t_indoor, self.indoor_w, p_outdoor)
        self.indoor_density = psychrolib.GetMoistAirDensity(t_indoor, self.indoor_w, p_outdoor)  # moist air, in kg/m^3
        self.indoor_wet_bulb = psychrolib.GetTWetBulbFromHumRatio(t_indoor, self.indoor_w, p_outdoor)  # in deg C

    def update_humidity(self, t_indoor, ach_indoor, t_outdoor, w_outdoor, p_outdoor):
        """
        Update dwelling humidity given:
            Occupancy metabolism
            Appliance latent gains
            Wind speed (air changes)
            Outside pressure
            HVAC latent cooling

            Inputs are in units of degC, kPa, fraction (for RH), W (for latent gains)
        """
        # FUTURE: Dehumidifier? Latent portion of HPWH gains?

        p_outdoor *= 1000  # kPa to Pa
        latent_gains_w = self.latent_gains * self.time_res.total_seconds() / 1000 / (
                self.indoor_density * self.indoor_volume * h_vap)  # unitless latent gains
        # w_outdoor = psychrolib.GetHumRatioFromRelHum(t_outdoor, rh_outdoor, p_outdoor)

        # Update moisture balance calculations
        hours = self.time_res.total_seconds() / 3600
        self.indoor_w += (latent_gains_w + ach_indoor * hours * (w_outdoor - self.indoor_w)) / humidity_cap_mult
        if self.indoor_w < 0:
            self.indoor_w = 0
            # TODO: add warnings back after running test suite (intgain test has high latent gains)
            # print("WARNING: Indoor Relative Humidity less than 0%, double check inputs.")

        self.indoor_rh = psychrolib.GetRelHumFromHumRatio(t_indoor, self.indoor_w, p_outdoor)
        if self.indoor_rh > 1:
            # print("WARNING: Indoor Relative Humidity greater than 100%, condensation is occuring.")
            self.indoor_rh = 1
            self.indoor_w = psychrolib.GetHumRatioFromRelHum(t_indoor, self.indoor_rh, p_outdoor)

        self.indoor_density = psychrolib.GetMoistAirDensity(t_indoor, self.indoor_w, p_outdoor)  # kg/m^3

        self.indoor_wet_bulb = psychrolib.GetTWetBulbFromHumRatio(t_indoor, self.indoor_w, p_outdoor)

    @staticmethod
    def get_dry_air_density(t, w, p):
        # calculate dry air density using moist air density and humidity ratio
        density = psychrolib.GetMoistAirDensity(t, w, p * 1000)  # moist air, in kg/m^3
        dry_density = density * 1 / (1 + w)
        return dry_density
