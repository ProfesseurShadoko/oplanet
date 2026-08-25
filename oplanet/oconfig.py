
from oakley import XConfig
import os
from copy import deepcopy

# ---------------- #
# !-- Filepath --! #
# ---------------- #

dirname = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(dirname, exist_ok=True)


# ---------------------- #
# !-- Default Config --! #
# ---------------------- #

default_config = {
    "system": {
        "references": [], # format "{author}_{year}", list of publications to prioritize when getting values in Nasa Exoplanet Archive
        "properties": {
            "parallax_mas":1, # which parameters you intend to use, and how much you want them to be not nans
            "distance_pc":1,
            "ra":1,
            "dec":1
        },
        "fallback": False,
        "order_authors": True
    },
    "star": {
        "references": [],
        "properties": {
            "age_myr":1,
            "mass_msun":1,
            "radius_rsun":1,
            "metallicity_dex":1,
            "system.distance_pc":1
        },
        "fallback": False,
        "order_authors": True
    },
    "planet": {
        "references": [],
        "properties": {
            "star.age_myr":1,
            "system.distance_pc":1,
            "orbital_period_yrs":2,
            "msini_mjup":1,
            "mass_mjup":3,
            "sma_au":2,
            "eccentricity":1,
            "inclination_deg":1,
            "arg_periastron_deg":1,
            "time_periastron_jd":1,
            "rv_amplitude_ms":1,
        },
        "fallback": False,
        "order_authors": True
    },
}

default_config = XConfig(filepath=os.path.join(dirname, "default_config.json"), default_config=default_config)

class OPlanetConfig(dict):
    def __init__(self):
        self.default_config = default_config
        super().__init__(deepcopy(default_config))

    def reset(self):
        """
        Resets the current configuration to be the default configuration.
        """
        self.clear()
        self.update(deepcopy(self.default_config))

    def dump(self):
        """
        Makes the current configuration become the default configuration
        for the future.
        """
        self.default_config.clear()
        self.default_config.update(deepcopy(self))
        self.default_config._dump() # save to file

    def reset_default(self):
        """
        Resets the default configuration to be the hard coded default configuration.
        """
        self.default_config.reset()

oplanet_config = OPlanetConfig()