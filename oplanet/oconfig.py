
from oakley import XConfig
import os

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
        "fallback": False
    },
    "star": {
        "references": [],
        "properties": {
            "age_myr":1,
            "mass_solar":1,
            "radius_solar":1,
            "metallicity_dex":1,
            "system.distance_pc":1
        },
        "fallback": False
    },
    "planet": {
        "references": [],
        "properties": {
            "star.age_myr":1,
            "system.distance_pc":1,
            "orbital_period_yrs":2,
            "mass_sini_mjup":1,
            "mass_mjup":3,
            "sma_au":2,
            "eccentricity":1,
            "inclination_deg":1,
            "arg_periastron_deg":1,
            "time_periastron_jd":1,
            "rv_amplitude_ms":1,
        },
        "fallback": False
    }
}


oplanet_config = XConfig(
    dirname, default_config=default_config
)
oplanet_temp_config = dict(oplanet_config)