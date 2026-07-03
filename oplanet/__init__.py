

__version__ = "1.4.11"
from .oconfig import oplanet_config, oplanet_temp_config, reset_config, update_default_config

# ----------------------- #
# !-- Simbad + Vizier --! #
# ----------------------- #

from .star_utils import get_photometry_jy, get_distance_pc, get_star_coords
from .star_utils import get_star_aliases, is_star_alias, get_star_name, parse_star_name


# ----------- #
# !-- SVO --! #
# ----------- #

from .sfilter import SFilter

for facility in ["2MASS", "WISE", "GAIA", "JWST"]:
    # check wether they are already loaded.
    if not SFilter.is_cached(facility):
        SFilter.download(facility)


# ------------------------------ #
# !-- NASA Exoplanet Archive --! #
# ------------------------------ #

from .nsystem import NSystem
