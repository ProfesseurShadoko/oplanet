

__version__ = "1.3.1"


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

from .data_loaders import check_if_old
check_if_old("nasa")
#check_if_old("eu")
from .nsystem import NSystem
#from .esystem import ESystem
