

__version__ = "1.3.0"


# ----------------------- #
# !-- Simbad + Vizier --! #
# ----------------------- #

from .star_utils import get_photometry_jy, get_distance_pc, get_star_coords
from .star_utils import get_star_aliases, is_star_alias, get_star_name, parse_star_name


# ----------- #
# !-- SVO --! #
# ----------- #

for facility in ["2MASS", "WISE", "GAIA", "JWST"]:
    for instrument in ["ALL"]:
        from .sfilter import SFilter
        SFilter.get_filters(facility, instrument)


# ------------------------------ #
# !-- NASA Exoplanet Archive --! #
# ------------------------------ #

from .data_loaders import check_if_old
check_if_old("nasa")
#check_if_old("eu")
from .nsystem import NSystem
#from .esystem import ESystem
