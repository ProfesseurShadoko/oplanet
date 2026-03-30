

__version__ = "0.1.0"


# ----------------------- #
# !-- Simbad + Vizier --! #
# ----------------------- #

from .star_utils import get_photometry_jy, get_distance_pc, get_star_coords
from .star_utils import get_star_aliases, is_star_alias, get_star_name, parse_star_name


# ------------------------------ #
# !-- NASA Exoplanet Archive --! #
# ------------------------------ #

from . import dataloader # downloads the database as csv file
from .utils import get_database # contains loader of the database
from .osystem import OSystem