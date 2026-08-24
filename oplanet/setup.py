from oakley import *


Message("(Re)initializing the datasets for oplanet.").par(
"""
The current function overwrites existing datasets by loading their new versions
from the internet (but won't erase anything if the download fails). This is useful
to ensure that the datasets are up to date (espacially for the NASA Exoplanet Archive
where new discoveries or measurements might be added frequently).
"""
)
ok = Message("Do you want to continue? [y/n]", "?").input(parser=lambda x: "y" in x.lower())

if not ok:
    Message("Aborting.", "!")
    exit(0)


Message.title("Nasa Exoplanet Archive")
from .nsystem import NSystem
NSystem.refresh()

Message.print()
Message.title("Inversion Tables")
from .einversion import EInversion
EInversion.download()

Message.print()
Message.title("SVO Filter Profile Service")

from .sfilter import SFilter
for facility in ["2MASS", "WISE", "GAIA", "JWST"]:
    SFilter.download(facility)

