

# --------------- #
# !-- Imports --! #
# --------------- #

import os
import pandas as pd


# ------------- #
# !-- Files --! #
# ------------- #


data_folder = os.path.join(
    os.path.dirname(__file__), "data"
)

# let's load all NASA Exoplanet Archive csvs
archive_filenames = [
    file for file in os.listdir(data_folder) if file.endswith("nasa-exoplanet-archive.csv")
]
assert len(archive_filenames) > 0, "No NASA Exoplanet Archive data found. This shouldn't be possible."

# sort them alphabetically (since this way the newest will be last)
archive_filenames.sort()
filepath = os.path.join(data_folder, archive_filenames[-1])


# ------------------- #
# !-- Data Loader --! #
# ------------------- #

database = None

def get_database() -> pd.DataFrame:
    """
    Loads (and caches) the Nasa Exoplanet Archive database as a pandas DataFrame.
    """
    global database
    if database is None:
        database = pd.read_csv(filepath)
    return database