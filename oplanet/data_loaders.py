

# --------------- #
# !-- Imports --! #
# --------------- #

import numpy as np
import pandas as pd
import requests
from oakley import *
import datetime
import os
from typing import Literal

# ------------- #
# !-- Paths --! #
# ------------- #

data_folder = os.path.join(
    os.path.dirname(__file__), "data"
)
os.makedirs(data_folder, exist_ok=True)

# ------------ #
# !-- URLs --! #
# ------------ #

eu_url = "https://exoplanet.eu/catalog/csv"
nasa_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
nasa_params = {
    "query": "select * from ps",
    "format": "csv",
}


# ------------------- #
# !-- Delete data --! #
# ------------------- #

def get_suffix(source: Literal["eu", "nasa"]):
    return {
        "nasa": "nasa-exoplanet-archive.csv",
        "eu": "exoplanet-eu.csv"
    }[source]


# ------------ #
# !-- Date --! #
# ------------ #

def get_today_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def get_filename(source: Literal["eu", "nasa"]):
    return f"{get_today_str()}_{get_suffix(source)}"

def archive_filename(source: Literal["eu", "nasa"]):
    suffix = get_suffix(source)
    archive_filenames = [
        file for file in os.listdir(data_folder) if file.endswith(suffix)
    ]
    assert len(archive_filenames) <= 1, f"More than one file found for source {source} in archive: {archive_filenames}. Should not happen."
    return archive_filenames[0] if len(archive_filenames) == 1 else None

def get_archive_date(source: Literal["eu", "nasa"]):
    filename = archive_filename(source)
    if filename is None:
        return None
    return datetime.datetime.strptime(filename.split("_")[0], "%Y-%m-%d")


# -------------------- #
# !-- NASA Archive --! #
# -------------------- #

def download_nasa_exoplanet_archive():
    with Task("Loading data from NASA Exoplanet Archive (should take about 60s...)"):
        response = requests.get(nasa_url, params=nasa_params)
    response.raise_for_status()
    
    with open(os.path.join(data_folder, get_filename("nasa")), "wb") as f:
        f.write(response.content)
        
        
# -------------------- #
# !-- Exoplanet EU --! #
# -------------------- #
    
def download_eu_exoplanet_catalog():
    with Task("Loading data from Exoplanet.eu Catalog (should take about 60s...)"):
        response = requests.get(eu_url)
    response.raise_for_status()
    
    with open(os.path.join(data_folder, get_filename("eu")), "wb") as f:
        f.write(response.content)


# --------------- #
# !-- Refresh --! #
# --------------- #

def refresh_data(source: Literal["eu", "nasa"]):
    # 1. Delete old files
    archive_file = archive_filename(source)
    if archive_file is not None:
        os.remove(os.path.join(data_folder, archive_file))
        Message(f"Deleted {os.path.basename(archive_file)}.", "?")
    
    # 2. Download new file
    if source == "nasa":
        download_nasa_exoplanet_archive()
    elif source == "eu":
        download_eu_exoplanet_catalog()

def check_if_old(source: Literal["eu", "nasa"], max_age_days: int = 50):
    archive_date = get_archive_date(source)
    if archive_date is None:
        Message(f"No archive file found for source {source}. Please refresh it with the `refresh()` static method.", "!")
    elif (datetime.datetime.now() - archive_date).days > max_age_days:
        Message(f"Archive file is older than {max_age_days} days. Please refresh it with the `refresh()` static method.", "!")


# ------------------- #
# !-- Data Loader --! #
# ------------------- #

database_eu = None
database_nasa = None

def get_database(source: Literal["eu", "nasa"]) -> pd.DataFrame:
    """
    Loads (and caches) the {source} database as a pandas DataFrame.
    """
    assert source in ["eu", "nasa"], f"Source must be 'eu' or 'nasa', got {source}."
    global database_eu, database_nasa
    if source == "eu":
        if database_eu is None:
            filepath = archive_filename("eu")
            assert filepath is not None, "No Exoplanet.eu catalog file found in archive. Please refresh it with the `refresh()` static method."
            database_eu = pd.read_csv(
                os.path.join(data_folder, filepath), low_memory=False
            )
        return database_eu
    elif source == "nasa":
        if database_nasa is None:
            filepath = archive_filename("nasa")
            assert filepath is not None, "No NASA Exoplanet Archive file found in archive. Please refresh it with the `refresh()` static method."
            database_nasa = pd.read_csv(os.path.join(data_folder, filepath), low_memory=False)
        return database_nasa