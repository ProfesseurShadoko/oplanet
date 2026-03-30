
# --------------- #
# !-- Imports --! #
# --------------- #

import numpy as np
import pandas as pd
import requests
from oakley import *
import datetime
import os

# ------------- #
# !-- Paths --! #
# ------------- #

data_folder = os.path.join(
    os.path.dirname(__file__), "data"
)
os.makedirs(data_folder, exist_ok=True)

# let's load all NASA Exoplanet Archive csvs
archive_filenames = [
    file for file in os.listdir(data_folder) if file.endswith("nasa-exoplanet-archive.csv")
]

# sort them alphabetically (since this way the newest will be last)
archive_filenames.sort()

# keep only the last file, erase all others
for filename in archive_filenames[:-1]:
    os.remove(os.path.join(data_folder, filename))

# check the date of the only remaning one, if it's older than 50 days, erase it
if len(archive_filenames) > 0:
    last_filename = archive_filenames[-1]
    last_file_date = datetime.datetime.strptime(
        last_filename.split("_")[0], "%Y-%m-%d"
    )
    if (datetime.datetime.now() - last_file_date).days > 50:
        Message(f"Last NASA Exoplanet Archive data is older than 50 days, deleting it and downloading a new one", "!")
        os.remove(os.path.join(data_folder, last_filename))
else:
    Message("No NASA Exoplanet Archive data found, downloading it", "!")

# count again how many files are left
archive_filenames = [
    file for file in os.listdir(data_folder) if file.endswith("nasa-exoplanet-archive.csv")
]
if len(archive_filenames) == 0:




    # --------------------- #
    # !-- Download data --! #
    # --------------------- #

    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    query = "select * from ps"

    params = {
        "query": query,
        "format": "csv"
    }

    with Task("Loading data from NASA Exoplanet Archive (should take about 60s...)"):
        response = requests.get(url, params=params)
    response.raise_for_status()

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    with open(os.path.join(
        data_folder, f"{today_str}_nasa-exoplanet-archive.csv"
    ), "wb") as f:
        f.write(response.content)