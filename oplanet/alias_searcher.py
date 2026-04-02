
# --------------- #
# !-- Imports --! #
# --------------- #

import oakley
from .star_utils import get_star_aliases, parse_star_name,  get_star_name
from .data_loaders import get_database, refresh_data

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import os
from oakley import *
from typing import Literal


# ---------------------- #
# !-- Load dataframe --! #
# ---------------------- #

df = get_database("eu")
Message(f"Number of planets in the Exoplanet.eu database: {len(df)}")
database_size = len(df)

df = pd.read_csv(
    "/home/kiwi/documents/these/oplanet/exoplanet_eu_sample.csv"
)

# -------------- #
# !-- Output --! #
# -------------- #

planet_name2_aliases_map = {}
# we will revert it to have aliases to planet names


# get alias for each row
with Task("Getting star aliases for rows..."):
    num_direct_successes = 0
    num_secondary_successs = 0
    num_fails = 0
    
    for i in ProgressBar(range(len(df))):
        row = df.iloc[i]
        star_name = row["star_name"]
        planet_name = row["name"]
        alternate_names = row["star_alternate_names"]
        
        Message.print()
        Message(f"Processing row {i}").list({
            "Star name": star_name,
            "Planet name": planet_name,
            "Alternate names": alternate_names
        })
        
        # get aliases
        if isinstance(star_name, str):
            with Message.mute():
                star_aliases = get_star_aliases(star_name)
        Message("Aliases:").list(star_aliases) # empty if not found
        
        if len(star_aliases) > 0:
            num_direct_successes += 1
            # fill the column with the aliases (as a comma-separated string)
            planet_name2_aliases_map[planet_name] = star_aliases
            continue
        
        # fall back on alternate names
        if isinstance(alternate_names, str):
            alternate_names = alternate_names.split(",")
            
            for alt_name in alternate_names:
                alt_name = alt_name.strip()
                with Message.mute():
                    star_aliases = get_star_aliases(alt_name)
                if len(star_aliases) > 0:
                    num_secondary_successs += 1
                    planet_name2_aliases_map[planet_name] = star_aliases
                    break
            else:
                Message(f"Could not find aliases for star: {star_name} (row {i}, planet {planet_name})", "!")
                num_fails += 1
                continue
        else:
            Message(f"Could not find aliases for star: {star_name} (row {i}, planet {planet_name})", "!")
            num_fails += 1
            continue
        
        
        
time_s = Task.last_task_runtime
total_time = database_size / len(df) * time_s
with Message("Estimated total time for aliases").tab():
    Message.print(Message.time(total_time))
Message("Summary of success rates").list({
    "Successes":f"{(num_direct_successes + num_secondary_successs) / len(df):.2%}",
    "Direct successes":f"{num_direct_successes / len(df):.2%}",
    "Secondary successes":f"{num_secondary_successs / len(df):.2%}",
    "Fails":f"{num_fails / len(df):.2%}"
})


# save to json
import json
with open("planet_name2_aliases_map.json", "w") as f:
    json.dump(planet_name2_aliases_map, f, indent=4)