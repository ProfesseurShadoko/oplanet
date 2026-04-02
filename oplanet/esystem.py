
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


# ------------ #
# !-- Data --! #
# ------------ #


class ESystem:
    """
    Class representing an exoplanetary system, with its star and its planets.
    Data is retrieved from the Exoplanet.eu database.
    """
    
    def __init__(self, star_name: str):
        """
        Initialize the ESystem object by retrieving data from the Exoplanet.eu database for the given star name.
        
        Parameters
        ----------
        star_name : str
            The name of the star for which to retrieve the exoplanetary system data.
        
        Attributes
        ----------
        star_name : str
            The name of the star, given by the user.
        star_aliases : list of str
            The list of aliases of the star, provided by Simbad.
        df : pd.DataFrame
            The DataFrame containing the exoplanetary system data for the star, as retrieved from the Exoplanet.eu database.
        df_star_name : str
            The name of the star, as it appears in the Exoplanet.eu database.
        """
        
        with Message.mute():
            self.star_name = parse_star_name(star_name)
        self.star_aliases = get_star_aliases(self.star_name)
        
        # load the database
        for alias in [
            self.star_name, *[
                star_alias for star_alias in self.star_aliases if star_alias != self.star_name
            ]
        ]:
            self.df_star_name = alias
            full_df = get_database("eu")
            self.df = full_df[full_df["star_name"] == alias].copy(deep=True)
            if len(self.df) > 0:
                break
        else:
            # retry by looking in the "star_alternate_names" column, which is
            # a string of comma separated star aliases
            for alias in [
                self.star_name, *[
                    star_alias for star_alias in self.star_aliases if star_alias != self.star_name
                ]
            ]:
                self.df_star_name = alias
                full_df = get_database("eu")
                self.df = full_df[full_df["star_alternate_names"].str.contains(alias, na=False, regex=False)].copy(deep=True)
                if len(self.df) > 0:
                    break
            else:
                raise ValueError(f"Star '{star_name}' or its aliases not found in the database.")
                
        




if __name__ == "__main__":
    esystem = ESystem("Beta Pictoris")
    print(esystem.df)
    