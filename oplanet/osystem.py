

# --------------- #
# !-- Imports --! #
# --------------- #

import oakley
from .star_utils import get_star_aliases, parse_star_name,  get_star_name
from .utils import get_database

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


class OSystem:
    """
    Class representing an exoplanetary system, with its star and its planets.
    """

    def __init__(self, star_name: str):
        """
        Initializes the OSystem object by loading the star and its planets from the database.

        Parameters
        ----------
        star_name : str
            Name of the star. Can be any alias of the star.
        
        Attributes
        ----------
        star_name : str
            The name of the star, in the format used in the database.
        star_aliases : list of str
            The list of aliases of the star, provided by Simbad.
        df : pd.DataFrame
            The dataframe containing the data of the star and its planets, as loaded from the database.
        df_star_name : str
            The name of the star as it appears in the dataframe.
        """
        self.star_name = parse_star_name(star_name)
        self.star_aliases = get_star_aliases(self.star_name)
        
        # load the database
        for alias in [
            star_name, *[star_alias for star_alias in self.star_aliases if star_alias != star_name]
        ]:
            self.df_star_name = star_name
            self.df = get_database()[
                get_database()["hostname"] == parse_star_name(alias)
            ].reset_index(drop=True).copy(deep=True)
            if len(self.df) > 0:
                break
        else:
            raise ValueError(f"Star '{star_name}' not found in the database.")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}({self.star_name}, {len(self.df)} rows)>"
    
    def copy(self) -> "OSystem":
        """
        Returns a copy of the OSystem object, without calling the __init__ method.
        """
        new_system = OSystem.__new__(OSystem)
        new_system.star_name = self.star_name
        new_system.star_aliases = self.star_aliases
        new_system.df_star_name = self.df_star_name
        new_system.df = self.df.copy(deep=True)
        return new_system
    
    def _get_column_prefix(self, prefix:str) -> str:
        """
        Returns the desired prefix. Translates the inputted prefix to one of "st_", "pl_" or "sy_".
        """
        prefix = prefix.lower().replace("_", "").replace("star", "st").replace("planet", "pl").replace("system", "sy")
        assert prefix in ["st", "pl", "sy"], f"Invalid prefix '{prefix}'. Must be one of {self.info_types} or 'star', 'planet', 'system', or 'st', 'pl', 'sy'."
        return prefix + "_"

    
    def restrict_to(self, prefix:str) -> "OSystem":
        """
        Restricts the dataframe's columns. Three types of columns exist:
        - star columns, with prefix "st_"
        - planet columns, with prefix "pl_"
        - system columns, with prefix "sy_"

        Other types of columns do exist, and we will always keep them. When we restrict to a prefix, we drop columns starting with the other prefixes.

        Parameters
        ----------
        prefix : str, optional
            The prefix to restrict to. Can be "st_", "pl_" or "sy_", or "star", "planet", "system", or "st", "pl", "sy".
        """
        prefix = self._get_column_prefix(prefix)
        other_prefixes = [p for p in ["st_", "pl_", "sy_"] if p != prefix]
        columns_to_drop = [col for col in self.df.columns if any(col.startswith(other_prefix) for other_prefix in other_prefixes)]
        new_df = self.df.drop(columns=columns_to_drop)
        new_system = self.copy()
        new_system.df = new_df
        return new_system
    
    def head(self, n:int = 5) -> pd.DataFrame:
        """
        Returns the first n rows of the dataframe, with only the columns relative to the system (i.e. with prefix "sy_").
        """
        return self.restrict_to("system").df.head(n)
    
  
    # --------------- #
    # !-- Columns --! #
    # --------------- #

    def _get_columns_with_prefix(self, prefix:Literal["system","star","planet"] = None) -> list:
        """
        Returns the list of column names that start with the given prefix, without the suffixes.

        Parameters
        ----------
        prefix : str, optional
            The prefix to filter by. Can be "system", "star" or "planet". If None, all columns are returned (but still without suffixes).
        """
        if prefix is None:
            prefix = ""
        else:
            prefix = self._get_column_prefix(prefix)
        columns = self.df.columns
        prefix_columns = set()
        for column in columns:
            if column.startswith(prefix):
                # remove the suffixes
                column = column.replace("err1", "").replace("err2", "").replace("str", "").replace("lim", "")
                prefix_columns.add(column)
        return [col for col in columns if col in prefix_columns] # keep the order
    
    @property
    def _star_columns(self) -> list:
        """
        The list of column names relative to the properties of the star.
        Columns are grouped together if they only differ by theit suffix.
        """
        return self._get_columns_with_prefix(prefix="st_")

    @property
    def _planet_columns(self) -> list:
        """
        The list of column names relative to the properties of the planets.
        Columns are grouped together if they only differ by theit suffix.
        """
        return self._get_columns_with_prefix(prefix="pl_")

    @property
    def _system_columns(self) -> list:
        """
        The list of column names relative to the properties of the system.
        Columns are grouped together if they only differ by theit suffix.
        """
        return self._get_columns_with_prefix(prefix="sy_")
    

    # ------------------ #
    # !-- Properties --! #
    # ------------------ #

    def print_column(self, column:str):
        """
        For the given column, prints the value of that column for each row in the dataframe.
        If they exist, the error columns and limit columns are also printed.
        """
        assert column in self.df.columns, f"Column '{column}' not found in the dataframe."
        columns2print = [
            column, *[
                column + suffix for suffix in ["err1", "err2", "lim"] if column + suffix in self.df.columns
            ]
        ]
        Message(f"Colmun {cstr(column):y} data:").print(self.df[columns2print])

    def _get_best_row_measure_for(self, column:str) -> np.ndarray:
        """
        For a given column, returns the row with the best measurement for that column,
        that is the measurement with lowest uncertainties (err1 and err2) and with a value (not NaN).
        If err1 and err2 are both NaN, the row with the lowest lim value is returned.

        We impose lim = 0, because lim=1 means that the value is an upper limit, not a real measurement.

        Returns a tuple of (best_value, err1, err2) with None if the value is nan.
        """
        assert column in self.df.columns, f"Column '{column}' not found in the dataframe."

        # 1. Loop over rows of the dataframe, and keep the best row and update it
        best_row_value = np.nan
        best_row_err1 = np.nan
        best_row_err2 = np.nan

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            row_value = np.nan
            row_err1 = np.nan
            row_err2 = np.nan

            # 1. Check for valid value
            if pd.isna(row[column]):
                continue
            # check that float or int or idk, but number
            if not isinstance(row[column], (float, int)):
                raise ValueError(f"Column '{column}' contains non-numeric values, which is not supported.")
            row_value = row[column]
            
            # 2. Check for valid lim value
            if column + "lim" in self.df.columns:
                if row[column + "lim"] != 0:
                    continue
            
            # 3. Check for valid err1 and err2 values
            if column + "err1" in self.df.columns and column + "err2" in self.df.columns:
                if pd.isna(row[column + "err1"]) or pd.isna(row[column + "err2"]):
                    continue
                row_err1 = row[column + "err1"]
                row_err2 = row[column + "err2"]
            
            # 4. Update best row if necessary
            if np.isnan(best_row_value):
                best_row_value = row_value
                best_row_err1 = row_err1
                best_row_err2 = row_err2
            else:
                if np.isnan(best_row_err1):
                    # do not update
                    continue
                elif np.isnan(row_err1):
                    # do not update, we don't have a valid error for this row
                    continue
                elif np.isnan(best_row_err2):
                    best_row_value = row_value
                    best_row_err1 = row_err1
                    best_row_err2 = row_err2
                else:
                    if np.abs(row_err1) + np.abs(row_err2) < np.abs(best_row_err1) + np.abs(best_row_err2):
                        print(row_err1, row_err2, best_row_err1, best_row_err2)
                        print()
                        best_row_value = row_value
                        best_row_err1 = row_err1
                        best_row_err2 = row_err2
        
        return np.array([best_row_value, best_row_err1, best_row_err2])
    


    # -------------- #
    # !-- System --! #
    # -------------- #

    @property
    def system(self) -> "OSystem":
        """
        Returns a new OSystem object with only the columns relative to the system.
        """
        out = self.copy()
        out = out.restrict_to("system")
        return out

    # ------------ #
    # !-- Star --! #
    # ------------ #

    @property
    def star(self) -> "OStar":
        """
        Returns a new OSystem object with only the columns relative to the star.
        """
        out = self.copy()
        out = out.restrict_to("star")
        return OStar(out)

    @property
    def a(self) -> "OStar":
        """
        An alias for `self.star`.
        """
        return self.star
    
    @property
    def A(self) -> "OStar":
        """
        An alias for `self.star`.
        """
        return self.star
    


    # --------------- #
    # !-- Planets --! #
    # --------------- #

    @property
    def n_planets(self) -> int:
        """
        Returns the number of planets in the system, by counting the number of unique values in the pl_letter column.
        """
        return self.df["pl_letter"].nunique()
    
    @property
    def planets(self) -> list:
        """
        Returns a list of OPlanet objects, one for each planet in the system.
        """
        return [self._get_planet(planet_letter) for planet_letter in sorted(self.df["pl_letter"].unique())]

    def _get_planet(self, planet_letter:str) -> "OPlanet":
        """
        Returns a new OSystem object with only the columns relative to the planet with the given letter.
        """
        out = self.copy()
        out = out.restrict_to("planet")
        # match pl_letter
        out.df = out.df[out.df["pl_letter"] == planet_letter].reset_index(drop=True)
        if len(out.df) == 0:
            raise ValueError(f"Planet with letter '{planet_letter}' not found in the system.")
        return OPlanet(out)

    @property
    def b(self) -> "OPlanet":
        return self._get_planet("b")
    
    @property
    def c(self) -> "OPlanet":
        return self._get_planet("c")
    
    @property
    def d(self) -> "OPlanet":
        return self._get_planet("d")
    
    @property
    def e(self) -> "OPlanet":
        return self._get_planet("e")
    
    @property
    def f(self) -> "OPlanet":
        return self._get_planet("f")
    
    @property
    def g(self) -> "OPlanet":
        return self._get_planet("g")
    
    @property
    def h(self) -> "OPlanet":
        return self._get_planet("h")

    @property
    def i(self) -> "OPlanet":
        return self._get_planet("i")
    
    @property
    def j(self) -> "OPlanet":
        return self._get_planet("j") # no system has 10 planets anyway




# ------------ #
# !-- Star --! #
# ------------ #

class OStar(OSystem):
    """
    Basically an alias for OSystem, but with additional properties to handle data for a single star.
    """
    def __init__(self, osystem: OSystem):
        self.df = osystem.df
        self.star_name = osystem.star_name
        self.star_aliases = osystem.star_aliases
        self.df_star_name = osystem.df_star_name

    
    


# -------------- #
# !-- Planet --! #
# -------------- #

class OPlanet(OSystem):
    """
    Basically an alias for OSystem, but with additional properties to handle data for a single planet.
    """
    def __init__(self, osystem: OSystem):
        self.df = osystem.df
        self.star_name = osystem.star_name
        self.star_aliases = osystem.star_aliases
        self.df_star_name = osystem.df_star_name

    
    # -------------------- #
    # !-- Boring Stuff --! #
    # -------------------- #

    @property
    def name(self) -> str:
        """
        Returns the name of the planet, in the format "Star letter", e.g. "TRAPPIST-1 b".
        """
        return self.df["pl_name"].iloc[0]
    
    @property
    def letter(self) -> str:
        """
        Returns the letter of the planet, e.g. "b".
        """
        return self.df["pl_letter"].iloc[0]
    
    @property
    def discovery_year(self) -> int:
        """
        Returns the discovery year of the planet.
        """
        return self.df["disc_year"].iloc[0]
    
    @property
    def discovery_method(self) -> str:
        """
        Returns the discovery method of the planet.
        """
        return self.df["discoverymethod"].iloc[0]
    

    # -------------------------- #
    # !-- Orbital properties --! #
    # -------------------------- #




if __name__ == "__main__":

    # --------------------- #
    # !-- Instantiation --! #
    # --------------------- #
    
    # 1. Load the data
    system = OSystem("TRAPPIST-1")

    Message("Looking at system properties").list({
        "Star name": system.star_name,
        "Number of planets": system.n_planets,
        "Dataframe shape": system.df.shape,
        "Star": system.star,
        "Planets": system.planets,
        "Planet mass": system.b._get_best_row_measure_for("pl_bmassj"),
    })

    


    # --------------- #
    # !-- Columns --! #
    # --------------- #

    system.b.print_column("pl_bmassj")

    #print(system.b.df[["pl_bmassj", "pl_bmassjerr1", "pl_bmassjerr2", "pl_bmassjlim"]])
    exit()


    Message("Planet columns").list(system._planet_columns)
    Message("Star columns").list(system._star_columns)
    Message("System columns").list(system._system_columns)

    print(system._get_best_row_for("pl_bmassj"))
