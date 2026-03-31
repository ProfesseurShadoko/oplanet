

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
        with Message.mute():
            self.star_name = parse_star_name(star_name)
        self.star_aliases = get_star_aliases(self.star_name)
        
        # load the database
        for alias in [
            self.star_name, *[star_alias for star_alias in self.star_aliases if star_alias != star_name]
        ]:
            self.df_star_name = alias
            self.df = get_database()[
                get_database()["hostname"] == alias
            ].reset_index(drop=True).copy(deep=True)
            if len(self.df) > 0:
                break
        else:
            # retry by looking at columns hd_name,hip_name,tic_id,gaia_dr2_id,gaia_dr3_id
            for helper_column, prefix in zip(["hd_name", "hip_name", "tic_id", "gaia_dr2_id", "gaia_dr3_id"], ["HD ", "HIP ", "TIC ", "Gaia DR2 ", "Gaia DR3 "]):
                # find the correct alias for this helper column
                matching_aliases = [
                    alias for alias in [self.star_name, *[star_alias for star_alias in self.star_aliases if star_alias != self.star_name]]
                    if alias.startswith(prefix)
                ]
                for alias in matching_aliases:
                    self.df_star_name = alias
                    self.df = get_database()[
                        get_database()[helper_column] == alias
                    ].reset_index(drop=True).copy(deep=True)
                    if len(self.df) > 0:
                        break
                else:
                    continue
                break
            else:
                raise ValueError(f"Star '{star_name}' or its aliases not found in the database.")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}({self.name}, {len(self.df)} rows)>"
    
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
    
    # ------------------------------ #
    # !-- Information Management --! #
    # ------------------------------ #
    
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
    
    @property
    def name(self) -> str:
        return self.star_name
    
  
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
        Message(f"Column {cstr(column):y} data:").print(self.df[columns2print])

    def _get_best_row_measure_for(self, column:str, _limit:Literal[-1, 0, 1] = 0) -> np.ndarray:
        """
        For a given column, returns the row with the best measurement for that column,
        that is the measurement with lowest uncertainties (err1 and err2) and with a value (not NaN).
        If err1 and err2 are both NaN, the row with the lowest lim value is returned.

        We impose lim = 0, because lim=1 means that the value is an upper limit, not a real measurement.
        When no value availalbe, we look for upper and lower limit and return [np.nan, upper_limit, lower_limit].

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
            #if not isinstance(row[column], (float, int)):
            if not np.isscalar(row[column]):
                raise ValueError(f"Column '{column}' contains non-numeric values, which is not supported.")
            row_value = row[column]
            
            # 2. Check for valid lim value
            if column + "lim" in self.df.columns:
                if row[column + "lim"] != _limit and not pd.isna(row[column + "lim"]):
                    continue
            
            # 3. Check for valid err1 and err2 values
            if column + "err1" in self.df.columns and column + "err2" in self.df.columns:
                row_err1 = row[column + "err1"]
                row_err2 = row[column + "err2"]
            
            # 4. Update best row if necessary
            if np.isnan(best_row_value):
                best_row_value = row_value
                best_row_err1 = row_err1
                best_row_err2 = row_err2
            else:
                if np.isnan(row_err1):
                    # do not update, we don't have a valid error for this row
                    continue
                elif np.isnan(best_row_err1):
                    # update, previous value doesn't have a valid error, but this one does
                    best_row_value = row_value
                    best_row_err1 = row_err1
                    best_row_err2 = row_err2
                else:
                    if np.abs(row_err1) + np.abs(row_err2) < np.abs(best_row_err1) + np.abs(best_row_err2):
                        best_row_value = row_value
                        best_row_err1 = row_err1
                        best_row_err2 = row_err2
        
        if np.isnan(best_row_value):
            # retry with limit = 1 and then with limit = -1
            if _limit == 0:
                # check if there is an upper limit and lower limit
                out_upper = self._get_best_row_measure_for(column, _limit=1)
                out_lower = self._get_best_row_measure_for(column, _limit=-1)
                return np.array([np.nan, out_upper[0], out_lower[0]])
        
        return np.array([best_row_value, best_row_err1, best_row_err2])
    

    def display(self):
        Message("Looking at system properties").list({
            "Star name": self.star_name,
            "Number of planets": self.n_planets,
            "Dataframe shape": self.df.shape,
            "Star": self.star,
            "Planets": self.planets,
            "Distance (pc)": self.distance_pc,
            "Parallax (mas)": self.parallax_mas,
        })
    
    

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
    

    @property
    def distance_pc(self) -> np.ndarray:
        """
        Returns the distance of the star in pc.
        """
        out = self._get_best_row_measure_for("sy_dist")
        return out
    
    @property
    def parallax_mas(self) -> np.ndarray:
        """
        Returns the parallax of the star in mas.
        """
        out = self._get_best_row_measure_for("sy_plx")
        return out
    


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
        Returns a list of OPlanet objects, one for each planet in the system. The return list is sorted
        by semi-major axis (inner to outer).
        """
        out = [self._get_planet(planet_letter) for planet_letter in sorted(self.df["pl_letter"].unique())]
        # sort them by: 1. semi axis (ineer to outer) 2. letter
        out.sort(key=lambda planet: (planet.sma_au[0], planet.letter))
        return out

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

    @property
    def age_myr(self) -> np.ndarray:
        """
        Returns the age of the star in Myr.
        """
        out = self._get_best_row_measure_for("st_age")
        conversion_factor = u.Gyr.to(u.Myr)
        return out * conversion_factor
    
    @property
    def mass_solar(self) -> np.ndarray:
        """
        Returns the mass of the star in solar masses.
        """
        out = self._get_best_row_measure_for("st_mass")
        return out
    
    @property
    def radius_solar(self) -> np.ndarray:
        """
        Returns the radius of the star in solar radii.
        """
        out = self._get_best_row_measure_for("st_rad")
        return out
    
    @property
    def luminosity_solar(self) -> np.ndarray:
        """
        Returns the luminosity of the star in solar luminosities.
        """
        out = self._get_best_row_measure_for("st_lum")
        return out
    
    @property
    def teff_k(self) -> np.ndarray:
        """
        Returns the effective temperature of the star in K.
        """
        out = self._get_best_row_measure_for("st_teff")
        return out
    
    @property
    def metallicity_dex(self) -> np.ndarray:
        """
        Returns the metallicity of the star in dex.
        """
        out = self._get_best_row_measure_for("st_met")
        return out
    
    @property
    def spectral_type(self) -> str:
        """
        Returns the spectral type of the star.
        """
        return self.df["st_spectype"].iloc[0]

    
    def display(self):
        Message(f"Star {cstr(self.star_name):y} properties:").list({
            "Age (Myr)": self.age_myr,
            "Mass (solar masses)": self.mass_solar,
            "Radius (solar radii)": self.radius_solar,
            "Luminosity (solar luminosities)": self.luminosity_solar,
            "Effective temperature (K)": self.teff_k,
            "Metallicity (dex)": self.metallicity_dex,
            "Spectral type": self.spectral_type,
        })
    
    


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
        return int(np.round(self.df["disc_year"].iloc[0]))
    
    @property
    def discovery_method(self) -> str:
        """
        Returns the discovery method of the planet.
        """
        return self.df["discoverymethod"].iloc[0]
    

    # -------------------------- #
    # !-- Orbital properties --! #
    # -------------------------- #

    @property
    def orbital_period_yrs(self) -> np.ndarray:
        """
        Returns the orbital period of the planet in years.
        """
        out = self._get_best_row_measure_for("pl_orbper")
        conversion_factor = u.day.to(u.yr)
        return out * conversion_factor

    @property
    def mass_sini_mjup(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_msinij")
    
    @property
    def mass_mjup(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_bmassj")
    
    @property
    def sma_au(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_orbsmax")
    
    @property
    def eccentricity(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_orbeccen")
    
    @property
    def inclination_deg(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_orbincl")
    
    @property
    def arg_periastron_deg(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_orblper")
    
    @property
    def time_periastron_jd(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_orbtper")
    
    @property
    def rv_amplitude_ms(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_rvamp")
    
    @property
    def radius_rjup(self) -> np.ndarray:
        return self._get_best_row_measure_for("pl_radj")
    
    

    # --------------- #
    # !-- Diplsay --! #
    # --------------- #

    def display(self):
        Message(f"Planet {cstr(self.name):y} properties:").list({
            "Letter": self.letter,
            "Discovery year": self.discovery_year,
            "Discovery method": self.discovery_method,
            "Orbital period (yrs)": self.orbital_period_yrs,
            "Mass (Mjup)": self.mass_mjup,
            "Mass sin(i) (Mjup)": self.mass_sini_mjup,
            "Semi-major axis (AU)": self.sma_au,
            "Eccentricity": self.eccentricity,
            "Inclination (deg)": self.inclination_deg,
            "Argument of periastron (deg)": self.arg_periastron_deg,
            "Time of periastron (JD)": self.time_periastron_jd,
            "RV amplitude (m/s)": self.rv_amplitude_ms,
            "Radius (Rjup)": self.radius_rjup,
        })




if __name__ == "__main__":

    # --------------------- #
    # !-- Instantiation --! #
    # --------------------- #
    
    # 1. Load the data
    system = OSystem("LHS 1140")

    Message("Looking at system properties").list({
        "Star name": system.star_name,
        "Number of planets": system.n_planets,
        "Dataframe shape": system.df.shape,
        "Star": system.star,
        "Planets": system.planets,
        "Planet mass": system.b._get_best_row_measure_for("pl_bmassj"),
        "Distance (pc)": system.distance_pc,
        "Parallax (mas)": system.parallax_mas,
    })

    
    # --------------- #
    # !-- Planets --! #
    # --------------- #

    system.b.display()

    # ------------ #
    # !-- Star --! #
    # ------------ #

    system.star.display()
