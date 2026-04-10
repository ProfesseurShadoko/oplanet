

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


class NSystem:
    """
    Class representing an exoplanetary system, with its star and its planets.
    Data is retrieved from the NASA Exoplanet Archive.
    """

    def __init__(self, star_name: str):
        """
        Initializes the NSystem object by loading the star and its planets from the database.

        Parameters
        ----------
        star_name : str
            Name of the star. Can be any alias of the star.
        
        Attributes
        ----------
        star_name : str
            The name of the star, given by the user.
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
            self.df = get_database("nasa")[
                get_database("nasa")["hostname"] == alias
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
                    self.df = get_database("nasa")[
                        get_database("nasa")[helper_column] == alias
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
    
    def copy(self) -> "NSystem":
        """
        Returns a copy of the NSystem object, without calling the __init__ method.
        """
        new_system = NSystem.__new__(NSystem)
        new_system.star_name = self.star_name
        new_system.star_aliases = self.star_aliases
        new_system.df_star_name = self.df_star_name
        new_system.df = self.df.copy(deep=True)
        return new_system
    
    @staticmethod
    def refresh():
        """
        Deletes database and triggers a download (of about 60s)
        of the Nasa Exoplanet Archive.
        """
        refresh_data("nasa")
    
    @staticmethod
    def get_database(source: Literal["eu", "nasa"] = "nasa") -> pd.DataFrame:
        """
        Returns the database as a pandas dataframe. If the database is not loaded, it is loaded from the archive file.

        Parameters
        ----------
        source : str, optional
            The source of the database. Can be "eu" for Exoplanet.eu catalog or "nasa" for NASA Exoplanet Archive. Default is "nasa".
        """
        return get_database(source)
    
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

    
    def restrict_to(self, prefix:str, keep_system_columns:bool = True) -> "NSystem":
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
        keep_system_columns : bool, optional
            Whether to keep system columns when restricting to a different prefix. Default is True.
        """
        prefix = self._get_column_prefix(prefix)
        other_prefixes = [p for p in ["st_", "pl_", "sy_"] if p != prefix]
        
        if keep_system_columns:
            other_prefixes = [p for p in other_prefixes if p != "sy_"]
            
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
        assert column in self.df.columns, f"Column '{column}' not found in the dataframe. Might be a consequence of the restriction of the dataframe to a specific prefix. For instance, `distance_pc` and `parallax_mas` are system properties, so they will not be available in `self.star` or `self.planets`."

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
        
        
    # ------------------- #
    # !-- Fill Values --! #
    # ------------------- #
    
    @staticmethod
    def fill(values:np.ndarray, rel_uncertainty:float = 0.1, default_value:float = None) -> np.ndarray:
        """
        Handles values and uncertainties when they are missing. If only the value is not None, uncertaintis
        are filled with the given relative uncertainty.
        
        Parameters
        ----------
        values : np.ndarray
            Values as outputted by the properties. If `val, val_upper, val_lower`, returned as such.
            If `val, np.nan, np.nan`, filled with `val, val*rel_uncertainty, -val*rel_uncertainty`.
            If `np.nan, val_upper, np.nan`, `val_upper` is an upper limit and not an uncertainty. We then return
            `val_upper - rel_uncertainty*val_upper, val_upper*rel_uncertainty, val_upper*rel_uncertainty`.
            If `np.nan, np.nan, val_lower`, `val_lower` is a lower limit and not an uncertainty. We then return
            `val_lower + rel_uncertainty*val_lower, val_lower*rel_uncertainty, val_lower*rel_uncertainty`.
        rel_uncertainty : float, optional
            The relative uncertainty to use when filling missing uncertainties. Default is 0.1 (10%).
        default_value : float, optional
            The default value to use when all values are missing. If None, an error will be raised. Default is None.
            
        Returns
        -------
        np.ndarray
            The filled values, in the format (value, err1, err2).
        """
        val, val_upper, val_lower = values
        
        # 1. If none of them are nans
        if not np.isnan(val):
            return values
        
        # 2. If all are nans
        if np.all(np.isnan(values)):
            if default_value is not None:
                val = default_value
            else:
                raise ValueError("All values are missing and no default value provided.")
        
        # 3. If uncertanties are missing, but value is not
        if not np.isnan(val):
            if val_upper == np.nan:
                val_upper = val * rel_uncertainty
            if val_lower == np.nan:
                val_lower = -val * rel_uncertainty
            return np.array([val, val_upper, val_lower])
    
        # 4. If value is missing, but there is an upper bound
        if np.isnan(val_lower):
            val = val_upper - rel_uncertainty * val_upper
            return np.array([val, val_upper * rel_uncertainty, val_upper * rel_uncertainty])
    
        # 5. If value is missing, but there is a lower bound
        if np.isnan(val_upper):
            val = val_lower + rel_uncertainty * val_lower
            return np.array([val, val_lower * rel_uncertainty, -val_lower * rel_uncertainty])
            
        
    
    

    # -------------- #
    # !-- System --! #
    # -------------- #

    @property
    def system(self) -> "NSystem":
        """
        Returns a new NSystem object with only the columns relative to the system.
        """
        out = self.copy()
        out = out.restrict_to("system")
        return out

    # ------------ #
    # !-- Star --! #
    # ------------ #

    @property
    def star(self) -> "NStar":
        """
        Returns a new NSystem object with only the columns relative to the star.
        """
        out = self.copy()
        out = out.restrict_to("star")
        return NStar(out)

    @property
    def a(self) -> "NStar":
        """
        An alias for `self.star`.
        """
        return self.star
    
    @property
    def A(self) -> "NStar":
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
    
    @property
    def aliases(self) -> list:
        """
        Returns the list of aliases of the star, provided by Simbad.
        """
        if not hasattr(self, "_aliases"):
            self._aliases = get_star_aliases(self.star_name)
        return self._aliases
    


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
        Returns a list of NPlanet objects, one for each planet in the system. The return list is sorted
        by semi-major axis (inner to outer).
        """
        out = [self._get_planet(planet_letter) for planet_letter in sorted(self.df["pl_letter"].unique())]
        # sort them by: 1. semi axis (ineer to outer) 2. letter
        out.sort(key=lambda planet: (planet.sma_au[0], planet.letter))
        return out

    def _get_planet(self, planet_letter:str) -> "NPlanet":
        """
        Returns a new NSystem object with only the columns relative to the planet with the given letter.
        """
        out = self.copy()
        out = out.restrict_to("planet")
        # match pl_letter
        out.df = out.df[out.df["pl_letter"] == planet_letter].reset_index(drop=True)
        if len(out.df) == 0:
            raise ValueError(f"Planet with letter '{planet_letter}' not found in the system.")
        return NPlanet(out)

    @property
    def b(self) -> "NPlanet":
        return self._get_planet("b")
    
    @property
    def c(self) -> "NPlanet":
        return self._get_planet("c")
    
    @property
    def d(self) -> "NPlanet":
        return self._get_planet("d")
    
    @property
    def e(self) -> "NPlanet":
        return self._get_planet("e")
    
    @property
    def f(self) -> "NPlanet":
        return self._get_planet("f")
    
    @property
    def g(self) -> "NPlanet":
        return self._get_planet("g")
    
    @property
    def h(self) -> "NPlanet":
        return self._get_planet("h")

    @property
    def i(self) -> "NPlanet":
        return self._get_planet("i")
    
    @property
    def j(self) -> "NPlanet":
        return self._get_planet("j") # no system has 10 planets anyway




# ------------ #
# !-- Star --! #
# ------------ #

class NStar(NSystem):
    """
    Basically an alias for NSystem, but with additional properties to handle data for a single star.
    """
    def __init__(self, osystem: NSystem):
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
    def Teff_k(self) -> np.ndarray:
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
            "Effective temperature (K)": self.Teff_k,
            "Irradiation temparature at 1 arcsec (K)": self.Tirr_k(1),
            "Metallicity (dex)": self.metallicity_dex,
            "Spectral type": self.spectral_type,
            
        })
    
    # ------------------------------- #
    # !-- Irradiation Temperature --! #
    # ------------------------------- #
    
    def Tirr_k(self, sep_arcsec: float | np.ndarray, albedo:float = 0) -> np.ndarray:
        """
        Returns the irradiation temperature of a planet at a given separation from the star, in K.
        The formula used is:
        ```python
        a = sep_arcsec * distance_pc # in AU
        Tirr = Teff_star * sqrt(R_star / (2 * a)) * (1 - albedo)**0.25 # R_star converted to AU
        ```
        Additionally, uncertainties are propagated.
        
        Parameters
        ----------
        sep_arcsec : float or np.ndarray
            The separation from the star in arcseconds. Can be a single value or an array of values.
        albedo : float, optional
            The albedo of the planet. Default is 0.

        Returns
        ------- 
        float or np.ndarray
            The irradiation temperature in K, at given separations, and uncertainties.
            
        Notes
        -----
        Several assumptions are made in this calculation. Going from separation to actual distance
        assumes a face-on circular orbit, or a planet at quadrature.
        No geometrical factor is added to the planet, meaning that heat is effectively well redistributed
        across the planet, and that there isn't a day-night contrast (which is invalid for Hot Jupiters for instance).
        """
        
        # 1. Check input validity
        if not (0 <= albedo < 1):
            raise ValueError("albedo must be in the range [0, 1).")

        sep_arcsec = np.asarray(sep_arcsec, dtype=float)
        if np.any(sep_arcsec <= 0):
            raise ValueError("sep_arcsec must be strictly positive.")

        # 2. Get measurements
        teff, teff_err1, teff_err2 = self.fill(self.Teff_k)
        radius_solar, radius_err1_solar, radius_err2_solar = self.fill(self.radius_solar)
        distance_pc, distance_err1_pc, distance_err2_pc = self.fill(self.distance_pc)

        # 3. Convert distances
        rstar_to_au = u.R_sun.to(u.AU) # a float
        radius_au = radius_solar * rstar_to_au
        radius_err1_au = radius_err1_solar * rstar_to_au
        radius_err2_au = radius_err2_solar * rstar_to_au

        sma_au = sep_arcsec * distance_pc
        sma_err1_au = sep_arcsec * distance_err1_pc
        sma_err2_au = sep_arcsec * distance_err2_pc
        
        # 4. Compute Tirr
        def T(teff, R, a):
            return teff * np.sqrt(R / (2*a)) * (1 - albedo)**0.25
        
        tirr = T(teff, radius_au, sma_au)
        
        # 5. Propagate uncertainties
        tirr_err2 = T(teff + teff_err2, radius_au + radius_err1_au, sma_au + sma_err2_au) - tirr
        tirr_err1 = T(teff + teff_err1, radius_au + radius_err2_au, sma_au + sma_err1_au) - tirr

        # return as array
        return np.stack([tirr, tirr_err1, tirr_err2], axis=-1)



# -------------- #
# !-- Planet --! #
# -------------- #

class NPlanet(NSystem):
    """
    Basically an alias for NSystem, but with additional properties to handle data for a single planet.
    """
    def __init__(self, osystem: NSystem):
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
    system = NSystem("LHS 1140")

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
    
    # Autimatically handling values
    system = NSystem("LHS 1140")
    Message("LHS 1140 age is only a lower value, but we can fill it this way").list({
        "Age output:": system.star.age_myr,
        "Filled age output:": NSystem.fill(system.star.age_myr),
    })
