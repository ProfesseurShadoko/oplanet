

# --------------- #
# !-- Imports --! #
# --------------- #

import oakley
from .star_utils import get_star_aliases, parse_star_name,  get_star_name
from .data_loaders import get_database, refresh_data
from .oconfig import oplanet_temp_config, reset_config

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
import os
from oakley import *
from typing import Literal
from bs4 import BeautifulSoup


# ------------ #
# !-- Data --! #
# ------------ #


class NSystem:
    """
    Class representing an exoplanetary system, with its star and its planets.
    Data is retrieved from the NASA Exoplanet Archive.
    """

    _prefix = "sy" # defines what is outputtd by "reference"
    _config_key = "system"


    # ---------------------- #
    # !-- Initialization --! #
    # ---------------------- #

    def __init__(
        self,
        star_name: str
    ):
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
        assert isinstance(star_name, (str, pd.DataFrame)), f"Invalid type for star_name: {type(star_name)}. Must be a string or a pandas dataframe."

        if isinstance(star_name, str):
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
        else:
            assert isinstance(star_name, pd.DataFrame), f"Invalid type for star_name: {type(star_name)}. Must be a string or a pandas dataframe."
            self.df = star_name
            self.star_name = self.df["hostname"].iloc[0]
            self.df_star_name = self.star_name
            self.star_aliases = get_star_aliases(self.star_name)

        # copy df, reset index
        self.df = self.df.reset_index(drop=True).copy(deep=True)
        self._choose_row()


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
        new_system.df = self.df.reset_index(drop=True).copy(deep=True)
        new_system._chosen_row = self._chosen_row
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

    @property
    def columns(self) -> list:
        """
        A list of all columns in the dataframe, but with the different suffixes removed (err1, err2, str, lim).
        """
        suffixes = ["err1", "err2", "str", "lim"]
        columns = set()
        for column in self.df.columns:
            for suffix in suffixes:
                if column.endswith(suffix):
                    columns.add(column[:-len(suffix)])
                    break
            else:
                columns.add(column)
        return sorted(columns)

    def head(self, n:int = 5, which:Literal["system", "star", "planet"] = None) -> pd.DataFrame:
        """
        Returns the first n rows of the dataframe, with only the columns relative to the
        specified type of information. If which is None, the head of the full dataframe
        is simply returned.

        Parameters
        ----------
        n : int, optional
            The number of rows to return. Default is 5.
        which : str, optional
            The type of information to return. Can be "system" for system properties, "star" for star properties, "planet" for planet properties.
            Default is None, which returns the full dataframe.
        """
        if which is None:
            return self.df.head(n)
        elif which == "system":
            columns = [column for column in self.df.columns if column.startswith("sy_")]
        elif which == "star":
            columns = [column for column in self.df.columns if column.startswith("st_")]
        elif which == "planet":
            columns = [column for column in self.df.columns if column.startswith("pl_")]
        else:
            raise ValueError(f"Invalid value for 'which': {which}. Must be one of None, 'system', 'star', 'planet'.")
        return self.df[columns].head(n)
    
    
    def set_config(
        self,
        references: list[str] = None,
        properties: dict[str, int] = None,
        fallback: bool = None
    ):
        """
        Sets the preferences for the system. These preferences are used to choose the row
        from which the measurments are taken, when there are multiple rows for the same system.

        Parameters
        ----------
        references : list of str, optional
            A list of references in the format "Author_Date". If a row matches one of these references, it will be automatically chosen over the others. Default is None (don't change).
        properties : dict of str to int, optional
            A dictionary mapping property names to look for to a weight to apply. Default is None (don't change).
        fallback : bool, optional
            Whether to use a fallback strategy when, in a given row, a value is missing (looks for a replacement in other rows). Default is None (don't change).

        Notes
        -----
        The preferences are defined for the whole class. This means that if you change the preferences
        for one system, it will change for all other systems as well.

        If preferences are updated, the data from the present object will be updated automatically.
        For other objects, run `obj._choose_row()` to update the data based on the new preferences.
        
        """
        if references is not None:
            # check that all references are in the format "Author_Date"
            for ref in references:
                if not isinstance(ref, str) or "_" not in ref:
                    raise ValueError(f"Invalid reference format: '{ref}'. Must be a string in the format 'Author_Date'.")
                # check that the date part is an integer, and the rest is string
                author, date = ref.rsplit("_", 1)
                if not date.isdigit() and date != "None":
                    raise ValueError(f"Invalid reference format: '{ref}'. Must be a string in the format 'Author_Date', where Author is a string and Date is an integer.")
            oplanet_temp_config[self._config_key]["references"] = references
        if properties is not None:
            # check that it corresponds to property names and integer values
            def safe_get(obj, path:str) -> bool:
                try:
                    for attr in path.split("."):
                        obj = getattr(obj, attr)
                    assert isinstance(obj, (int, float, list, np.ndarray, str))
                    return True
                except:
                    return False
                
            for prop, value in properties.items():
                assert isinstance(prop, str), f"Invalid property name: '{prop}'. Must be a string."
                assert isinstance(value, int), f"Invalid property value: '{value}' for property '{prop}'. Must be an integer."
                assert safe_get(self, prop), f"Invalid property name: '{prop}'. Must be a valid property of the NSystem or its subclasses. Current class: {self.__class__.__name__}."

            oplanet_temp_config[self._config_key]["properties"] = properties

        if fallback is not None:
            assert isinstance(fallback, bool), f"Invalid value for fallback: '{fallback}'. Must be a boolean."
            oplanet_temp_config[self._config_key]["fallback"] = fallback

        if references is not None or properties is not None or fallback is not None:
            self._choose_row() # choose again, based on the updated preferences
    
    def add_reference_priority(
        self,
        author:str,
        date:int|None = None
    ):
        """
        Adds a reference to the priority list. This is a shortcut for `set_config(references=...)`.

        Parameters
        ----------
        author : str
            The author of the reference to add, as a string. Can be a substring of the actual author name in the reference (for instance "Smith" will match "Smith et al. 2020").
        date : int, optional
            The date of the reference to add, as an integer year. If None, any date will match.
        """
        reference = f"{author}_{date}"
        current_references = oplanet_temp_config[self._config_key]["references"]
        self.set_config(references=sorted(set(current_references + [reference])))


    def remove_reference_priority(
        self,
        author:str,
        date:int
    ):
        """
        Removes a reference from the priority list. This is a shortcut for `set_config(references=...)`.

        Parameters
        ----------
        author : str
            The author of the reference to remove, as a string.
        date : int
            The date of the reference to remove, as an integer year.
        """
        reference = f"{author}_{date}"
        self.set_config(references=[ref for ref in oplanet_temp_config[self._config_key]["references"] if ref != reference])

    def add_property_priority(
        self,
        property_name:str,
        weight:int
    ):
        """
        Adds a property to the priority list. This is a shortcut for `set_config(properties=...)`.

        Parameters
        ----------
        property_name : str
            The name of the property to add, as a string. Must be a valid property of the NSystem or its subclasses.
        weight : int
            The weight to apply to this property when choosing the row. Must be an integer.
        """
        current_properties = oplanet_temp_config[self._config_key]["properties"]
        current_properties[property_name] = weight
        self.set_config(properties=current_properties)

    def remove_property_priority(
        self,
        property_name:str
    ):
        """
        Removes a property from the priority list. This is a shortcut for `set_config(properties=...)`.

        Parameters
        ----------
        property_name : str
            The name of the property to remove, as a string. Must be a valid property of the NSystem or its subclasses.
        """
        current_properties = oplanet_temp_config[self._config_key]["properties"]
        if property_name in current_properties:
            del current_properties[property_name]
            self.set_config(properties=current_properties)

    def set_fallback(
        self,
        fallback:bool
    ):
        """
        Sets the fallback strategy. This is a shortcut for `set_config(fallback=...)`.

        Parameters
        ----------
        fallback : bool
            Whether to use a fallback strategy when, in a given row, a value is missing (looks for a replacement in other rows).
        """
        self.set_config(fallback=fallback)


    def _choose_row(self) -> None:
        """
        Among all rows of the dataframe, chooses the one to use for the system properties.
        """
        # disable fallback when looking for best rows, otherwise values will be filled by default
        fallback = oplanet_temp_config[self._config_key]["fallback"]
        oplanet_temp_config[self._config_key]["fallback"] = False

        def safe_get(obj, path:str) -> bool:
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                assert isinstance(obj, (int, float, list, np.ndarray, str))
                return obj
            except:
                return np.nan
            
        def score_row(row:int) -> tuple:
            self._chosen_row = row

            reference_matching = 0
            existence = 0
            error_existence = 0
            error_precision = 0

            for priority_reference in oplanet_temp_config[self._config_key]["references"]:
                # get date as integer
                date_str = priority_reference.rsplit("_", 1)[-1].strip()
                date = int(date_str) if date_str.isdigit() else None
                # get author name lowercase without spaces, or -, or _
                author = priority_reference.rsplit("_", 1)[0].lower().replace(" ", "").replace("-", "").replace("_", "")
                row_date = self.reference_date
                row_author = self.reference_author.lower().replace(" ", "").replace("-", "").replace("_", "") if self.reference_author is not None else ""
        
                if author in row_author and (date is None or row_date == date):
                    reference_matching += 1

            for prop, weight in oplanet_temp_config[self._config_key]["properties"].items():
                data = safe_get(self, prop)

                # 1. Existence
                if data is None:
                    continue
                if isinstance(data, str): # no errorbars, just add to existence
                    if data.strip() == "":
                        continue
                    existence += weight
                    continue # no errorbars
                if isinstance(data, (int, float)):
                    existence += weight
                    continue # no errorbars
                if isinstance(data, (list, np.ndarray)):
                    # length should be 3, otherwise ignore
                    if len(data) != 3:
                        continue
                    value, err1, err2 = data
                    if not np.isnan(value):
                        existence += weight
                    if not np.isnan(err1) or not np.isnan(err2):
                        error_existence += weight * 2
                    if not np.isnan(err1) and not np.isnan(err2):
                        error_existence += weight / 2
                        # error_precision += weight * np.abs(value) / (np.abs(err1) + np.abs(err2)) if (err1 != 0 or err2 != 0) else 0 # this can overweight certain parameters
                        # let's count how many rows are beaten by this one on this parameter.
                        error_range = np.abs(err1) + np.abs(err2)
                        for i in range(len(self.df)):
                            previous_chosen_row = self._chosen_row
                            self._chosen_row = i
                            other_data = safe_get(self, prop)
                            if np.any(np.isnan(other_data)):
                                error_precision += weight / len(self.df)
                            else:
                                other_value, other_err1, other_err2 = other_data
                                other_error_range = np.abs(other_err1) + np.abs(other_err2)
                                if error_range < other_error_range:
                                    error_precision += weight / len(self.df)
                            self._chosen_row = previous_chosen_row
                            

            return (reference_matching, existence, error_existence, float(error_precision))
        # save priorities to a list: index (reference) --> tuple (reference_matching, existence, error_existence, error_precision)
        self._priorities = []
        for i in range(len(self.df)):
            self._chosen_row = i
            reference = self.reference
            self._priorities.append((i, reference, score_row(i)))
        # sort by priority
        self._priorities.sort(key=lambda x: x[2], reverse=True)

        self._chosen_row = self._priorities[0][0] # choose the row with the best priority
        # reset fallback to its original value
        oplanet_temp_config[self._config_key]["fallback"] = fallback

    def set_row(self, row:int):
        """
        Sets the row to use for the system properties. This can be used to manually choose a specific row, for instance if the user knows that a specific reference is more reliable than the others.
        """
        assert isinstance(row, int), f"Invalid type for row: {type(row)}. Must be an integer."
        assert 0 <= row < len(self.df), f"Invalid value for row: {row}. Must be between 0 and {len(self.df)-1}."
        self._chosen_row = row


    def _get(self, column:str, _allow_fallback:bool = True) -> np.ndarray:
        """
        Returns the value for a given column at the chosen row.
        """

        # 1. Locate the columns
        column_err1 = column + "err1"
        column_err2 = column + "err2"
        column_lim = column + "lim"

        # 2. Get values
        if not column in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the dataframe.")
        value = self.df[column].iloc[self._chosen_row]
        err1 = self.df[column_err1].iloc[self._chosen_row] if column_err1 in self.df.columns else np.nan
        err2 = self.df[column_err2].iloc[self._chosen_row] if column_err2 in self.df.columns else np.nan
        
        # 3. Check for limits
        if column_lim in self.df.columns:
            lim = self.df[column_lim].iloc[self._chosen_row]
            if lim==-1:
                # lower limit
                value, err2 = np.nan, value
            elif lim==1:
                # upper limit
                value, err1 = np.nan, value


        # 4. Check for fallback if necessary
        if np.isnan(value):
            if oplanet_temp_config[self._config_key]["fallback"] and _allow_fallback: # check wether any row has a value for this column. pick the one with lowest errors
                original_chosen_row = self._chosen_row
                best_value = np.nan
                best_err1 = np.nan
                best_err2 = np.nan

                for i in range(len(self.df)):
                    self._chosen_row = i
                    value, err1, err2 = self._get(column, _allow_fallback=False) # avoid infinite recursion
                    
                    # a. If the values of the other row are nans, skip
                    if np.isnan(err1) and np.isnan(err2) and np.isnan(value):
                        continue
                    # b. If already valid value, skip
                    if np.isnan(value) and not np.isnan(best_value):
                        continue
                    # c. If new valid value, replace
                    if not np.isnan(value) and np.isnan(best_value):
                        best_value, best_err1, best_err2 = value, err1, err2
                        continue

                    # here, both value and best_value are either both valid or both nans
                    # d. Compare errors
                    if not np.isnan(value) and not np.isnan(best_value):
                        # check for better errors
                        if np.isnan(best_err1) or np.isnan(best_err2):
                            if not np.isnan(err1) and not np.isnan(err2):
                                best_value, best_err1, best_err2 = value, err1, err2
                                continue
                            else:
                                continue
                        else:
                            if np.isnan(err1) or np.isnan(err2):
                                continue
                            # here, neither best_err1 nor best_err2 are nans
                            # pick the one with the lowest erros
                            error_range = np.abs(err1) + np.abs(err2)
                            best_error_range = np.abs(best_err1) + np.abs(best_err2)
                            if error_range < best_error_range:
                                best_value, best_err1, best_err2 = value, err1, err2
                                continue
                    else:
                        # here, both value and best_value are both nans
                        # e. Check for limits (upper or lower)
                        if not np.isnan(err1):
                            if np.isnan(best_err1) or err1 < best_err1:
                                best_err1 = err1
                        if not np.isnan(err2):
                            if np.isnan(best_err2) or err2 < best_err2:
                                best_err2 = err2
                
                value, err1, err2 = best_value, best_err1, best_err2
                self._chosen_row = original_chosen_row # reset to original chosen row

        return np.array([value, err1, err2])
    

    @staticmethod
    def reset_config():
        """
        Resets the temporary configuration to the default
        configuration (stored in the oplanet_config object,
        synchronized with the json file).
        """
        # print(f"Resetting ID: {id(oplanet_temp_config)}")
        reset_config()

    def display_config(self):
        """
        Displays the current configuration for the object's properties.
        """
        config = oplanet_temp_config[self._config_key]
        with Message(f"Current configuration for '{self._config_key}':"):
            Message("References priority:").list(config["references"])
            Message("Properties priority:").list(config["properties"])
            Message(f"Fallback: {config['fallback']}")

    def display_priorities(self):
        """
        Displays the priorities of each row in the dataframe (as computed by the last call to the _choose_row method).
        """
        Message(f"Row priorities for {self._config_key}: (ref-match, val-notna, err-notna, err-range)", "?").list([
            f"Row {i}: {cstr(reference).italic()} --> {priority}" for i, reference, priority in self._priorities
        ])



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
        Message(f"Column {cstr(column):y} data:").print(self.df[columns2print].columns)

    
    def display(self, row:int = None):
        if row is not None:
            original_chosen_row = self._chosen_row
            self._chosen_row = row % len(self.df)

        Message("Looking at system properties").list({
            "Star name": self.star_name,
            "Number of planets": self.n_planets,
            "Dataframe shape": self.df.shape,
            "Star": self.star,
            "Planets": self.planets,
            "Distance (pc)": self.distance_pc,
            "Parallax (mas)": self.parallax_mas,
            "Coordinates (RA, Dec)": f"({self.ra[0]:.4f}, {self.dec[0]:.4f})",
            "Reference": self.reference,
            "Row": f"{self._chosen_row}/{len(self.df)-1}"
        })

        if row is not None:
            self._chosen_row = original_chosen_row # reset to original chosen row
        
        
    # ------------------- #
    # !-- Fill Values --! #
    # ------------------- #
    
    @staticmethod
    def fill(values:np.ndarray, rel_uncertainty:float = 0.1, default_value:float = None) -> np.ndarray:
        """
        Handles values and uncertainties when they are missing. If only the value is not None, uncertainties
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
            if np.isnan(val_upper):
                val_upper = val * rel_uncertainty
            if np.isnan(val_lower):
                val_lower = -val * rel_uncertainty
            return np.array([val, val_upper, val_lower])
        
        # 2. If all are nans
        if np.all(np.isnan(values)):
            if default_value is not None:
                val = default_value
            else:
                raise ValueError("All values are missing and no default value provided.")
        
        # 3. If uncertanties are missing, but value is not
        if not np.isnan(val):
            if np.isnan(val_upper):
                val_upper = val * rel_uncertainty
            if np.isnan(val_lower):
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
            


    # ------------------ #
    # !-- References --! #
    # ------------------ #

    @staticmethod
    def _parse_reference(ref:str) -> str:
        """
        Takes the value of a reference column as provided by the Exoplanet Archive
        (format <a>...</a>) and returns a dictionary:
        ```
        {
            "date": ...,
            "author": ...,
            "href": ...
        }
        ```
        """
        # 1. Parse the reference string to BeautifulSoup object
        soup = BeautifulSoup(ref, "html.parser").find("a")

        # 2. Extract date from the refstr
        ref_str = soup["refstr"]
        last_part = ref_str.split("_")[-1].strip()
        if last_part and last_part.isdigit():
            date = int(last_part)
        else:
            date = None # this happens when source is a catalog
        
        # 3. Extract author from the refstr
        date_as_str = str(date) if date is not None else "NO_DATE_:("
        author = ref_str.replace(date_as_str, "")
        # split and join around "_"
        author = " ".join([
            part for part in author.split("_") if part != ""
        ])
        # remove ET AL, AMP, AND
        author = author.split(" ET AL")[0].split(" AMP ")[0].split(" AND ")[0].split(" & ")[0].strip()

        # 4. Handle name anomalies
        if " " in author:
            # assume two names are displayed --> fall back on the text of the link, which might be in a better format
            author:str = soup.text.strip()
            # replace date if it is in the text
            if date is not None and str(date) in author:
                author = author.replace(str(date), "").strip()
                author = author.upper() # match format of ref str
            # remove ET AL, AMP, AND again
            author = author.split(" ET AL ")[0].split(" ET. AL.")[0].split(" AMP ")[0].split(" AND ")[0].split(" & ")[0].strip()
        
        # 5. Extract href and text
        href = soup["href"]
        text = soup.text.strip()

        # 6. Return the result as a dictionary
        return {
            "date": date, # int
            "author": author, # str
            "href": href, # str
            "reference": text # str
        }
    
    def get_reference(self) -> dict:
        """
        Returns the reference for the returned properties, as a dictionary:

        ```python
        {
            "date": ..., # (int)
            "author": ..., # (str)
            "href": ..., # url (str)
            "reference": ... # written as in Exoplanet Archive (str) 
        }

        They can also be accessed separately with the properties `reference_date`, `reference_author` and `reference_url`.
        """
        reference = self.df[f"{self._prefix}_refname"].iloc[self._chosen_row]
        return self._parse_reference(reference)
    
    # ------------------ #
    # !-- Properties --! #
    # ------------------ #

    @property
    def name(self) -> str:
        return self.star_name
    
    @property
    def distance_pc(self) -> np.ndarray:
        """
        Returns the distance of the star in pc.
        """
        out = self._get("sy_dist")
        return out
    
    @property
    def parallax_mas(self) -> np.ndarray:
        """
        Returns the parallax of the star in mas.
        """
        out = self._get("sy_plx")
        return out
    
    @property
    def ra(self):
        """
        Returns the right ascension of the star in degrees.
        """
        return self._get("ra")
    
    @property
    def dec(self):
        """
        Returns the declination of the star in degrees.
        """
        return self._get("dec")
    
    @property
    def aliases(self) -> list:
        """
        Returns the list of aliases of the star, provided by Simbad.
        """
        if not hasattr(self, "_aliases"):
            self._aliases = get_star_aliases(self.star_name)
        return self._aliases
    
    @property
    def reference_date(self) -> int:
        """
        Returns the date of the reference for the measurments, as an integer year.
        Might return None if the reference is a catalog for instance.
        """
        return self.get_reference()["date"]
    
    @property
    def reference_author(self) -> str:
        """
        Returns the author of the reference for the properties, as a string.
        """
        author = self.get_reference()["author"]
        return author.title() if author is not None else None
    
    @property
    def reference_url(self) -> str:
        """
        Returns the URL of the reference for the measurments, as a string.
        """
        return self.get_reference()["href"]
    
    @property
    def reference(self) -> str:
        """
        Returns the reference for the measurments, as a string usually in the format "Author et. al. (Date)".
        """
        return self.get_reference()["reference"]



    # ------------ #
    # !-- Star --! #
    # ------------ #

    @property
    def star(self) -> "NStar":
        """
        Returns a new NSystem object with properties relative to the star.
        """
        out = self.copy()
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
    _prefix = "st"
    _config_key = "star"


    def __init__(self, osystem: NSystem):
        self.df = osystem.df
        self.star_name = osystem.star_name
        self.star_aliases = osystem.star_aliases
        self.df_star_name = osystem.df_star_name
        self._choose_row()

    @property
    def age_myr(self) -> np.ndarray:
        """
        Returns the age of the star in Myr.
        """
        out = self._get("st_age")
        conversion_factor = u.Gyr.to(u.Myr)
        return out * conversion_factor
    
    @property
    def mass_solar(self) -> np.ndarray:
        """
        Returns the mass of the star in solar masses.
        """
        out = self._get("st_mass")
        return out
    
    @property
    def radius_solar(self) -> np.ndarray:
        """
        Returns the radius of the star in solar radii.
        """
        out = self._get("st_rad")
        return out
    
    @property
    def luminosity_solar(self) -> np.ndarray:
        """
        Returns the luminosity of the star in solar luminosities.
        """
        out = self._get("st_lum")
        return out
    
    @property
    def Teff_k(self) -> np.ndarray:
        """
        Returns the effective temperature of the star in K.
        """
        out = self._get("st_teff")
        return out
    
    @property
    def metallicity_dex(self) -> np.ndarray:
        """
        Returns the metallicity of the star in dex.
        """
        out = self._get("st_met")
        return out
    
    @property
    def spectral_type(self) -> str:
        """
        Returns the spectral type of the star.
        """
        return self.df["st_spectype"].iloc[0]

    
    def display(self, row:int = None):
        if row is not None:
            original_chosen_row = self._chosen_row
            self._chosen_row = row % len(self.df)
        Message(f"Star {cstr(self.star_name):y} properties:").list({
            "Age (Myr)": self.age_myr,
            "Mass (solar masses)": self.mass_solar,
            "Radius (solar radii)": self.radius_solar,
            "Luminosity (solar luminosities)": self.luminosity_solar,
            "Effective temperature (K)": self.Teff_k,
            "Irradiation temparature at 1 arcsec (K)": self.Tirr_k(1),
            "Metallicity (dex)": self.metallicity_dex,
            "Spectral type": self.spectral_type,
            "Reference": self.reference,
            "Row": f"{self._chosen_row}/{len(self.df)-1}"
        })
        if row is not None:
            self._chosen_row = original_chosen_row # reset to original chosen row
    
    # ------------------- #
    # !-- System Data --! #
    # ------------------- #

    @property
    def system(self) -> NSystem:
        """
        Return a `NSystem` object containing the properties of the system, 
        for the same row as the planet properties. This might be usefull is the 
        properties of the planet are derived using specific properties of the system.
        """
        out = self.copy()
        # keep only the chosen row
        out.df = self.df.iloc[[self._chosen_row]].reset_index(drop=True).copy(deep=True)
        return NSystem(out)
    
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

    _prefix = "pl"
    _config_key = "planet"

    

    def __init__(self, osystem: NSystem):
        self.df = osystem.df
        self.star_name = osystem.star_name
        self.star_aliases = osystem.star_aliases
        self.df_star_name = osystem.df_star_name
        self._choose_row()

    
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
        out = self._get("pl_orbper")
        conversion_factor = u.day.to(u.yr)
        return out * conversion_factor

    @property
    def mass_sini_mjup(self) -> np.ndarray:
        return self._get("pl_msinij")
    
    @property
    def mass_mjup(self) -> np.ndarray:
        return self._get("pl_bmassj")
    
    @property
    def sma_au(self) -> np.ndarray:
        return self._get("pl_orbsmax")
    
    @property
    def eccentricity(self) -> np.ndarray:
        return self._get("pl_orbeccen")
    
    @property
    def inclination_deg(self) -> np.ndarray:
        return self._get("pl_orbincl")
    
    @property
    def arg_periastron_deg(self) -> np.ndarray:
        return self._get("pl_orblper")
    
    @property
    def time_periastron_jd(self) -> np.ndarray:
        return self._get("pl_orbtper")
    
    @property
    def rv_amplitude_ms(self) -> np.ndarray:
        return self._get("pl_rvamp")
    
    @property
    def radius_rjup(self) -> np.ndarray:
        return self._get("pl_radj")
    

    # ----------------- #
    # !-- Star data --! #
    # ----------------- #

    @property
    def star(self) -> NStar:
        """
        Return a `NStar` object containing the properties of the star, 
        for the same row as the planet properties. This might be usefull is the 
        properties of the planet are derived using specific properties of the star.
        """
        out = self.copy()
        # keep only the chosen row
        out.df = self.df.iloc[[self._chosen_row]].reset_index(drop=True).copy(deep=True)
        return NStar(out)

    @property
    def system(self) -> NSystem:
        """
        Return a `NSystem` object containing the properties of the system, 
        for the same row as the planet properties. This might be usefull is the 
        properties of the planet are derived using specific properties of the system.
        """
        out = self.copy()
        # keep only the chosen row
        out.df = self.df.iloc[[self._chosen_row]].reset_index(drop=True).copy(deep=True)
        return NSystem(out.df)
    
    

    # --------------- #
    # !-- Diplsay --! #
    # --------------- #

    def display(self, row:int = None):
        if row is not None:
            original_chosen_row = self._chosen_row
            self._chosen_row = row % len(self.df)
            # disable fallback to avoid picking values from other rows
            fallback = ["fallback"]
            oplanet_temp_config[self._config_key]["fallback"] = False
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
            "Reference": self.reference,
            "Row": f"{self._chosen_row}/{len(self.df)-1}"
        })
        if row is not None:
            self._chosen_row = original_chosen_row # reset to original chosen row
            oplanet_temp_config[self._config_key]["fallback"] = fallback # reset fallback to its original value



if __name__ == "__main__":
    system = NSystem("LHS 1140")

    system.df.to_csv("lhs1140.csv", index=False)

    with Message("Planet Reference"):
        system.print_column("pl_refname")
