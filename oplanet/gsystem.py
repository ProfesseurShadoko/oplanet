
from .star_utils import get_star_aliases
from oakley import *
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
import os

# ------------- #
# !-- Files --! #
# ------------- #

data_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data"
)
os.makedirs(data_folder, exist_ok=True)
cache_path = os.path.join(data_folder, "gsystem_cache.csv")


class GStar:

    cache_max_nstars = 1000 # a df of 1000 rows, for quick access

    # ------------- #
    # !-- Query --! #
    # ------------- #

    def __init__(
        self,
        star_name: str
    ):
        """
        Initializes a GStar object with the given star name,
        loading stellar parameters from Gaia DR3.

        Parameters
        ----------
        star_name : str
            The name of the star to initialize the GStar object with.
        """

        # 1. Obtain the Gaia name
        aliases = get_star_aliases(star_name)
        gaia_name = [alias for alias in aliases if alias.startswith("Gaia DR3")]

        if not len(gaia_name) == 1:
            Message(f"Unable to find a unique Gaia DR3 name for {star_name}. Matching aliases:", "!").list(gaia_name)
            assert len(gaia_name) > 0, f"Unable to find a Gaia DR3 name for {star_name}."
        gaia_name = gaia_name[0]

        self.star_name = star_name
        self.gaia_name = gaia_name
        self.gaia_id = int(self.gaia_name.split(" ")[-1])

        if self._is_cached():
            return self.load()

        with Task(f"Querying Gaia DR3 for {repr(self)}"):
            # 2. Query Gaia DR3 for stellar parameters
            query_flame = f"""
                SELECT
                    source_id,
                    flags_flame,
                    age_flame,
                    age_flame_lower,
                    age_flame_upper,
                    mass_flame,
                    mass_flame_lower,
                    mass_flame_upper,
                    radius_flame,
                    radius_flame_lower,
                    radius_flame_upper,
                    lum_flame,
                    lum_flame_lower,
                    lum_flame_upper,
                    evolstage_flame
                FROM gaiadr3.astrophysical_parameters
                WHERE source_id = {self.gaia_id}
            """

            query_spec = f"""
                SELECT
                    source_id,

                    teff_gspphot,
                    teff_gspphot_lower,
                    teff_gspphot_upper,

                    logg_gspphot,
                    logg_gspphot_lower,
                    logg_gspphot_upper,

                    mh_gspphot,
                    mh_gspphot_lower,
                    mh_gspphot_upper,

                    ag_gspphot,
                    ag_gspphot_lower,
                    ag_gspphot_upper,

                    ebpminrp_gspphot,
                    ebpminrp_gspphot_lower,
                    ebpminrp_gspphot_upper

                FROM gaiadr3.astrophysical_parameters
                WHERE source_id = {self.gaia_id}
            """

            query_source = f"""
                SELECT
                    source_id,
                    phot_g_mean_mag,
                    phot_bp_mean_mag,
                    phot_rp_mean_mag,
                    ruwe,
                    bp_rp

                FROM gaiadr3.gaia_source
                WHERE source_id = {self.gaia_id}
            """

            job_flame = Gaia.launch_job_async(query_flame)
            job_spec = Gaia.launch_job_async(query_spec)
            job_source = Gaia.launch_job_async(query_source)
            df_flame: pd.DataFrame = job_flame.get_results().to_pandas()
            df_spec: pd.DataFrame = job_spec.get_results().to_pandas()
            df_source: pd.DataFrame = job_source.get_results().to_pandas()
            duplicate_columns = ["source_id"]
            df_spec = df_spec.drop(columns=duplicate_columns)
            df_source = df_source.drop(columns=duplicate_columns)
            self.df = pd.concat([df_flame, df_spec, df_source], axis=1)
            self.dump() # to cache
            Message("Information retrieved and cached successfully.", "#")


    def __repr__(self) -> str:
        return f"<GStar({self.star_name}):{self.gaia_id}>"

    def __str__(self) -> str:
        return repr(self)[1:-1]


    # ------------- #
    # !-- FLAME --! #
    # ------------- #

    def _get_flame(self, param: str) -> np.ndarray:
        """
        Helper function to retrieve a stellar parameter from the Gaia DR3 FLAME model.

        Parameters
        ----------
        param : str
            The name of the stellar parameter to retrieve (e.g., "age", "mass", "radius", "lum").

        Returns
        -------
        np.ndarray
            The value of the stellar parameter, positive uncertainty, negative uncertainty.
        """
        return np.array([
            self.df[f"{param}_flame"].values[0],
            self.df[f"{param}_flame_upper"].values[0] - self.df[f"{param}_flame"].values[0],
            self.df[f"{param}_flame_lower"].values[0] - self.df[f"{param}_flame"].values[0]
        ])

    def age_myr(self) -> np.ndarray:
        """
        Estimated age of the star in Myr, based on the luminosity
        model by GaiaDR3. 

        Returns
        -------
        np.ndarray
            Estimated age of the star in Myr, positive uncertainty, negative uncertainty.
        """
        return self._get_flame("age") * 1e3 # Gyr to Myr

    def mass_msun(self) -> np.ndarray:
        """
        Estimated mass of the star in solar masses, based on the luminosity
        model by GaiaDR3. 

        Returns
        -------
        np.ndarray
            Estimated mass of the star in solar masses, positive uncertainty, negative uncertainty.
        """
        return self._get_flame("mass")

    def radius_rsun(self) -> np.ndarray:
        """
        Estimated radius of the star in solar radii, based on the luminosity
        model by GaiaDR3. 

        Returns
        -------
        np.ndarray
            Estimated radius of the star in solar radii, positive uncertainty, negative uncertainty.
        """
        return self._get_flame("radius")

    def luminosity_lsun(self) -> np.ndarray:
        """
        Estimated luminosity of the star in solar luminosities, based on the luminosity
        model by GaiaDR3. 

        Returns
        -------
        np.ndarray
            Estimated luminosity of the star in solar luminosities, positive uncertainty, negative uncertainty.
        """
        return self._get_flame("lum")


    # --------------------------- #
    # !-- Spectral Parameters --! #
    # --------------------------- #

    def _get_spec(self, param: str) -> np.ndarray:
        """
        Helper function to retrieve a stellar parameter from the Gaia DR3 spectral model.

        Parameters
        ----------
        param : str
            The name of the stellar parameter to retrieve (e.g., "teff", "logg", "mh", "ag", "ebpminrp").

        Returns
        -------
        np.ndarray
            The value of the stellar parameter, positive uncertainty, negative uncertainty.
        """
        return np.array([
            self.df[f"{param}_gspphot"].values[0],
            self.df[f"{param}_gspphot_upper"].values[0] - self.df[f"{param}_gspphot"].values[0],
            self.df[f"{param}_gspphot_lower"].values[0] - self.df[f"{param}_gspphot"].values[0]
        ])

    def teff_k(self) -> np.ndarray:
        """
        Estimated effective temperature of the star in Kelvin.

        Returns
        -------
        np.ndarray
            Estimated effective temperature of the star in Kelvin, positive uncertainty, negative uncertainty.
        """
        return self._get_spec("teff")

    def logg_cgs(self) -> np.ndarray:
        """
        Estimated surface gravity of the star in log10(cm/s^2).

        Returns
        -------
        np.ndarray
            Estimated surface gravity of the star in log10(cm/s^2), positive uncertainty, negative uncertainty.
        """
        return self._get_spec("logg")

    def metallicity_dex(self) -> np.ndarray:
        """
        Estimated metallicity of the star in [M/H], in log10(Z/Z_sun).

        Returns
        -------
        np.ndarray
            Estimated metallicity of the star in [M/H], positive uncertainty, negative uncertainty.
        """
        return self._get_spec("mh")

    def ebprp_mag(self) -> np.ndarray:
        """
        Estimated reddening of the star in E(BP-RP) magnitude (aka excess of BP-RP color).

        Returns
        -------
        np.ndarray
            Estimated reddening of the star in E(BP-RP) magnitude, positive uncertainty, negative uncertainty.
        """
        return self._get_spec("ebpminrp")


    # ------------------- #
    # !-- Gaia Source --! #
    # ------------------- #

    def _get_source(self, param: str) -> np.ndarray:
        """
        Helper function to retrieve a measurement from the Gaia DR3 source catalog.

        Parameters
        ----------
        param : str
            The name of the stellar parameter to retrieve (e.g., "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "ruwe", "bp_rp").

        Returns
        -------
        np.ndarray
            The value of the stellar parameter, no uncertainty (nans)
        """
        return np.array([self.df[f"{param}"].values[0], np.nan, np.nan])

    def g_mag(self) -> np.ndarray:
        """
        Gaia G-band mean magnitude.

        Returns
        -------
        np.ndarray
            Gaia G-band mean magnitude, no uncertainty (nans)
        """
        return self._get_source("phot_g_mean_mag")

    def bp_mag(self) -> np.ndarray:
        """
        Gaia BP-band mean magnitude.

        Returns
        -------
        np.ndarray
            Gaia BP-band mean magnitude, no uncertainty (nans)
        """
        return self._get_source("phot_bp_mean_mag")

    def rp_mag(self) -> np.ndarray:
        """
        Gaia RP-band mean magnitude.

        Returns
        -------
        np.ndarray
            Gaia RP-band mean magnitude, no uncertainty (nans)
        """
        return self._get_source("phot_rp_mean_mag")

    def ruwe(self) -> np.ndarray:
        """
        Gaia Renormalized Unit Weight Error (RUWE).

        Returns
        -------
        np.ndarray
            Gaia RUWE, no uncertainty (nans)
        """
        return self._get_source("ruwe")

    def bp_rp(self) -> np.ndarray:
        """
        Gaia BP-RP color.

        Returns
        -------
        np.ndarray
            Gaia BP-RP color, no uncertainty (nans)
        """
        return self._get_source("bp_rp")



    # ------------- #
    # !-- Cache --! #
    # ------------- #

    def _is_cached(self) -> bool:
        """
        Checks if the GStar object is cached in the cache file.

        Returns
        -------
        bool
            True if the GStar object is cached, False otherwise.
        """
        if not os.path.exists(cache_path):
            return False

        cache_df = pd.read_csv(cache_path)
        return self.gaia_id in cache_df["source_id"].values

    def clear_cache(self):
        """
        Clears the cache file.
        """
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def load(self):
        """
        Loads the GStar object from the cache file.
        """
        cache_df = pd.read_csv(cache_path)
        self.df = cache_df[cache_df["source_id"] == self.gaia_id]
        return

    def dump(self):
        """
        Dumps the GStar object to the cache file.
        """
        if os.path.exists(cache_path):
            if self._is_cached():
                return
            cache_df = pd.read_csv(cache_path)
            cache_df = pd.concat([cache_df, self.df], ignore_index=True)
            if len(cache_df) > self.cache_max_nstars:
                cache_df = cache_df.tail(self.cache_max_nstars)
            cache_df.to_csv(cache_path, index=False)
        else:
            self.df.to_csv(cache_path, index=False)

    def display(self):
        with Message(str(self), "#"):
            Message("FLAME Parameters:", "#").list({
                "Age (Myr)": self.age_myr(),
                "Mass (Msun)": self.mass_msun(),
                "Radius (Rsun)": self.radius_rsun(),
                "Luminosity (Lsun)": self.luminosity_lsun()
            })
            Message("Spectral Parameters:", "#").list({
                "Teff (K)": self.teff_k(),
                "logg (cgs)": self.logg_cgs(),
                "Metallicity [M/H]": self.metallicity_dex(),
                "E(BP-RP) (mag)": self.ebprp_mag()
            })
            Message("Source Parameters:", "#").list({
                "G (mag)": self.g_mag(),
                "BP (mag)": self.bp_mag(),
                "RP (mag)": self.rp_mag(),
                "RUWE": self.ruwe(),
                "BP-RP (mag)": self.bp_rp()
            })





if __name__ == "__main__":
    star_name = "Beta Pictoris"
    GStar(star_name).display()