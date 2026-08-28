
from .star_utils import get_star_aliases
from oakley import *
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
import os
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u

from typing import Literal
import matplotlib.pyplot as plt

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
    cache_df = None

    def __init__(self, star_name: str):
        """
        Initializes a GStar object, loading several essential
        parameters from the Gaia DR3 database and storing
        them in cache.
        """
        # 1. Get the Gaia name
        if not star_name.startswith("Gaia DR3"):
            aliases = get_star_aliases(star_name)
            gaia_names = [alias for alias in aliases if alias.startswith("Gaia DR3")]

            if not len(gaia_names) == 1:
                Message(f"Unable to find a unique Gaia DR3 name for {star_name}. Matching aliases:", "!").list(gaia_names)
                assert len(gaia_names) > 0, f"Unable to find a Gaia DR3 name for {star_name}."
            gaia_name = gaia_names[0]
        else:
            gaia_name = star_name

        self.star_name = star_name
        self.gaia_name = gaia_name
        self.gaia_id = int(self.gaia_name.split(" ")[-1])

        self.df = self.query_id(self.gaia_id)

    @staticmethod
    def from_id(gaia_id: int) -> "GStar":
        """
        Initializes a GStar object from a Gaia DR3 source_id.
        """
        return GStar(f"Gaia DR3 {gaia_id}")


    # ---------------------- #
    # !-- Identification --! #
    # ---------------------- #

    @property
    def id(self) -> int:
        """Gaia DR3 identifier."""
        return self.gaia_id

    @property
    def epoch(self) -> int:
        """
        Reference epoch of the Gaia DR3 astrometric solution for this star.
        """
        return int(self.df["ref_epoch"].values[0])


    # ---------------- #
    # !-- Position --! #
    # ---------------- #

    def get_position_deg(self, date:str) -> tuple[float, float]:
        """
        Returns the right ascension and declination of the star at a given date, in degrees, from the current
        date and the object's proper motion.

        Parameters
        ----------
        date : str
            Date in the format "YYYY-MM-DD".
        """
        obs_time = Time(date, format='iso', scale='utc')
    
        # Handle NaNs for parallax and RV safely (assuming distant/0 for missing data)
        plx = self.parallax_mas[0]
        if plx <= 0 or np.isnan(plx):
            return self.ra_deg[0], self.dec_deg[0]  # super far, so no proper motion effect
        plx = 1e-5 if np.isnan(plx) else plx
        
        rv = self.rv_kms[0]
        rv = 0.0 if np.isnan(rv) else rv
        
        pmra = self.pmra_masyr[0]
        pmdec = self.pmdec_masyr[0]
        
        coord_ref = SkyCoord(
            ra = self.ra_deg[0] * u.deg,
            dec = self.dec_deg[0] * u.deg,
            distance = Distance(parallax=plx * u.mas, allow_negative=True),
            pm_ra_cosdec = (0.0 if np.isnan(pmra) else pmra) * u.mas / u.yr,
            pm_dec = (0.0 if np.isnan(pmdec) else pmdec) * u.mas / u.yr,
            radial_velocity = rv * u.km / u.s,
            obstime = Time(self.epoch, format='jyear'),
            frame = 'icrs'
        )
        
        coord_obs = coord_ref.apply_space_motion(new_obstime=obs_time)
        return coord_obs.ra.deg, coord_obs.dec.deg


    def _get_val_err(self, val:str) -> np.ndarray:
        """
        Returns a numpy array containing the value, lower error and upper error
        for a given parameter in the Gaia DR3 database.
        """
        return np.array([self.df[val].values[0], self.df[f"{val}_error"].values[0], -self.df[f"{val}_error"].values[0]])

    @property
    def ra_deg(self) -> np.ndarray:
        """Right Ascension (ICRS) in degrees."""
        return self._get_val_err("ra")

    @property
    def dec_deg(self) -> np.ndarray:
        """Declination (ICRS) in degrees."""
        return self._get_val_err("dec")

    @property
    def pmra_masyr(self) -> np.ndarray:
        """Proper motion in Right Ascension (ICRS) in mas/yr."""
        return self._get_val_err("pmra")

    @property
    def pmdec_masyr(self) -> np.ndarray:
        """Proper motion in Declination (ICRS) in mas/yr."""
        return self._get_val_err("pmdec")

    @property
    def rv_kms(self) -> np.ndarray:
        """Radial velocity in km/s."""
        return self._get_val_err("radial_velocity")

    @property
    def parallax_mas(self) -> np.ndarray:
        """Parallax in mas."""
        return self._get_val_err("parallax")

    @property
    def distance_pc(self) -> np.ndarray:
        """Distance in parsecs, derived from parallax."""
        parallax_mas = self.parallax_mas[0]
        if parallax_mas <= 0:
            return np.array([np.nan, np.nan, np.nan])
        distance = 1000.0 / parallax_mas
        distance_err = (self.parallax_mas[1] / parallax_mas**2) * 1000.0
        return np.array([distance, distance_err, -distance_err])


    # ------------------ #
    # !-- Photometry --! #
    # ------------------ #

    def _get_valnan(self, val:str) -> np.ndarray:
        """
        Returns a numpy array containing the value, lower error and upper error
        for a given parameter in the Gaia DR3 database. If the parameter is not
        available, returns NaN.
        """
        return np.array([self.df[val].values[0], np.nan, np.nan])

    @property
    def g_mag(self) -> np.ndarray:
        """Gaia G-band mean magnitude."""
        return self._get_valnan("phot_g_mean_mag")

    @property
    def bp_mag(self) -> np.ndarray:
        """Gaia BP-band mean magnitude."""
        return self._get_valnan("phot_bp_mean_mag")

    @property
    def rp_mag(self) -> np.ndarray:
        """Gaia RP-band mean magnitude."""
        return self._get_valnan("phot_rp_mean_mag")

    @property
    def bprp(self) -> np.ndarray:
        """Gaia BP-RP color index."""
        return self._get_valnan("bp_rp")

    @property
    def ruwe(self) -> np.ndarray:
        """Renormalized Unit Weight Error (RUWE) for Gaia DR3 astrometry."""
        return self._get_valnan("ruwe")


    # ------------------ #
    # !-- Morphology --! #
    # ------------------ #

    @property
    def classprob_galaxy(self) -> np.ndarray:
        """Probability of being a galaxy according to the DSC classifier."""
        return self._get_valnan("classprob_dsc_combmod_galaxy")

    @property
    def classprob_quasar(self) -> np.ndarray:
        """Probability of being a quasar according to the DSC classifier."""
        return self._get_valnan("classprob_dsc_combmod_quasar")

    @property
    def classprob_star(self) -> np.ndarray:
        """Probability of being a star according to the DSC classifier."""
        return self._get_valnan("classprob_dsc_combmod_star")

    @property
    def morphology(self) -> str:
        """Returns the most probable morphological classification."""
        probs = {
            "star": self.classprob_star[0],
            "galaxy": self.classprob_galaxy[0],
            "quasar": self.classprob_quasar[0]
        }
        return max(probs, key=probs.get)


    # ------------------ #
    # !-- Parameters --! #
    # ------------------ #

    def _get_vallim(self, valname:str) -> np.ndarray:
        """
        Returns a numpy array containing the value, lower error and upper error
        for a given parameter in the Gaia DR3 database. If the parameter is not
        available, returns NaN.
        """
        val = self.df[valname].values[0]
        return np.array([val, self.df[f"{valname}_upper"].values[0] - val, self.df[f"{valname}_lower"].values[0] - val])

    @property
    def teff_k(self) -> np.ndarray:
        """Effective temperature (GSP-Phot) in Kelvin."""
        return self._get_vallim("teff_gspphot")

    @property
    def logg_cgs(self) -> np.ndarray:
        """Surface gravity (GSP-Phot) in log10(cm/s^2)."""
        return self._get_vallim("logg_gspphot")

    @property
    def metallicity_dex(self) -> np.ndarray:
        """Metallicity (GSP-Phot) in dex."""
        return self._get_vallim("mh_gspphot")

    @property
    def ag_mag(self) -> np.ndarray:
        """Extinction in the G-band (GSP-Phot) in magnitudes."""
        return self._get_vallim("ag_gspphot")

    @property
    def ebpminrp_mag(self) -> np.ndarray:
        """Color excess E(BP-RP) (GSP-Phot) in magnitudes."""
        return self._get_vallim("ebpminrp_gspphot")


    # --------------- #
    # !-- Display --! #
    # --------------- #

    def __str__(self) -> str:
        return f"GStar({self.star_name}, {self.gaia_name})"

    def __repr__(self) -> str:
        return f"<GStar({self.id})>"

    def display(self) -> None:
        with Message(str(self), "#"):
            Message("Position (ICRS):", "?").list({
                "RA (deg)": self.ra_deg,
                "Dec (deg)": self.dec_deg,
                "PM RA (mas/yr)": self.pmra_masyr,
                "PM Dec (mas/yr)": self.pmdec_masyr,
                "Radial Velocity (km/s)": self.rv_kms
            })
            Message("Photometry:", "?").list({
                "G (mag)": self.g_mag,
                "BP (mag)": self.bp_mag,
                "RP (mag)": self.rp_mag,
                "BP-RP (mag)": self.bprp,
                "RUWE": self.ruwe
            })
            Message("Morphology:", "?").list({
                "Morphology": self.morphology,
                "Prob. Star": self.classprob_star,
                "Prob. Galaxy": self.classprob_galaxy,
                "Prob. Quasar": self.classprob_quasar
            })
            Message("Astrophysical Parameters (GSP-Phot):", "?").list({
                "Teff (K)": self.teff_k,
                "logg (dex)": self.logg_cgs,
                "Metallicity (dex)": self.metallicity_dex,
                "A_G (mag)": self.ag_mag,
                "E(BP-RP) (mag)": self.ebpminrp_mag
            })


    # ------------- #
    # !-- Cache --! #
    # ------------- #

    @staticmethod
    def _add2cache(df:pd.DataFrame) -> None:
        """
        Adds a pandas DataFrame to the cache.
        """
        if GStar.cache_df is None:
            if os.path.exists(cache_path):
                GStar.cache_df = pd.read_csv(cache_path)
            else:
                GStar.cache_df = df

        GStar.cache_df = pd.concat([GStar.cache_df, df], ignore_index=True)
        GStar.cache_df.drop_duplicates(subset=["source_id"], keep="last", inplace=True)
        GStar.cache_df.to_csv(cache_path, index=False)

    @staticmethod
    def _getfromcache(gaia_ids:list[int]) -> pd.DataFrame:
        """
        Retrieves a pandas DataFrame from the cache.
        """
        if GStar.cache_df is None:
            if os.path.exists(cache_path):
                GStar.cache_df = pd.read_csv(cache_path)
            else:
                return None

        return GStar.cache_df[GStar.cache_df["source_id"].isin(gaia_ids)]


    # --------------- #
    # !-- Plotter --! #
    # --------------- #

    def plot(
        self,
        date:str,
        rotation_rad: float = 0,
        radius_arcsec: float = 20,
        radius_arcsec_query: float = None,
        axis_unit:Literal["arcsec", "au"] = "arcsec",
        **kwargs
    ) -> None:
        """
        On the current axis, plots the position of all stars in the Gaia DR3 database within a circular
        region of a given radius around the star, at a given date.

        Parameters
        ----------
        date : str
            Date in the format "YYYY-MM-DD". Important to account for proper motion of the objects.
        rotation_rad : float, optional
            Rotation angle to apply to the coordinates, in radians. Default is 0 (meaning the plot
            is expected to be north alinged). You can use the WCS information of the image to find the rotation angle.
        radius_arcsec : float, optional
            Radius of the circular region to plot, in arcseconds. Default is 20.
        radius_arcsec_query : float, optional
            Radius of the circular region to query the Gaia DR3 database, in arcseconds. Default to None (use radius_arcsec).
            Indeed, because of proper motion and all, ou might need to query a larger region than the one you want to plot, 
            and then only plot the objects within the smaller radius.
        axis_unit : str, optional
            Unit of the axis. Can be "arcsec" or "au". Default is "arcsec".
        **kwargs
            Additional keyword arguments to pass to plt.scatter().

        Notes
        -----
        Here is my personal function to find the rotation angle from the WCS of an image:
        ```python
        def get_rotation(wcs):
            # Use a reference pixel near the image center, then measure the local
            # sky direction of +North in pixel space from the WCS.
            x0 = (self.shape[1] - 1) / 2
            y0 = (self.shape[0] - 1) / 2

            sky0 = SkyCoord.from_pixel(x0, y0, wcs)
            sky_north = sky0.directional_offset_by(0 * u.deg, 1 * u.arcsec)
            xN, yN = sky_north.to_pixel(self.wcs)

            north_vec = np.array([xN - x0, yN - y0])

            # Angle between the measured north direction and the positive vertical axis.
            rotation_angle_rad = np.arctan2(north_vec[0], north_vec[1])
            return -rotation_angle_rad
        ```
        """
        if not "color" in kwargs:
            kwargs["color"] = "black"
        if radius_arcsec_query is None:
            radius_arcsec_query = radius_arcsec

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        ra_origin, dec_origin = self.get_position_deg(date)

        # 1. Collect all Gaia DR3 source ids
        gaia_ids = self.query_region(self.ra_deg[0], self.dec_deg[0], radius_arcsec_query)
        # remove self
        gaia_ids = [id for id in gaia_ids if id != self.id]
        if len(gaia_ids) == 0:
            return

        # 2. Query Gaia DR3 database
        df = self.query_id(gaia_ids) # this will trigger one big query at once
        # which will be cached

        # 3. Plot each object one by one
        used_markers = set()
        for gaia_id in gaia_ids:

            # a. Get coords
            star = GStar.from_id(gaia_id)
            ra, dec = star.get_position_deg(date)

            # b. Convert to projected relative coords
            cos_dec = np.cos(np.radians(dec_origin)) 
            x_unrotated = - (ra - ra_origin) * 3600.0 * cos_dec # because east is left in the plots
            y_unrotated = (dec - dec_origin) * 3600.0

            # c. Apply rotation
            sep_x = x_unrotated * np.cos(rotation_rad) - y_unrotated * np.sin(rotation_rad)
            sep_y = x_unrotated * np.sin(rotation_rad) + y_unrotated * np.cos(rotation_rad)

            # skip if outside the radius
            if np.sqrt(sep_x**2 + sep_y**2) > radius_arcsec:
                continue

            # d. Convert to au if needed
            if axis_unit.lower().strip() == "au":
                sep_x *= self.distance_pc[0]
                sep_y *= self.distance_pc[0]

            # e. Choose maker based on morphology
            if star.morphology == "star":
                marker = "*"
            elif star.morphology == "galaxy":
                marker = "h"
            elif star.morphology == "quasar":
                marker = "d"
            else:
                marker = "o" # should not happen
            used_markers.add(marker)

            # f. Plot
            plt.scatter(sep_x, sep_y, marker=marker, **kwargs)

        # 4. Add points to the legend
        marker2label = {
            "*": "Star",
            "h": "Galaxy",
            "d": "Quasar",
            "o": "Unknown"
        }
        for marker in used_markers:
            plt.scatter([], [], marker=marker, label=marker2label[marker], **kwargs)
        # plt.legend() # let the user decide

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax) # avoid changing the limits of the plot
            


    # --------------- #
    # !-- Queries --! #
    # --------------- #

    @staticmethod
    def query_id(gaia_ids:list[int] | int) -> pd.DataFrame:
        """
        Collects various essential information about an object from
        the Gaia DR3 database (photometry, astrometry and astrophysical
        parameters) and stores it in a pandas DataFrame.
        """
        if not isinstance(gaia_ids, list):
            gaia_ids = [gaia_ids]

        # 1. Check cache
        cached_df = GStar._getfromcache(gaia_ids)
        # filter out the ids that are already in the cache
        if cached_df is not None:
            cached_ids = cached_df["source_id"].tolist()
            gaia_ids = [id for id in gaia_ids if id not in cached_ids]
        if len(gaia_ids) == 0:
            return cached_df

        # 2. Query Gaia DR3 database
        query = f"""
            SELECT
                -- 1. Id
                gs.source_id,
                gs.ref_epoch,

                -- 2. Astrometry
                gs.ra, gs.ra_error,
                gs.dec, gs.dec_error,
                gs.parallax, gs.parallax_error,
                gs.pmra, gs.pmra_error,
                gs.pmdec, gs.pmdec_error,
                gs.radial_velocity, gs.radial_velocity_error,

                -- 3. Photometry
                gs.phot_g_mean_mag,
                gs.phot_bp_mean_mag,
                gs.phot_rp_mean_mag,
                gs.bp_rp,
                gs.ruwe,

                -- 4. Morphological Classifiers
                gs.classprob_dsc_combmod_galaxy,
                gs.classprob_dsc_combmod_quasar,
                gs.classprob_dsc_combmod_star,

                -- 5. Astrophysical Parameters
                ap.teff_gspphot, ap.teff_gspphot_lower, ap.teff_gspphot_upper,
                ap.logg_gspphot, ap.logg_gspphot_lower, ap.logg_gspphot_upper,
                ap.mh_gspphot, ap.mh_gspphot_lower, ap.mh_gspphot_upper,
                ap.ag_gspphot, ap.ag_gspphot_lower, ap.ag_gspphot_upper,
                ap.ebpminrp_gspphot, ap.ebpminrp_gspphot_lower, ap.ebpminrp_gspphot_upper

                FROM gaiadr3.gaia_source AS gs
                LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap
                    ON gs.source_id = ap.source_id
                WHERE gs.source_id IN ({', '.join(map(str, gaia_ids))})
        """
        job = Gaia.launch_job_async(query)
        df:pd.DataFrame = job.get_results().to_pandas()

        # 3. Merge with cache if needed
        if cached_df is not None:
            if not df.empty:
                df = pd.concat([cached_df, df], ignore_index=True)
                df.drop_duplicates(subset=["source_id"], keep="last", inplace=True)
            else:
                df = cached_df

        # 4. Add to cache
        GStar._add2cache(df)


        return df


    @staticmethod
    def query_region(
        ra_deg: float, dec_deg: float, radius_arcsec: float
    ) -> list[int]:
        """
        Retrieves all Gaia DR3 source_ids within a circular sky region.

        Parameters
        ----------
        ra_deg : float
            Right Ascension of the center in degrees (ICRS).
        dec_deg : float
            Declination of the center in degrees (ICRS).
        radius_arcsec : float
            Search cone radius in arcseconds.

        Returns
        -------
        list[int]
            List of integer Gaia DR3 source ids.
        """
        radius_deg = radius_arcsec / 3600.0

        query = f"""
            SELECT source_id
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
            )
        """

        job = Gaia.launch_job(query)
        results = job.get_results()

        if len(results) == 0:
            return []
        return np.array(results['source_id'], dtype=np.int64).astype(int).tolist()
