

# --------------- #
# !-- Imports --! #
# --------------- #


from oakley import *

try:
    from astroquery.simbad import Simbad
    from astroquery.vizier import Vizier
    Simbad.add_votable_fields('parallax')
    Vizier.ROW_LIMIT = -1
    vizier_wise = Vizier(columns=["*"], catalog="II/328/allwise")
    vizier_2mass = Vizier(columns=["*"], catalog="II/246/out")
    vizier_gaia = Vizier(columns=["*"], catalog="I/350/gaiaedr3")
except:
    Message("Failed to import Simbad or Vizier. Check your internet connection.", "!")

from astropy import units as u
from astropy import constants as const
from astropy.coordinates import SkyCoord
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

from astropy.table import Table
from urllib.parse import quote

from oakley import Message, ProgressBar, cstr, Task
import matplotlib.pyplot as plt

# find the directory of the current python file
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "data", "photometry")
os.makedirs(cache_dir, exist_ok=True)
import json


class StarInfoRetriever:
    photometric_points = ["WISE:W1", "WISE:W2", "WISE:W3", "WISE:W4", "2MASS:J", "2MASS:H", "2MASS:Ks"]
    catalog_priorities = ["HD", "HIP", "GJ", "LHS", "HR", "TOI", "LTT", "L", "2MASS", "TYC", "BD", "SAO"]
    radius = 2
    spline_order = 2
    _cache = {}
    _star_distance_cache = {}
    _star_aliases_cache = {}
    _star_coords_cache = {}
        
    # ---------------- #
    # !-- Distance --! #
    # ---------------- #

    @staticmethod
    def get_star_distance_pc(star:str) -> u.Quantity:
        """
        Get the distance to a star in parsecs using Simbad. Retrieved from parallax.

        Parameters
        ----------
        star : str
            Name of the star. Must be compatible with Simbad database.

        Returns
        -------
        float
            Distance to the star in parsecs.
        """
        star = StarInfoRetriever.get_star_name(star)
        
        # 1. Check cache
        if StarInfoRetriever.star_to_cache_name(star) in StarInfoRetriever._star_distance_cache:
            return StarInfoRetriever._star_distance_cache[StarInfoRetriever.star_to_cache_name(star)]
        
        # 2. Query Simbad
        result = Simbad.query_object(star)
        
        # 3. Extract parallax and compute distance
        plx_kwd = "PLX_VALUE"
        try:
            result[plx_kwd]
        except KeyError:
            plx_kwd = "plx_value"
        parallax_mas = result[plx_kwd][0]
        distance_pc = 1e3 / parallax_mas
        StarInfoRetriever._star_distance_cache[StarInfoRetriever.star_to_cache_name(star)] = distance_pc * u.pc

        # 4. Return
        return StarInfoRetriever.get_star_distance_pc(star)
    
    @staticmethod
    def get_star_coords(star:str) -> tuple[float, float]:
        """
        Get the RA and Dec of a star in degrees using Simbad.

        Parameters
        ----------
        star : str
            Name of the star. Must be compatible with Simbad database.

        Returns
        -------
        float, float
            RA and Dec of the star in degrees.
        """
        star = StarInfoRetriever.get_star_name(star)
        
        # 1. Check cache
        if StarInfoRetriever.star_to_cache_name(star) in StarInfoRetriever._star_coords_cache:
            return StarInfoRetriever._star_coords_cache[StarInfoRetriever.star_to_cache_name(star)]
        
        # Query Simbad
        result = Simbad.query_object(star)
        if len(result) == 0:
            raise ValueError(f"Star {star} not found in Simbad database.")
    
        ra_kwd, dec_kwd = "RA", "DEC"
        try:
            result[ra_kwd], result[dec_kwd]
        except KeyError:
            ra_kwd, dec_kwd = "ra", "dec"
        coords = (result[0][ra_kwd], result[0][dec_kwd])
        StarInfoRetriever._star_coords_cache[StarInfoRetriever.star_to_cache_name(star)] = coords
        return StarInfoRetriever.get_star_coords(star)
    
    
    @staticmethod
    def get_star_name(star_name:str) -> str:
        """
        Takes a string and translates it to try to make it compatible with Simbad.
        Applies `upper()` and handles hypens and underscores.
        """
        star_name = star_name.replace("_", " ").strip()
        while len(star_name) > len(star_name.replace("  ", " ")):
            star_name = star_name.replace("  ", " ")
        star_name = star_name.upper()
        
        # handle hyphens carefully
        import re
        def replace_hyphen(s: str) -> str:
            """
            Replace '-' with ' ' unless it is between two digits.
            """
            return re.sub(r'(?<!\d)-(?!\d)|(?<!\d)-(?=\d)|(?<=\d)-(?!\d)', ' ', s)
        star_name = replace_hyphen(star_name)
        
        # for trappist we actually leave the hyphen
        star_name = star_name.replace("TRAPPIST ", "TRAPPIST-")
        if star_name.startswith("CD "):
            star_name = star_name.replace("-", " ")
            star_name = star_name.replace("CD ", "CD-")
        
        return star_name
        
    
    @staticmethod
    def get_star_aliases(star_name: str) -> list[str]:
        """
        Given a star name, returns a list of other names/identifiers in different catalogs.
        
        Parameters
        ----------
        star_name : str
            Name of the star to query.
            
        Returns
        -------
        list[str]
            List of aliases for the star.
        """
        # modify star name to remove extra spaces, transform hypens and underscores to spaces
        star_name = StarInfoRetriever.get_star_name(star_name)
        
        # Check cache first
        if StarInfoRetriever.star_to_cache_name(star_name) in StarInfoRetriever._star_aliases_cache:
            return StarInfoRetriever._star_aliases_cache[StarInfoRetriever.star_to_cache_name(star_name)]
        
        
        # Configure Simbad to return all identifiers
        custom_simbad = Simbad()
        custom_simbad.TIMEOUT = 60
        custom_simbad.add_votable_fields('ids')  # 'ids' is the cross-id list

        result = custom_simbad.query_object(star_name)
        
        if result is None:
            return []  # star not found

        # 'IDS' column contains all aliases as a single string separated by '|'
        id_kwd = "IDS"
        try:
            result[id_kwd]
        except KeyError:
            id_kwd = "ids"
        if len(result[id_kwd]) == 0 or result[id_kwd][0] is None:
            Message(f"Unknown star {cstr(star_name):rb}.", "!")
            return []
        
        ids_str = result[id_kwd][0]
        aliases = [s.strip() for s in ids_str.split('|') if s.strip()]
        
        for i in range(len(aliases)):
            while len(aliases[i]) > len(aliases[i].replace("  ", " ")):
                aliases[i] = aliases[i].replace("  ", " ")
                
        # sort the aliases by priority
        def alias_priority(alias: str) -> int:
            for i, prefix in enumerate(StarInfoRetriever.catalog_priorities):
                if alias.lower().startswith(prefix.lower()):
                    return i
            return len(StarInfoRetriever.catalog_priorities) # lowest priority if not found
        aliases.sort(key=alias_priority)

        StarInfoRetriever._star_aliases_cache[StarInfoRetriever.star_to_cache_name(star_name)] = aliases
        return aliases
    
    @staticmethod
    def is_alias(star_name: str, alias: str) -> bool:
        """
        Check if `alias` is an alias of `star_name`.

        Parameters
        ----------
        star_name : str
            Name of the star.
        alias : str
            Potential alias.
            
        Returns
        -------
        bool
            True if `alias` is an alias of `star_name`, False otherwise.
        """
        aliases = StarInfoRetriever.get_star_aliases(star_name)
        
        # make all aliases lowercase, remove spaces and underscores and hypens for comparison
        aliases = [a.lower().replace(" ", "").replace("_", "").replace("-", "") for a in aliases]
        alias = alias.lower().replace(" ", "").replace("_", "").replace("-", "")
        
        return alias in aliases
    
    
    
    # -------------------- #
    # !-- Stellar Flux --! #
    # -------------------- #
    
    
    def __init__(self, star:str):
        self.star = StarInfoRetriever.get_star_name(star)

        if StarInfoRetriever.star_to_cache_name(star) in StarInfoRetriever._cache:
            self._load_from_cache()
        else:
            self._load_photometry()
            self._check_wise4()
            self.add_to_cache()
        self._fit_photometry()
        
        
    def _load_photometry(self) -> None:
        """
        Download photometry from Vizier and remove anomalous points.
        Sets `self.df` and `self.wavelengths`, `self.fluxes`.
        """
        
        with Message(f"Retrieving photometry for star {cstr(self.star).green()} with radius {cstr(self.radius).green()} arcsec", "#"):
            # 1. Query SED from Vizier
            try:
                sed = Table.read(
                    f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={quote(self.star)}&-c.rs={self.radius}",
                    format="votable"
                )
            except Exception as e:
                Message("An exceptiuon occured while retrieving photometry. Trying different star names.", "!")
                aliases = StarInfoRetriever.get_star_aliases(self.star)
                for alias in aliases:
                    try:
                        sed = Table.read(
                            f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={quote(alias)}&-c.rs={self.radius}",
                            format="votable"
                        )
                        break
                    except Exception as e:
                        continue
                else:
                    Message(f"Unable to retrieve photometry for {self.star}.", "!")
                    raise e

            self.df = sed.to_pandas()
            Message.print(f"{len(self.df)} photometric measurements retrieved.")
            
            # 2. Keep only relevant filters
            mask = self.df["sed_filter"].isin(StarInfoRetriever.photometric_points)
            self.df = self.df[mask].reset_index(drop=True)
            Message.print(f"{(~mask).sum()} points dropped (keeping only {StarInfoRetriever.photometric_points[0]}, {StarInfoRetriever.photometric_points[1]}, ...).")

            # 3. Keep only relevant columns
            self.df = self.df[["sed_filter", "sed_freq", "sed_flux"]]
            
            # 4. Remove duplicate measurements
            is_duplicate = self.df.duplicated(keep='first') # no subset => remove identical rows
            self.df = self.df[~is_duplicate].reset_index(drop=True)
            Message.print(f"{is_duplicate.sum()} duplicate points dropped.")
            
            self.df = self.df.copy(deep=True)
            
            # 5. Sort by frequency
            self.df.sort_values(by=["sed_freq"], inplace=True)
            self.df = self.df[::-1].reset_index(drop=True) # sort by wavelength = frequency^-1
            
            # 6. Get sorted filter names
            self.filters = self.df["sed_filter"].unique().tolist()
            
            # 7. Define loss function
            def loss(_df):
                loss_value = 0.0
                
                # compute discrete second derivative at each filter
                for i in range(len(self.filters) - 2):
                    filter1 = self.filters[i]
                    filter2 = self.filters[i+1]
                    filter3 = self.filters[i+2]
                    
                    freq1 = _df[_df["sed_filter"] == filter1]["sed_freq"].values[0]
                    freq2 = _df[_df["sed_filter"] == filter2]["sed_freq"].values[0]
                    freq3 = _df[_df["sed_filter"] == filter3]["sed_freq"].values[0]
                    
                    dfreq12 = freq2 - freq1
                    dfreq23 = freq3 - freq2
                    dfreq13 = freq3 - freq1
                    
                    # gather all measurements for filter1 and filter2
                    mask1 = _df["sed_filter"] == filter1
                    mask2 = _df["sed_filter"] == filter2
                    mask3 = _df["sed_filter"] == filter3
                    
                    flux1 = _df[mask1]["sed_flux"].values
                    flux2 = _df[mask2]["sed_flux"].values
                    flux3 = _df[mask3]["sed_flux"].values
                    
                    # compute all second derivative combinations
                    for f1 in flux1: # this is time consuming, TODO: think of something better (maybe don't recompute every second derivative)
                        for f2 in flux2:
                            for f3 in flux3:
                                d12 = (f2 - f1) / dfreq12
                                d23 = (f3 - f2) / dfreq23
                                dd13 = (d23 - d12) / dfreq13
                                if dd13 > 0:
                                    loss_value += dd13
                                    # i want a concave function => only negative second derivatives
                                    # => penalize positive second derivatives
                                
                                else:
                                    loss_value += 1e-3 * abs(dd13) # small penalty for negative second derivatives --> I don't want a too strong second derivative
                return loss_value

            self._original_df = self.df.copy(deep=True)

            # 8. Clean data to enforce concavity
            with Message(f"Choosing {len(self.filters)} points out of {len(self.df)} to enforce concavity of SED..."):
                # how many points need to be dropped?
                n_droped = len(self.df) - len(self.filters) # we want only one point per filter
                for _ in ProgressBar(range(n_droped), size=n_droped):
                    duplicate_filters = self.df["sed_filter"][self.df["sed_filter"].duplicated(keep=False)].unique().tolist()
                    
                    # find the best point to drop
                    minimal_loss = np.inf
                    best_removal = None
                    
                    # gather all indices of filters with duplicates
                    all_indices = self.df.index[self.df["sed_filter"].isin(duplicate_filters)].tolist()
                    for index in all_indices:
                        _df = self.df.drop(index)
                        current_loss = loss(_df)
                        if current_loss < minimal_loss:
                            minimal_loss = current_loss
                            best_removal = index
                    self.df.drop(best_removal, inplace=True)
            
            # 9. Compute wavelengths
            self.df["wavelength_um"] = ( const.c.to("um/s") / (self.df["sed_freq"].values * u.GHz) ).to("um").value
            self.df["flux_Jy"] = self.df["sed_flux"].values# * u.Jy
            self._original_df["wavelength_um"] = ( const.c.to("um/s") / (self._original_df["sed_freq"].values * u.GHz) ).to("um").value
            self._original_df["flux_Jy"] = self._original_df["sed_flux"].values

    def _check_wise4(self) -> None:
        """
        Sometimes, WISE_W4 photometry is completely off. This function checks if the WISE_W4 point is an outlier (ie greater than W3)
        """        
        # sort df by wavelength
        self.df.sort_values(by=["wavelength_um"], inplace=True)
        self.df = self.df.reset_index(drop=True)
        
        # check whether the last point as greater flux than the second last point, if so drop the last point
        if self.df["sed_filter"].values[-1] == "WISE:W4":
            if self.df["flux_Jy"].values[-1] > self.df["flux_Jy"].values[-2]:
                Message.print(f"WISE W4 photometry for star {cstr(self.star).green()} is an outlier and will be dropped.", "!")
                self.df = self.df[:-1].reset_index(drop=True)
    
       
    def _load_from_cache(self) -> None:
        data = StarInfoRetriever._cache[StarInfoRetriever.star_to_cache_name(self.star)]
        self.df = pd.DataFrame({
            "sed_filter": list(data.keys()),
            "wavelength_um": [data[filter_name]["wavelength_um"] for filter_name in data.keys()],
            "flux_Jy": [data[filter_name]["flux_Jy"] for filter_name in data.keys()],
        })
        
        Message(f"Photometry for star {cstr(self.star).green()} loaded from cache.", "#")     
            

    def _fit_photometry(self) -> None:
        """
        Fit the photometry loaded by load_photometry() with a spline.
        """
        x = np.log(self.df["wavelength_um"].values)
        y = np.log(self.df["flux_Jy"].values)
        
        # sort them (even though they should already be sorted)
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]

        with Task("Fitting spline to photometry..."):
            spline = UnivariateSpline(x, y, k=min(self.spline_order, len(x)-1), s=0)
        
        # we do not save the spline, we save fluxes on a finer grid instead.
        # then, we will interpolate linearly on that grid when needed.
        self.wl_grid = np.logspace(x.min(), x.max(), 500, base=np.e) * u.micron
        log_flux_grid = spline(np.log(self.wl_grid.to(u.micron).value))
        self.flux_grid = np.exp(log_flux_grid) * u.Jy
        
        
        
    def get_photometry(self, wavlengths: u.Quantity) -> u.Quantity:
        """
        Get the stellar flux at given wavelengths by interpolating the fitted photometry.

        Parameters
        ----------
        wavlengths : u.Quantity
            Wavelengths at which to get the stellar flux.

        Returns
        -------
        u.Quantity
            Stellar flux at the given wavelengths.
        """
        interp_func = interp1d(
            np.log(self.wl_grid.to(u.micron).value),
            np.log(self.flux_grid.to(u.Jy).value),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )
        fluxes = np.exp(interp_func(np.log(wavlengths.to(u.micron).value))) * u.Jy
        return fluxes
    

    # ------------ #
    # !-- Plot --! #
    # ------------ #
    
    def plot(self, show:bool = True, close:bool = True) -> None:
        plt.figure(figsize=(15,8))
        
        # 1. Scatter original data
        if hasattr(self, "_original_df"):
            plt.scatter(
                self._original_df["wavelength_um"].values,
                self._original_df["flux_Jy"].values,
                color="gray",
                label="Original Data",
                alpha=0.5,
                s=50,
                zorder=100
            )
        
        # 2. Scatter cleaned data on top
        plt.scatter(
            self.df["wavelength_um"].values,
            self.df["flux_Jy"].values,
            color="darkblue",
            label="Cleaned Data",
            s=20,
            zorder = 101
        )
        
        # 3. Plot fitted spline below
        plt.plot(
            self.wl_grid.to(u.micron).value,
            self.flux_grid.to(u.Jy).value,
            color="red",
            label="Fitted Spline",
            alpha=0.7,
            zorder=10,
        )
        
        # 4. Add extrapolated regions
        log_extrapolation = np.log10(2) # if extrapolation must be 10 times smaller/larger than the min/max wavelength, put 10 here
        
        x_extrapolated = np.logspace(np.log10(self.wl_grid.min().to(u.micron).value) - log_extrapolation, np.log10(self.wl_grid.min().to(u.micron).value), 100) * u.micron
        y_extrapolated = self.get_photometry(x_extrapolated)
        plt.plot(
            x_extrapolated.to(u.micron).value,
            y_extrapolated.to(u.Jy).value,
            color="red",
            alpha=0.3,
            linestyle="--",
            label="Extrapolation",
            zorder=5,
        )
        x_extrapolated = np.logspace(np.log10(self.wl_grid.max().to(u.micron).value), np.log10(self.wl_grid.max().to(u.micron).value) + log_extrapolation, 100) * u.micron
        y_extrapolated = self.get_photometry(x_extrapolated)
        plt.plot(
            x_extrapolated.to(u.micron).value,
            y_extrapolated.to(u.Jy).value,
            color="red",
            alpha=0.3,
            linestyle="--",
            zorder=5,
        )
        
        
        # 5. Finalize plot
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Wavelength [micron]")
        plt.ylabel("Flux [Jy]")
        plt.title(f"Stellar Photometry for {self.star}")    
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)

        if show:
            plt.show()
        else:
            if close:
                plt.close()
    
    @staticmethod
    def plot_cache() -> None:
        plt.figure(figsize=(15,8))
        
        colors = ["darkblue", "darkgreen", "red", "goldenrod", "purple", "teal", "orange", "magenta", "slategray", "crimson"]
        colors = np.random.permutation(colors)
        
        # sort StarInfoRetriever._cache by max flux descending
        def get_max_flux(data:dict) -> float:
            filters = list(data.keys())
            fluxes = [data[filter_name]["flux_Jy"] for filter_name in filters]
            return max(fluxes)
        
        sorted_cache_items = sorted(
            StarInfoRetriever._cache.items(), # item is star_name, dict
            key=lambda item: get_max_flux(item[1]),
            reverse=True
        )
        
        for i, (star, data) in enumerate(sorted_cache_items):
            wavelengths = np.array([data[filter_name]["wavelength_um"] for filter_name in data.keys()]) * u.um
            fluxes = np.array([data[filter_name]["flux_Jy"] for filter_name in data.keys()]) * u.Jy
            plt.plot(
                wavelengths.to(u.micron).value,
                fluxes.to(u.Jy).value,
                marker="o",
                color=colors[i % len(colors)],
                label=star
            )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Wavelength [micron]")
        plt.ylabel("Flux [Jy]")
        plt.title("Stellar Photometry from Cache")
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)        
        plt.show()
        
        
    # ---------- #
    # !-- IO --! #
    # ---------- #
    
    def add_to_cache(self) -> None:
        data = {
            filter_name: {
                "wavelength_um": float(self.df[self.df["sed_filter"] == filter_name]["wavelength_um"].values[0]),
                "flux_Jy": float(self.df[self.df["sed_filter"] == filter_name]["flux_Jy"].values[0]),
            } for filter_name in self.df["sed_filter"].unique()
        }
        StarInfoRetriever._cache[StarInfoRetriever.star_to_cache_name(self.star)] = data
        StarInfoRetriever.dump_json()
        
    @staticmethod
    def dump_json():
        """
        Dumps `StarInfoRetriever._cache` to a JSON file.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "star_photometry_cache.json")
        with open(cache_path, "w") as f:
            json.dump(StarInfoRetriever._cache, f, indent=3)
    
    @staticmethod
    def load_json():
        """
        Loads `StarInfoRetriever._cache` from a JSON file.
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "star_photometry_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                StarInfoRetriever._cache = json.load(f)
        else:
            StarInfoRetriever._cache = {}
            
    @staticmethod
    def star_to_cache_name(star:str) -> str:
        """
        Converts a star name to a cache-friendly format. Removes spaces and hyphens.
        """
        return star.replace(" ", "").replace("-", "")
    

StarInfoRetriever.load_json()



def get_photometry_jy(star:str, wavelength_um: float, show:bool = False) -> float:
    """
    Convenience function to get the stellar flux (in Jy) at given wavelengths for a given star.

    Parameters
    ----------
    star : str
        Name of the star.
    wavelength : float
        Wavelength(s) at which to get the stellar flux. In microns
    show : bool, optional
        Whether to show the photometry plot, by default False.

    Returns
    -------
    float
        Stellar flux at the given wavelengths. In Jy.
    """

    retriever = StarInfoRetriever(star)
    
    wavelength = wavelength_um * u.um
        
    flux_jy = retriever.get_photometry(wavelength)
    
    if show:
        retriever.plot(show=False, close=False)
        # add the requested wavelength and retrieved flux
        plt.scatter(
            wavelength.to(u.micron).value,
            flux_jy.to(u.Jy).value,
            color="goldenrod",
            marker="+",
            s=100,
            label=f"Requested Point ({wavelength.to(u.micron).value:.2f} um, {flux_jy.to(u.Jy).value:.2f} Jy)",
            zorder=200
        )
        plt.legend()
        plt.show()
    return flux_jy.to(u.Jy).value


def plot_photometry_cache() -> None:
    """
    Convenience function to plot the fluxes of all stars in the photometry cache.
    """
    StarInfoRetriever.plot_cache()
    

def get_distance_pc(star:str) -> float:
    """
    Convenience function to get the distance to a star in parsecs.

    Parameters
    ----------
    star : str
        Name of the star.

    Returns
    -------
    u.Quantity
        Distance to the star in parsecs.
    """
    return StarInfoRetriever.get_star_distance_pc(star).to(u.pc).value



def get_star_aliases(star_name: str) -> list[str]:
    """
    Given a star name, returns a list of other names/identifiers in different catalogs.
    
    Parameters
    ----------
    star_name : str
        Name of the star to query.
        
    Returns
    -------
    list[str]
        List of aliases for the star.
    """
    return StarInfoRetriever.get_star_aliases(star_name)
    
def is_star_alias(star_name: str, alias: str) -> bool:
    """
    Check if `alias` is an alias of `star_name`.

    Parameters
    ----------
    star_name : str
        Name of the star.   
    alias : str
        Potential alias.
        
    Returns
    -------
    bool
        True if `alias` is an alias of `star_name`, False otherwise.
    """
    return StarInfoRetriever.is_alias(star_name, alias)
   

def get_star_name(star_name:str) -> str:
    """
    Takes a string and translates it to try to make it compatible with Simbad.
    Applies `upper()` and handles hypens and underscores.

    Parameters
    ----------
    star_name : str
        Name of the star.

    Returns
    -------
    str
        Parsed star name.
    """
    return StarInfoRetriever.get_star_name(star_name)

from difflib import get_close_matches

def parse_star_name(star_name:str) -> str:
    """
    Takes a string and retrieves all star aliases. Returns the alias closest
    to the original string.
    
    Parameters
    ----------
    star_name : str
        Name of the star to parse.
        
    Returns
    -------
    str
        The closest alias to the original star name. If no aliases are found, returns the original star.
    """
    aliases = get_star_aliases(star_name)
    if len(aliases) == 0:
        Message(f"Unable to parse star name {cstr(star_name):rb}. No aliases found.", "!")
        return star_name
    closest_alias = get_close_matches(star_name, aliases, n=1)
    if len(closest_alias) == 0:
        Message(f"Unable to parse star name {cstr(star_name):rb}. No close alias found among {aliases}.", "!")
        return star_name
    return closest_alias[0]
    
    
    


def get_star_coords(star:str) -> tuple[float, float]:
    """
    Get the RA and Dec of a star in degrees using Simbad.

    Parameters
    ----------
    star : str
        Name of the star. Must be compatible with Simbad database.

    Returns
    -------
    float, float
        RA and Dec of the star in degrees.
    """
    return StarInfoRetriever.get_star_coords(star) 




# ------------------------------ #
# !-- Nasa Exoplanet Archive --! #
# ------------------------------ #

# create a function to retrieve star information from nasa exoplanet database
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

def get_star_info_NAE(star_name):
    
    # 1. Parse the star name, to make sure it is compatible with the database
    star_name = parse_star_name(star_name)
    
    # 2. Query the database for the star information
    for star_name in [star_name, *get_star_aliases(star_name)]:
        table = NasaExoplanetArchive.query_object(star_name, table="pscomppars")
        table = table.to_pandas()
        if len(table) == 0:
            Message(f"Unable to find star information for {cstr(star_name):r}, trying next alias...", "!")
            continue
        return table
    
def get_star_age(star_name, uncertainty=False) -> float | np.ndarray:
    """
    For a star with a known planet (such as stars from transit surveys), 
    retrieves the age of the star from the NASA Exoplanet Archive. If the age is not available, returns None.
    
    The age is returned in Myr.
    
    Parameters:
    -----------
    star_name: str
        The name of the star to query. The function will look for aliases as well in the database.
    uncertainty: bool, optional
        If True, returns a numpy array of (age, age_error_plus, age_error_minus). If False, returns just the age. Default is False.
    
    Returns:
    --------
    float or np.ndarray or None
        The age of the star in Myr, or a numpy array of (age, age_error_plus, age_error_minus) if uncertainty is True, or None if the age is not available.
    """
    table = get_star_info_NAE(star_name)
    if table is None:
        return np.nan if not uncertainty else np.full((3,), np.nan)
    if uncertainty:
        return np.array([table["st_age"].values[0]*1e3, table["st_ageerr1"].values[0]*1e3, table["st_ageerr2"].values[0]*1e3])
    return table["st_age"].values[0]*1e3

def get_star_mass(star_name, uncertainty=False) -> float | np.ndarray:
    """
    For a star with a known planet (such as stars from transit surveys), 
    retrieves the mass of the star from the NASA Exoplanet Archive. If the mass is not available, returns None.
    
    The mass is returned in solar masses.
    
    Parameters:
    -----------
    star_name: str
        The name of the star to query. The function will look for aliases as well in the database.
    uncertainty: bool, optional
        If True, returns a numpy array of (mass, mass_error_plus, mass_error_minus). If False, returns just the mass. Default is False.
    
    Returns:
    --------
    float or np.ndarray or None
        The mass of the star in solar masses, or a numpy array of (mass, mass_error_plus, mass_error_minus) if uncertainty is True, or None if the mass is not available.
    """
    table = get_star_info_NAE(star_name)
    if table is None:
        return np.nan if not uncertainty else np.full((3,), np.nan)
    if uncertainty:
        return np.array([table["st_mass"].values[0], table["st_masserr1"].values[0], table["st_masserr2"].values[0]])
    return table["st_mass"].values[0]



if __name__ == "__main__":
    # plot cached photometries
    StarInfoRetriever.plot_cache()
        
if __name__ == "__main_2_":
    
    star = "LHS 1140"
    with Message(f"Retrieving photometry for star {cstr(star).green()}"):
        phot = get_photometry_jy(star, "F1130W", show=True)
        Message.print(f"Photometry at F1130W: {phot.to(u.uJy):.2e} (expected ~ 1.47e+04 uJy at 11.56 um)")
        phot1156 = get_photometry_jy(star, 11.56 * u.um)
        Message.print(f"Photometry at 11.56 um: {phot1156.to(u.uJy):.2e} (expected ~ 1.47e+04 uJy at 11.56 um)")
    