
from oakley import *
from astroquery.svo_fps import SvoFps
from astropy import units as u
from astropy import constants as const
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt


# ------------------- #
# !-- Cache Setup --! #
# ------------------- #

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, "data")
cache_path = os.path.join(cache_dir, "svo_filters_cache.json")
os.makedirs(cache_dir, exist_ok=True)
import json
if not os.path.exists(cache_path):
    with open(cache_path, "w") as f:
        json.dump({}, f)


# -------------- #
# !-- Filter --! #
# -------------- #


class SFilter:
    """
    Class representing a filter transmission profile (data retrieved from SVO database).
    Retrieved profiles are cached locally, to avoid issues when the SVO database is down.
    """

    def __init__(
        self,
        filter_name:str,
        facility:str = None,
        instrument:str = None,
    ):
        """
        Initialize an SFilter instance. Loads the data from all filters for the specified facility (e.g. JWST).
        This data is cached, so subsequent calls won't require a connection to the SVO database.

        Parameters
        ----------
        filter_name : str
            Name of the filter (e.g. "F1140C"). For some facilities, the filter name is not unique (e.g. GAIA).
            In this case, the filter name is not sufficient to identify a unique filter, and the instrument must be specified.
        facility : str
            Facility for the filter (e.g. "JWST"). All filters from this facility will be loaded and cached (even when a specific
            instrument is specified). Defaults to None (in which case the filter name must be unique across all facilities, and already loaded).
        instrument : str, optional
            Instrument for the filter (e.g. "GAIA3"). This is only necessary when the filter name is not unique within the specified facility (e.g. GAIA).
        
        Notes
        -----
        Be carefull with the filter profiles retrieved from SVO for JWST/MIRI. The scaling
        of the profiles is incorrect and inconsistent with the JWST documentation. However the
        overall shape of the profiles is correct, so anything related to the normalized transmission
        (e.g. photomatry) should be unaffected. For anything related to the absolute transmission
        (e.g. photon count), use data from `pandeia` instead.
        """
        self._filter_name = filter_name
        self._facility = facility
        self._instrument = instrument

        self.load()
    
    # ----------- #
    # !-- API --! #
    # ----------- #

    @staticmethod
    def _custom_json_dump(obj, indent=None):
        """
        Custom json dump function. Add idnent to dict only, 
        not lists (e.g. for wl and tr) to avoid creating huge files.
        """
        space = " " * indent
        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                items.append(f'\n{space}"{k}": {SFilter._custom_json_dump(v, indent+4)}')
            return "{" + ",".join(items) + f"\n{space}" + "}"
        elif isinstance(obj, list):
            return "[" + ", ".join(SFilter._custom_json_dump(x, 0) for x in obj) + "]"
        else:
            return json.dumps(obj)

    def load(self, _recursion:bool = True) -> None:
        """
        Load the filter transmission profile from cache, or SVO
        database if not cached.        
        """

        # 1. Check cache
        cache = json.load(open(cache_path, "r"))
        # loop over cached data to find the filter name with specified facility (and instrument, if specified)
        matching_data = []
        for id, filter_info in cache.items():
            filter = filter_info["filter"]
            if filter.lower() != self._filter_name.lower():
                continue

            if self._facility is not None:
                facility = filter_info["facility"]
                if facility.lower() != self._facility.lower():
                    continue
            
            if self._instrument is not None:
                instrument = filter_info["instrument"]
                if instrument.lower() != self._instrument.lower():
                    continue
            
            matching_data.append(id)
        
        if len(matching_data) > 1:
            raise ValueError(f"Multiple filters found in cache matching the specified criteria (name={self._filter_name}, facility={self._facility}, instrument={self._instrument}). Please specify more precise criteria. Matching filters: {matching_data}")
        elif len(matching_data) == 1:
            self._filer_info = cache[matching_data[0]]
            return
        if not _recursion:
            raise ValueError(f"Filter not found in SVO. Please check that the filter exists in the SVO database and that the specified criteria are correct (name={self._filter_name}, facility={self._facility}, instrument={self._instrument}).")
            # this means we already downloaded what was necessary, but filter was not found
        
        self._build_dictionnaries_from_svo()
        self.load(_recursion=False) # build saved to cache, now we can load again!
        


    def _build_dictionnaries_from_svo(self):
        """
        For every retrieved filter from SVO, build a dictionnary containing all relevant information
        and cache it.
        """
        # 1. Check that facility is specified (we need it to retrieve the filters from SVO)
        assert self._facility is not None, "Facility must be specified to retrieve filter data from SVO database (e.g. 'JWST'). Provided filter name is not known yet."

        # 2. Retrieve filter IDs from SVO
        try:
            fps = SvoFps()
            filter_list = fps.get_filter_list(self._facility).to_pandas()
        except Exception as e:
            raise ConnectionError(f"Failed to retrieve filter data from SVO database. Error: {e}")
        assert len(filter_list) > 0, f"No filter found for specified criteria (facility={self._facility}). Please check that the specified facility and instrument are correct and that filters exist for them in the SVO database."

        # 3. Loop over retrieved filters, build dictionnary for each filter, and cache it
        cache = json.load(open(cache_path, "r"))

        Message(f"Found {len(filter_list)} filters for specified criteria (facility={self._facility}). Building dictionnaries and caching them locally...")

        with Task(f"Loading transmission curves for {self._facility}..."):
            for i, filter_info in ProgressBar(filter_list.iterrows(), size=len(filter_list)):

                # 1. Load filter information
                filter_id = filter_info["filterID"]
                instrument = filter_id.split("/")[1].split(".")[0]
                facility = filter_id.split("/")[0]
                filter = filter_id.split(".")[-1]
                wavelength_ref = filter_info["WavelengthRef"] * 1e-10 # A to meter
                wavelength_mean = filter_info["WavelengthMean"] * 1e-10
                wavelength_eff = filter_info["WavelengthEff"] * 1e-10
                wavelength_min = filter_info["WavelengthMin"] * 1e-10
                wavelength_max = filter_info["WavelengthMax"] * 1e-10
                effective_bandwidth = filter_info["WidthEff"] * 1e-10
                wavelength_central = filter_info["WavelengthCen"] * 1e-10
                wavelength_pivot = filter_info["WavelengthPivot"] * 1e-10
                wavelength_peak = filter_info["WavelengthPeak"] * 1e-10
                wavelength_phot = filter_info["WavelengthPhot"] * 1e-10
                fwhm = filter_info["FWHM"] * 1e-10
                mag_sys = filter_info["MagSys"]
                zero_point_jy = filter_info["ZeroPoint"]
                detector_type = "photon_counter" if filter_info["DetectorType"] == "0" else "energy_counter"

                # 2. Load transmission data
                table = fps.get_transmission_data(filter_id).to_pandas()
                wl = table["Wavelength"].values * 1e-10 # A to meter
                tr = table["Transmission"].values

                # 3. Build dictionnary and cache it
                filter_dict = {
                    "filter_id": filter_id,
                    "filter": filter,
                    "facility": facility,
                    "instrument": instrument,
                    "wavelength_ref": wavelength_ref, # same as pivot, don't care
                    "wavelength_mean": wavelength_mean,
                    "wavelength_eff": wavelength_eff,
                    "wavelength_min": wavelength_min,
                    "wavelength_max": wavelength_max,
                    "effective_bandwidth": effective_bandwidth,
                    "wavelength_central": wavelength_central,
                    "wavelength_pivot": wavelength_pivot,
                    "wavelength_peak": wavelength_peak,
                    "wavelength_phot": wavelength_phot,
                    "fwhm": fwhm,
                    "mag_sys": mag_sys,
                    "zero_point_jy": zero_point_jy,
                    "wl": list(wl),
                    "tr": list(tr),
                    "detector_type": detector_type
                }
                cache[filter_id] = filter_dict

        with open(cache_path, "w") as f:
            f.write(
                self._custom_json_dump(cache, indent=4)
            )

    @staticmethod
    def reset() -> None:
        """
        Reset the cache by replacing the file by an empty dictionnary.
        """
        with open(cache_path, "w") as f:
            json.dump({}, f)
 




    @property
    def name(self) -> str:
        """
        Name of the filter (e.g. "F1140C").
        """
        return self._filer_info["filter"]

    @property
    def id(self) -> str:
        """
        Unique SVO identifier for the filter (e.g. "JWST/MIRI.F1500W").
        """
        return self._filer_info["filter_id"]

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self):
        return f"SFilter(name={self.name}, id={self.id})"
    
    @staticmethod
    def from_id(filter_id:str) -> "SFilter":
        """
        Create an SFilter instance from a unique SVO identifier (e.g. "JWST/MIRI.F1500W").
        The code will effectively retrieve the filter name, facility and instrument from 
        the string, load every single filter from the specified facility, and then find
        the one matching the specified filter ID.

        This is equivalent to directly creating an SFilter instance with the filter name, facility and instrument.
        """
        filter_name = filter_id.split(".")[-1]
        facility = filter_id.split("/")[0]
        instrument = filter_id.split("/")[1].split(".")[0]
        return SFilter(filter_name=filter_name, facility=facility, instrument=instrument)


    # ------------------- #
    # !-- Wavelengths --! #
    # ------------------- #

    @property
    def wl_mean(self) -> float:
        """
        Mean wavelength of the filter (in meters). This is the wavelength that corresponds
        to a weighted average of the transmission curve.
        """
        return self._filer_info["wavelength_mean"]

    @property
    def wl_pivot(self) -> float:
        """
        Pivot wavelength of the filter (in meters). This is the wavelength to use when 
        converting between flux density per unit wavelength and flux density per unit frequency.
        """
        return self._filer_info["wavelength_pivot"]

    @property
    def wl_eff(self) -> float:
        """
        Effective wavelength of the filter (in meters). This is the wavelength that corresponds
        to a weighted average of the transmission curve mulitplied by an effective stellar spectrum.
        This is physically closer to the real wavelength of a filter.
        """
        return self._filer_info["wavelength_eff"]
    
    @property
    def normalized_bandwidth(self) -> float:
        """
        The normalized bandwidth (in meters).
        """

    @property
    def bandwidth(self) -> float:
        """
        The bandwidth (in meters)
        """
    

    @property
    def fwhm(self) -> float:
        """
        Full width at half maximum (FWHM) of the filter (in meters).
        """
        return self._filer_info["fwhm"]
    
    @property
    def wl_min(self) -> float:
        """
        Minimum wavelength (reached when transmission is 1% of maximum transmission) of the filter (in meters).
        """
        return self._filer_info["wavelength_min"]

    @property
    def wl_max(self) -> float:
        """
        Maximum wavelength (reached when transmission is 1% of maximum transmission) of the filter (in meters).
        """
        return self._filer_info["wavelength_max"]
    
    @property
    def wl_central(self) -> float:
        """
        Central wavelength of the filter (in meters). This is the (unweighted)
        middle of the FWHM of the filter. This is physically not very meaningfull.
        """
        return self._filer_info["wavelength_central"]
    
    @property
    def wl_peak(self) -> float:
        """
        Peak wavelength of the filter (in meters). This is the wavelength at which the transmission is maximum.
        """
        return self._filer_info["wavelength_peak"]
    
    @property
    def detector_type(self) -> Literal["photon_counter", "energy_counter"]:
        """
        Type of detector associated with the filter. This is important for photometry calculations.
        It can be either "photon_counter" or "energy_counter".
        """
        return self._filer_info["detector_type"]


    # -------------------- #
    # !-- Transmission --! #
    # -------------------- #

    @property
    def wl(self) -> np.ndarray:
        """
        Wavelengths of the transmission curve (in meters).
        """
        return np.array(self._filer_info["wl"])
    
    @property
    def tr(self) -> np.ndarray:
        """
        Transmission values of the transmission curve (between 0 and 1).
        For a given wavelength, the transmission value represents the fraction
        of incoming light at that wavelength that is transmitted through the filter.
        """
        return np.array(self._filer_info["tr"])


    # ------------------ #
    # !-- Photometry --! #
    # ------------------ #

    def photometry(self, wavelengths: np.ndarray, flux: np.ndarray, flux_type:Literal["lambda", "nu"]) -> float:
        """
        Computes the photometry of a given spectrum through this filter.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelengths of the spectrum (in meters).
        flux : np.ndarray
            Flux values of the spectrum. In any desired unit.
        flux_type : Literal["lambda", "nu"]
            Type of the flux values. "lambda" if the flux is given in units of flux density per unit wavelength (e.g. W/m^2/m), "nu" if the flux
            is given in units of flux density per unit frequency (e.g. W/m^2/Hz). This is important for the photometry calculation, as it determines
            how the flux values are weighted by the filter transmission curve.
            Returned flux is in same units as input flux.
        Notes
        -----
        The photometry is computed in two different ways, depending on the
        type of detector associated with the filter (photon counter or energy counter).
        For energy counters, photometry is computed as:

        ```python
        photometry = np.trapz(flux * self.tr, self.wl) / np.trapz(self.tr, self.wl)
        ```

        This is simply a weighted average of the flux. For photon counters, photometry
        is computed as:

        ```python
        photometry = np.trapz(flux * self.tr * self.wl, self.wl) / np.trapz(self.tr * self.wl, self.wl)
        ```

        This is again a weighted average of the flux, but with an additional weighting by wavelength.
        
        Indeed, for energy counters, the total energy recieved is just `E = time * flux * filter_transmission`.
        For photon counters, it is rather `N = time * flux/E_photon * filter_transmission`, where `E_photon = h * c / wavelength`.

        The type of detector is specified in the SVO database.
        """

        if flux_type == "nu":
            # convert to flux per unit of wavelength
            flux = flux / wavelengths**2 # don't care about the constant

        # let's interpolate the filter profile to the input wavelengths
        tr_interpolated = np.interp(wavelengths, self.wl, self.tr, left=0, right=0)
        
        match self.detector_type:
            case "energy_counter":
                photometry = np.trapezoid(flux * tr_interpolated, wavelengths) / np.trapezoid(tr_interpolated, wavelengths)
            case "photon_counter":
                photometry = np.trapezoid(flux * tr_interpolated * wavelengths, wavelengths) / np.trapezoid(tr_interpolated * wavelengths, wavelengths)
            case _:
                raise ValueError(f"Unknown detector type: {self.detector_type}. This should not happen. Please check the SVO database for the filter {self.id}.")
        
        if flux_type == "nu":
            #convert back to flux per unit of frequency
            photometry = photometry * self.wl_pivot**2 # here the omitted c before cancels out
        
        return photometry


    # ------------- #
    # !-- Plots --! #
    # ------------- #

    def display(self) -> None:
        """
        Display basic information about the filter.
        """
        Message(f"Filter {cstr(self.name):y} properties:").list({
            "ID": self.id,
            "Facility": self._filer_info["facility"],
            "Instrument": self._filer_info["instrument"],
            "Mean wavelength (m)": f"{self.wl_mean:.4e}",
            "Pivot wavelength (m)": f"{self.wl_pivot:.4e}",
            "Effective wavelength (m)": f"{self.wl_eff:.4e}",
            "Detector type": self.detector_type
        })

    def plot(self) -> None:
        """
        Show a plot of the transmission curve of the filter.
        """
        plt.figure(figsize=(6,4))
        plt.plot(self.wl, self.tr)
        plt.xlabel("Wavelength (m)")
        plt.ylabel("Transmission")
        plt.title(f"Transmission curve of {self.name} filter")
        plt.grid()
        plt.show()

    @staticmethod
    def plot_all(filter_ids:list[str] | str) -> None:
        """
        Show a plot with the transmission curves of multiple filters.

        Parameters
        ----------
        filter_ids : list of str or str
            List of filter IDs (e.g. ["JWST/MIRI.F1500W", "JWST/MIRI.F1140C"]) or instrument name
            (in which case all filters are plotted).
        """
        if isinstance(filter_ids, str):
            filters = SFilter.get_filters(filter_ids)
            filter_ids = [filter.id for filter in filters]
        
        plt.figure(figsize=(15,6)) # very horizontal
        plt.title("Transmission curves of filters")

        # 1. load all filters
        filters = [SFilter.from_id(filter_id) for filter_id in filter_ids]

        # 2. Find minimum and maximum effective wavelengths
        wl_min = min(filter.wl_eff for filter in filters)
        wl_max = max(filter.wl_eff for filter in filters)

        # 3. Create a violet to red colormap (spanning spectral colors)
        colormap = plt.get_cmap("rainbow")
        colors = [colormap((filter.wl_eff - wl_min) / (wl_max - wl_min)) for filter in filters]

        # 4. Plot each filter with corresponding color
        for filter, color in zip(filters, colors):
            plt.plot(filter.wl, filter.tr, label=filter.name, color=color)
            # fill between as well, with lower alpha
            plt.fill_between(filter.wl, filter.tr, alpha=0.3, color=color)
        
        plt.xlabel("Wavelength (m)")
        plt.ylabel("Transmission")
        plt.legend()
        plt.grid()
        plt.ylim(0, None)
        plt.show()

    @staticmethod
    def get_filters(
        facility:str,
        instrument:str = None,
    ) -> list["SFilter"]:
        """
        Get a list of SFilter instances for all filters matching the specified criteria.

        Parameters
        ----------
        facility : str
            Facility for the filters (e.g. "JWST"). All filters from this facility will be loaded and cached (even when a specific
            instrument is specified).
        instrument : str, optional
            Wether to only plot filters for a given instrument. If not specified, filters from all instruments will be plotted.
        """

        # 1. Check wether data is in cache, otherwise load
        cache = json.load(open(cache_path, "r"))
        for filter_id, filter_info in cache.items():
            if filter_info["facility"].lower() == facility.lower():
                break
        else:
            # Load the data for the facility in cache
            try:
                SFilter(filter_name="NONE", facility=facility)
            except ValueError:
                # this is expected as filter wont be found, but data will be loaded.
                pass

        # 2. Loop over cache!
        cache = json.load(open(cache_path, "r"))
        filters = []
        for id, filter_info in cache.items():
            filter_rfacility = filter_info["facility"]
            filter_instrument = filter_info["instrument"]

            if facility.lower() != filter_rfacility.lower():
                continue
            
            if instrument is not None and instrument.lower() != filter_instrument.lower():
                continue
            
            filters.append(id)
        
        return [SFilter.from_id(filter_id) for filter_id in filters]
        
    @staticmethod
    def download(facility:str|list[str]) -> None:
        """
        Download all filters for a given facility and cache them. 

        Parameters
        ----------
        facility : str or list of str
            Facility (or facilities) for which to download the filters (e.g. "JWST"). All filters from this facility will be loaded and cached.
            If a list of facilities is provided, filters for all specified facilities will be downloaded and cached.
        """
        if isinstance(facility, str):
            facility = [facility]
        for f in facility:
            try:
                SFilter(filter_name="NONE", facility=f)
            except ValueError:
                # this is expected as filter wont be found, but data will be loaded.
                pass

    @staticmethod
    def is_cached(facility:str) -> bool:
        """
        Check if filters for a given facility are already cached.

        Parameters
        ----------
        facility : str
            Facility for which to check the filters (e.g. "JWST").
        
        Returns
        -------
        bool
            True if filters for the specified facility are already cached, False otherwise.
        """
        cache = json.load(open(cache_path, "r"))
        for filter_id, filter_info in cache.items():
            if filter_info["facility"].lower() == facility.lower():
                return True
        return False

    
if __name__ == "__main__":
    #filter = SFilter("F1500W", "JWST")
    #filter.display()
    #filter = SFilter.from_id("GAIA/GAIA2.G")
    #filter.display()
    #filter = SFilter("F1140C") # should be already downlaoded
    #filter.display()
    

    miri_filters = SFilter.get_filters("JWST", "MIRI")
    SFilter.plot_all([filter.id for filter in miri_filters])
    SFilter.download(["2MASS", "WISE", "GAIA"])
    SFilter.plot_all("WISE")