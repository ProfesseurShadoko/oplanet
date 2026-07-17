

# --------------- #
# !-- Imports --! #
# --------------- #

from oakley import *
import os

import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from typing import Literal

model_folder = os.path.join(os.path.dirname(__file__), "data", "evolutionary_models")
os.makedirs(model_folder, exist_ok=True)

# ------------- #
# !-- Files --! #
# ------------- #

atmo_csv = os.path.join(model_folder, "atmo_2020_grid_oplanet.csv")
hades_csv = os.path.join(model_folder, "hades_2026_grid_oplanet.csv")
linder_csv = os.path.join(model_folder, "linder_2019_grid_oplanet.csv")
sonora_csv = os.path.join(model_folder, "sonora_2021_grid_oplanet.csv")

if not all(os.path.exists(f) for f in [atmo_csv, hades_csv, linder_csv, sonora_csv]):
    Message("Evolutionnary files are missing. Please run `EInversion.download()` or `python -m oplanet.einversion` to download them.", "!")

# -------------------- #
# !-- Interpolator --! #
# -------------------- #

class EInversion:

    def __init__(
        self,
        model: Literal["atmo", "hades", "linder", "sonora"] = "hades",
        scheme: str = "f1140c,age,met->mass"
    ):
        """
        Notes
        -----
        The units are the following:
        - photometry: Jy
        - age: Myr
        - metallicity: dex (log10(Z/Z_sun))
        - mass: MJup
        - radius: RJup
        - temperature: K

        The `scheme` is a string that defines the input parameters and the output of the interpolation.
        The input parameters are seprated by commas, and the output is separated by an arrow (`->`).
        The `scheme` will be converted internally to the column names present in the CSV files.

        Only one output parameter is allowed, but multiple input parameters are allowed. The recommended
        schemes are `filter,age,met -> any` or `any,age,met -> filter`. Having more or less than three input 
        parameters is not recommended, as the grid is basically three dimensional.        
        """
        self.model = model.lower()

        # 1. Load the grid of models
        self.df = pd.read_csv(
            {"atmo": atmo_csv, "hades": hades_csv, "linder": linder_csv, "sonora": sonora_csv}[self.model]
        )
        
        # 2. Parse the scheme string into input and output parameters
        self.inputs, self.output = self.parse_scheme(scheme)

        # 3. Drop zero dimensional columns
        self._init_inputs = [i for i in self.inputs]
        for col in self.inputs:
            if self.df[col].nunique() == 1:
                self.inputs.remove(col)
        
        # 5. Drop unused columns
        columns_to_keep = self.inputs + [self.output] + [
            col + "_upper" for col in self.columns
            if col + "_upper" in self.df.columns
        ] + [
            col + "_lower" for col in self.columns
            if col + "_lower" in self.df.columns
        ]
        self.df = self.df[columns_to_keep].dropna()
        
        # 4. Create the interpolators
        self.fit()


    def __str__(self):
        return f"EInversion({cstr(self.model.upper()):r}:{cstr(self.make_scheme(self.inputs, self.output)):y})"

    def __repr__(self):
        return f"EInversion({self.model.upper()} - {self.output})>"
    
    @property
    def columns(self):
        """
        Returns the list of columns used in the interpolation.
        """
        return self.inputs + [self.output]


    # ---------------- #
    # !-- Training --! #
    # ---------------- #

    def fit(self):
        """
        Pretrains the interpolator(s).
        """
        # 1. Go the latent space
        X, y = self.transform(self.df)

        # 2. Fit the interpolator
        self.median_model = LinearNDInterpolator(X, y, fill_value=np.nan)

        # 3. Create upper and lower models for uncertainty estimation
        upper_df = self.df.copy(deep=True)
        lower_df = self.df.copy(deep=True)
        # rename clumns that have an _upper or _lower suffix to the original column name
        for col in self.columns:
            if col+"_upper" in self.df.columns:
                upper_df[col] = self.df[col+"_upper"]
            if col+"_lower" in self.df.columns:
                lower_df[col] = self.df[col+"_lower"]
        upper_df = upper_df[self.columns].dropna()
        lower_df = lower_df[self.columns].dropna()
        self.upper_model = LinearNDInterpolator(*self.transform(upper_df), fill_value=np.nan)
        self.lower_model = LinearNDInterpolator(*self.transform(lower_df), fill_value=np.nan)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Do not use directly. This function is called by the `__call__` method.
        """
        assert hasattr(self, "median_model"), "You need to fit the model before using it!"
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame."

        # 1. Go to latent space and make the prediction
        X, _ = self.transform(df)
        y_median = self.median_model(X)
        y_upper = self.upper_model(X)
        y_lower = self.lower_model(X)

        # 2. Return to physical space
        y_median, y_upper, y_lower = self.inverse(y_median), self.inverse(y_upper), self.inverse(y_lower)

        # 3. Sort the quantiles to avoid crossings
        y_lower, y_median, y_upper = np.sort([y_lower, y_median, y_upper], axis=0)

        return y_median, y_lower, y_upper
    
    def __call__(self, *inputs:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        *inputs : np.ndarray | float
            Input parameters in the order defined by the scheme.
        
        Returns
        -------
        y_median : np.ndarray
            Median prediction of the output parameter.
        sigma_upper : np.ndarray
            Upper prediction of the output parameter (84th percentile - median).
        sigma_lower : np.ndarray
            Lower prediction of the output parameter (16th percentile - median), hence is negative.
        
        Uncertainties are only available in the HADES grid. Otherwise,
        the lower and upper predictions will be zero.
        """
        # 1. Check that the number of inputs is correct
        assert len(inputs) == len(self._init_inputs), f"Number of inputs must be {len(self._init_inputs)}, got {len(inputs)}."
        scalar_output = all(np.ndim(inp) == 0 for inp in inputs)

        # 2. Create a dataframe from the inputs
        inputs = [np.atleast_1d(inp) for inp in inputs]
        # broadcast all shapes together
        inputs = np.broadcast_arrays(*inputs)
        output_shape = inputs[0].shape
        inputs = [inp.flatten() for inp in inputs]
        
        df = pd.DataFrame({col: inp for col, inp in zip(self._init_inputs, inputs)})

        # 3. Predict the output parameter
        y_median, y_lower, y_upper = self.predict(df)
        y_median = y_median.reshape(output_shape)
        y_lower = y_lower.reshape(output_shape)
        y_upper = y_upper.reshape(output_shape)

        # 4. Return the median and the uncertainties
        if scalar_output:
            return y_median.item(), y_upper.item() - y_median.item(), y_lower.item() - y_median.item()
        else:
            return y_median, y_upper - y_median, y_lower - y_median
    
    @staticmethod
    def sample(
        median: np.ndarray,
        sigma_upper: np.ndarray,
        sigma_lower: np.ndarray,
        distribution: Literal["normal", "laplace", "uniform"] = "laplace"
    ) -> np.ndarray:
        """
        Returns an altered copy of the input median array, where each element is randomly sampled from a distribution defined by the corresponding upper and lower uncertainties.

        Parameters
        ----------
        median : np.ndarray
            Array of median values.
        sigma_upper : np.ndarray
            Array of upper uncertainties (84th percentile - median), positive.
        sigma_lower : np.ndarray
            Array of lower uncertainties (16th percentile - median), negative.

        Returns
        -------
        np.ndarray
            Array of sampled values. Take its median and quantiles to get the median and uncertainties of the sampled distribution.
        """
        assert distribution in ["normal", "laplace", "uniform"], f"Distribution must be one of ['normal', 'laplace', 'uniform'], got {distribution}."
        if distribution == "normal":
            return EInversion.split_gaussian(median, median + sigma_lower, median + sigma_upper)
        elif distribution == "laplace":
            return EInversion.split_laplace(median, median + sigma_lower, median + sigma_upper)
        elif distribution == "uniform":
            return np.random.uniform(median + sigma_lower, median + sigma_upper)



    # -------------------- #
    # !-- Latent Space --! #
    # -------------------- #
        
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms the input parameters into a latent space
        (log space for everyone except metallicity) for interpolations.
    
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with columns containing the input and output parameters.
            The column names must match the scheme.
        
        Returns
        -------
        inputs : np.ndarray
            Array of shape (n_samples, n_inputs) containing the input parameters in latent space.
        output : np.ndarray
            Array of shape (n_samples,) containing the output parameter in latent space.
        """
        # 1. Load columns from the dataframe
        inputs = [
            df[col].to_numpy() for col in self.inputs
        ]
        output = df[self.output].to_numpy() if self.output in df.columns else np.full(len(df), np.nan)

        for i, col in enumerate(self.inputs):
            if col != "metallicity_solar_dex":
                inputs[i] = np.log(inputs[i])
        output = np.log(output) if self.output != "metallicity_solar_dex" else output

        return np.column_stack(inputs), output
    
    def inverse(self, outputs: np.ndarray) -> np.ndarray:
        """
        Inverse transforms the output parameter from latent space
        back to the original space.
        """
        return np.exp(outputs) if self.output != "metallicity_solar_dex" else outputs


    
    
    # -------------- #
    # !-- Scheme --! #
    # -------------- #
    
    @staticmethod
    def make_scheme(
        inputs: list[str], output: str
    ):
        """
        Make a scheme string from the input and output parameters.
        """
        return ",".join(inputs) + " => " + output

    
    def parse_scheme(
        self, scheme: str,
    ):
        """
        Parse a scheme string into input and output parameters.

        Returns
        -------
        inputs : list[str]
            List of input parameters (column names matching csv file).
        output : str
            Output parameter (column name matching csv file).
        """
        # 0. Find the arrow in the scheme string
        arrow_candidates = []
        # find caracter >
        for i, char in enumerate(scheme):
            if char == ">":
                # find the start of the arrow (look for = or - or > before the >)
                start = i
                while start > 0 and scheme[start - 1] in ["=", "-", ">"]:
                    start -= 1
                end = i
                arrow_candidates.append((start, end))
        assert len(arrow_candidates) == 1, f"Scheme string must contain exactly one arrow, found {len(arrow_candidates)}: {[scheme[start:end+1] for start, end in arrow_candidates]}"
        arrow = scheme[arrow_candidates[0][0]:arrow_candidates[0][1] + 1]

        # 1. Extract from the scheme string the input and output parameters
        inputs, output = scheme.split(arrow)
        inputs = inputs.split(",")

        # 2. Clean the input and output parameters
        inputs = [i.strip().lower() for i in inputs]
        output = output.strip().lower()

        # 3. Make inputs and outputs to column names
        colnames2aliases = {
            "t_int_k": ["t", "t_int", "tint", "t_eff", "teff", "temp", "temperature", "t_int_k", "t_eff_k", "teff_k", "temp_k", "temperature_k"],
            "age_myr": ["age", "age_myr", "a"],
            "metallicity_solar_dex": ["met", "metallicity", "metallicity_solar_dex", "met_dex", "z"],
            "req_rjup": ["req", "radius", "radius_eq", "radius_eq_rjup", "req_rjup", "r"],
            "mass_mjup": ["mass", "mass_mjup", "m"],
        } # + all filters

        def get_colname(alias: str):
            # 1. Check wether it is a standard column name
            for colname, aliases in colnames2aliases.items():
                if alias in aliases:
                    return colname
            # 2. Check wether it is a filter
            filter_alias = alias.split(".")[-1]
            filter_column = f"{filter_alias.lower().replace(' ', '')}_jy_10pc"
            if filter_column in self.df.columns:
                return filter_column
            with Message(f"Alias '{alias}' not found in column names.", "!").tab():
                Message("Column name to alias mapping:").list(colnames2aliases)
                Message("Available filters:").list([c.replace("_jy_10pc", "") for c in self.df.columns if c.endswith("_jy_10pc")])
            raise ValueError(f"Alias '{alias}' not found in column names.")
    
        inputs = [get_colname(i) for i in inputs]
        output = get_colname(output)

        # check that all columns are in csv
        assert all(i in self.df.columns for i in inputs), f"Some input columns are not in the CSV file: {inputs}"
        assert output in self.df.columns, f"Output column '{output}' is not in the CSV file: {self.df.columns}"

        # remove duplicates
        inputs = list(dict.fromkeys(inputs))

        return inputs, output


    # ---------------- #
    # !-- Sampling --! #
    # ---------------- #

    @staticmethod
    def split_laplace(
        mean:float | np.ndarray | pd.Series,
        q16:float | np.ndarray | pd.Series,
        q84:float | np.ndarray | pd.Series
    ) -> np.ndarray:
        """
        Samples from a split Laplace distribution given the mean and the 16th
        and 84th percentiles. This is useful for sampling from the predicted
        uncertainties of the model.

        Parameters
        ----------
        mean : float
            The mean of the distribution (the median prediction).
        q16 : float
            The 16th percentile of the distribution (lower uncertainty).
        q84 : float
            The 84th percentile of the distribution (upper uncertainty).

        Returns
        -------
        float
            A random sample from the split Laplace distribution.
        """

        # 1. Convert q16 and q84 to b parameters
        b_lower = (mean - q16) / -np.log(0.32)
        b_upper = (q84 - mean) / -np.log(0.32)
        p_left = b_lower / (b_lower + b_upper)

        # 2. Sample from uniform distribution
        u = np.random.uniform(0, 1, size=np.shape(mean))

        # 3. Sample from split Laplace
        sample = np.where(
            u < p_left,
            mean + b_lower * np.log(u / p_left),
            mean - b_upper * np.log((1 - u) / (1 - p_left))
        )
        return sample
    
    @staticmethod
    def split_gaussian(
        mean:float | np.ndarray | pd.Series,
        q16:float | np.ndarray | pd.Series,
        q84:float | np.ndarray | pd.Series
    ) -> np.ndarray:
        """
        Samples from a split Gaussian distribution given the mean and the 16th
        and 84th percentiles. This is useful for sampling from inputs of the model.

        Parameters
        ----------
        mean : float
            The mean of the distribution (the median prediction).
        q16 : float
            The 16th percentile of the distribution (lower uncertainty).
        q84 : float
            The 84th percentile of the distribution (upper uncertainty).

        Returns
        -------
        float
            A random sample from the split Gaussian distribution.
        """

        # 1. Convert q16 and q84 to sigma parameters
        sigma_lower = (mean - q16) / 1.0
        sigma_upper = (q84 - mean) / 1.0
        p_left = sigma_lower / (sigma_lower + sigma_upper)

        # 2. Sample from uniform distribution
        u = np.random.uniform(0, 1, size=np.shape(mean))
        z = np.abs(np.random.normal(0, 1, size=np.shape(mean)))
        sample = np.where(
            u < p_left,
            mean - z * sigma_lower,
            mean + z * sigma_upper
        )
        return sample


    # ---------------- #
    # !-- Download --! #
    # ---------------- #

    @staticmethod
    def download():
        """
        Downloads (and overwrites) the local evolutionnary models, so that they can be used.
        """
        from huggingface_hub import HfApi, hf_hub_download
        import time
        REPOSITORY_ID = "ProfShadoko/mirix"
        api = HfApi()

        # 1. List all files in the repository (there is a bunch of stuff)
        files = api.list_repo_files(REPOSITORY_ID, repo_type="dataset")
        files = [f for f in files if f.endswith("grid_oplanet.csv")]

        Message("Files to download:").list(files)

        # 2. Download them
        with Task("Downloading files..."):
            for f in files:
                time.sleep(1)
                hf_hub_download(
                    repo_id = REPOSITORY_ID,
                    repo_type = "dataset",
                    filename = f,
                    local_dir = model_folder,
                    force_download = True
                )






if __name__ == "__main__":

    Message.title("Download")
    EInversion.download()

    Message.title("Test")
    for model in ["hades", "atmo", "linder", "sonora"]:
        xi = EInversion(model=model, scheme="m,a,z -> f1500w")
        print(xi)