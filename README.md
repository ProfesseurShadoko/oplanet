# oplanet

Lightweight Python wrapper around the NASA Exoplanet Archive with practical helpers for stellar aliases, photometry, and system/planet properties.

This package is built to make exoplanet metadata easier to use in notebooks and scripts by exposing a simple object API and automatically selecting the most reliable published values.

## What it is

`oplanet` wraps data from:

- NASA Exoplanet Archive (`ps` table)
- Simbad / Vizier utilities for star aliases, coordinates, and photometry

For parameters with multiple entries in the archive, it chooses the best measurement by preferring rows with the smallest uncertainty bars (when available), with fallback to limits when no direct value exists.

## Installation

### From GitHub (recommended)

```bash
pip install "git+https://github.com/ProfesseurShadoko/oplanet.git"
```

### From source

```bash
git clone https://github.com/ProfesseurShadoko/oplanet.git
cd oplanet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

Dependencies are listed in `requirements.txt`:

- `oakley`
- `numpy`
- `pandas`
- `astroquery`
- `astropy`
- `scipy`
- `matplotlib`

## Quick examples

### 1. Resolve star names and aliases

```python
from oplanet import parse_star_name, get_star_aliases

print(parse_star_name("TOI 1478"))
print(get_star_aliases("TOI 1478"))
```

### 2. Explore a system, its star, and its planets

```python
from oplanet.osystem import OSystem

system = OSystem("LHS 1140")

print(system.star_name)
print(system.n_planets)

# Returns [value, err1, err2] when available,
# or [nan, upper_limit, lower_limit] when only limits exist
print(system.distance_pc)
print(system.star.age_myr)

print(system.b.mass_mjup)
print(system.b.orbital_period_yrs)
```

### 3. Get stellar photometry at a wavelength

```python
from oplanet import get_photometry_jy

flux_jy = get_photometry_jy("LHS 1140", 11.56)
print(flux_jy)
```

## Data behavior

On import, the loader keeps a local CSV cache in `oplanet/data`, removes older archive snapshots, and refreshes stale files automatically.

## Notes

- Internet access is needed for Simbad/Vizier queries and first-time archive download.
- This repository is currently source-first (requirements-driven), not a published PyPI package.

## License

MIT
