# oplanet

Lightweight Python wrapper around the NASA Exoplanet Archive and the SVO database with practical helpers for stellar aliases, photometry, and system/planet properties.

This package is built to make exoplanet metadata easier to use in notebooks and scripts by exposing a simple object API and automatically selecting the most reliable published values.

## Installation

### From pip (recommended)

```bash
pip install oplanet
```

### From GitHub 

```bash
pip install "git+https://github.com/ProfesseurShadoko/oplanet.git"
```

## Quick examples

For a guided walkthrough with explanations and runnable cells, see [examples.ipynb](examples.ipynb).

### 1. Resolve star names and aliases

```python
from oplanet import parse_star_name, get_star_aliases

print(parse_star_name("TOI 1478"))
print(get_star_aliases("TOI 1478"))
```

### 2. Explore a system, its star, and its planets

```python
from oplanet import NSystem

system = NSystem("LHS 1140")

print(system.star_name)
print(system.n_planets)

# Returns [value, err1, err2] when available,
# or [nan, upper_limit, lower_limit] when only limits exist
print(system.distance_pc)
print(system.star.age_myr)

print(system.b.mass_mjup)
print(system.b.orbital_period_yrs)
```

### 3. One-line import for common API

```python
from oplanet import NSystem, get_star_aliases, get_photometry_jy, SFilter
```



### 4. Pretty-print helpers (`display` and `print_column`)

```python
from oplanet import NSystem

system = NSystem("LHS 1140")

# Human-readable summary of system properties
system.display()

# Human-readable summary for star and a planet
system.star.display()
system.b.display()
```

## Returned values

Most numeric property getters in the object API return a NumPy array with 3 entries:

- `value`: best selected value
- `err_pos`: positive uncertainty
- `err_neg`: negative uncertainty

When no direct value is available and only limits are present, the returned array is:

- `nan, limit_upper, limit_lower`

Examples:

```python
system = NSystem("LHS 1140")

age = system.star.age_myr
distance = system.distance_pc

print(age)       # e.g. [value, err_pos, err_neg] or [nan, upper, lower]
print(distance)  # same convention
```

## Data behavior

On import, the loader keeps a local CSV cache in `oplanet/data`, removes older archive snapshots, and refreshes stale files automatically. Filters retrieved from the SVO database are also stored locally.

## Notes

- Internet access is needed for Simbad/Vizier queries and first-time archive download.
- The support for [exoplanet.eu](https://exoplanet.eu/catalog/all_fields/) will be added once the database becomes more reliable and *code friendly*.

## License

MIT
