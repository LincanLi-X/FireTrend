
# FireCast Dataset

FireCast dataset integrates multi-source environmental, vegetation, topography, fuel, meteorological, and wildfire risk data into a unified spatial-temporal HDF5 format for wildfire modeling and analysis.

### Repository Dataset Structure

```

data/
│
├── FireCast-CA/
│   └── CA_wildfire_grid_ERA5_LANDFIRE_aligned.h5
│
├── FireCast-FL/
│   └── FL_wildfire_grid_ERA5_LANDFIRE_aligned.h5
|
├── FireCast-OR/
    └── OR_wildfire_grid_ERA5_LANDFIRE_aligned.h5 #Full dataset provided
```

## Full Dataset Availability

⚠️NOTE that the **complete FireCast dataset** is too large to be uploaded directly in the GitHub repository. Therefore, The full dataset is provided via an **anonymized Google Drive link**:


> https://drive.google.com/drive/folders/13_1l7uCxD6APLFWe4ZLUAhM1f_Lnz1ie?usp=sharing




## How to Load the Dataset

Example using `h5py`:

```python
import h5py

with h5py.File("CA_wildfire_grid_ERA5_LANDFIRE_aligned.h5", "r") as f:
  print(list(f.keys()))
  wildfire = f["wildfire_risk"][:]
````


# Dataset Structure

Each `.h5` file contains multiple named datasets (subsets). Each subset corresponds to one feature and is stored independently inside the HDF5 file.All spatial features share the same grid.

### Data Shape

Unless otherwise specified, the `shape of each feature` is **(time, latitude_index_feature_value, longitude_index_feature_value)**.

For the full dataset, the **time span from Feb 2023 to Dec 2025, with daily resolution.**


# Data Sources

The dataset integrates information from:

- **LANDFIRE** (Vegetation, Topography, Fuel)
- **ERA5 Reanalysis** (Meteorological variables)
- **Wildfire Risk Observations** (NASA Satellite Observations)



# Included Features

Below is the complete list of subsets stored inside each `.h5` file:

---

## 🌿 LANDFIRE – Vegetation Features

| Subset Name | Full Name | Description |
|-------------|-----------|-------------|
| `EVH` | Existing Vegetation Height | Height of existing vegetation |
| `EVC` | Existing Vegetation Cover | Percentage vegetation cover |
| `EVT` | Existing Vegetation Type | Categorical vegetation type classification |

**Source:** LANDFIRE Vegetation Layers

---

## 🏔 LANDFIRE – Topography Features

| Subset Name | Full Name | Description |
|-------------|-----------|-------------|
| `Aspect` | Terrain Aspect | Direction slope faces |
| `Slope` | Terrain Slope | Gradient of terrain |
| `Elevation` | Terrain Elevation | Elevation above sea level |

**Source:** LANDFIRE Topographic Layers

---

## 🔥 LANDFIRE – Fuel Features

| Subset Name | Full Name | Description |
|-------------|-----------|-------------|
| `CBD` | Forest Canopy Bulk Density | Canopy fuel density |
| `FVH` | Fuel Vegetation Height | Height of fuel vegetation |
| `FVC` | Fuel Vegetation Cover | Coverage of fuel vegetation |
| `FVT` | Fuel Vegetation Type | Fuel vegetation classification |

**Source:** LANDFIRE Fuel Layers

---

## 🌦 ERA5 – Meteorological Features

All meteorological features are derived from the **ERA5 reanalysis dataset** (ECMWF).

ERA5 originally provides 6-hour resolution data.  
In FireCast, the data is aggregated to **daily resolution**.

| Subset Name | Full Name | Description |
|-------------|-----------|-------------|
| `d2m` | 2m Dewpoint Temperature | Near-surface dew point temperature |
| `msl` | Mean Sea Level Pressure | Surface pressure at sea level |
| `sp` | Surface Pressure | Surface atmospheric pressure |
| `stl1` | Soil Temperature Level 1 | Top-layer soil temperature |
| `swvl1` | Soil Water Volumetric Level 1 | Soil moisture (top layer) |
| `t2m` | 2m Temperature | Near-surface air temperature |
| `u10` | 10m U Wind Component | Zonal wind component |
| `v10` | 10m V Wind Component | Meridional wind component |

**Source:** ERA5 Reanalysis (ECMWF)

---

## 🔥 Wildfire Label

|  Subset Name  |   Description   |
|---------------|-----------------|
| `wildfire_risk` | Daily wildfire risk score |

This serves as the target variable for modeling.

---

## Coordinate Information

| Subset Name | Description |
|-------------|------------|
| `valid_time` | Daily timestamps (YYYY-MM-DD) |
| `latitude` | Latitude grid values |
| `longitude` | Longitude grid values |

## Valid-region mask

Each HDF5 file contains a static `valid_region_mask` dataset with shape
`(latitude, longitude)` and `uint8` values:

| Mask value | Meaning |
|---:|---|
| `1` | Pixel center lies inside the target state's Census land boundary |
| `0` | Invalid target pixel: outside-state, coastal water, or rectangular padding |

The mask is generated from the U.S. Census Bureau 2025 state Cartographic
Boundary File at 1:500,000 scale using the stored latitude/longitude grid
centers (`all_touched=False`). It is static and broadcasts across all time
steps. The continuous `wildfire_risk` dataset is preserved unchanged.

For Low/Medium/High classification, invalid target pixels are assigned the
derived label `-100`. This is an ignore label, not a fourth class. Invalid
pixels must be excluded from classification loss, class-frequency estimates,
and all reported metrics.

Generation and acceptance artifacts:

- `VALID_REGION_MASK_DEVELOPMENT_PLAN.md`: implementation and acceptance plan.
- `valid_region_mask_audit.json`: machine-readable provenance and checksums.
- `../scripts/add_valid_region_masks.py`: reproducible mask generator/updater.
- `../scripts/validate_valid_region_masks.py`: independent acceptance validator.
- `backups_pre_valid_region_mask/`: recoverable pre-mask HDF5 files.



