
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

with h5py.File("FL_wildfire_grid_ERA5_LANDFIRE_aligned.h5", "r") as f:
  print(list(f.keys()))
  wildfire = f["wildfire_risk"][:]
````


# Dataset Structure

Each `.h5` file contains multiple named datasets (subsets). Each subset corresponds to one feature and is stored independently inside the HDF5 file.All spatial features share the same grid.

### Data Shape

Unless otherwise specified, the `shape of each feature` is **(time_step, latitude_grid_count, longitude_grid_count)**.

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





