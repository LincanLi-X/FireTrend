# 🔥FireTrend

### FireTrend: Physics-Guided Contrastive Learning for Large-Scale Wildfire Risk Forecasting and Management

<p align="center">
     <img src="assets/firetrend_model.png" alt="FireTrend" width="800px">
</p>

> FireTrend is an end to end multimodal framework for regional spatiotemporal wildfire risk forecasting. It integrates meteorological records, satellite based fire observations, and geospatial information through a multimodal spatial-temporal Transformer encoder. To improve representation quality, FireTrend introduces a multi-view contrastive learning schema that enforces temporal consistency, spatial coherence, and cross-modal alignment in the latent space. On top of the learned representations, a physics-guided module named **PyroCast** models environmental-driven fire spread using directional dynamic convolutions. The entire model is trained jointly with prediction, contrastive, and physics consistency objectives to produce accurate and physically plausible wildfire risk forecasts.

## FireCast Dataset

We provide two well-formatted `FireCast-CA` and `FireCast-FL` subsets of FireCast-US dataset in this package, namely:

* **FireCast-CA**
  `CA_wildfire_grid_ERA5aligned_0p25deg_padded.h5`

* **FireCast-FL**
  `FL_wildfire_grid_ERA5aligned_0p25deg_padded.h5`

Each file contains daily grid-based wildfire risk labels and aligned multi-modal features at a spatial resolution of 0.25° × 0.25°.

The dataset is constructed through a unified pipeline consisting of the following stages:


<p align="center">
     <img src="assets/dataset_flow_chart.png" alt="dataset_flowchart" width="650px">
</p>


### Step 1: Data Acquisition

We collect data from multiple authoritative sources covering wildfire activity, meteorology, and geospatial context.

#### (1) Wildfire Observations

* **Source**: NASA FIRMS active fire products, derived from MODIS sensors onboard Terra and Aqua satellites and VIIRS sensors onboard Suomi-NPP and NOAA-20 platforms.
* **Raw attributes**: Coordinate (latitude,longitude) of active fire points, Detection time, Fire confidence score.


#### (2) Meteorological Data (9-dim)

* **Source:** ERA5 Reanalysis Data(ECMWF).
* **Variables:** Include Near-surface temperature, Relative humidity, Precipitation, Wind speed and wind direction, Soil moisture and drought-related indicators.
* **Temporal Resolution**: Hourly.

#### (3) Geospatial and Topographic Data (5-dim)

* **Sources**: NLCD and LANDFIRE land cover products.
* **Features**: Elevation, Slope and Aspect, Land cover type, Fuel model category.

#### (4) Vegetation and Fuel Indicators (5-dim)

* **Source**: LANDFIRE
* **Temporal resolution**: The original resolution is xx. We interpolated to daily frequency.
* **Features**: Existing Vegetation Type (EVT), Existing Vegetation Cover (EVC), Existing Vegetation Height (EVH), Surface Fuel (SF), Canopy Fuel (CF).


### Step 2: Spatial Standardization

All data modalities are projected to the same **common geographic reference system**:

* **Coordinate system**: WGS84
* **Spatial grid**: Regular latitude-longitude grid at 0.25° × 0.25°
* **Study regions**: Each State of Continental U.S.



### Step 3: Grid-Level Feature Integration

For each grid cell i and day t, all aligned features are concatenated into a unified multi-channel representation:
$$
\mathbf{x}_{i,t} =
\left[
\mathbf{f}^{\text{fire}}_{i,t},
\mathbf{f}^{\text{meteo}}_{i,t},
\mathbf{f}^{\text{geo}}_{i,t}
\right]
$$


### Step 4: Wildfire Risk Label Generation

Wildfire risk labels are derived from NASA fire confidence score (continuous value). Each grid cell is assigned one of three risk levels:
  * Low
  * Medium
  * High
Labels are generated daily based on aggregated fire confidence within the grid cell.



### Dataset Statistics

Key statistics of the released subsets are summarized in the table below.

| Statistic                    | FireCast-CA         | FireCast-FL         |
| ---------------------------- | ------------------- | ------------------- |
| Time Duration                | Feb 2023 – Dec 2025 | Feb 2023 – Dec 2025 |
| Total Grid Regions           | 2,496               | 1,216               |
| Spatial Resolution           | 0.25° × 0.25°       | 0.25° × 0.25°       |
| Temporal Resolution          | One day             | One day             |
| Total Feature Dimension      | 20                  | 20                  |
| Avg. Fire Ratio per Map      | 0.9362              | 0.9075              |
| Avg. Non-Fire Ratio per Map  | 0.0638              | 0.0925              |
| Fire Confidence Distribution | Low / Mid / High    | Low / Mid / High    |


## Getting Started with the FireTrend Model


### Install necessary packages
```sh
pip install -r requirements.txt
```

#### Required Dependencies

```
python>=3.10
numpy>=1.23.0
scipy>=1.10.0
pandas>=2.0.0

# Deep Learning Framework
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
einops>=0.7.0         # safer reshaping
timm>=0.9.12           # pretrained vision transformers
torchinfo>=1.8.0       # model summary & dimension debugging
transformers>=4.39.0 

# Data Handling
h5py>=3.9.0
xarray>=2023.8.0       # spatio-temporal dataset handling
netCDF4>=1.6.4         # dealing with ERA5 meteorological data
rasterio>=1.3.8        # geospatial raster reading
geopandas>=0.14.0
shapely>=2.0.2

# Visualization
matplotlib>=3.8.0
seaborn>=0.12.2
plotly>=5.18.0
cartopy>=0.22.0        # map projections for wildfire visualization

# Training Utilities
tqdm>=4.66.0
tensorboard>=2.15.0
pyyaml>=6.0.2
wandb>=0.16.0           # experiment tracking

# Evaluation & Metrics
scikit-learn>=1.4.0
opencv-python>=4.9.0.80
pytorch-lightning>=2.2.0  # optional training loop simplification

```

### Train FireTrend Model

Use the provided ERA5 + wildfire grid datasets in `0_FireTrend/data` to train the model.

### 1. Configure the Training Parameters

Open `config.yaml` and update the following:
- `data.root_dir` → point to your local dataset folder (e.g., `./data`)
- `data.region` → `"california"` or `"florida"`
- `data.seq_length`, `data.pred_horizon`
- `training.epochs`, `training.batch_size`, `training.lr`
- **Important:** update `model.height` / `model.width` to match your data resolution (California default is `49×53`, Florida likely `64×64`)

### 2. Start Training
Run the main script:
```sh
python main.py --config config.yaml --train --region california
```
Checkpoints will be saved to:
- `./outputs/checkpoints/{region}_model_epoch_XXX.pth`
- `./outputs/checkpoints/{region}_model_best.pth`
Logs are written to `./outputs/logs/`

**Optional:** use the bash helper script:
```sh
bash scripts/train_firetrend.sh
```


### Test FireTrend Model

Use a trained checkpoint to evaluate the model on the same region dataset.

1. **Configure the Testing Parameters**
- Confirm `data.root_dir` and `data.region` in `config.yaml`
- Select the correct checkpoint path (e.g. best model)

2. **Run Testing**
```sh
python main.py --config config.yaml --test --region california --checkpoint ./outputs/checkpoints/california_model_best.pth
```
The script will load the checkpoint, run evaluation, and print metrics (IoU/AUC/F1) to logs.



### Experimental Results


<p align="center">
     <br/> Table 1. Comparison of different methods on the FireCast-California and Florida subsets. The best results are highlighted in bold.
     <br/>
     <img src="assets/Experiment_Table1.png" alt="dataset_flowchart" width="600px">
</p>

<p align="center">
     <br/> Table 2. Performance comparison of different methods on WildfireSpreadTS dataset. The best results are highlighted in bold.
     <br/>
     <img src="assets/Experiment_Table2.png" alt="dataset_flowchart" width="400px">
</p>

<!-- <p align="center">
  <img src="assets/original_images.png" alt="Sim2Real-Fire dataset" width="800px">
  <br/> Examples of satellite images with the real fire areas.
</p>



<br>

<p align="center">
  <img src="assets/examples.png" alt="Sim2Real-Fire dataset" width="800px">
  <br/> Topography, vegetation, fuel, weather, and the satellite data in the Sim2Real-Fire dataset.
</p>

<br>

<p align="center">
     <img src="assets/pie.png" alt="Sim2Real-Fire dataset distribution" width="800px">
     <br/> (a) Distribution of vegetation covers and types. (b) Distribution of fuel types. (c) Distribution topography data. (d) Distribution of weather data.
</p> -->


### PyroCast: Physics-Guided Directional Aware Convolution

#### Mathematical Formulation

#### Performance  w and w/o PyroCast Module

**Performance of Baseline Methods with and without Pyrocast Module**

**FireTrend with and without PyroCast Module**


#### PyroCast (directional-aware Conv) V.S. Standard Convolution)



### Acknowledgements
```
@inproceedings{FireTrend2026,
  title={FireTrend: Physics-Guided Contrastive Learning for Large-Scale Wildfire Risk Forecasting and Management},
  author={Anonymous A, Anonymous B, Anonymous C, Anonymous D, Anonymous E, Anonymous F},
  booktitle={Submission to Conference}
}
```





