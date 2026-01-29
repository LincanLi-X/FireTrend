import os
import torch
import numpy as np
import h5py
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from utils.data_augmentation import apply_augmentations, normalize_batch


class FireDataset(Dataset):
    """
    FireTrend Dataset Loader (Two-Scale Temporal Encoder version)
    Combines wildfire history (.h5) + ERA5 meteorological (.nc) + ERA5 geographic (.nc)
    Supports ERA5 6-hourly frequency with 4 sub-steps per day.
    """

    def __init__(
        self,
        data_root,
        region="California",
        seq_length=4,
        pred_horizon=1,
        transform=None,
        normalize=True,
    ):
        super().__init__()

        self.region = region
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.normalize = normalize
        self.transform = transform

        # ------------------------------------------------------------
        # File paths
        # ------------------------------------------------------------
        if region.lower().startswith("ca"):
            self.fire_file = os.path.join(data_root, "CA_wildfire_grid_ERA5aligned_0p25deg_padded.h5")
            self.era5_file = os.path.join(data_root, "ERA5_CA_2023Feb_2025Oct_0.25deg_Full.nc")
        elif region.lower().startswith("fl"):
            self.fire_file = os.path.join(data_root, "FL_wildfire_grid_ERA5aligned_0p25deg_padded.h5")
            self.era5_file = os.path.join(data_root, "ERA5_FL_2023Feb_2025Oct_0.25deg_Full.nc")
        else:
            raise ValueError(f"Unknown region: {region}")

        # ------------------------------------------------------------
        # Load wildfire history (.h5)
        # ------------------------------------------------------------
        print(f"🔥 Loading wildfire data from {self.fire_file}")
        with h5py.File(self.fire_file, "r") as f:
            fire_keys = list(f.keys())
            if len(fire_keys) == 0:
                raise ValueError(f"No datasets found in {self.fire_file}")
            fire_key = fire_keys[0]
            self.fire_data = np.array(f[fire_key])  # [T_days, H, W]

        n_days = self.fire_data.shape[0]
        self.fire_times = np.arange(n_days)

        # ------------------------------------------------------------
        # Load ERA5 meteorological + geographic features (.nc)
        # ------------------------------------------------------------
        print(f"Loading ERA5 data from {self.era5_file}")
        self.era5_ds = xr.open_dataset(self.era5_file)

        # Handle missing or non-standard time dimension safely
        dims = list(self.era5_ds.dims.keys())
        if "time" in dims:
            print("ERA5 dataset already has 'time' dimension. Skipping rename.")
        elif "valid_time" in dims:
            print("Renaming 'valid_time' → 'time'.")
            self.era5_ds = self.era5_ds.rename({"valid_time": "time"})
        else:
            first_dim = dims[0]
            print(f"⚠️ Unknown time dimension '{first_dim}', renaming it → 'time'.")
            self.era5_ds = self.era5_ds.rename({first_dim: "time"})

        # Expected ERA5 features
        all_vars = ['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'stl1', 'swvl1']
        for v in all_vars:
            if v not in self.era5_ds.data_vars:
                raise ValueError(f"Missing expected variable '{v}' in ERA5 dataset")

        self.meteo_vars = all_vars[:6]
        self.geo_vars = all_vars[6:]

        # ------------------------------------------------------------
        # Temporal grouping: ERA5 6-hourly → group every 4 records as 1 day
        # ------------------------------------------------------------
        n_era5_steps = self.era5_ds.sizes["time"]
        if n_era5_steps % 4 != 0:
            print(f"⚠️ Warning: ERA5 steps ({n_era5_steps}) not divisible by 4, truncating tail")
            n_era5_steps = (n_era5_steps // 4) * 4
            self.era5_ds = self.era5_ds.isel(time=slice(0, n_era5_steps))

        n_days_era5 = n_era5_steps // 4
        if n_days_era5 != n_days:
            print(f"⚠️ Wildfire days ({n_days}) ≠ ERA5_days ({n_days_era5}), aligning to min length")
            n_days = min(n_days, n_days_era5)
            self.fire_data = self.fire_data[:n_days]

        # Extract ERA5 variables into array
        arr = np.stack([self.era5_ds[v].values for v in all_vars], axis=1)  # [T_era5, 8, H, W]
        H, W = arr.shape[-2], arr.shape[-1]
        arr = arr[:n_days * 4]  # ensure divisible alignment
        arr = arr.reshape(n_days, 4, 8, H, W)  # [days, 4, 8, H, W]
        self.era5_hourly = torch.tensor(np.nan_to_num(arr), dtype=torch.float32)  # 6-hour steps

        # Static geo features (repeated daily)
        self.geo_data = torch.tensor(
            np.stack([self.era5_ds[v].isel(time=0).values for v in self.geo_vars], axis=0),
            dtype=torch.float32
        )  # [2, H, W]

        # Wildfire
        self.fire_data = torch.tensor(np.nan_to_num(self.fire_data), dtype=torch.float32)

        self.num_samples = n_days - seq_length - pred_horizon
        print(f"✅ Loaded {region}: ERA5 hourly shape={self.era5_hourly.shape}, Fire shape={self.fire_data.shape}")

    # ------------------------------------------------------------
    # Dataset Access
    # ------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            x_seq: [T, 9, H, W]
            y_target: [1, H, W]
        """
        # Past wildfire [T, H, W]
        fire_seq = self.fire_data[idx:idx + self.seq_length]

        # Past ERA5 hourly [T, 4, 6, H, W]
        meteo_seq = self.era5_hourly[idx:idx + self.seq_length, :, :6, :, :]

        # Static geo features [T, 4, 2, H, W]
        geo_feat = self.era5_hourly[idx:idx + self.seq_length, :, 6:, :, :]

        # Combine wildfire + ERA5 + geo per time step
        fire_seq_expanded = fire_seq.unsqueeze(1).unsqueeze(2).repeat(1, 4, 1, 1, 1)  # [T, 4, 1, H, W]
        x_seq = torch.cat([fire_seq_expanded, meteo_seq, geo_feat], dim=2)  # [T, 4, 9, H, W]

        # # Collapse 4×6-hour substeps into a daily feature tensor to match model input
        # # Expected by model: [B, T, C, H, W]
        # x_seq = x_seq.mean(dim=1)  # [T, 9, H, W]

        # Target wildfire
        y_target = self.fire_data[idx + self.seq_length + self.pred_horizon - 1]

        # Optional augment & normalization
        if self.transform:
            x_seq = apply_augmentations(x_seq.unsqueeze(0), self.transform).squeeze(0)
        if self.normalize:
            x_seq = normalize_batch(x_seq.unsqueeze(0)).squeeze(0)

        return x_seq, y_target.unsqueeze(0)  # [T, 4, 9, H, W], [1, H, W]


# ================================================================
# Dataloader Builder
# ================================================================
def create_dataloader(
    data_root,
    region="California",
    seq_length=4,
    pred_horizon=1,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    transform=None,
    normalize=True,
):
    dataset = FireDataset(
        data_root=data_root,
        region=region,
        seq_length=seq_length,
        pred_horizon=pred_horizon,
        transform=transform,
        normalize=normalize,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
