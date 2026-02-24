import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from utils.data_augmentation import apply_augmentations


class FireDataset(Dataset):
    """
    FireTrend dataset loader for FireCast v2.
    Uses a single daily-resolution HDF5 file per region.
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
        self.eps = 1e-6

        # ------------------------------------------------------------
        # File paths (single HDF5 per region)
        # ------------------------------------------------------------
        if region.lower().startswith("ca"):
            self.data_file = os.path.join(data_root, "CA_wildfire_grid_ERA5_LANDFIRE_aligned_2.h5")
        elif region.lower().startswith("fl"):
            self.data_file = os.path.join(data_root, "FL_wildfire_grid_ERA5_LANDFIRE_aligned_2.h5")
        else:
            raise ValueError(f"Unknown region: {region}")

        # Meteorology branch (ERA5, 8 channels)
        self.meteo_vars = ["u10", "v10", "d2m", "t2m", "msl", "sp", "stl1", "swvl1"]
        # Geo/LANDFIRE branch (10 channels): vegetation + topography + fuel
        self.geo_vars = ["EVH", "EVC", "EVT", "Aspect", "Slope", "Elevation", "CBD", "FVH", "FVC", "FVT"]

        print(f"🔥 Loading FireCast v2 data from {self.data_file}")
        with h5py.File(self.data_file, "r") as f:
            required = ["wildfire_risk"] + self.meteo_vars + self.geo_vars
            missing = [k for k in required if k not in f]
            if missing:
                raise ValueError(f"Missing keys in {self.data_file}: {missing}")

            self.fire_data = np.array(f["wildfire_risk"], dtype=np.float32)  # [T, H, W]
            meteo_data = np.stack([np.array(f[v], dtype=np.float32) for v in self.meteo_vars], axis=1)  # [T, 8, H, W]
            geo_data = np.stack([np.array(f[v], dtype=np.float32) for v in self.geo_vars], axis=1)      # [T, 10, H, W]

            self.valid_time = np.array(f["valid_time"]) if "valid_time" in f else None
            self.latitude = np.array(f["latitude"]) if "latitude" in f else None
            self.longitude = np.array(f["longitude"]) if "longitude" in f else None

        n_days = self.fire_data.shape[0]
        if meteo_data.shape[0] != n_days or geo_data.shape[0] != n_days:
            n_days = min(n_days, meteo_data.shape[0], geo_data.shape[0])
            self.fire_data = self.fire_data[:n_days]
            meteo_data = meteo_data[:n_days]
            geo_data = geo_data[:n_days]

        self.fire_data = torch.tensor(np.nan_to_num(self.fire_data), dtype=torch.float32)
        self.meteo_data = torch.tensor(np.nan_to_num(meteo_data), dtype=torch.float32)
        self.geo_data = torch.tensor(np.nan_to_num(geo_data), dtype=torch.float32)

        # ------------------------------------------------------------
        # Unified normalization stats
        # X channels: wildfire(1) + meteo(8) + geo(10) = 19
        # Y target: wildfire_risk in [0, 100]
        # ------------------------------------------------------------
        fire_min, fire_max = self.fire_data.min(), self.fire_data.max()
        meteo_min = self.meteo_data.amin(dim=(0, 2, 3))
        meteo_max = self.meteo_data.amax(dim=(0, 2, 3))
        geo_min = self.geo_data.amin(dim=(0, 2, 3))
        geo_max = self.geo_data.amax(dim=(0, 2, 3))

        self.x_min = torch.cat([fire_min.view(1), meteo_min, geo_min], dim=0).float()  # [9]
        self.x_max = torch.cat([fire_max.view(1), meteo_max, geo_max], dim=0).float()  # [9]
        self.x_range = torch.clamp(self.x_max - self.x_min, min=self.eps)              # [9]

        # Fixed target scaling per task definition: wildfire_risk in [0, 100]
        self.y_min = torch.tensor(0.0, dtype=torch.float32)
        self.y_max = torch.tensor(100.0, dtype=torch.float32)
        self.y_range = torch.tensor(100.0, dtype=torch.float32)

        self.height = self.fire_data.shape[-2]
        self.width = self.fire_data.shape[-1]

        self.num_samples = n_days - seq_length - pred_horizon + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough days ({n_days}) for seq_length={seq_length}, pred_horizon={pred_horizon}"
            )

        print(
            f"✅ Loaded {region}: fire={tuple(self.fire_data.shape)}, "
            f"meteo={tuple(self.meteo_data.shape)}, geo={tuple(self.geo_data.shape)}"
        )

    # ------------------------------------------------------------
    # Dataset Access
    # ------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            x_seq: [T, 19, H, W]
            y_target: [1, H, W]
        """
        fire_seq = self.fire_data[idx:idx + self.seq_length]      # [T, H, W]
        meteo_seq = self.meteo_data[idx:idx + self.seq_length]    # [T, 8, H, W]
        geo_seq = self.geo_data[idx:idx + self.seq_length]        # [T, 10, H, W]

        fire_seq = fire_seq.unsqueeze(1)                           # [T, 1, H, W]
        x_seq = torch.cat([fire_seq, meteo_seq, geo_seq], dim=1)  # [T, 19, H, W]

        # Target wildfire
        y_target = self.fire_data[idx + self.seq_length + self.pred_horizon - 1]

        # Optional augment
        if self.transform:
            x_seq = apply_augmentations(x_seq.unsqueeze(0), self.transform).squeeze(0)
        # Unified normalization:
        # - Inputs use per-channel min-max normalization
        # - Target wildfire_risk uses fixed [0,100] -> [0,1]
        if self.normalize:
            x_min = self.x_min.view(1, -1, 1, 1)      # [1, 19, 1, 1]
            x_range = self.x_range.view(1, -1, 1, 1)  # [1, 19, 1, 1]
            x_seq = (x_seq - x_min) / x_range
            x_seq = torch.clamp(x_seq, 0.0, 1.0)

            y_target = (y_target - self.y_min) / self.y_range
            y_target = torch.clamp(y_target, 0.0, 1.0)

        return x_seq, y_target.unsqueeze(0)  # [T, 19, H, W], [1, H, W]

    def denormalize_y(self, y):
        """Map normalized wildfire_risk [0,1] back to real scale [0,100]."""
        return y * self.y_range + self.y_min


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
