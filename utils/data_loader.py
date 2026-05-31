from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.data_augmentation import apply_augmentations


class FireDataset(Dataset):
    """
    FireCast HDF5 loader for FireTrend.

    Returns historical risk-score/covariate sequences plus ordinal target
    labels derived from the provider-defined wildfire risk score.
    """

    meteo_vars = ["u10", "v10", "d2m", "t2m", "msl", "sp", "stl1", "swvl1"]
    geo_vars = ["EVH", "EVC", "EVT", "Aspect", "Slope", "Elevation", "CBD", "FVH", "FVC", "FVT"]

    def __init__(
        self,
        data_root,
        region: str = "california",
        seq_length: int = 8,
        pred_horizon: int = 1,
        transform=None,
        normalize: bool = True,
        risk_thresholds: tuple[float, float] | list[float] = (33.3333, 66.6667),
        split: str | None = None,
        split_ratios: tuple[float, float, float] | list[float] = (0.7, 0.1, 0.2),
        return_dict: bool = True,
    ):
        super().__init__()
        self.region = region.lower()
        self.seq_length = int(seq_length)
        self.pred_horizon = int(pred_horizon)
        self.normalize = bool(normalize)
        self.transform = transform
        self.return_dict = bool(return_dict)
        self.eps = 1e-6

        if len(risk_thresholds) != 2:
            raise ValueError("risk_thresholds must contain [low_mid, mid_high].")
        low_mid, mid_high = float(risk_thresholds[0]), float(risk_thresholds[1])
        if not low_mid < mid_high:
            raise ValueError(f"Invalid risk thresholds: {risk_thresholds}")
        self.risk_thresholds = (low_mid, mid_high)

        self.data_file = self._resolve_data_file(data_root, region)
        print(f"Loading FireCast data from {self.data_file}")

        with h5py.File(self.data_file, "r") as f:
            required = ["wildfire_risk"] + self.meteo_vars + self.geo_vars
            missing = [key for key in required if key not in f]
            if missing:
                raise ValueError(f"Missing keys in {self.data_file}: {missing}")

            fire_data = np.array(f["wildfire_risk"], dtype=np.float32)
            meteo_data = np.stack([np.array(f[var], dtype=np.float32) for var in self.meteo_vars], axis=1)
            geo_data = np.stack([np.array(f[var], dtype=np.float32) for var in self.geo_vars], axis=1)

            self.valid_time = np.array(f["valid_time"]) if "valid_time" in f else None
            self.latitude = np.array(f["latitude"]) if "latitude" in f else None
            self.longitude = np.array(f["longitude"]) if "longitude" in f else None

        n_days = min(fire_data.shape[0], meteo_data.shape[0], geo_data.shape[0])
        fire_data = fire_data[:n_days]
        meteo_data = meteo_data[:n_days]
        geo_data = geo_data[:n_days]

        self.fire_data = torch.as_tensor(np.nan_to_num(fire_data), dtype=torch.float32)
        self.meteo_data = torch.as_tensor(np.nan_to_num(meteo_data), dtype=torch.float32)
        self.geo_data = torch.as_tensor(np.nan_to_num(geo_data), dtype=torch.float32)

        self.height = self.fire_data.shape[-2]
        self.width = self.fire_data.shape[-1]

        self._build_normalization_stats()
        self.num_total_samples = n_days - self.seq_length - self.pred_horizon + 1
        if self.num_total_samples <= 0:
            raise ValueError(
                f"Not enough days ({n_days}) for seq_length={seq_length}, pred_horizon={pred_horizon}"
            )

        all_indices = np.arange(self.num_total_samples, dtype=np.int64)
        self.sample_indices = self._split_indices(all_indices, split, split_ratios)
        self.class_counts, self.class_weights = self._compute_class_stats(self.sample_indices)

        print(
            f"Loaded {region}: fire={tuple(self.fire_data.shape)}, "
            f"meteo={tuple(self.meteo_data.shape)}, geo={tuple(self.geo_data.shape)}, "
            f"samples={len(self.sample_indices)}"
        )

    @staticmethod
    def _resolve_data_file(data_root, region: str) -> str:
        root = Path(data_root)
        region_l = region.lower()
        if region_l.startswith("ca") or region_l == "california":
            candidates = [
                "CA_wildfire_grid_ERA5_LANDFIRE_aligned.h5",
                "CA_wildfire_grid_ERA5_LANDFIRE_aligned_2.h5",
            ]
        elif region_l.startswith("fl") or region_l == "florida":
            candidates = [
                "FL_wildfire_grid_ERA5_LANDFIRE_aligned.h5",
                "FL_wildfire_grid_ERA5_LANDFIRE_aligned_2.h5",
            ]
        else:
            raise ValueError(f"Unknown region: {region}. FireCast-OR is intentionally not wired here yet.")

        for name in candidates:
            path = root / name
            if path.exists():
                return str(path)
        raise FileNotFoundError(f"No FireCast HDF5 file found for region={region} under {root}")

    @staticmethod
    def _split_indices(indices: np.ndarray, split: str | None, ratios) -> np.ndarray:
        if split is None or str(split).lower() in {"all", "none"}:
            return indices
        split = str(split).lower()
        ratios = np.asarray(ratios, dtype=np.float64)
        ratios = ratios / ratios.sum()
        n = len(indices)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        if split == "train":
            return indices[:n_train]
        if split in {"val", "valid", "validation"}:
            return indices[n_train : n_train + n_val]
        if split == "test":
            return indices[n_train + n_val :]
        raise ValueError(f"Unsupported split: {split}")

    def _build_normalization_stats(self) -> None:
        fire_min, fire_max = self.fire_data.min(), self.fire_data.max()
        meteo_min = self.meteo_data.amin(dim=(0, 2, 3))
        meteo_max = self.meteo_data.amax(dim=(0, 2, 3))
        geo_min = self.geo_data.amin(dim=(0, 2, 3))
        geo_max = self.geo_data.amax(dim=(0, 2, 3))

        self.x_min = torch.cat([fire_min.view(1), meteo_min, geo_min], dim=0).float()
        self.x_max = torch.cat([fire_max.view(1), meteo_max, geo_max], dim=0).float()
        self.x_range = torch.clamp(self.x_max - self.x_min, min=self.eps)

        self.y_min = torch.tensor(0.0, dtype=torch.float32)
        self.y_max = torch.tensor(100.0, dtype=torch.float32)
        self.y_range = torch.tensor(100.0, dtype=torch.float32)

    def score_to_class(self, score_map: torch.Tensor) -> torch.Tensor:
        low_mid, mid_high = self.risk_thresholds
        cls = torch.zeros_like(score_map, dtype=torch.long)
        cls[(score_map >= low_mid) & (score_map < mid_high)] = 1
        cls[score_map >= mid_high] = 2
        return cls

    def _compute_class_stats(self, sample_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        target_indices = sample_indices + self.seq_length + self.pred_horizon - 1
        target_scores = self.fire_data[torch.as_tensor(target_indices, dtype=torch.long)]
        target_cls = self.score_to_class(target_scores)
        counts = torch.bincount(target_cls.reshape(-1), minlength=3).float()
        weights = counts.sum() / (counts.clamp_min(1.0) * 3.0)
        weights = weights / weights.mean().clamp_min(self.eps)
        return counts, weights

    def __len__(self) -> int:
        return int(len(self.sample_indices))

    def __getitem__(self, idx: int):
        base_idx = int(self.sample_indices[idx])
        end_idx = base_idx + self.seq_length
        target_idx = end_idx + self.pred_horizon - 1

        fire_seq = self.fire_data[base_idx:end_idx].unsqueeze(1)
        meteo_seq = self.meteo_data[base_idx:end_idx]
        geo_seq = self.geo_data[base_idx:end_idx]
        x_seq = torch.cat([fire_seq, meteo_seq, geo_seq], dim=1)

        # Raw wind preserves direction. Temperature/humidity proxy are min-max
        # normalized for the alpha modulation term in PyroCast.
        u10 = meteo_seq[:, 0:1]
        v10 = meteo_seq[:, 1:2]
        humidity_proxy = self._normalize_meteo_channel(meteo_seq[:, 2:3], channel_idx=2)
        temperature = self._normalize_meteo_channel(meteo_seq[:, 3:4], channel_idx=3)
        drivers = torch.cat([u10, v10, temperature, humidity_proxy], dim=1)

        y_score = self.fire_data[target_idx]
        y_class = self.score_to_class(y_score)
        y_score_norm = torch.clamp((y_score - self.y_min) / self.y_range, 0.0, 1.0).unsqueeze(0)

        if self.transform:
            x_seq = apply_augmentations(x_seq.unsqueeze(0), self.transform).squeeze(0)

        if self.normalize:
            x_min = self.x_min.view(1, -1, 1, 1)
            x_range = self.x_range.view(1, -1, 1, 1)
            x_seq = torch.clamp((x_seq - x_min) / x_range, 0.0, 1.0)

        if self.return_dict:
            return {
                "x": x_seq,
                "y_class": y_class,
                "y_score": y_score_norm,
                "drivers": drivers,
                "target_index": torch.tensor(target_idx, dtype=torch.long),
            }
        return x_seq, y_class, y_score_norm, drivers

    def _normalize_meteo_channel(self, tensor: torch.Tensor, channel_idx: int) -> torch.Tensor:
        ch_min = self.x_min[1 + channel_idx].view(1, 1, 1, 1)
        ch_range = self.x_range[1 + channel_idx].view(1, 1, 1, 1)
        return torch.clamp((tensor - ch_min) / ch_range, 0.0, 1.0)

    def denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_range + self.y_min


def create_dataloader(
    data_root,
    region: str = "california",
    seq_length: int = 8,
    pred_horizon: int = 1,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    normalize: bool = True,
    risk_thresholds=(33.3333, 66.6667),
    split: str | None = None,
    split_ratios=(0.7, 0.1, 0.2),
    return_dict: bool = True,
):
    dataset = FireDataset(
        data_root=data_root,
        region=region,
        seq_length=seq_length,
        pred_horizon=pred_horizon,
        transform=transform,
        normalize=normalize,
        risk_thresholds=risk_thresholds,
        split=split,
        split_ratios=split_ratios,
        return_dict=return_dict,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
