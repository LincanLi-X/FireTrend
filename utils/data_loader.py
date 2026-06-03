from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.data_augmentation import apply_firetrend_augmentations


class FireDataset(Dataset):
    """
    FireCast HDF5 loader for FireTrend.
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
        risk_label_protocol: str = "equal_width_0_100",
        split: str | None = None,
        split_ratios: tuple[float, float, float] | list[float] = (0.7, 0.1, 0.2),
        normalization_stats_split: str | None = "train",
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

        self.risk_label_protocol = str(risk_label_protocol).lower()
        if self.risk_label_protocol not in {"equal_width_0_100", "fixed", "train_quantile"}:
            raise ValueError(f"Unsupported risk_label_protocol: {risk_label_protocol}")
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

        self.num_total_samples = n_days - self.seq_length - self.pred_horizon + 1
        if self.num_total_samples <= 0:
            raise ValueError(
                f"Not enough days ({n_days}) for seq_length={seq_length}, pred_horizon={pred_horizon}"
            )

        all_indices = np.arange(self.num_total_samples, dtype=np.int64)
        self.sample_indices = self._split_indices(all_indices, split, split_ratios)
        stats_indices = self._split_indices(all_indices, normalization_stats_split, split_ratios)
        if len(stats_indices) == 0:
            stats_indices = self.sample_indices
        self._build_normalization_stats(stats_indices)
        if self.risk_label_protocol == "train_quantile":
            self.risk_thresholds = self._compute_quantile_thresholds(stats_indices)
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
                "CA_wildfire_grid_ERA5_LANDFIRE_aligned_gzip.h5",
                "CA_wildfire_grid_ERA5_LANDFIRE_aligned_2.h5",
            ]
        elif region_l.startswith("fl") or region_l == "florida":
            candidates = [
                "FL_wildfire_grid_ERA5_LANDFIRE_aligned.h5",
                "FL_wildfire_grid_ERA5_LANDFIRE_aligned_gzip.h5",
                "FL_wildfire_grid_ERA5_LANDFIRE_aligned_2.h5",
            ]
        elif region_l.startswith("or") or region_l == "oregon":
            candidates = [
                "OR_wildfire_grid_ERA5_LANDFIRE_aligned.h5",
                "OR_wildfire_grid_ERA5_LANDFIRE_aligned_gzip.h5",
            ]
        else:
            raise ValueError(f"Unknown region: {region}.")

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

    def _input_day_indices(self, sample_indices: np.ndarray) -> torch.Tensor:
        days = [np.arange(int(i), int(i) + self.seq_length, dtype=np.int64) for i in sample_indices]
        if not days:
            return torch.arange(self.fire_data.shape[0], dtype=torch.long)
        return torch.as_tensor(np.unique(np.concatenate(days)), dtype=torch.long)

    def _build_normalization_stats(self, sample_indices: np.ndarray) -> None:
        day_indices = self._input_day_indices(sample_indices)
        fire_ref = self.fire_data[day_indices]
        meteo_ref = self.meteo_data[day_indices]
        geo_ref = self.geo_data[day_indices]

        fire_min, fire_max = fire_ref.min(), fire_ref.max()
        meteo_min = meteo_ref.amin(dim=(0, 2, 3))
        meteo_max = meteo_ref.amax(dim=(0, 2, 3))
        geo_min = geo_ref.amin(dim=(0, 2, 3))
        geo_max = geo_ref.amax(dim=(0, 2, 3))

        self.x_min = torch.cat([fire_min.view(1), meteo_min, geo_min], dim=0).float()
        self.x_max = torch.cat([fire_max.view(1), meteo_max, geo_max], dim=0).float()
        self.x_range = torch.clamp(self.x_max - self.x_min, min=self.eps)

        self.y_min = torch.tensor(0.0, dtype=torch.float32)
        self.y_max = torch.tensor(100.0, dtype=torch.float32)
        self.y_range = torch.tensor(100.0, dtype=torch.float32)

    def _compute_quantile_thresholds(self, sample_indices: np.ndarray) -> tuple[float, float]:
        target_indices = sample_indices + self.seq_length + self.pred_horizon - 1
        target_scores = self.fire_data[torch.as_tensor(target_indices, dtype=torch.long)].reshape(-1)
        foreground = target_scores[target_scores > 0]
        ref = foreground if foreground.numel() >= 3 else target_scores
        q = torch.quantile(ref.float(), torch.tensor([1.0 / 3.0, 2.0 / 3.0], device=ref.device)).cpu()
        low_mid, mid_high = float(q[0]), float(q[1])
        if not low_mid < mid_high:
            low_mid, mid_high = self.risk_thresholds
        return low_mid, mid_high

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

        drivers = self._build_driver_tensor(meteo_seq)
        future_meteo = self.meteo_data[end_idx : target_idx + 1]
        future_drivers = self._build_driver_tensor(future_meteo)

        y_score = self.fire_data[target_idx]
        y_class = self.score_to_class(y_score)
        y_score_norm = torch.clamp((y_score - self.y_min) / self.y_range, 0.0, 1.0).unsqueeze(0)

        if self.transform:
            x_seq, drivers, y_class, y_score_norm, future_drivers = apply_firetrend_augmentations(
                x_seq, drivers, y_class, y_score_norm, self.transform, future_drivers=future_drivers
            )

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
                "future_drivers": future_drivers,
                "target_index": torch.tensor(target_idx, dtype=torch.long),
            }
        return x_seq, y_class, y_score_norm, drivers

    @staticmethod
    def _relative_humidity_from_dewpoint_temperature(d2m: torch.Tensor, t2m: torch.Tensor) -> torch.Tensor:
        d_c = d2m - 273.15
        t_c = t2m - 273.15
        rh = 100.0 * torch.exp((17.625 * d_c) / (243.04 + d_c) - (17.625 * t_c) / (243.04 + t_c))
        return torch.clamp(rh / 100.0, 0.0, 1.0)

    def _build_driver_tensor(self, meteo_seq: torch.Tensor) -> torch.Tensor:
        u10 = meteo_seq[:, 0:1]
        v10 = meteo_seq[:, 1:2]
        d2m = meteo_seq[:, 2:3]
        t2m = meteo_seq[:, 3:4]
        temperature = self._normalize_meteo_channel(t2m, channel_idx=3)
        relative_humidity = self._relative_humidity_from_dewpoint_temperature(d2m, t2m)
        return torch.cat([u10, v10, temperature, relative_humidity], dim=1)

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
    risk_label_protocol: str = "equal_width_0_100",
    split: str | None = None,
    split_ratios=(0.7, 0.1, 0.2),
    normalization_stats_split: str | None = "train",
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
        risk_label_protocol=risk_label_protocol,
        split=split,
        split_ratios=split_ratios,
        normalization_stats_split=normalization_stats_split,
        return_dict=return_dict,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
