import torch
import random
import numpy as np

def apply_augmentations(x, config):
    """
    Apply data augmentations consistently across time and modalities.

    Args:
        x (torch.Tensor): Input tensor [B, T, C, H, W]
        config (dict): augmentation settings from config.yaml
    Returns:
        torch.Tensor: augmented tensor [B, T, C, H, W]
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    aug_cfg = config.get("augmentations", {})

    # Random horizontal flip
    if aug_cfg.get("random_flip", False) and random.random() > 0.5:
        x = torch.flip(x, dims=[-1])  # horizontal flip on W axis

    # Random vertical flip
    if aug_cfg.get("random_vertical_flip", False) and random.random() > 0.5:
        x = torch.flip(x, dims=[-2])  # vertical flip on H axis

    # Random rotation (90, 180, 270)
    if aug_cfg.get("random_rotation", False) and random.random() > 0.5:
        k = random.choice([1, 2, 3])
        x = torch.rot90(x, k=k, dims=[-2, -1])

    # Temporal shuffle (only if explicitly enabled)
    if aug_cfg.get("temporal_shuffle", False) and random.random() > 0.5:
        perm = torch.randperm(x.shape[1])
        x = x[:, perm, :, :, :]

    # Add Gaussian noise
    if aug_cfg.get("gaussian_noise", False) and random.random() > 0.5:
        noise_std = aug_cfg.get("noise_std", 0.01)
        x = x + noise_std * torch.randn_like(x)

    return x


def apply_firetrend_augmentations(x, drivers, y_class, y_score, config, future_drivers=None):
    """
    Apply spatial augmentations to FireTrend inputs, targets, and drivers.

    Args:
        x: [T, C, H, W]
        drivers: [T, 4, H, W] with [u, v, temperature, humidity]
        future_drivers: optional [Horizon, 4, H, W]
        y_class: [H, W]
        y_score: [1, H, W]
    """
    aug_cfg = config.get("augmentations", {}) if config else {}
    if aug_cfg.get("random_rotation", False):
        raise ValueError(
            "random_rotation is disabled for FireTrend samples because wind-vector "
            "rotation and target alignment must be handled explicitly."
        )
    if aug_cfg.get("temporal_shuffle", False):
        raise ValueError("temporal_shuffle is incompatible with chronological forecasting targets.")

    if aug_cfg.get("random_flip", False) and random.random() > 0.5:
        x = torch.flip(x, dims=[-1])
        drivers = torch.flip(drivers, dims=[-1])
        drivers[:, 0:1] = -drivers[:, 0:1]
        if future_drivers is not None:
            future_drivers = torch.flip(future_drivers, dims=[-1])
            future_drivers[:, 0:1] = -future_drivers[:, 0:1]
        y_class = torch.flip(y_class, dims=[-1])
        y_score = torch.flip(y_score, dims=[-1])

    if aug_cfg.get("random_vertical_flip", False) and random.random() > 0.5:
        x = torch.flip(x, dims=[-2])
        drivers = torch.flip(drivers, dims=[-2])
        drivers[:, 1:2] = -drivers[:, 1:2]
        if future_drivers is not None:
            future_drivers = torch.flip(future_drivers, dims=[-2])
            future_drivers[:, 1:2] = -future_drivers[:, 1:2]
        y_class = torch.flip(y_class, dims=[-2])
        y_score = torch.flip(y_score, dims=[-2])

    if aug_cfg.get("gaussian_noise", False) and random.random() > 0.5:
        noise_std = aug_cfg.get("noise_std", 0.01)
        x = x + noise_std * torch.randn_like(x)

    return x, drivers, y_class, y_score, future_drivers


class Normalize:
    """
    Normalize input data based on provided mean and std.
    Works across temporal dimension.

    Example:
        norm = Normalize(mean=[0.5], std=[0.2])
        x = norm(x)
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, 1, -1, 1, 1)

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return (x - self.mean.to(x.device)) / (self.std.to(x.device) + 1e-6)


def normalize_batch(x, mean=None, std=None):
    """Quick normalization for tensors"""
    if mean is None or std is None:
        mean = x.mean(dim=(0, 1, 3, 4), keepdim=True)
        std = x.std(dim=(0, 1, 3, 4), keepdim=True)
    return (x - mean) / (std + 1e-6)
